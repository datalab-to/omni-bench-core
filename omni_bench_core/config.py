"""A run as a YAML file: the question a benchmark is asked and the system that answers it, every key written out.

- `Config`: a config, checked for its shape: every key given, of its kind, and nothing else;
- `load` and `parse`: a config from its YAML file with each `--set path=value` applied (`apply`), or from a dict;
- `scorers`, `system` and `benchmark`: what a config asks, built from a package's own scorers and systems, each
  checked by its own class;
- `plan`: the run a config asks of a benchmark (`Benchmark.plan`); and
- `resolved` and `write`: the config as a run keeps it, the system's options as it takes them.

    dataset: hf://datasets/datalab-to/omni_parse_bench   a folder or an hf:// address
    manifest: manifest.parquet     which of its rows, with which files
    scorers:                       what grades the run, each with its options (a list for several of one)
      unit_tests: {}
    rows:                          which rows: `sample` N evenly spaced, `limit` the first N (0 for no limit), `only`
      sample: 0                    these ids (null for any)
      limit: 0
      only: null
    timeout: 1800.0                seconds each row may take
    workers: 0                     rows in flight at once; 0 for the system's own
    system:                        the one system, and its options
      datalab:
        mode: balanced

A config never refers to another file and nothing in it has a default, so the file says the whole run. `--set`
changes a value the file has, at its path (`system.datalab.mode=accurate`); a path the file hasn't is refused, so a
mistyped one never adds a key. The value is read as YAML: `150`, `true`, `[a, b]`, `{exact_text: {}}`. A package adds
what is its own by subclassing `Config` (another key, a check across keys); its scorers and systems are named by it.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pydantic

from omni_bench_core import adapter, scorer, task
from omni_bench_core import benchmark as core

STRICT = pydantic.ConfigDict(extra="forbid", frozen=True, strict=True)
SAID = {"missing": "missing; a config gives every key", "extra_forbidden": "not a key a config has"}
FILE = "config.yaml"  # a run's config, in its folder


class Rows(pydantic.BaseModel):
    """Which rows a run asks for: `sample` N evenly spaced, `limit` the first N (0 for either means every row), or
    `only` these ids (null for any)."""

    model_config = STRICT
    sample: pydantic.NonNegativeInt
    limit: pydantic.NonNegativeInt
    only: list[str] | None


class Config(pydantic.BaseModel):
    """One run, written out whole (the module docstring says each key). `scorers` and `system` name what the package
    running it has: `scorers`, `system` and `benchmark` build them, each checked by its own class."""

    model_config = STRICT
    dataset: str = pydantic.Field(min_length=1)
    manifest: str = pydantic.Field(min_length=1)
    scorers: dict[str, dict[str, Any] | list[dict[str, Any]]] = pydantic.Field(min_length=1)
    rows: Rows
    timeout: pydantic.PositiveFloat
    workers: pydantic.NonNegativeInt
    system: dict[str, dict[str, Any]]

    @pydantic.field_validator("system")
    @classmethod
    def _one(cls, given: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if len(given) != 1:
            raise ValueError(f"system is one system and its options, such as {{datalab: {{mode: balanced}}}}; "
                             f"not {len(given)}")
        return given


def load[C: Config](path: str | Path, sets: list[str] | None = None, model: type[C] = Config, **context: Any) -> C:
    """The config in the YAML file at `path`, each of `sets` (`path=value`) applied, checked as `model` (a package's
    subclass, given `context` while it checks)."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no config at {p}; give a YAML file")
    try:
        c = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{p} isn't YAML: {exc}") from None
    if not isinstance(c, dict):
        # Note: a ValueError, as for every mistake in a user's file.
        raise ValueError(f"{p}: a config is a mapping of {', '.join(model.model_fields)}")  # noqa: TRY004
    return parse(apply(c, sets or []), str(p), model, **context)


def parse[C: Config](c: dict[str, Any], where: str, model: type[C] = Config, **context: Any) -> C:
    """`c`, checked as `model`; each thing wrong with it said at its key, with the value."""
    try:
        return model.model_validate(c, context=context)
    except pydantic.ValidationError as exc:
        said = []
        for e in exc.errors():
            at = ".".join(map(str, e["loc"]))
            msg = SAID.get(e["type"], e["msg"]).removeprefix("Value error, ")
            if e["type"] not in (*SAID, "value_error"):
                msg += f"; it is {e['input']!r}"
            said.append(f"{at}: {msg}" if at else msg)
        raise ValueError(f"{where}: " + "; ".join(said)) from None


def apply(c: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    """`c` with each of `sets`, `path=value`, in place of the value at its dotted path. A path `c` hasn't is
    refused, with the nearest it has; the value is read as YAML."""
    import yaml

    out = copy.deepcopy(c)
    for s in sets:
        path, eq, raw = s.partition("=")
        if not eq or not path:
            raise ValueError(f"--set {s!r}: write path=value, such as rows.sample=5")
        node, here, rest = out, [], path.split(".")
        while rest:
            # Note: the longest key that matches, so a key with a dot in it (a model id) is reachable.
            n = next((n for n in range(len(rest), 0, -1) if isinstance(node, dict) and ".".join(rest[:n]) in node), 0)
            if not n:
                k = rest[0]
                have = sorted(node) if isinstance(node, dict) else []
                near = difflib.get_close_matches(k, have, n=1)
                raise ValueError(
                    f"--set {path}: the config has no {'.'.join([*here, k])}"
                    + (f"; did you mean {'.'.join([*here, near[0]])}?" if near
                       else f"; {'.'.join(here) or 'it'} has {', '.join(have) or 'nothing'}")
                )
            k, rest = ".".join(rest[:n]), rest[n:]
            if rest:
                node, here = node[k], [*here, k]
            else:
                node[k] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def scorers(c: Config, known: Mapping[str, type[scorer.Scorer]]) -> list[scorer.Scorer]:
    """The config's scorers, from those the package has (`known`, by name), each built with its options; a list of
    options is several configurations of one scorer."""
    if unknown := sorted(set(c.scorers) - set(known)):
        raise ValueError(f"no scorer {unknown[0]!r}; the scorers are {', '.join(sorted(known))}")
    return [adapter.configure(known[s], o)
            for s, opts in c.scorers.items() for o in (opts if isinstance(opts, list) else [opts])]


def system(c: Config, make: Callable[[str, dict[str, Any]], adapter.Adapter]) -> adapter.Adapter:
    """The config's system, built by the package's `make(name, options)`, which refuses a name it hasn't."""
    ((name, opts),) = c.system.items()
    return make(name, opts)


def benchmark(c: Config, t: task.Task, known: Mapping[str, type[scorer.Scorer]], *, dataset: str | Path | None = None,
              dataset_name: str | None = None) -> core.Benchmark:
    """The benchmark a config asks: task `t` on its dataset's manifest (`dataset` in place of the config's, a copy of
    it named `dataset_name` in each run's question), graded by its scorers."""
    return core.Benchmark(t, dataset or c.dataset, scorers(c, known), c.manifest, dataset_name=dataset_name)


def plan(c: Config, b: core.Benchmark, s: adapter.Adapter, out: str | Path, **kw: Any) -> core.Plan:
    """The run the config asks of `b` with system `s` under `out`: its rows, timeout and workers; `kw` the rest of
    `Benchmark.plan`'s (`resume`, `retry_failed`, `launch`)."""
    return b.plan(s, out, limit=c.rows.limit, sample=c.rows.sample, only=c.rows.only, timeout=c.timeout,
                  workers=c.workers, **kw)


def resolved(c: Config, s: adapter.Adapter) -> dict[str, Any]:
    """The config as plain JSON, its system's options as the system takes them (its defaults filled in)."""
    return {**c.model_dump(mode="json"), "system": {next(iter(c.system)): s.options()}}


def write(c: Config, s: adapter.Adapter, folder: str | Path) -> Path:
    """The config resolved (`resolved`), as YAML, in `folder`'s `FILE`, so the run says the whole of what it ran
    and `load` reruns it."""
    import yaml

    path = Path(folder) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved(c, s), sort_keys=False, width=1000, allow_unicode=True))
    return path


def template(make: Callable[[str, dict[str, Any]], adapter.Adapter], name: str, *, dataset: str, manifest: str,
             scorers: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """A config to start from: system `name` at its defaults (built by `make`, so every option is written out), on
    `dataset`'s `manifest`, graded by `scorers`, every row, an 1800 s timeout and the system's own workers."""
    s = make(name, {})
    return {"dataset": dataset, "manifest": manifest, "scorers": dict(scorers),
            "rows": {"sample": 0, "limit": 0, "only": None}, "timeout": 1800.0, "workers": 0,
            "system": {name: s.options()}}

