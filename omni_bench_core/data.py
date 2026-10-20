"""Datasets: how they are found, downloaded and read.

A dataset is a folder. It can also be a HuggingFace dataset, which is downloaded whole the first time it is
used. Either way, it holds three things:

    manifest.parquet       one row per thing to predict: its id, its suite, its input files and its gold file
    <any folder>/<file>    the input files, with any names; several rows may share one
    <any folder>/<id>.json each row's gold, named after its row

A task says which columns its rows have, and turns each row into its own frozen dataclass. This module
checks only what every task needs:

- each id is a file name, and no two ids are the same ignoring case, because an id names the row's files
  (`predictions/<id>.json`) and some disks ignore case;
- each suite is a string. A missing, empty or null suite becomes "other";
- each path is relative and inside the dataset, so the dataset can be moved as one folder. A gold file is
  named `<id>.json`, so a gold can't be graded against another row; and
- every other column is kept, unread, in the row's `extra`, so a dataset can carry more for scorers of
  its own.

Everywhere in this package, `dataset` is either a folder's path or `hf://datasets/<org>/<name>[@<revision>]`.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from omni_bench_core import errors

log = logging.getLogger(__name__)

HF_PREFIX = "hf://datasets/"
MANIFEST = "manifest.parquet"
NO_SUITE = "other"
DOWNLOAD_ATTEMPTS = 5


class Row(Protocol):
    """What the shared code reads of a row. The task's own row type holds everything else.

    A row has two fields that every task shares:

    - `id`: names the row's files (its prediction, record and gold), so an answer is never graded against
      another row's gold; and
    - `suite`: the kind of row it is, such as a scan, a form or a language. Summaries are broken down by
      suite, so a drop on one kind of row isn't hidden in the average."""

    @property
    def id(self) -> str: ...

    @property
    def suite(self) -> str: ...


def hf_dataset(dataset: str | Path) -> tuple[str, str | None] | None:
    """The repo and revision of an `hf://` address, or None when `dataset` is a folder."""
    s = str(dataset)
    form = f"{HF_PREFIX}<org>/<name>[@<revision>]"
    if isinstance(dataset, Path) and s.startswith("hf:"):
        raise errors.UsageError(f"the dataset {s!r} was made a Path, which folds the '//' out of an address; "
                                f"pass it as a str: {form}")
    if not s.startswith("hf://"):
        return None
    org, _, name = s.removeprefix(HF_PREFIX).partition("/")
    name, at, revision = name.partition("@")
    if not s.startswith(HF_PREFIX) or not org or not name or "/" in name or (at and not revision):
        raise errors.UsageError(f"{s!r} does not name a HuggingFace dataset; write it as {form}")
    return f"{org}/{name}", revision or None


@functools.cache
def download_dataset(dataset: str | Path) -> Path:
    """The dataset as a folder on disk.

    A folder is returned as it is. A HuggingFace dataset is downloaded whole, at one commit, into the
    HuggingFace cache, even when only a few rows are run: a run's folder and its grading need the whole
    dataset. A later run finds it there and downloads nothing, and a second call in the same process doesn't
    even ask the Hub: planning, predicting and grading all need the folder."""
    if (at := hf_dataset(dataset)) is None:
        root = Path(dataset).expanduser()
        if not root.is_dir():
            raise errors.UsageError(f"no dataset folder at {root}; pass a folder or {HF_PREFIX}<org>/<name>")
        return root
    repo, revision = at
    huggingface_hub = hub()
    from huggingface_hub import errors as hf_errors

    cached = _cached(repo, revision)
    # Note: a cached dataset still asks the Hub for its latest commit, but shows no progress bars for files
    # it already has.
    if cached: huggingface_hub.utils.disable_progress_bars()
    if huggingface_hub.get_token() is None and not cached:
        log.info("downloading %s, all of it, once; later runs use the copy in the HuggingFace cache. Tip: set "
                 "HF_TOKEN (or run `hf auth login`) to download faster and avoid the Hub's rate limits", dataset)

    # Note: we try again because a dataset is hundreds of files, the Hub drops some requests under that load,
    # and a new attempt resumes where the last one stopped. A missing repo or missing access fails at once,
    # because it would fail the same way every time.
    try:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                return Path(huggingface_hub.snapshot_download(repo, repo_type="dataset", revision=revision,
                                                              max_workers=4))
            except (hf_errors.RepositoryNotFoundError, hf_errors.GatedRepoError,
                    hf_errors.RevisionNotFoundError) as exc:
                raise errors.UsageError(f"can't read dataset {dataset}: {type(exc).__name__}; check the repo and "
                                        "revision, and that you're logged in with access (hf auth login)") from None
            except (hf_errors.LocalEntryNotFoundError, hf_errors.OfflineModeIsEnabled) as exc:
                how = ("HF_HUB_OFFLINE is set; unset it" if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
                       else "the Hub can't be reached; connect")
                raise errors.UsageError(f"can't read dataset {dataset}: it isn't cached and {how}, or pass a folder "
                                        f"({type(exc).__name__})") from None
            except Exception as exc:
                if attempt == DOWNLOAD_ATTEMPTS - 1: raise
                log.warning("downloading %s failed (%s: %s); trying again in %ds", dataset, type(exc).__name__,
                            exc, 30 * (attempt + 1))
            time.sleep(30 * (attempt + 1))
    finally:
        if cached: huggingface_hub.utils.enable_progress_bars()
    raise AssertionError("unreachable")


def hub() -> Any:
    """The `huggingface_hub` module, which reading an `hf://` dataset needs."""
    try:
        import huggingface_hub
    except ImportError:
        raise errors.MissingDependency("reading an hf:// dataset needs huggingface_hub: pip install huggingface_hub "
                                       "(a benchmark package's [benchmark] extra includes it), or pass a dataset "
                                       "folder") from None
    return huggingface_hub


def _cached(repo: str, revision: str | None) -> bool:
    """Whether the dataset is already in the HuggingFace cache, asking nothing of the Hub."""
    try:
        hub().snapshot_download(repo, repo_type="dataset", revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 -- any reason it isn't there means it isn't cached
        return False
    return True


def dataset_id(dataset: str | Path) -> str:
    """Which dataset this is, ignoring its revision: the repo of an `hf://` dataset, or a folder's full path.
    A run folder records it, to refuse being used with another dataset."""
    if (at := hf_dataset(dataset)) is not None:
        return f"{HF_PREFIX}{at[0]}"
    return str(Path(dataset).expanduser().resolve())


def commit(dataset: str | Path, root: Path) -> str | None:
    """The commit an `hf://` dataset was read at, which is the name of its downloaded folder. None for a
    folder, which has no commits."""
    return root.name if hf_dataset(dataset) is not None else None


def manifest_path(manifest: str) -> str:
    """A manifest's path inside its dataset, always written the same way. Refuses a path outside the dataset."""
    m = PurePosixPath(manifest)
    if m.is_absolute() or ".." in m.parts:
        raise errors.UsageError(f"manifest {manifest} must be a path inside the dataset, relative to it, "
                                "e.g. manifests/sample.parquet")
    return m.as_posix()


@functools.cache
def read_manifest(dataset: str | Path, manifest: str = MANIFEST, *, columns: tuple[str, ...],
                  id: str = "id", inputs: tuple[str, ...] = (), gold: str | None = "gt_path"
                  ) -> tuple[Path, tuple[dict[str, Any], ...]]:
    """The dataset's folder, and its manifest's rows, checked, in the manifest's order.

    It reads the manifest, then checks each row (`row`) and that no id appears twice (`unique`).

    Args:
      columns: the columns every row must have, besides the ones named below.
      id: the column holding each row's id.
      inputs: the columns holding paths to input files. They become full paths on this machine.
      gold: the column holding the path to each row's gold file. It becomes a full path on this machine.

    Returns:
      The folder, and each row as a dict. The dict holds the row's id, `suite`, `inputs`, `gold` and
      `columns`, and `extra`, which holds every other column of the manifest, unread.
    """
    import pyarrow.parquet as pq

    manifest = manifest_path(manifest)
    _on_hub(dataset, manifest)
    root = download_dataset(dataset)
    path = root / manifest
    need = tuple(dict.fromkeys((id, *inputs, *([gold] if gold else []), *columns)))
    if not path.exists():
        raise errors.UsageError(f"no manifest at {path}; a manifest is a parquet file inside the dataset with "
                                f"one row per input, with columns {', '.join(need)}")
    table = pq.read_table(path)
    if absent := [c for c in need if c not in table.column_names]:
        raise errors.UsageError(f"{path} has no column {', '.join(absent)}; its rows need {', '.join(need)}, "
                                f"and it has {', '.join(table.column_names) or 'none'}")
    rows = tuple(row(r, n, path, root, need, id=id, inputs=inputs, gold=gold)
                 for n, r in enumerate(table.to_pylist()))
    if not rows:
        raise errors.UsageError(f"the manifest {path} names no rows")
    unique([r[id] for r in rows], path)
    return root, rows


def row(r: dict[str, Any], n: int, path: Path, root: Path, need: tuple[str, ...], *, id: str,
        inputs: tuple[str, ...], gold: str | None) -> dict[str, Any]:
    """Manifest row `n`, checked. Its paths become full paths under `root`, and its other columns go into
    `extra`. `path` is the manifest's path, used only in messages."""
    r = {"suite": None, **r}
    i = r[id]
    if not isinstance(i, str) or not i or i in (".", "..") or Path(i).name != i:
        raise errors.UsageError(f"{path}: row {n + 1} has {id} {i!r}, which is not a file name; it names the "
                                "row's prediction and record, so it is one path component")
    if r["suite"] is not None and not isinstance(r["suite"], str):
        raise errors.UsageError(f"{path}: row {i} has suite {r['suite']!r}; give a string, or null for {NO_SUITE!r}")
    for c in inputs:
        if not _inside(r[c]):
            raise errors.UsageError(f"{path}: row {i} has {c} {r[c]!r}; give a relative path to a file inside the "
                                    "dataset")
        r[c] = root / r[c]
    if gold:
        if not _inside(r[gold], i):
            raise errors.UsageError(f"{path}: row {i} has {gold} {r[gold]!r}; give a relative path inside the "
                                    f"dataset named for its row, e.g. gold/{i}.json")
        r[gold] = root / r[gold]
    r["suite"] = r["suite"] if (r["suite"] or "").strip() else NO_SUITE
    keep = (*need, "suite")
    return {**{c: r[c] for c in keep}, "extra": {k: v for k, v in r.items() if k not in keep}}


def unique(ids: Sequence[str], path: Path) -> None:
    """Refuses an id that appears more than once, ignoring case.

    An id names files, and on a disk that ignores case (macOS, Windows) two ids that differ only in case
    would be one file."""
    by: dict[str, list[str]] = {}
    for i in ids: by.setdefault(i.casefold(), []).append(i)
    if twice := [xs for xs in by.values() if len(xs) > 1]:
        raise errors.UsageError(f"{path}: ids more than once, ignoring case: "
                                + "; ".join(" and ".join(xs) for xs in twice[:3]))


def selection(suites: Iterable[str] | None = None, limit: int = 0, sample: int = 0,
              only: Iterable[str] | None = None) -> dict[str, Any]:
    """A choice of rows, written the same way every time: only what was given, names sorted. It is part of
    a run's question, so it names the run's folder, and grading reads it back to know which rows the run
    answers. `{}` is every row."""
    out: dict[str, Any] = {}
    if (s := _names(suites, "suites")) is not None: out["suites"] = sorted(s)
    if (o := _names(only, "only")) is not None: out["only"] = sorted(o)
    if sample: out["sample"] = sample
    if limit: out["limit"] = limit
    return out


def select[R: Row](rows: Sequence[R], suites: Iterable[str] | None = None, limit: int = 0, sample: int = 0,
                   only: Iterable[str] | None = None) -> tuple[R, ...]:
    """The rows to run, in their order. The filters apply in this order:

    1. `suites`: only rows of these suites;
    2. `only`: only rows with these ids;
    3. `sample`: this many rows, spread evenly over the suites; and
    4. `limit`: the first this many rows.

    `sample` takes one row from each suite in turn, evenly spaced within the suite. So a run sees every kind
    of row, and sees the same rows each time: two systems run with the same `sample` answer the same rows.
    `sample` and `limit` are two ways to say how many rows, so only one may be given."""
    suites, only = _names(suites, "suites"), _names(only, "only")
    for name, n in (("limit", limit), ("sample", sample)):
        if isinstance(n, bool) or not isinstance(n, int): raise TypeError(f"{name} is a whole number, not {n!r}")
        if n < 0: raise errors.UsageError(f"{name} is {n}; it counts rows, so give 0 (no limit) or more")
    if limit and sample:
        raise errors.UsageError(f"give sample ({sample}) or limit ({limit}), not both: sample spreads its rows over "
                                "every suite, limit takes the first ones")
    xs = tuple(rows)
    known = {r.suite for r in xs}
    if suites is not None and (bad := sorted(suites - known)):
        raise errors.UsageError(f"unknown suites {bad}; the dataset's suites are {sorted(known)}")
    xs = tuple(r for r in xs if suites is None or r.suite in suites)
    if only is not None:
        if bad := sorted(only - {r.id for r in xs}):
            raise errors.UsageError(f"rows asked for that aren't in the dataset: {errors.counted(len(bad), 'row')}, "
                                    f"e.g. {bad[:3]}")
        xs = tuple(r for r in xs if r.id in only)
    if sample and xs:
        by: dict[str, list[R]] = {}
        for r in xs: by.setdefault(r.suite, []).append(r)
        k = -(-sample // len(by))
        spread = [[ys[i * len(ys) // min(k, len(ys))] for i in range(min(k, len(ys)))] for ys in by.values()]
        chosen = {x.id for x in [ys[i] for i in range(k) for ys in spread if i < len(ys)][:sample]}
        xs = tuple(r for r in xs if r.id in chosen)
    if not (xs := xs[:limit] if limit else xs):
        raise errors.UsageError("no rows selected; check suites, only, sample and limit")
    return xs


def _names(xs: Iterable[str] | None, what: str) -> frozenset[str] | None:
    if isinstance(xs, str):
        raise TypeError(f"{what} is a list of names, not the string {xs!r}; pass [{xs!r}]")
    return None if xs is None else frozenset(xs)


def _inside(path: Any, id: str | None = None) -> bool:
    """Whether `path` is a relative path to a file inside the dataset. Given `id`, also whether it is named
    `<id>.json`."""
    if not isinstance(path, str) or not path.strip(): return False
    p = Path(path)
    return (not p.is_absolute() and ".." not in p.parts and bool(p.suffix)
            and (id is None or (p.stem == id and p.suffix == ".json")))


def _on_hub(dataset: str | Path, manifest: str) -> None:
    """Refuses an `hf://` dataset that has no `manifest`, before downloading gigabytes of it.

    The Hub answers a private dataset that this machine's token can't read as if it didn't exist, so a dataset
    the Hub doesn't show is reported as that, not as a missing manifest. When the Hub doesn't answer, the
    download goes ahead, and a manifest that really is missing is reported after it."""
    if (at := hf_dataset(dataset)) is None: return
    huggingface_hub = hub()

    try:
        if not huggingface_hub.repo_exists(at[0], repo_type="dataset"):
            raise errors.UsageError(f"the Hugging Face Hub shows no dataset {at[0]}: it doesn't exist, or it is private "
                                    "and this machine has no token that can read it (set HF_TOKEN)")
        missing = not huggingface_hub.file_exists(at[0], manifest, repo_type="dataset", revision=at[1])
    except OSError:
        missing = False
    if missing:
        raise errors.UsageError(f"there is no {manifest} in {dataset}; name another manifest, or another dataset")
