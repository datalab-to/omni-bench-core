"""Runs: one system's answers, kept in a folder.

- `Run`: the folder and the bookkeeping over it;
- `identity` and `folder_name`: what a run answers, which a resume must match, and a new run's folder name;
- `total`: recorded costs added up; and
- `write_json`, `write_text` and `write_lines`: write a file whole, so a crash never leaves half a file.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import re
import secrets
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from omni_bench_core import adapter, data

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Run:
    """One system's answers to one question, kept in a folder, and the system that predicts into it.

    The question is the system and its options, the dataset, manifest and task, and the choice of rows
    (`identity`), kept in settings.json. The folder is named after the system and a run id (`folder_name`): every
    launch is a new run in a new folder, and only a launch that names this folder (`resume`) continues it, and only
    for the same question. The folder holds:

        <folder>/
          settings.json           the question, the timeout, and the launch (when, and what the caller added)
          server.json             for a served system: what its server said at each launch (`served_by`)
          records/<id>.json       everything `adapter.predict` returned for each row: the source of truth
          predictions/<id>.json   each row's output alone, which is all grading reads
          <scorer>/               each scorer's scores.jsonl, summary.json and details/

    A `Run` is an address, like a `Path`: it holds whatever its folder holds when read. A folder of your own
    predictions is a run with no system (`Run.at`, `brought`), graded over the whole manifest. Each row is:

        answered   it has a prediction;
        failed     it has a record of a `SystemFailed`, which fails it; or
        missing    it has neither (never predicted, or its answer is still unread after a bug in `read`).

    A run is finished when no row is missing; grading refuses one that isn't."""

    folder: Path
    system: adapter.Adapter | None = None

    @classmethod
    def at(cls, folder: str | Path) -> Run:
        """The run kept in `folder`, with no system: a folder of your own predictions, or a run to grade again."""
        f = Path(folder)
        if not f.is_dir():
            raise ValueError(f"no folder at {f}; pass the folder holding predictions/<id>.json")
        if not (f / "predictions").is_dir():
            hint = f"; pass its parent, {f.parent}" if f.name == "predictions" else ""
            raise ValueError(f"{f} has no predictions/ folder; a folder to grade holds "
                             f"predictions/<id>.json{hint}")
        return cls(f)

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def predictions(self) -> Path:
        return self.folder / "predictions"

    @property
    def records(self) -> Path:
        return self.folder / "records"

    def settings(self) -> dict[str, Any]:
        """The folder's settings.json, or {} when it has none."""
        path = self.folder / "settings.json"
        try:
            said = json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path} isn't readable JSON ({exc}); fix it, or remove it") from None
        if not isinstance(said, dict):
            raise ValueError(f"{path} holds a JSON {type(said).__name__}; settings.json is an object")
        return said

    def brought(self) -> bool:
        """Whether the folder holds predictions you brought: its settings.json names no system."""
        return not self.settings().get("system")

    def selection(self) -> dict[str, Any]:
        """The choice of rows the run answers (`data.selection`); `{}` for every row."""
        return self.settings().get("rows", {})

    def check_settings(self, question: Mapping[str, Any]) -> None:
        """Refuses grading a run against another dataset, manifest or task than the one it answers: a row's files
        are named by its id alone, so they would be graded against the wrong gold."""
        said = self.settings()
        for key in ("dataset", "manifest", "task"):
            if (was := said.get(key)) and was != question[key]:
                raise ValueError(f"{self.folder} answers {key} {was}, not {question[key]}: "
                                 f"grade it with the {key} it was run on")

    def prepare(self, question: Mapping[str, Any], dataset_commit: str | None, timeout: float,
                launch: Mapping[str, Any] | None = None) -> None:
        """Makes the run's folders, and writes settings.json before the first call: the question, the dataset's
        commit (`dataset_commit`, for an `hf://` dataset), the timeout, where the system is (`located`), and the launch
        (`started`, UTC, and `launch`, the caller's own record of it). A resumed run keeps its first launch's
        settings.json but the timeout and where the system is, which may change, and refuses a dataset whose commit changed since: its rows would be answered against two
        versions of the data."""
        assert self.system is not None, "only a run with a system predicts"
        if (said := self.settings()):
            if (was := said.get("dataset_commit")) and dataset_commit and was != dataset_commit:
                raise ValueError(f"{self.folder} was run on the dataset at commit {was}, and it is now at {dataset_commit}; "
                                 "start a new run (leave out resume)")
            now = {"timeout_s": timeout, **self._located()}
            if any(said.get(k) != v for k, v in now.items()):
                write_json(self.folder / "settings.json", {**said, **now})
            return
        self.predictions.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)
        started = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        write_json(self.folder / "settings.json", {"run": self.name, **identity(self.system, question),
                                                   **({"dataset_commit": dataset_commit} if dataset_commit else {}), "timeout_s": timeout,
                                                   **self._located(),
                                                   "started": started, "launch": dict(launch or {})})

    def served_by(self, said: Mapping[str, Any]) -> None:
        """Appends what a served system's server said at this launch (`adapter.check_served`) to server.json, with
        when and where it was: `{"launches": [{"started", "located", "server", "details"}, ...]}`. One entry a
        launch, since a resume may meet another machine; the server's identity is checked the same at each."""
        path = self.folder / "server.json"
        before = json.loads(path.read_text())["launches"] if path.is_file() else []
        now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        entry = {"started": now, "located": self._located().get("located", {}), **said}
        write_json(path, {"launches": [*before, entry]})

    def _located(self) -> dict[str, Any]:
        """Where the system is, as of the latest launch (its `located` options), when it says: `{"located": {...}}`."""
        assert self.system is not None
        o = self.system.options()
        return {"located": {k: o[k] for k in self.system.located}} if self.system.located else {}

    def record(self, id: str) -> dict[str, Any] | None:
        """The row's record, or None when it has none or the file can't be read."""
        path = self.records / f"{id}.json"
        try:
            got = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("%s isn't readable JSON (%s); it is treated as missing", path, exc)
            return None
        return got if isinstance(got, dict) else None

    def prediction_files(self) -> frozenset[str]:
        """The file names in predictions/. A prediction is read only under its exact name, so a disk that ignores
        case grades the same files as any other."""
        return frozenset(f.name for f in self.predictions.iterdir()) if self.predictions.is_dir() else frozenset()

    def attempted(self, id: str, files: frozenset[str]) -> bool:
        """Whether the row has a prediction (in `files`) or a record of a read answer or of a failure."""
        if f"{id}.json" in files: return True
        return (rec := self.record(id)) is not None and not rec.get("unread")

    def needs(self, id: str, retry_failed: bool = False) -> bool:
        """Whether to call the system for this row: when it has no record or, with `retry_failed`, a failure that
        may not recur (`adapter.retryable`). Never when it has an answer."""
        if (rec := self.record(id)) is None: return True
        return retry_failed and adapter.retryable(rec.get("error"))

    def save(self, id: str, rec: dict[str, Any]) -> None:
        """Writes a row's record from `adapter.predict`, and its prediction when it has an answer."""
        # Note: the prediction is written first: `needs` reads the record, so a kill between the two writes
        # predicts the row again rather than leaving a record with no prediction.
        if rec["error"] is None and not rec.get("unread"):
            self.write_prediction(id, rec["output"])
        else:
            (self.predictions / f"{id}.json").unlink(missing_ok=True)
        write_json(self.records / f"{id}.json", rec)

    def write_prediction(self, id: str, output: Any) -> None:
        """Writes the row's output to predictions/<id>.json."""
        self.predictions.mkdir(parents=True, exist_ok=True)
        write_json(self.predictions / f"{id}.json", output)

    def lost(self, id: str) -> dict[str, Any] | None:
        """The row's record, when it holds an answer but the prediction file is gone; else None."""
        if (self.predictions / f"{id}.json").exists(): return None
        rec = self.record(id)
        if rec is None or rec.get("error") is not None or rec.get("unread"): return None
        if "output" not in rec:
            log.warning("%s: the record of %s has no output to rebuild its prediction from", self.folder, id)
            return None
        return rec

    def restore(self, rows: Sequence[data.Row]) -> list[str]:
        """Rebuilds every lost prediction from its record, with no call, and returns their ids."""
        out = []
        for r in rows:
            if (rec := self.lost(r.id)) is not None:
                self.write_prediction(r.id, rec["output"])
                out.append(r.id)
        return out

    def prediction(self, id: str) -> Any:
        """The row's prediction, read as JSON. Raises `FileNotFoundError` when there is none, and `ValueError` when
        the file isn't JSON."""
        raw = (self.predictions / f"{id}.json").read_bytes()
        if not raw.strip(): raise ValueError("the prediction file is empty")
        try:
            # Note: we read utf-8-sig so that a file saved with a byte-order mark still reads as JSON.
            return json.loads(raw.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise ValueError("the prediction file isn't UTF-8 text") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"the prediction file isn't JSON ({exc.msg}, line {exc.lineno})") from None

    def costs(self, ids: Iterable[str]) -> Iterator[dict[str, Any]]:
        """Each row's recorded cost, one record at a time; rows with none are left out."""
        for i in ids:
            if c := (self.record(i) or {}).get("cost"): yield c


def identity(system: adapter.Adapter, question: Mapping[str, Any]) -> dict[str, Any]:
    """What a run of `system` on `question` answers, as its settings.json keeps it: the system, every option but
    where it is (`Adapter.asked`), and the question. A resume must match it exactly."""
    # Note: every option, not only those that differ from the defaults, since a default can change.
    return {"system": system.name, "options": system.asked(), **question}


def folder_name(system: adapter.Adapter, id: str) -> str:
    """The folder name of one run of `system`: its `label`, then the run id (its time and 4 hex)."""
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', system.label)}-{id}"


def new_id() -> str:
    """A new run id: the launch time (UTC, to the second), then four random hex digits, `20261021-160012-3f9a`.
    Two launches in one second still differ."""
    return f"{datetime.datetime.now(datetime.UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def total(costs: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """What the calls cost together, in dollars and credits; None, not 0, when no call reported it."""
    usd = credits = None
    for c in costs:
        if c.get("usd") is not None: usd = (usd or 0.0) + c["usd"]
        if c.get("credits") is not None: credits = (credits or 0.0) + c["credits"]
    return {"usd": None if usd is None else round(usd, 4), "credits": None if credits is None else round(credits, 4)}


def write_json(path: Path, obj: Any) -> None:
    """Writes `obj` as JSON, whole."""
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    """Writes `text` to a temporary file, then moves it into place, so a crash never leaves half a file."""
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_lines(path: Path, xs: Iterable[Any]) -> None:
    """Writes each of `xs` as a line of JSON, one at a time, then moves the file into place."""
    tmp = path.with_name(f".{path.name}.partial")
    with tmp.open("w") as f:
        for x in xs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
