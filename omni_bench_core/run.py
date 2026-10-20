"""Runs: one system's answers to a dataset, kept in a folder.

`Run` is the folder and the bookkeeping over it. Its docstring says what the folder holds and the rules it
follows. `folder_name` names a run's folder after the question it answers. `total` adds up costs.
`write_json`, `write_text` and `write_lines` write a file whole, so a
crash never leaves half a file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from omni_bench_core import adapter, data, errors

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Run:
    """One system's answers to one question, kept in a folder, and the system that predicts into it.

    A run answers one exact question: one system with its options, on one dataset, manifest and task, for
    one choice of rows (`data.selection`). The folder is named after all of it (`folder_name`), so asking
    another question uses another folder, and asking the same one again resumes this one. The folder holds:

        <folder>/
          settings.json           the question (system, options, dataset, manifest, task, rows) and the timeout
          records/<id>.json       everything `adapter.predict` returned for each row: the source of truth
          predictions/<id>.json   each row's output alone, which is all that grading reads
          <scorer>/               each scorer's scores.jsonl, summary.json and details/

    Answers cost money and time, so each one is written to a file as it arrives. A run that stops resumes
    where it stopped, and any scorer can grade a run again without calling the system.

    A `Run` is an address, like a `Path`. What it holds is whatever its folder holds when it is read. A
    folder of your own predictions is a run with no system (`Run.at`). Its settings.json names no system,
    which is how the two kinds are told apart (`brought`). It is graded over the whole manifest.

    Each row of a run is in one of three states:

        answered   it has a prediction
        failed     it has a record of a failed call: the system's failure, which counts as 0
        missing    it has neither: it was never predicted, or, in a folder of your own, it has no prediction

    A run is finished when none of its rows is missing. Grading refuses an unfinished run (`Scorer.grade`).

    Its methods are the bookkeeping that any loop over rows needs: what was recorded, what still needs a
    call, and what is missing. They live here, once, so every task follows the same rules."""

    folder: Path
    system: adapter.Adapter | None = None

    @classmethod
    def at(cls, folder: str | Path) -> Run:
        """The run kept in `folder`, as it is now. Use it for a folder of your own predictions, or to grade a
        run again. Grading doesn't need the system."""
        f = Path(folder)
        if not f.is_dir():
            raise errors.UsageError(f"no folder at {f}; pass the folder holding predictions/<id>.json")
        if not (f / "predictions").is_dir():
            hint = f"; pass its parent, {f.parent}" if f.name == "predictions" else ""
            raise errors.UsageError(f"{f} has no predictions/ folder; a folder to grade holds "
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
            raise errors.UsageError(f"{path} isn't readable JSON ({exc}); fix it, or remove it") from None
        if not isinstance(said, dict):
            raise errors.UsageError(f"{path} holds a JSON {type(said).__name__}; settings.json is an object")
        return said

    def brought(self) -> bool:
        """Whether the folder holds predictions you brought, rather than a run made by predicting. A folder you
        brought has no records, no question and no rows of its own: it is graded over the whole manifest."""
        return not self.settings().get("system")

    def selection(self) -> dict[str, Any]:
        """The choice of rows the run answers (`data.selection`), as its settings.json says: `{}` for every row,
        and for a folder you brought."""
        return self.settings().get("rows", {})

    def check_settings(self, question: Mapping[str, Any]) -> None:
        """Refuses grading a run with another dataset, manifest or task than the one it answers.

        A row's files are named by its id alone. So answers for another dataset would be graded against this
        dataset's gold, and answers to another task (another variant of the input) against the wrong outputs."""
        said = self.settings()
        for key in ("dataset", "manifest", "task"):
            if (was := said.get(key)) and was != question[key]:
                raise errors.UsageError(f"{self.folder} answers {key} {_spelled(was)}, not {_spelled(question[key])}: "
                                        f"grade it with the {key} it was run on")

    def prepare(self, question: Mapping[str, Any], commit: str | None, timeout: float) -> None:
        """Makes the run's folders, and writes its question, the dataset's commit and the timeout to
        settings.json. This happens before the first call, so even a run stopped at its first row says what it
        was."""
        assert self.system is not None, "only a run with a system predicts"
        self.predictions.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)
        write_json(self.folder / "settings.json", {"run": self.name, "system": self.system.name,
                                                   "options": self.system.options(), **question,
                                                   **({"commit": commit} if commit else {}), "timeout_s": timeout})

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
        """The file names in predictions/, listed once per grading.

        A prediction is only read under its exact name, so a disk that ignores case grades the same files
        as any other disk."""
        return frozenset(f.name for f in self.predictions.iterdir()) if self.predictions.is_dir() else frozenset()

    def attempted(self, id: str, files: frozenset[str]) -> bool:
        """Whether the row was attempted, meaning it has a prediction (in `files`) or a record."""
        return f"{id}.json" in files or (self.records / f"{id}.json").exists()

    def needs(self, id: str, retry_failed: bool = False) -> bool:
        """Whether to call the system for this row.

        Yes when the row has no record. With `retry_failed`, also yes when its record is a failure that may
        not happen again (`adapter.retryable`). Never when it has an answer, so no answer is paid for twice."""
        if (rec := self.record(id)) is None: return True
        return retry_failed and adapter.retryable(rec.get("error"))

    def save(self, id: str, rec: dict[str, Any]) -> None:
        """Writes a row's record from `adapter.predict`: first its prediction, if it has an answer, then the
        record.

        The order matters. `needs` reads the record, so if the process is killed between the two writes, the
        row is predicted again rather than left with a record and no prediction. predictions/ holds answers
        only; a failure is written in the record alone."""
        if rec["error"] is None:
            self.write_prediction(id, rec["output"])
        else:
            (self.predictions / f"{id}.json").unlink(missing_ok=True)
        write_json(self.records / f"{id}.json", rec)

    def write_prediction(self, id: str, output: Any) -> None:
        """Writes the row's output to predictions/<id>.json."""
        self.predictions.mkdir(parents=True, exist_ok=True)
        write_json(self.predictions / f"{id}.json", output)

    def lost(self, id: str) -> dict[str, Any] | None:
        """The row's record, when it holds an answer but the prediction file is gone. None otherwise, and also
        when the record has no `output` to rebuild from (one written by an older version), which is logged."""
        if (self.predictions / f"{id}.json").exists(): return None
        rec = self.record(id)
        if rec is None or rec.get("error") is not None: return None
        if "output" not in rec:
            log.warning("%s: the record of %s has no output to rebuild its prediction from", self.folder, id)
            return None
        return rec

    def restore(self, rows: Sequence[data.Row]) -> list[str]:
        """Rebuilds every lost prediction from its record, with no call, and returns their ids.

        The record is the source of truth, so a deleted prediction is never paid for twice. It goes one row
        at a time: `lost`, then `write_prediction`."""
        out = []
        for r in rows:
            if (rec := self.lost(r.id)) is not None:
                self.write_prediction(r.id, rec["output"])
                out.append(r.id)
        return out

    def prediction(self, id: str) -> Any:
        """The row's prediction, read as JSON.

        Raises `FileNotFoundError` when the row has none, and ValueError, saying why, when the file isn't
        JSON."""
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
        """Each row's recorded cost, reading one record at a time. Rows with no cost are left out."""
        for i in ids:
            if c := (self.record(i) or {}).get("cost"): yield c


def _spelled(x: Any) -> str:
    """A dataset, manifest or task as a message writes it: a task as its name and options, `parse (outputs=html)`."""
    if not isinstance(x, dict): return str(x)
    opts = ", ".join(f"{k}={' '.join(map(str, v)) if isinstance(v, list) else v}"
                     for k, v in x.get("options", {}).items())
    return f"{x.get('name')}" + (f" ({opts})" if opts else "")


def folder_name(system: adapter.Adapter, question: Mapping[str, Any]) -> str:
    """The name of the folder of `system`'s run on `question`: the system's `label`, then a digest of the system,
    every one of its options, and the question (dataset, manifest, task and rows).

    The digest covers every option, not only those that differ from the defaults. Defaults change, and then
    two configurations would share a folder and be averaged into one number."""
    spelled = json.dumps({"system": system.name, "options": system.options(), **question}, sort_keys=True,
                         default=str)
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', system.label)}-{hashlib.sha256(spelled.encode()).hexdigest()[:8]}"


def total(costs: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """What the calls cost together, in dollars and in credits.

    A total is None, not 0, when no call reported it, because a system billed some other way isn't free."""
    usd = credits = None
    for c in costs:
        if c.get("usd") is not None: usd = (usd or 0.0) + c["usd"]
        if c.get("credits") is not None: credits = (credits or 0.0) + c["credits"]
    return {"usd": None if usd is None else round(usd, 4), "credits": None if credits is None else round(credits, 4)}


def write_json(path: Path, obj: Any) -> None:
    """Writes `obj` as JSON: first to a temporary file, then moved into place, so a crash never leaves half a
    file."""
    write_text(path, json.dumps(obj, default=str, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    """Writes `text` the same way as `write_json`."""
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_lines(path: Path, xs: Iterable[Any]) -> None:
    """Writes each of `xs` as one line of JSON, one at a time, then moves the file into place whole."""
    tmp = path.with_name(f".{path.name}.partial")
    with tmp.open("w") as f:
        for x in xs: f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
