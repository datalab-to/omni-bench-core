"""Scorers: how a task's predictions are graded.

- `Scorer`: the class to subclass for a way of grading;
- `Score`: what `Scorer.score` returns for a scored row; and
- `ask`: sends a judge (an adapter) one question, under the same rules as a system's calls.
"""

from __future__ import annotations

import abc
import concurrent.futures as cf
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import re
import shutil
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, final

from omni_bench_core import adapter, data, errors
from omni_bench_core import run as runs

log = logging.getLogger(__name__)

type Line = dict[str, Any]
type Summary = dict[str, Any]

SHOWN = 10   # how many ids a message about unfinished rows names
POOLS = ("process", "thread")


class Score(NamedTuple):
    """A scored row's score. `result` is a small dict of the row's values, mostly numbers, which its line in
    scores.jsonl holds; it may not use `id`, `status` or `row`, which the core writes. `details` is an optional
    larger breakdown (a verdict per field, a judge's reasoning), written to `details/<id>.jsonl` and never read
    by `summarize`."""

    result: dict[str, Any]
    details: list[dict[str, Any]] | None = None


class _Failed(NamedTuple):
    """A row with no usable prediction, which counts 0, as the scorer's `zero` scores it if it gives one. Only
    `grade` makes one: from the row's record, a missing prediction, or `prediction` raising `SystemFailed` (with
    its `traceback`)."""

    error: str
    traceback: str | None = None
    zero: Score | None = None


@dataclasses.dataclass(frozen=True)
class Scorer[R: data.Row, G, P](abc.ABC):
    """One way of grading. A scorer is a frozen dataclass whose fields are its options:

        @dataclasses.dataclass(frozen=True)
        class Exact(scorer.Scorer):
            def gold(self, row):                  # the row's gold, read and checked
                return json.loads(row.gt_path.read_text())["text"]

            def score(self, pred, gold):          # one row's numbers
                return scorer.Score({"exact": pred["text"] == gold})

            def summarize(self, lines):           # the scored rows' lines as one summary
                return {"headline": sum(x["exact"] for x in lines) / len(lines), "headline_is": "exact"}

    `gold` and `summarize` are abstract; `score` is required unless `score_all` is written instead. Optional,
    with defaults:

        name            the scorer's name; the class's name in snake case
        prediction      what the scorer reads of a prediction; the prediction itself
        row_fields      what of the row each line carries under "row"; nothing
        zero            the score a failed row counts as; None, which leaves failed rows to coverage
        check_gold      refuses a dataset before any call; reads every gold
        score_all       many rows' scores at once; `score` for each row
        workers         rows scored at once; 1
        pool            "process" for scoring that computes, "thread" for scoring that waits on calls; "process"
        cache           keep each row's score while its prediction, gold, options and code are unchanged; False
        batch_size      rows read and given to `score_all` at once; 500, or None for all

    A scorer scores a row (a `Score`) or leaves it unscored (`errors.Unscored`); only the core fails a row.
    A judge is an adapter field, asked through `ask`, with `cache = True`, `pool = "thread"` and many
    `workers`. A cached scorer's predictions, golds and results must be JSON.

    The options are written to the summary and digested into the results folder's name,
    `<run>/<name>-<digest>/`, so two configurations never overwrite each other. `grade` holds the rules and
    can't be overridden:

        grade(run, rows)                    reads files, applies the rules, writes
          gold(row), prediction(output)     each row's gold and prediction, read once
          score_all(rows, preds, golds)     `batch_size` rows at a time, without kept scores
            score(pred, gold)               one row
          zero(row, gold)                   a failed row's score, if the scorer gives one
          row_fields(row)                   each line's "row"
          summarize(lines)                  the scored (and zeroed) rows' lines as one summary
    """

    name: ClassVar[str]
    workers: ClassVar[int] = 1
    pool: ClassVar[str] = "process"
    cache: ClassVar[bool] = False
    batch_size: ClassVar[int | None] = 500

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        if "grade" in cls.__dict__:
            raise TypeError(f"{cls.__name__} overrides grade, which holds the rules every scorer follows (finished, "
                            "failed, the headline, the cache); override score_all to change how rows are "
                            "scored, or summarize to change the summary")

    @property
    def folder_name(self) -> str:
        """The scorer's results folder in a run: its name, then a digest of its options if it has any."""
        o = self.options()
        if not o: return self.name
        return f"{self.name}-{hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:8]}"

    def options(self) -> dict[str, Any]:
        """The scorer's options, as its summary states them; a judge appears as its name and options."""
        errors.check_dataclass(type(self), Scorer)
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = {"system": v.name, **v.options()} if isinstance(v, adapter.Adapter) else v
        errors.plain(out, f"{type(self).__name__}'s options")
        return out

    @abc.abstractmethod
    def gold(self, row: R) -> G:
        """The row's gold, read and checked. Raise when it can't be read: grading stops and names the row."""

    def prediction(self, output: Any) -> P:
        """What the scorer reads of one prediction (`Adapter.read`'s output). Raise `errors.SystemFailed` for an
        output it can't use: the row fails and counts 0. Anything else it raises stops grading."""
        return output

    def score(self, pred: P, gold: G) -> Score:
        """One row's score. Keep it pure (no files, no calls), or set `cache = True` when it asks a judge. Raise
        `errors.Unscored` only when asking again may give a score (the judge timed out or was down): the row
        waits for the next grading. Not abstract, since `score_all` can replace it; a scorer with neither is
        refused."""
        raise NotImplementedError(f"{type(self).__name__} has no score(pred, gold)")

    def row_fields(self, row: R) -> dict[str, Any]:
        """What of the row its line carries under `"row"`: fields `summarize` groups by (a document, a language)."""
        return {}

    @abc.abstractmethod
    def summarize(self, lines: Sequence[Line]) -> Summary:
        """The scored rows' lines as one summary. Each line is `{"id", "row" (if any), ...the result's values}`;
        a failed row is given only with the score `zero` gave it, and it isn't called when no row has a score.

        Return `headline`, a number from 0 to 1 where higher is better, and `headline_is`, what it is
        ("accuracy"); anything else is the scorer's own. `grade` keeps it under the summary's `scored` and
        multiplies the headline by coverage."""

    def zero(self, row: R, gold: G) -> Score | None:
        """The score a failed row counts as, in the scorer's own units (every test failed), given to `summarize`
        with the scored rows' lines. None, the default, leaves failed rows out, and the headline's coverage counts
        each as 0: the same thing when the headline is a mean over rows, but not when rows weigh differently (a
        page with 40 tests and a page with 2)."""
        return None

    def check_gold(self, rows: Sequence[R]) -> None:
        """Refuses, before any call, a dataset whose gold this scorer can't read. Reads every gold by default."""
        for r in rows: self._gold(r)

    def score_all(self, rows: Sequence[R], preds: Sequence[P], golds: Sequence[G]) -> list[Score | errors.Unscored]:
        """Many rows' scores, in order: a `Score` for each row, or an `errors.Unscored` (returned, not raised) for
        a row it couldn't score. Override it to score rows together (a batch to a judge). It gets only rows with a
        prediction. By default it calls `score` for each row, `workers` at a time, in `pool`."""
        if self.pool not in POOLS:
            raise TypeError(f"{type(self).__name__}.pool is {self.pool!r}; it is one of {POOLS}")
        ids = [r.id for r in rows]
        if (n := min(self.workers, len(rows))) <= 1:
            return [_score_one(self, i, p, g) for i, p, g in zip(ids, preds, golds, strict=True)]
        make = cf.ProcessPoolExecutor if self.pool == "process" else cf.ThreadPoolExecutor
        try:
            with make(n) as ex:
                return list(ex.map(functools.partial(_score_one, self), ids, preds, golds, chunksize=8))
        except cf.process.BrokenProcessPool as exc:
            raise ValueError(f"{self.name} scores in {n} processes, and they couldn't start or died. In a "
                             "script, put the code that runs the benchmark under `if __name__ == "
                             "\"__main__\":`, because each process imports the script again (macOS and "
                             "Windows); otherwise a process ran out of memory") from exc

    @final
    def grade(self, run: runs.Run, rows: Sequence[R], *, partial: bool = False, details: bool = True,
              write: bool = True) -> Summary:
        """Grades `run` on `rows`, writes the results under `<run>/<folder_name>/`, and returns the summary. In
        order, it:

        1. refuses a folder of your own that names none of the rows, and rebuilds lost predictions from records;
        2. refuses rows with neither an answer nor a recorded failure, unless `partial`;
        3. for each batch of `batch_size` rows: fails the rows the core knows failed, reuses kept scores,
           scores the rest (`score_all`), and writes their details;
        4. refuses unscored rows, unless `partial`, which counts them failed;
        5. summarizes the scored lines and the failed rows `zero` scores (`summarize`), checks the summary, and
           multiplies its headline by coverage, the share of rows summarized; and
        6. writes scores.jsonl and summary.json, only when grading finishes.

        `write=False` writes no results, though it may still rebuild a prediction or fill the cache."""
        files, brought = run.prediction_files(), run.brought()
        if brought and not files & {f"{r.id}.json" for r in rows}:
            raise ValueError(f"none of the {errors.counted(len(files), 'file')} in {run.predictions} "
                             f"names a row of the {len(rows)} graded (e.g. {rows[0].id}.json): is the "
                             "dataset right?")
        if rebuilt := run.restore(rows):
            log.info("%s: %s rebuilt from records, e.g. %s", run.folder, errors.counted(len(rebuilt), "prediction"),
                     rebuilt[:3])
        missing = [r.id for r in rows if not run.attempted(r.id, files)]
        if missing and not partial:
            raise ValueError(_unfinished(run, missing, [], brought))
        home = run.folder / self.folder_name
        code = self._code() if self.cache else ""
        lines: list[Line] = []
        zeroed: set[str] = set()
        _discard(home)
        try:
            # Note: one batch's predictions, golds and details in memory at a time.
            n = self.batch_size or len(rows) or 1
            for at in range(0, len(rows), n):
                chunk = rows[at:at + n]
                failed, ready = self._read(run, chunk)
                keys = self._keys(ready, code) if self.cache else {}
                kept = _kept(home, keys)
                fresh = self._scored([t for t in ready if t[0].id not in kept])
                if self.cache:
                    fresh = {i: self._as_json(x) for i, x in fresh.items()}
                    _keep(home, keys, fresh)
                scores = failed | kept | fresh
                zeroed |= {i for i, x in failed.items() if x.zero is not None}
                lines += [self._line(r, scores[r.id]) for r in chunk]
                if write and details: _write_details(home, [(r.id, scores[r.id]) for r in chunk])
        except BaseException:
            _discard(home)
            raise
        unscored = [x["id"] for x in lines if x["status"] == "unscored"]
        if unscored and not partial:
            _discard(home)
            raise ValueError(_unfinished(run, [], [(x["id"], x["error"]) for x in lines
                                                    if x["status"] == "unscored"], brought))
        lines = [{**x, "status": "failed", "error": f"not scored: {x['error']}"} if x["status"] == "unscored" else x
                 for x in lines]
        core = {"scored": ("status",), "failed": ("status", "error", "traceback")}
        ys = [{k: v for k, v in x.items() if k not in core[x["status"]]} for x in lines
              if x["status"] == "scored" or x["id"] in zeroed]
        summary = self._framed(self.summarize(ys) if ys else None, lines, len(zeroed), run.name)
        # Note: counts and a few ids, not every id, which a run stopped early would have thousands of.
        if missing or unscored:
            summary["partial"] = {"no_answer": len(missing), "not_scored": len(unscored),
                                  "e.g.": [*missing, *unscored][:SHOWN]}
        if write: _write(home, lines, summary)
        else: _discard(home)
        return summary

    def _read(self, run: runs.Run, rows: Sequence[R]) -> tuple[dict[str, _Failed], list[tuple[R, P, G]]]:
        """The rows that failed (by their record, a missing or unreadable prediction, or `prediction`), and the
        others with their predictions and golds, ready to score."""
        failed: dict[str, _Failed] = {}
        ready: list[tuple[R, P, G]] = []
        for r in rows:
            if err := (run.record(r.id) or {}).get("error"):
                failed[r.id] = self._failed(r, f"{err['type']}: {err['message']}")
                continue
            try:
                output = run.prediction(r.id)
            except FileNotFoundError:
                failed[r.id] = self._failed(r, f"no prediction: predictions/{r.id}.json")
                continue
            except ValueError as exc:   # Note: the core's own check that the file is JSON
                failed[r.id] = self._failed(r, str(exc))
                continue
            try:
                pred = self.prediction(output)
            except errors.SystemFailed as exc:
                failed[r.id] = self._failed(r, str(exc), "".join(traceback.format_exception(exc)))
                continue
            except Exception as exc:
                errors.noted(exc, type(self), f"reading the prediction of {r.id}")
                raise
            ready.append((r, pred, self._gold(r)))
        return failed, ready

    def _failed(self, r: R, error: str, tb: str | None = None) -> _Failed:
        """A failed row, with the score `zero` gives it; its gold is read only for a scorer that gives one."""
        if type(self).zero is Scorer.zero: return _Failed(error, tb)
        z = self.zero(r, self._gold(r))
        if not (z is None or isinstance(z, Score)):
            raise TypeError(f"{type(self).__name__}.zero must return a scorer.Score or None; for row {r.id} it "
                            f"returned a {type(z).__name__}")
        if z is not None: _check_score(type(self).__name__, r.id, z)
        return _Failed(error, tb, z)

    def _scored(self, ready: Sequence[tuple[R, P, G]]) -> dict[str, Score | errors.Unscored]:
        """`score_all` of the rows ready to score, by id, checked."""
        if not ready: return {}
        rows, preds, golds = (list(xs) for xs in zip(*ready, strict=True))
        try:
            got = list(self.score_all(rows, preds, golds))
        except Exception as exc:
            row = getattr(exc, "row", None)
            what = f"scoring {row}" if row else f"scoring {errors.counted(len(rows), 'row')} from {rows[0].id}"
            errors.noted(exc, type(self), what)
            raise
        # Note: a scorer can't fail a row (only `_Failed`, from the core, does), so the one rule decides every 0.
        if len(got) != len(rows) or not all(isinstance(x, (Score, errors.Unscored)) for x in got):
            raise TypeError(f"{type(self).__name__}.score_all must return, for each of its {len(rows)} rows, a "
                            f"scorer.Score, or an errors.Unscored for a row it couldn't score; got {len(got)} values: "
                            f"{sorted({type(x).__name__ for x in got})}")
        for r, x in zip(rows, got, strict=True):
            if isinstance(x, Score): _check_score(type(self).__name__, r.id, x)
        return {r.id: x for r, x in zip(rows, got, strict=True)}

    def _keys(self, ready: Sequence[tuple[R, P, G]], code: str) -> dict[str, str]:
        """Each ready row's cache key: a digest of its prediction, gold, the scorer's options and code. A gold that
        isn't JSON is refused; a prediction that isn't gets no key, so it is scored every time."""
        out = {}
        for r, p, g in ready:
            rest = [g, self.options(), code]
            _digest(self, "its gold", rest)
            try:
                out[r.id] = _digest(self, "the prediction", [p, *rest])
            except TypeError:
                log.info("%s: the prediction of %s isn't plain JSON; it is scored but not cached", self.name, r.id)
        return out

    def _as_json(self, x: Score | errors.Unscored) -> Score | errors.Unscored:
        """A score converted to JSON and back, so a fresh score and a kept one hold the same values."""
        if not isinstance(x, Score): return x
        _digest(self, "its results", [x.result, x.details])
        return Score(*json.loads(json.dumps([x.result, x.details])))

    def _line(self, r: R, x: Score | errors.Unscored | _Failed) -> Line:
        """A row's line: what the row is (id, status, row fields), then its score's numbers, or why it has none."""
        row = {"row": f} if (f := self.row_fields(r)) else {}
        errors.plain(f, f"{self.name}.row_fields for row {r.id}")
        match x:
            case Score():
                return {"id": r.id, "status": "scored", **row, **self._result(r, x, ())}
            case errors.Unscored():
                return {"id": r.id, "status": "unscored", **row, "error": errors.capped(str(x))}
            case _Failed():
                zero = self._result(r, x.zero, ("error", "traceback")) if x.zero is not None else {}
                return {"id": r.id, "status": "failed", **row, **zero, "error": errors.capped(x.error),
                        **({"traceback": errors.capped(x.traceback)} if x.traceback else {})}
            case _:
                raise AssertionError(x)

    def _result(self, r: R, x: Score, kept: tuple[str, ...]) -> dict[str, Any]:
        """A score's values for the row's line, refused if they use a key the line keeps for the core (`id`,
        `status`, `row`, and `kept`)."""
        if clash := sorted(set(x.result) & {"id", "status", "row", *kept}):
            raise ValueError(f"{self.name}'s result for row {r.id} holds {clash}, which its line keeps for the core; "
                             "name them otherwise")
        return x.result

    def _framed(self, summary: Summary | None, lines: Sequence[Line], zeroed: int, run: str) -> Summary:
        """The grading's summary: the core's fields, and `summarize`'s summary under `scored`. The headline is
        `scored`'s times coverage, or 0 when no row has a score (`scored` is None)."""
        cover = coverage(lines, zeroed)
        if summary is not None: _check_summary(type(self).__name__, summary)
        return {"scorer": self.name, "options": self.options(), "run": run,
                "headline": 0.0 if summary is None else summary["headline"] * cover,
                "headline_is": "scored.headline × coverage", "rows": len(lines),
                "failed": sum(x["status"] == "failed" for x in lines), "zeroed": zeroed, "coverage": cover,
                "scored": summary}

    def _code(self) -> str:
        """A digest of the code the scorer and its judges run: their classes' names and every Python file of
        their packages (or a class's own file when it is in none), so a fix in a helper module invalidates the
        cache too."""
        cls = [c for c in type(self).__mro__ if issubclass(c, Scorer) and c is not Scorer]
        cls += [c for f in dataclasses.fields(self) if isinstance(v := getattr(self, f.name), adapter.Adapter)
                for c in type(v).__mro__ if issubclass(c, adapter.Adapter) and c is not adapter.Adapter]
        files: set[Path] = set()
        try:
            for c in cls:
                top = sys.modules[c.__module__.split(".")[0]]
                if hasattr(top, "__path__") and top.__name__ != "omni_bench_core":
                    files |= {f for p in top.__path__ for f in Path(p).rglob("*.py")}
                else:
                    files.add(Path(inspect.getfile(c)))
            names = "".join(f"{c.__module__}.{c.__qualname__}" for c in cls).encode()
            return hashlib.sha256(names + b"".join(f.read_bytes() for f in sorted(files))).hexdigest()
        except (OSError, TypeError):
            raise TypeError(f"{self.name} caches its scores, so its code must be readable to tell a change to it; "
                            "define it in a file, not in an interactive session") from None

    def _gold(self, row: R) -> G:
        try:
            return self.gold(row)
        except Exception as exc:
            exc.add_note(f"{self.name} can't read the gold of row {row.id}: either the dataset holds another kind of "
                         f"gold than {self.name} reads, or {type(self).__name__}.gold has a bug")
            raise


def _unfinished(run: runs.Run, missing: Sequence[str], unscored: Sequence[tuple[str, str]], brought: bool) -> str:
    """Why grading refuses `run`, naming the rows and how to finish them."""
    def ids(xs: Sequence[str]) -> str:
        return ", ".join(xs[:SHOWN]) + (f" and {len(xs) - SHOWN} more" if len(xs) > SHOWN else "")

    out = [f"{run.folder} isn't finished, so it has no headline yet:"]
    if missing:
        out.append(f"  {errors.counted(len(missing), 'row')} "
                   + (f"with no file in {run.predictions}" if brought
                      else "never predicted, or answered but not yet read")
                   + f": {ids(missing)}")
    if unscored:
        out.append(f"  {errors.counted(len(unscored), 'row')} not scored ({unscored[0][1][:120]}): "
                   f"{ids([i for i, _ in unscored])}")
    how = []
    if missing: how.append("add their predictions" if brought else "run the command that made this run again, "
                                                                   "which predicts only them")
    if unscored: how.append("grade again to score them (scores already kept aren't paid for again)")
    out.append(f"{' and '.join(how)}; or grade with --partial (partial=True in Python) to count them as failed")
    return "\n".join(out)


def _kept(home: Path, keys: dict[str, str]) -> dict[str, Score]:
    """The kept scores in `<home>/cache/` whose key still matches; a missing or unreadable one is left out."""
    out = {}
    for i, k in keys.items():
        try:
            was = json.loads((home / "cache" / f"{i}.json").read_text())
            if was["key"] == k: out[i] = Score(was["result"], was["details"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return out


def _keep(home: Path, keys: dict[str, str], scores: dict[str, Score | errors.Unscored]) -> None:
    """Writes each scored row's score to `<home>/cache/`, with its key."""
    for i, x in scores.items():
        if not isinstance(x, Score) or i not in keys: continue
        (home / "cache").mkdir(parents=True, exist_ok=True)
        runs.write_json(home / "cache" / f"{i}.json", {"key": keys[i], "result": x.result, "details": x.details})


def _write_details(home: Path, scores: Sequence[tuple[str, Any]]) -> None:
    """Writes each scored (or zeroed) row's details to `<home>/.details.partial/`, which `_write` moves into
    place."""
    for i, x in scores:
        if isinstance(x, _Failed): x = x.zero
        if not isinstance(x, Score) or x.details is None: continue
        (home / ".details.partial").mkdir(parents=True, exist_ok=True)
        runs.write_lines(home / ".details.partial" / f"{i}.jsonl", x.details)


def _write(home: Path, lines: Sequence[Line], summary: Summary) -> None:
    """Moves the details into place, then writes the lines, then the summary, last, so it always stands beside
    the lines it summarizes."""
    home.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(home / "details", ignore_errors=True)
    if (home / ".details.partial").is_dir(): (home / ".details.partial").rename(home / "details")
    runs.write_lines(home / "scores.jsonl", lines)
    runs.write_json(home / "summary.json", summary)


def _discard(home: Path) -> None:
    """Removes the details of a grading that didn't finish."""
    shutil.rmtree(home / ".details.partial", ignore_errors=True)


def _score_one(s: Scorer, id: str, pred: Any, gold: Any) -> Score | errors.Unscored:
    """`s.score` of one row, or its `errors.Unscored`. At module level so a process pool can send it."""
    try:
        got = s.score(pred, gold)
    except errors.Unscored as exc:
        return exc
    except Exception as exc:
        exc.row = id   # Note: kept on the exception, which a process pool sends back whole, to name the row
        raise
    if not isinstance(got, Score):
        raise TypeError(f"{type(s).__name__}.score must return a scorer.Score, not a {type(got).__name__}")
    return got


def _check_score(name: str, id: str, x: Score) -> None:
    """Refuses a `Score` whose `result` isn't a dict, or whose `details` isn't None or a list of dicts."""
    if not isinstance(x.result, dict):
        raise TypeError(f"{name} scored row {id} with a result that is a {type(x.result).__name__}; a result is a "
                        "dict of the row's values, e.g. {\"exact\": True}")
    if x.details is not None and not (isinstance(x.details, list) and all(isinstance(d, dict) for d in x.details)):
        raise TypeError(f"{name} scored row {id} with details that aren't a list of dicts; give a list of dicts, "
                        "one per thing checked, or None")
    errors.plain(x.result, f"{name}'s result for row {id}")
    errors.plain(x.details, f"{name}'s details for row {id}")


def _digest(s: Scorer, what: str, x: Any) -> str:
    """A digest of `x`, which must be JSON, the one form that is the same in every process."""
    try:
        return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{s.name} caches its scores, so {what} must be JSON (str, numbers, bools, None, lists, "
                        f"dicts with str keys): {exc}") from None


def ask[I](judge: adapter.Adapter[I], x: I, *, timeout: float = 300.0) -> dict[str, Any]:
    """Asks `judge` about `x` under a system's rules (`adapter.predict`), and returns its record. Raises
    `errors.Unscored` when the judge didn't answer, so the row waits for a later grading."""
    rec = adapter.predict(judge, x, timeout=timeout)
    if (e := rec["error"]) is not None:
        raise errors.Unscored(f"the judge {judge.name} didn't answer: {e['type']}: {e['message']}")
    return rec


def coverage(lines: Sequence[Line], zeroed: int = 0) -> float:
    """The share of rows `summarize` was given: scored, or failed and `zeroed` by the scorer; unscored rows count
    as neither."""
    n = sum(x["status"] in ("scored", "failed") for x in lines)
    return (sum(x["status"] == "scored" for x in lines) + zeroed) / n if n else 0.0


def _check_summary(name: str, s: Any) -> None:
    """Refuses a summary without a `headline` from 0 to 1 and a `headline_is`, or that isn't plain JSON, before
    anything is written."""
    errors.plain(s, f"{name}.summarize's summary")
    if not isinstance(s, dict) or "headline" not in s:
        raise ValueError(f'{name}.summarize must return a dict with a "headline" key, the number runs are compared '
                         f"by; it returned {s!r:.120}")
    h = s["headline"]
    if isinstance(h, bool) or not isinstance(h, (int, float)) or not 0 <= h <= 1:
        raise ValueError(f'{name}.summarize returned "headline": {h!r}; it must be a number from 0 to 1 where higher '
                         "is better, because a failed row counts as 0 (for an error rate, return 1 - rate)")
    if not (isinstance(s.get("headline_is"), str) and s["headline_is"].strip()):
        raise ValueError(f'{name}.summarize must return a "headline_is" key saying what its headline is '
                         '(e.g. "accuracy")')
