"""Scorers: how a task's predictions are graded.

- `Scorer`: the class to subclass for a way of grading;
- `Score`: what `Scorer.score` returns for a scored row;
- `check`: refuses scorers a run can't be graded by; and
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
# Note: why a row failed. "system": its record says the call failed; "unusable": its prediction can't be used
# (`prediction` raised `SystemFailed`, or the file isn't JSON); "no_answer": it has no prediction; "not_scored": a
# partial grading left it unscored. A finished run's rows fail only by the first two.
CAUSES = ("system", "unusable", "no_answer", "not_scored")
FRAME = ("scorer", "options", "run", "rows", "failed", "coverage", "partial")   # the summary's keys the core writes


class Score(NamedTuple):
    """A scored row's score. `result` is a small dict of the row's values, mostly numbers, which its line in
    scores.jsonl holds; it may not use `id`, `status` or `row`, which the core writes. `details` is an optional
    larger breakdown (a verdict per field, a judge's reasoning), written to `details/<id>.jsonl` and never read
    by `summarize`."""

    result: dict[str, Any]
    details: list[dict[str, Any]] | None = None


class _Failed(NamedTuple):
    """A row with no usable prediction, scored by the scorer's `zero`. Only `grade` makes one: from the row's
    record, a missing prediction, `prediction` raising `SystemFailed` (with its `traceback`), or a partial
    grading's unscored row. `cause` is one of `CAUSES`."""

    cause: str
    error: str
    zero: Score
    traceback: str | None = None


@dataclasses.dataclass(frozen=True)
class Scorer[R: data.Row, G, P](abc.ABC):
    """One way of grading. A scorer is a frozen dataclass whose fields are its options:

        @dataclasses.dataclass(frozen=True)
        class Exact(scorer.Scorer):
            def gold(self, row):                  # the row's gold, read and checked
                return json.loads(row.gt_path.read_text())["text"]

            def score(self, pred, gold):          # one row's numbers
                return scorer.Score({"exact": pred["text"] == gold})

            def zero(self, row, gold):            # a failed row, as the worst score
                return scorer.Score({"exact": False})

            def summarize(self, lines):           # every row's line as one summary
                return {"headline": sum(x["exact"] for x in lines) / len(lines), "headline_is": "exact"}

    `gold`, `zero` and `summarize` are abstract; `score` is required unless `score_all` is written instead.
    Optional, with defaults:

        name            the scorer's name; the class's name in snake case
        prediction      what the scorer reads of a prediction; the prediction itself
        row_fields      what of the row each line carries under "row"; nothing
        detail_fields   what of the row each of its detail lines carries; nothing
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
          zero(row, gold)                   a failed row's score
          row_fields(row)                   each line's "row"
          detail_fields(row)                each detail line's fields from the row
          summarize(lines)                  every row's line, scored or zeroed, as one summary
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
                            "failed, the cache); override score_all to change how rows are scored, zero to change "
                            "what a failed row counts as, or summarize to change the summary")

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
        output it can't use: the row fails. Anything else it raises stops grading."""
        return output

    def score(self, pred: P, gold: G) -> Score:
        """One row's score. Keep it pure (no files, no calls), or set `cache = True` when it asks a judge. Raise
        `errors.Unscored` only when asking again may give a score (the judge timed out or was down): the row
        waits for the next grading. Not abstract, since `score_all` can replace it; a scorer with neither is
        refused."""
        raise NotImplementedError(f"{type(self).__name__} has no score(pred, gold)")

    def detail_fields(self, row: R) -> dict[str, Any]:
        """What of the row each of its detail lines carries (a page, a document), so the details can be sliced
        after grading without the dataset; nothing by default. A detail line that holds one of these keys is
        refused."""
        return {}

    def row_fields(self, row: R) -> dict[str, Any]:
        """What of the row its line carries under `"row"`: fields `summarize` groups by (a document, a language)."""
        return {}

    @abc.abstractmethod
    def zero(self, row: R, gold: G) -> Score:
        """A failed row's score: the worst score the row can get, in the scorer's own units (every test failed,
        every gold character missed). `summarize` gets it as it gets a scored row's, so every number the summary
        holds counts the failure, weighed as the scorer weighs rows. Not `score` of an empty answer, which can
        score above the worst (an empty page passes a test that a text is absent)."""

    @abc.abstractmethod
    def summarize(self, lines: Sequence[Line]) -> Summary:
        """Every row's line as one summary, a failed row's with the score `zero` gave it. Each line is `{"id",
        "row" (if any), ...the result's values}`, without its status: a summary is the same whether a row failed
        or answered as badly as possible.

        Return `headline`, a number from 0 to 1 where higher is better, and `headline_is`, what it is
        ("accuracy"); anything else is the scorer's own, except the keys the core writes (`FRAME`)."""

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
        3. for each batch of `batch_size` rows: fails the rows the core knows failed (`zero`), reuses kept scores,
           scores the rest (`score_all`), and writes their details;
        4. refuses unscored rows, unless `partial`, which fails them;
        5. summarizes every row's line (`summarize`), checks the summary, and adds the core's fields: `rows`,
           `failed` by cause (`CAUSES`), and `coverage`, the share of rows scored; and
        6. writes scores.jsonl and summary.json, only when grading finishes.

        `write=False` writes no results, though it may still rebuild a prediction or fill the cache."""
        if not rows: raise ValueError(f"{run.folder} has no rows to grade")
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
        unscored: list[tuple[str, str]] = []
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
                now = [r for r in chunk if isinstance(scores[r.id], errors.Unscored)]
                unscored += [(r.id, str(scores[r.id])) for r in now]
                if partial: scores |= {r.id: self._failed(r, "not_scored", f"not scored: {scores[r.id]}") for r in now}
                lines += [self._line(r, scores[r.id]) for r in chunk]
                if write and details: _write_details(home, [(r.id, self._details(r, scores[r.id])) for r in chunk])
        except BaseException:
            _discard(home)
            raise
        if unscored and not partial:
            _discard(home)
            raise ValueError(_unfinished(run, [], unscored, brought))
        ys = [{k: v for k, v in x.items() if k not in ("status", "cause", "error", "traceback")} for x in lines]
        summary = self._framed(self.summarize(ys), lines, run.name)
        # Note: counts and a few ids, not every id, which a run stopped early would have thousands of.
        if missing or unscored:
            summary["partial"] = {"no_answer": len(missing), "not_scored": len(unscored),
                                  "e.g.": [*missing, *(i for i, _ in unscored)][:SHOWN]}
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
                failed[r.id] = self._failed(r, "system", f"{err['type']}: {err['message']}")
                continue
            try:
                output = run.prediction(r.id)
            except FileNotFoundError:
                failed[r.id] = self._failed(r, "no_answer", f"no prediction: predictions/{r.id}.json")
                continue
            except ValueError as exc:   # Note: the core's own check that the file is JSON
                failed[r.id] = self._failed(r, "unusable", str(exc))
                continue
            try:
                pred = self.prediction(output)
            except errors.SystemFailed as exc:
                failed[r.id] = self._failed(r, "unusable", str(exc), "".join(traceback.format_exception(exc)))
                continue
            except Exception as exc:
                errors.noted(exc, type(self), f"reading the prediction of {r.id}")
                raise
            ready.append((r, pred, self._gold(r)))
        return failed, ready

    def _failed(self, r: R, cause: str, error: str, tb: str | None = None) -> _Failed:
        """A failed row, with the score `zero` gives it."""
        assert cause in CAUSES, cause
        z = self.zero(r, self._gold(r))
        if not isinstance(z, Score):
            raise TypeError(f"{type(self).__name__}.zero must return a scorer.Score; for row {r.id} it returned a "
                            f"{type(z).__name__}")
        _check_score(type(self).__name__, r.id, z)
        return _Failed(cause, error, z, tb)

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
                return {"id": r.id, "status": "failed", "cause": x.cause, **row,
                        **self._result(r, x.zero, ("cause", "error", "traceback")), "error": errors.capped(x.error),
                        **({"traceback": errors.capped(x.traceback)} if x.traceback else {})}
            case _:
                raise AssertionError(x)

    def _details(self, r: R, x: Score | errors.Unscored | _Failed) -> list[dict[str, Any]] | None:
        """A row's detail lines, a scored row's or a failed row's from `zero`, each with the row's `detail_fields`."""
        if isinstance(x, _Failed): x = x.zero
        if not isinstance(x, Score) or x.details is None: return None
        if not (f := self.detail_fields(r)): return x.details
        errors.plain(f, f"{self.name}.detail_fields for row {r.id}")
        if clash := sorted({k for d in x.details for k in d} & set(f)):
            raise ValueError(f"{self.name}'s details for row {r.id} hold {clash}, which its detail_fields add; name "
                             "them otherwise")
        return [f | d for d in x.details]

    def _result(self, r: R, x: Score, kept: tuple[str, ...]) -> dict[str, Any]:
        """A score's values for the row's line, refused if they use a key the line keeps for the core (`id`,
        `status`, `row`, and `kept`)."""
        if clash := sorted(set(x.result) & {"id", "status", "row", *kept}):
            raise ValueError(f"{self.name}'s result for row {r.id} holds {clash}, which its line keeps for the core; "
                             "name them otherwise")
        return x.result

    def _framed(self, summary: Summary, lines: Sequence[Line], run: str) -> Summary:
        """The grading's summary: the core's fields (`FRAME`), then `summarize`'s, checked."""
        _check_summary(type(self).__name__, summary)
        if clash := sorted(set(summary) & set(FRAME)):
            raise ValueError(f"{type(self).__name__}.summarize returned {clash}, which the core writes; name them "
                             "otherwise")
        failed = [x["cause"] for x in lines if x["status"] == "failed"]
        return {"scorer": self.name, "options": self.options(), "run": run, "rows": len(lines),
                "failed": {c: failed.count(c) for c in CAUSES},
                "coverage": coverage(lines), **summary}

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


def _write_details(home: Path, details: Sequence[tuple[str, list[dict[str, Any]] | None]]) -> None:
    """Writes each row's detail lines, if it has any, to `<home>/.details.partial/`, which `_write` moves into
    place."""
    for i, ds in details:
        if ds is None: continue
        (home / ".details.partial").mkdir(parents=True, exist_ok=True)
        runs.write_lines(home / ".details.partial" / f"{i}.jsonl", ds)


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


def check(scorers: Sequence[Scorer]) -> None:
    """Refuses scorers a run can't be graded by, before any call: none, one that isn't a `Scorer` dataclass or has
    neither `score` nor `score_all`, or two that write one results folder."""
    for s in scorers:
        errors.check_dataclass(type(s), Scorer)
        # Note: `score` can't be abstract, since a scorer that scores rows together writes `score_all` instead;
        # so a scorer with neither is refused here, before any call, not after the answers are paid for.
        if type(s).score is Scorer.score and type(s).score_all is Scorer.score_all:
            raise TypeError(f"{type(s).__name__} has neither score(pred, gold) nor score_all(rows, preds, golds); "
                            "write score, or score_all to score rows together")
    if not scorers:
        raise ValueError("a benchmark needs at least one scorer")
    if len(names := [s.folder_name for s in scorers]) != len(set(names)):
        raise ValueError(f"scorers {names} name one results folder twice: the same scorer with the "
                         "same options")


def ask[I](judge: adapter.Adapter[I], x: I, *, timeout: float = 300.0) -> dict[str, Any]:
    """Asks `judge` about `x` under a system's rules (`adapter.predict`), and returns its record. Raises
    `errors.Unscored` when the judge didn't answer, so the row waits for a later grading."""
    rec = adapter.predict(judge, x, timeout=timeout)
    if (e := rec["error"]) is not None:
        raise errors.Unscored(f"the judge {judge.name} didn't answer: {e['type']}: {e['message']}")
    return rec


def coverage(lines: Sequence[Line]) -> float:
    """The share of rows scored: the rest failed, and counted as `zero` scored them."""
    return sum(x["status"] == "scored" for x in lines) / len(lines)


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
                         "is better, so every headline reads one way (for an error rate, return 1 - rate)")
    if not (isinstance(s.get("headline_is"), str) and s["headline_is"].strip()):
        raise ValueError(f'{name}.summarize must return a "headline_is" key saying what its headline is '
                         '(e.g. "accuracy")')
