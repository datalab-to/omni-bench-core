"""Scorers: how a task's predictions are graded.

`Scorer` is the class you subclass to add a way of grading, and its docstring says how. The rest of the
module:

- `Score`: what `Scorer.score` returns for one row;
- `ask`: sends a judge (an adapter) one question, under the same rules as a system's calls; and
- `coverage`, `scored` and `per`: helpers for writing your own `summarize`.
"""

from __future__ import annotations

import collections
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
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar, NamedTuple

from omni_bench_core import adapter, data, errors
from omni_bench_core import run as runs

log = logging.getLogger(__name__)

type Line = dict[str, Any]
type Summary = dict[str, Any]

LINE = ("id", "suite", "status", "row")   # what a line holds besides its row's result
FRAME = ("scorer", "options", "run", "rows", "failed", "coverage", "headline_is", "partial")   # what grade adds
SHOWN = 10   # how many ids a message about unfinished rows names
POOLS = ("process", "thread")


class Score(NamedTuple):
    """One row's score. `result` is its numbers, one line of scores.jsonl. `details` is an optional breakdown
    kept beside the lines, for finding where a system went wrong, never read by `summarize`."""

    result: dict[str, Any]
    details: list[dict[str, Any]] | None = None

    @classmethod
    def failed(cls, why: str) -> Score:
        """The score of a row with no usable prediction: the system's failure, counted as 0."""
        return cls({"status": "failed", "error": why[:300]})

    @classmethod
    def unscored(cls, why: str) -> Score:
        """The score of a row the scorer couldn't score, for a reason that isn't the system's. Grading isn't
        finished until a later grading scores it."""
        return cls({"status": "unscored", "error": why[:300]})


@dataclasses.dataclass(frozen=True)
class Scorer[R: data.Row, G, P]:
    """One way of grading. Its fields are its options.

    The smallest scorer is a frozen dataclass with two methods:

        @dataclasses.dataclass(frozen=True)
        class Exact(scorer.Scorer):
            def gold(self, row):                  # the row's gold, read and checked
                return json.loads(row.gt_path.read_text())["text"]

            def score(self, pred, gold):          # one row's numbers
                return scorer.Score({"exact": pred["text"] == gold})

    Its summary is then the mean of each number over the scored rows, overall and per suite, and its
    headline is the mean of the first number.

    Everything else is optional and has a default:

        name            the scorer's name; by default the class's name, in snake case
        prediction      what the scorer reads of a prediction; by default the prediction as it is
        row_fields      what of the row each line carries, under "row", for a summary to group by; nothing
        summarize       all the lines as one summary; by default the mean of each number
        headline_field  which number the default `summarize` makes the headline; by default the first
        check_gold      refuses a dataset before any call; by default it reads every gold
        score_all       many rows' scores at once; by default `score` for each row
        workers         how many rows are scored at once; 1
        pool            what scores them at once: "process" for scoring that computes, "thread" for
                        scoring that waits on calls; "process"
        cache           whether to keep each row's score and reuse it while nothing it depends on changed;
                        False
        batch_size      how many rows are read and given to `score_all` at once; 500, or None for every row

    What a score holds. `score` returns a `Score` with two parts. `result` is the row's numbers, which
    become its line in scores.jsonl. `details` is optional: a breakdown too large for a line, such as one
    verdict per field, a diff, or a judge's reasoning. It is written to `details/<id>.jsonl`. Every row's
    result is kept in memory until the summary is made, so a result should be a few numbers. Anything large
    goes in `details`, which is written after each batch and never held for the whole run.

    Options. A scorer's fields are its options, as for an adapter. They are written to its summary, and they
    are part of its results folder's name, `<run>/<name>-<digest>/` (or `<run>/<name>/` when it has none).
    So two configurations of one scorer never overwrite each other.

    There are four ways to grade, from most to least common:

    1. `gold` and `score`, plus `summarize` when a mean isn't the right summary (for example an error rate
       over all characters, or a score built from counts). Use this when each row's score depends only on
       its prediction and its gold;
    2. the same, with `workers = N`, when scoring is slow to compute. The scorer, and each prediction and
       gold, are sent to another process, so define the scorer in a module, or under
       `if __name__ == "__main__":` in a script;
    3. a judge. `score` asks a model with `ask`, through an adapter that is one of the scorer's fields. Set
       `cache = True`, so each answer is paid for once and grading again gives the same scores. Set
       `pool = "thread"` and `workers = N`, because a judge waits on calls. When the judge doesn't answer,
       the row is unscored (`ask` raises `errors.Unscored`), and the next grading asks again for those rows
       only:

           @dataclasses.dataclass(frozen=True)
           class Faithful(scorer.Scorer):
               cache: ClassVar[bool] = True
               pool: ClassVar[str] = "thread"
               workers: ClassVar[int] = 16
               judge: adapter.Adapter = OpenAI(model="gpt-5.5")

               def score(self, pred, gold):
                   rec = scorer.ask(self.judge, Prompt(pred, gold))
                   return scorer.Score({"faithful": rec["output"]["verdict"] == "yes",
                                        "judge_usd": rec["cost"]["usd"]}, [rec["output"]])

    4. override `score_all(rows, preds, golds)`, when rows are scored together rather than one by one: a
       judge or a model sent a batch, or a score that needs the other rows. It gets the same values `score`
       would, for every row that has a prediction to score, and returns one `Score` per row
       (`Score.unscored` for a row it couldn't score). It is given `batch_size` rows at a time; a score that
       needs every row at once sets `batch_size = None`. With `cache = True` it is given only the rows
       without a kept score.

    How the methods fit together. `grade` does all the reading and writing, and applies every rule. The
    methods it calls only compute:

        grade(run, rows)                              reads files, applies the rules, writes; not overridden
          gold(row), prediction(output)               each row's gold and prediction, read once
          score_all(rows, preds, golds)               `batch_size` rows at a time, without kept scores
            score(pred, gold)                         one row; by default called for each row
          row_fields(row)                             each line's "row"
          summarize(lines)                            the summary, before the rules are applied to it

    `grade` can't be overridden; a subclass that tries is refused. It holds the rules that make every
    scorer's headline mean the same thing:

    - only a finished run is graded: every row has an answer or a recorded failure, and a score. A row with
      neither (never predicted, or no prediction in a folder of your own), or one the scorer couldn't score
      (a judge that didn't answer), makes grading refuse (`errors.Unfinished`), naming the rows and how to
      finish them. `partial=True` grades anyway, counts those rows as failed, and says so in the summary;
    - a row with no usable answer fails and counts as 0 in the headline: `summarize` computes the headline over
      the scored rows, and `grade` multiplies it by `coverage`, the share of rows scored rather than failed,
      so failing on hard rows can't raise it. The summary's `headline_is` says so ("accuracy × coverage").
      Each failed row is a line of scores.jsonl with its `error`, and its record holds the whole failure, so
      failures can be found and audited;
    - the same inputs get the same score. A scorer scores every row again on every grading. A cached scorer
      keeps a row's score while the prediction, the gold, the scorer's options and the code of the scorer
      and its judge (every Python file of the packages they come from) are unchanged; and
    - a grading's results are written only when it finishes. A cached scorer's cache is written as each row
      is scored, so an interrupted or unfinished grading keeps what its judge was paid for; and
    - a scorer that raises anything else has a bug: grading stops, and prints its whole traceback and whose
      code it is.

    A cached scorer's predictions, golds and results must be JSON (strings, numbers, booleans, None, lists
    and dicts). That way a row's key is the same in every process, and a kept score is exactly the score it
    replaced.

    A headline is a number from 0 to 1 where higher is better, so every headline reads the same way.
    """

    name: ClassVar[str]
    headline_field: ClassVar[str | None] = None
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
        """The name of the scorer's results folder in a run: its name, then a digest of its options if it has
        any."""
        o = self.options()
        if not o: return self.name
        return f"{self.name}-{hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:8]}"

    def options(self) -> dict[str, Any]:
        """Every option of the scorer, as its summary states them. A judge adapter appears as its name and
        options."""
        errors.check_dataclass(type(self), Scorer)
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = {"system": v.name, **v.options()} if isinstance(v, adapter.Adapter) else v
        return out

    def gold(self, row: R) -> G:
        """The row's gold, read and checked.

        Each scorer writes its own, because the shape of the gold belongs to the scorer. The dataset says
        where a row's gold is (a gold file, a column), and the scorer says what it holds (unit tests, a
        transcription, an extraction). Raise when it can't be read: the dataset is wrong for this scorer, so
        grading stops and names the row."""
        raise NotImplementedError(f"{type(self).__name__} has no gold(row)")

    def prediction(self, output: Any) -> P:
        """What the scorer reads of one prediction, which is `Adapter.read`'s output.

        It is separate from `score` for two reasons. First, scorers of one task read different parts of one
        output (the html, or the layout blocks). Second, a prediction that can't be read is the system's
        failure, not a bug: raise ValueError, and that row fails (counts as 0) while grading goes on."""
        return output

    def score(self, pred: P, gold: G) -> Score:
        """One row's score. This is the comparison itself, and the one method every scorer writes.

        It should be pure: it reads no files, makes no calls, and uses nothing but its arguments. Then the
        same answers always get the same scores, in any order and in any process. A scorer that asks a judge
        isn't pure, so it sets `cache = True`, which gives the same guarantee. Raise `errors.Unscored` when the
        row can't be scored for a reason that isn't the system's fault, such as a judge that didn't answer.
        The row is then left for a later grading instead of counting against the system."""
        raise NotImplementedError(f"{type(self).__name__} has no score(pred, gold)")

    def row_fields(self, row: R) -> dict[str, Any]:
        """What of the row its line carries, under `"row"`. Use it for fields a summary breaks down by that the
        row holds and the score doesn't, such as a source document, tags, or a language in `row.extra`.

        It is separate from `gold` because nothing is graded against it. It goes on the line, rather than
        being handed to `summarize` with the rows, so any part of a scores.jsonl can be summarized on its own."""
        return {}

    def summarize(self, lines: Sequence[Line]) -> Summary:
        """All the lines as one summary, with a `headline`: the one number runs are compared by.

        Each scorer can write its own, because rows combine in different ways: a mean of each row's score,
        or one rate over all rows' counts. It reads the scored rows; `grade` makes a failed row count as 0 in the
        headline. The default takes the mean of each number over the scored rows (a boolean counts as 0 or 1),
        overall and per suite. Its headline is the mean of `headline_field`, or of the first number."""
        ys = scored(lines)
        keys = [k for k, v in ys[0].items() if k not in LINE and isinstance(v, (int, float))] if ys else []
        if ys and not keys:
            raise TypeError(f"{self.name}'s results hold no numbers to average (e.g. {ys[0]}); write its summarize")
        head = self.headline_field or (keys[0] if keys else None)
        if ys and head not in keys:
            raise TypeError(f"{self.name}.headline_field is {head!r}, but its results' numbers are {keys}")

        def over(xs: Sequence[Line]) -> Summary:
            zs = scored(xs)

            def mean(k: str) -> float | None:
                vs = [float(z[k]) for z in zs if z.get(k) is not None]
                return fmean(vs) if vs else None

            return {"scored": len(zs), **{k: mean(k) for k in keys}}

        out = per(lines, "suite", over)
        # Note: when rows were graded but none scored, every row failed, so the headline is 0. When no rows
        # were graded at all, there is no headline.
        return {"headline": out[head] if ys else (0.0 if lines else None), **out}

    def check_gold(self, rows: Sequence[R]) -> None:
        """Refuses a dataset whose gold this scorer can't read. `plan` calls it before any call, so a wrong
        dataset is found before answers are paid for. It reads every gold; override it when there is a
        quicker check that is just as sure."""
        for r in rows: self._gold(r)

    def score_all(self, rows: Sequence[R], preds: Sequence[P], golds: Sequence[G]) -> list[Score]:
        """Many rows' scores, in order. It is to many rows what `score` is to one.

        It is the one place to change how rows are scored together: a batch sent to a judge, a model run
        over many rows at once, or a score that needs the other rows. It gets values, never files, and only
        for rows that have a prediction to score. Deciding which rows those are, and which rows are failed
        or unfinished, is `grade`'s job. Return `Score.unscored` for a row it can't score. By default it calls
        `score` for each row, `workers` at a time, in the scorer's `pool`."""
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
            raise errors.UsageError(f"{self.name} scores in {n} processes, and they couldn't start or died. In a "
                                    "script, put the code that runs the benchmark under `if __name__ == "
                                    "\"__main__\":`, because each process imports the script again (macOS and "
                                    "Windows); otherwise a process ran out of memory") from exc

    def grade(self, run: runs.Run, rows: Sequence[R], *, partial: bool = False, details: bool = True,
              write: bool = True) -> Summary:
        """Grades `run` on `rows`, writes the results under `<run>/<folder_name>/`, and returns the summary.

        Every read and every rule is here. In order, it:

        1. refuses a folder of your own whose files name none of the rows;
        2. rebuilds lost predictions from their records;
        3. refuses rows with neither an answer nor a recorded failure, unless `partial`;
        4. for each batch: reads which rows failed, reads the others' predictions and golds, reuses kept
           scores (`cache`), scores the rest (`score_all`), and writes their details;
        5. refuses rows it couldn't score, unless `partial`;
        6. makes the lines (with `row_fields`), and summarizes them (`summarize`);
        7. applies the headline rule; and
        8. writes the lines and the summary.

        `rows` are the rows the run answers. `write=False` grades by the same rules and writes no results,
        for trying a scorer out. It may still rebuild a lost prediction and fill the cache, because both keep
        what was paid for."""
        files, brought = run.prediction_files(), run.brought()
        if brought and not files & {f"{r.id}.json" for r in rows}:
            raise errors.UsageError(f"none of the {errors.counted(len(files), 'file')} in {run.predictions} "
                                    f"names a row of the {len(rows)} graded (e.g. {rows[0].id}.json): is the "
                                    "dataset right?")
        if rebuilt := run.restore(rows):
            log.info("%s: %s rebuilt from records, e.g. %s", run.folder, errors.counted(len(rebuilt), "prediction"),
                     rebuilt[:3])
        missing = [r.id for r in rows if not run.attempted(r.id, files)]
        if missing and not partial:
            raise errors.Unfinished(_unfinished(run, missing, [], brought))
        home = run.folder / self.folder_name
        code = self._code() if self.cache else ""
        lines: list[Line] = []
        _discard(home)
        try:
            # Note: we go `batch_size` rows at a time so that no more than one batch's predictions, golds and
            # details are in memory at once. Each batch's details are written before the next batch is read.
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
                lines += [self._line(r, scores[r.id]) for r in chunk]
                if write and details: _write_details(home, [(r.id, scores[r.id]) for r in chunk])
        except BaseException:
            _discard(home)
            raise
        unscored = [x["id"] for x in lines if x["status"] == "unscored"]
        if unscored and not partial:
            _discard(home)
            raise errors.Unfinished(_unfinished(run, [], [(x["id"], x["error"]) for x in lines
                                                           if x["status"] == "unscored"], brought))
        lines = [{**x, "status": "failed", "error": f"not scored: {x['error']}"} if x["status"] == "unscored" else x
                 for x in lines]
        summary = self._framed(self.summarize(lines), lines, run.name)
        if missing or unscored: summary["partial"] = {"no_answer": missing, "not_scored": unscored}
        if write: _write(home, lines, summary)
        else: _discard(home)
        return summary

    def _read(self, run: runs.Run, rows: Sequence[R]) -> tuple[dict[str, Score], list[tuple[R, P, G]]]:
        """Reads some rows. Returns the scores of the rows that failed (the call failed, or the prediction is
        missing or can't be read), and the other rows with their predictions and golds, ready to score."""
        failed: dict[str, Score] = {}
        ready: list[tuple[R, P, G]] = []
        for r in rows:
            if err := (run.record(r.id) or {}).get("error"):
                failed[r.id] = Score.failed(f"{err['type']}: {err['message']}")
                continue
            try:
                pred = self.prediction(run.prediction(r.id))
            except FileNotFoundError:
                failed[r.id] = Score.failed(f"no prediction: predictions/{r.id}.json")
                continue
            except ValueError as exc:
                failed[r.id] = Score.failed(str(exc))
                continue
            ready.append((r, pred, self._gold(r)))
        return failed, ready

    def _scored(self, ready: Sequence[tuple[R, P, G]]) -> dict[str, Score]:
        """`score_all` of the rows ready to score, by id. Checks that it gave one `Score` per row. A bug in the
        scorer stops grading (`errors.Bug`)."""
        if not ready: return {}
        rows, preds, golds = (list(xs) for xs in zip(*ready, strict=True))
        try:
            got = list(self.score_all(rows, preds, golds))
        except (errors.UsageError, errors.Stop):
            raise
        except Exception as exc:
            row = getattr(exc, "row", None)
            what = f"scoring {row}" if row else f"scoring {errors.counted(len(rows), 'row')} from {rows[0].id}"
            raise errors.bug(what, exc, type(self)) from None
        if len(got) != len(rows) or not all(isinstance(x, Score) for x in got):
            raise TypeError(f"{type(self).__name__}.score_all must return one scorer.Score per row: {len(rows)} "
                            f"rows, got {len(got)} values")
        return {r.id: x for r, x in zip(rows, got, strict=True)}

    def _keys(self, ready: Sequence[tuple[R, P, G]], code: str) -> dict[str, str]:
        """Each ready row's cache key: a digest of its prediction, its gold, the scorer's options and the code.

        A gold that isn't JSON is refused: it is the scorer's own choice, and would never be cached. A prediction
        that isn't (a NaN a system answered) is the system's: that row gets no key, so it is scored every time
        and never cached."""
        out = {}
        for r, p, g in ready:
            rest = [g, self.options(), code]
            _digest(self, "its gold", rest)
            try:
                out[r.id] = _digest(self, "the prediction", [p, *rest])
            except TypeError:
                log.info("%s: the prediction of %s isn't plain JSON; it is scored but not cached", self.name, r.id)
        return out

    def _as_json(self, x: Score) -> Score:
        """A score converted to JSON and back, so that a fresh score and a kept one hold the same values."""
        if x.result.get("status", "scored") != "scored": return x
        _digest(self, "its results", [x.result, x.details])
        return Score(*json.loads(json.dumps([x.result, x.details])))

    def _line(self, r: R, x: Score) -> Line:
        """A row's line: what the row is (id, suite, status, row fields), then its score's numbers."""
        if clash := sorted(set(x.result) & {"id", "suite", "row"}):
            raise TypeError(f"{self.name}'s result for row {r.id} holds {clash}, which a line keeps for the row")
        return {"id": r.id, "suite": r.suite, "status": "scored", **({"row": f} if (f := self.row_fields(r)) else {}),
                **x.result}

    def _framed(self, summary: Summary, lines: Sequence[Line], run: str) -> Summary:
        """`summarize`'s summary, with what every summary holds added, and the headline rule applied: the
        headline is multiplied by coverage, so a failed row counts as 0. When no row was scored, it is 0."""
        if "headline" not in summary:
            raise TypeError(f"{type(self).__name__}.summarize must return a summary with a headline")
        if clash := sorted(set(summary) & set(FRAME)):
            raise TypeError(f"{type(self).__name__}.summarize returns {clash}, which grade sets: name them otherwise")
        cover = coverage(lines)
        head = summary["headline"]
        return {"scorer": self.name, "options": self.options(), "run": run, "rows": len(lines),
                "failed": sum(x["status"] == "failed" for x in lines), "coverage": cover,
                "headline_is": f"{self._headline_name(lines)} × coverage", **summary,
                "headline": 0.0 if head is None else head * cover}

    def _headline_name(self, lines: Sequence[Line]) -> str:
        """What the headline is before coverage: `headline_field`, or the default summary's first number."""
        if self.headline_field: return self.headline_field
        if type(self).summarize is not Scorer.summarize: return "the summary's headline"
        ys = scored(lines)
        return next((k for k, v in ys[0].items() if k not in LINE and isinstance(v, (int, float))), "headline") if ys \
            else "headline"

    def _code(self) -> str:
        """A digest of the code the scorer and its judges run: their classes' names, and every Python file of the
        packages their classes come from, or the file a class is defined in when it is in no package. A change
        to how it scores (a prompt, a parser, a rule, or a helper in another module of its package) changes
        every row's cache key.

        It reads whole packages, not only the classes, because a scorer often hands the comparison to helper
        modules, and a fix there must not leave stale scores behind. A change elsewhere in the package only
        costs a rescoring."""
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
        except errors.UsageError:
            raise
        except Exception as exc:
            raise errors.UsageError(f"{self.name} can't read the gold of row {row.id}: {exc}; the dataset may hold "
                                    f"another kind of gold than {self.name} reads, or this gold is malformed") from None


def _unfinished(run: runs.Run, missing: Sequence[str], unscored: Sequence[tuple[str, str]], brought: bool) -> str:
    """Why grading refuses `run`, naming the rows and how to finish them."""
    def ids(xs: Sequence[str]) -> str:
        return ", ".join(xs[:SHOWN]) + (f" and {len(xs) - SHOWN} more" if len(xs) > SHOWN else "")

    out = [f"{run.folder} isn't finished, so it has no headline yet:"]
    if missing:
        out.append(f"  {errors.counted(len(missing), 'row')} "
                   + (f"with no file in {run.predictions}" if brought else "never predicted") + f": {ids(missing)}")
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
    """The kept scores in `<home>/cache/` whose key still matches the row's. A missing or unreadable one is
    left out, so that row is scored again."""
    out = {}
    for i, k in keys.items():
        try:
            was = json.loads((home / "cache" / f"{i}.json").read_text())
            if was["key"] == k: out[i] = Score(was["result"], was["details"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return out


def _keep(home: Path, keys: dict[str, str], scores: dict[str, Score]) -> None:
    """Writes each scored row's score to `<home>/cache/`, with its key."""
    for i, x in scores.items():
        if x.result.get("status", "scored") != "scored" or i not in keys: continue
        (home / "cache").mkdir(parents=True, exist_ok=True)
        runs.write_json(home / "cache" / f"{i}.json", {"key": keys[i], "result": x.result, "details": x.details})


def _write_details(home: Path, scores: Sequence[tuple[str, Score]]) -> None:
    """Writes each row's details to `<home>/.details.partial/`. `_write` moves them into place when grading
    finishes."""
    for i, x in scores:
        if x.details is None: continue
        (home / ".details.partial").mkdir(parents=True, exist_ok=True)
        runs.write_lines(home / ".details.partial" / f"{i}.jsonl", x.details)


def _write(home: Path, lines: Sequence[Line], summary: Summary) -> None:
    """Moves the details written during grading into place, then writes the lines, then the summary. The
    summary is last, so it always stands beside the lines it summarizes."""
    home.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(home / "details", ignore_errors=True)
    if (home / ".details.partial").is_dir(): (home / ".details.partial").rename(home / "details")
    runs.write_lines(home / "scores.jsonl", lines)
    runs.write_json(home / "summary.json", summary)


def _discard(home: Path) -> None:
    """Removes the details of a grading that didn't finish."""
    shutil.rmtree(home / ".details.partial", ignore_errors=True)


def _score_one(s: Scorer, id: str, pred: Any, gold: Any) -> Score:
    """`s.score` of one row. A row it can't score is unscored, and anything else it raises is a bug. It is at
    the top of the module so a process pool can send it to another process."""
    try:
        got = s.score(pred, gold)
    except errors.Unscored as exc:
        return Score.unscored(str(exc))
    except (errors.UsageError, errors.Stop):
        raise
    except Exception as exc:
        exc.add_note(f"{s.name}: scoring row {id} raised, a bug in the scorer; nothing was written")
        exc.row = id   # Note: kept on the exception, which a process pool sends back whole, to name the row
        raise
    if not isinstance(got, Score):
        raise TypeError(f"{type(s).__name__}.score must return a scorer.Score, not a {type(got).__name__}")
    return got


def _digest(s: Scorer, what: str, x: Any) -> str:
    """A digest of `x`, which must be JSON. Anything else has no written form that is the same in every
    process."""
    try:
        return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{s.name} caches its scores, so {what} must be JSON (str, numbers, bools, None, lists, "
                        f"dicts with str keys): {exc}") from None


def ask[I](judge: adapter.Adapter[I], x: I, *, timeout: float = 300.0) -> dict[str, Any]:
    """Asks `judge` about `x`, and returns the judge's record, which holds its answer.

    The call follows the same rules as a system's calls (`adapter.predict`): a time budget, retries, and a
    stop on an account failure. Raises `errors.Unscored` when the judge didn't answer, so the row is left
    for a later grading rather than failed."""
    rec = adapter.predict(judge, x, timeout=timeout)
    if (e := rec["error"]) is not None:
        raise errors.Unscored(f"the judge {judge.name} didn't answer: {e['type']}: {e['message']}")
    return rec


def coverage(lines: Sequence[Line]) -> float:
    """The share of rows that were scored rather than failed. Unscored rows count as neither."""
    n = sum(x["status"] in ("scored", "failed") for x in lines)
    return sum(x["status"] == "scored" for x in lines) / n if n else 0.0


def scored(lines: Sequence[Line]) -> list[Line]:
    """The lines of rows that were scored. A summary's numbers are computed over these."""
    return [x for x in lines if x["status"] == "scored"]


def per(lines: Sequence[Line], field: str, over: Callable[[Sequence[Line]], Summary]) -> Summary:
    """`over` of all the lines, then of each group of lines under `per_<field>`.

    Lines are grouped by `field`, which is either a field of the line (`"suite"`), or one of its row fields
    (`"row.language"`, from `Scorer.row_fields`)."""
    by = collections.defaultdict(list)
    for x in lines: by[_field(x, field)].append(x)
    name = field.removeprefix("row.")
    return {**over(lines), f"per_{name}": {g: over(xs) for g, xs in sorted(by.items(), key=lambda kv: str(kv[0]))}}


def _field(x: Line, field: str) -> Any:
    if not field.startswith("row."): return x[field]
    return x.get("row", {}).get(field.removeprefix("row."))
