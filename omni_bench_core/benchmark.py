"""Benchmarks: a task's dataset and the scorers that grade it, run on predictive systems.

`Benchmark` is the class to use, and its docstring says how. `Plan` is what `Benchmark.plan` checks and
`Benchmark.predict` carries out. `predict_row` is the work for one row, which an executor may run elsewhere.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from omni_bench_core import adapter, data, errors, progress, scorer, task
from omni_bench_core import run as runs

log = logging.getLogger(__name__)


class Plan(NamedTuple):
    """One run, before any call: the run, the question it answers, the rows it chooses, and how. `Benchmark.plan`
    builds and checks it, so `predict` has nothing left to refuse."""

    run: runs.Run
    question: Mapping[str, Any]      # the dataset, manifest, task and choice of rows the run answers
    rows: tuple[data.Row, ...]
    timeout: float
    workers: int                     # rows in flight at once
    retry_failed: bool


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """A task, its dataset and manifest, and the scorers that grade it.

        oeb = benchmark.Benchmark(Extract(), "hf://datasets/datalab-to/omni_extract_bench", [FieldMatching()])
        oeb.run(Datalab(mode="accurate"), out="runs", sample=20)
        oeb.grade("mine/")                  # a folder of your own predictions, predicting nothing

    A benchmark is what stays fixed while systems change: one task, one dataset and manifest, and the
    scorers. So every system run on it is graded on the same rows by the same rules, and runs of different
    systems can be compared.

    A run is one system on the benchmark: to compare systems, run each, with the same choice of rows. A run has
    two steps. Predicting sends each row to the system and saves its answer. Grading compares each
    saved answer with its gold, once per scorer. Predicting costs money and time, and grading is free, so
    the two are separate: an answer is paid for once and graded as often as we like.

    Three rules say what happens in every case:

    1. a run folder answers one exact question: the system and its options, the dataset, the manifest, the
       task and the choice of rows (`suites`, `sample`, `limit`, `only`). Another question gets another
       folder; the same question resumes the same one;
    2. grading reports only a finished run, one where every row it chose has an answer or a recorded failure,
       and a score. Otherwise it refuses, and names the rows and how to finish them; `partial=True` grades
       anyway, counting them as failed; and
    3. a failure that is the system's (a refusal, an error, an answer that can't be read, a timeout) is
       recorded, and the row counts 0. A failure that is ours (a missing key, an account that can't pay, a
       bug, an interrupt) stops the run with nothing recorded, so running the same command again does the
       row. A bug prints its whole traceback, and whose code it is.

    It has six methods:

        plan      the run, checked before any call: the options, the keys, the rows, their input files and
                  each scorer's gold
        predict   sends every row the run has no record of to its system; a stopped run resumes
        grade     each scorer on a finished run, over the rows the run answers
        run       plan, then predict, then grade
        reread    reads a run's saved answers again with its adapter's `read`, after a fix to it, with no call
        check     checks the dataset (manifest, input files, golds) without calling anything

    The dataset is a folder or an `hf://` address, never rows made in code, because a run folder records which
    dataset its answers are for.
    """

    task: task.Task
    dataset: str | Path
    scorers: Sequence[scorer.Scorer]
    manifest: str = data.MANIFEST

    def __post_init__(self) -> None:
        errors.check_dataclass(type(self.task), task.Task)
        for s in self.scorers: errors.check_dataclass(type(s), scorer.Scorer)
        if not self.scorers:
            raise errors.UsageError("a benchmark needs at least one scorer")
        if len(names := [s.folder_name for s in self.scorers]) != len(set(names)):
            raise errors.UsageError(f"scorers {names} name one results folder twice: the same scorer with the "
                                    "same options")

    def rows(self) -> tuple[data.Row, ...]:
        """Every row of the manifest."""
        return self.task.rows(self.dataset, self.manifest)

    def question(self, selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What a run on this benchmark answers, besides its system: the dataset, the manifest, the task and the
        choice of rows (`data.selection`)."""
        return {"dataset": data.dataset_id(self.dataset), "manifest": data.manifest_path(self.manifest),
                "task": self.task.identity(), "rows": dict(selection or {})}

    def plan(self, system: adapter.Adapter, out: str | Path, *, suites: Iterable[str] | None = None,
             limit: int = 0, sample: int = 0, only: Iterable[str] | None = None,
             timeout: float = adapter.DEFAULT_TIMEOUT, workers: int = 0, retry_failed: bool = False) -> Plan:
        """The run of `system`, in its own folder under `out`, checked before any call.

        Args:
          suites, only, sample, limit: choose the rows (`data.select`). They are part of the run's question,
            so another choice of rows is another run, and grading covers exactly these rows.
          timeout: the seconds each row may take.
          workers: how many rows are in flight at once; 0 for the system's own `Adapter.workers`.
          retry_failed: predict again the rows whose recorded failure might not happen again.
        """
        if not isinstance(system, adapter.Adapter):
            raise TypeError(f"{system!r} is not one adapter instance; make one, e.g. Datalab(mode='accurate'). A run "
                            "is one system: to compare systems, run each")
        if not timeout > 0:
            raise errors.UsageError(f"timeout is {timeout}; give the seconds a row may take, a positive number")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise errors.UsageError(f"workers is {workers!r}; give a whole number, or 0 for the system's own")
        if missing := system.missing_keys():
            raise errors.MissingCredential(f"{system.label} needs {', '.join(missing)}: "
                                           + " and ".join(f"export {k}=..." for k in missing) + ", then run again")
        every = self.rows()
        selection = data.selection(suites, limit, sample, only)
        chosen = data.select(every, **selection)
        # Note: a limit or sample that takes every row asks the same question as no limit, so it is the same run.
        if len(chosen) == len(every) and set(selection) <= {"limit", "sample"}: selection = {}
        q = self.question(selection)
        self._inputs(chosen)
        for s in self.scorers: s.check_gold(every)
        return Plan(runs.Run(Path(out) / runs.folder_name(system, q), system), q, chosen, timeout,
                    workers or type(system).workers, retry_failed)

    def check(self) -> dict[str, int]:
        """Checks the dataset before it is used or published, and returns how many rows and suites it has.

        It reads every row (`Task.rows` checks the manifest), builds every row's input (`Task.input`, which
        raises when an input file is missing), and reads every gold (each scorer's `check_gold`). It makes no
        call and runs no scorer."""
        every = self.rows()
        self._inputs(every)
        for s in self.scorers: s.check_gold(every)
        return {"rows": len(every), "suites": len({r.suite for r in every})}

    def describe(self, p: Plan) -> str:
        """What `predict(p)` would do, without calling anything. Read it before paying for a run."""
        def said(v: Any) -> str:
            if not isinstance(v, list): return str(v)
            return ", ".join(v) if len(v) <= 3 else f"{len(v)} of them"

        chose = "; ".join(f"{k} {said(v)}" for k, v in p.question["rows"].items())
        out = [f"{self.task.name} on {self.dataset} ({self.manifest}): {errors.counted(len(p.rows), 'row')}"
               f"{f' ({chose})' if chose else ''}, graded by {', '.join(s.name for s in self.scorers)}"]
        out.append(f"  {p.run.folder}: {sum(p.run.needs(x.id, p.retry_failed) for x in p.rows)} to predict, "
                   f"{p.workers} at a time")
        return "\n".join(out)

    def predict(self, p: Plan, executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor) -> None:
        """Sends every row the run lacks to its system, `p.workers` at a time, and saves each answer as it arrives.

        A failure that is the same for every row (a `Stop`), or a bug, stops the run once the calls in flight
        are saved. An interrupt does the same. Either way, running the same command again resumes.

        `executor` makes the pool, given its number of workers: threads on this machine by default. Another
        executor (processes, a cluster, a GPU container next to a model's server) runs `predict_row` for each
        row wherever it likes; this machine still decides which rows are needed and saves every record. It is
        also how several runs share one rate limit: give each run a factory that returns one shared pool."""
        r = p.run
        r.prepare(p.question, data.commit(self.dataset, data.download_dataset(self.dataset)), p.timeout)
        self._read_unread(r, p.rows)
        work = collections.deque(x for x in p.rows if r.needs(x.id, p.retry_failed))
        live: dict[cf.Future, data.Row] = {}
        failure: BaseException | None = None
        if done := len(p.rows) - len(work):
            log.info("%s: %s", r.name, f"resuming: {done} of {errors.counted(len(p.rows), 'row')} done in an earlier "
                     f"run, {len(work)} to predict" if work else f"all {errors.counted(len(p.rows), 'row')} done in an "
                     "earlier run; nothing to predict")
        with progress.Board() as board, executor(p.workers) as ex:
            board.start(r.name, len(p.rows), len(work))
            try:
                while True:
                    while work and len(live) < p.workers and failure is None:
                        x = work.popleft()
                        live[ex.submit(predict_row, self.task, r.system, x, p.timeout)] = x
                    if not live: break
                    finished, _ = cf.wait(live, return_when=cf.FIRST_COMPLETED)
                    for f in finished: failure = _saved(r, live.pop(f), f, board) or failure
            except KeyboardInterrupt:
                # Note: the calls in flight are paid for, so they are saved before the interrupt goes on.
                log.warning("interrupted: saving the calls in flight; run the same command again to resume")
                for f in cf.as_completed(live): _saved(r, live[f], f, board)
                raise
        if failure is not None: raise failure

    def grade(self, run: runs.Run | str | Path, *, partial: bool = False, details: bool = True,
              write: bool = True) -> dict[str, Any]:
        """Grades `run`, or the folder of your own predictions at that path, with each scorer. Returns the run's
        name, its cost and each scorer's summary by folder name.

        A run is graded over the rows it answers (its settings.json's `rows`); a folder of your own, over the
        whole manifest. Grading refuses a run that isn't finished (`errors.Unfinished`); `partial=True` grades
        it anyway, counting its unfinished rows as failed. `write=False` writes no results (`Scorer.grade`)."""
        r = run if isinstance(run, runs.Run) else runs.Run.at(run)
        every = self.rows()
        if not r.brought(): r.check_settings(self.question())
        rows = data.select(every, **r.selection())
        if stray := sorted(r.prediction_files() - {f"{x.id}.json" for x in every}):
            log.warning("%s: files that match no row of the manifest are ignored: %d, e.g. %s", r.predictions,
                        len(stray), stray[:3])
        graded = {s.folder_name: s.grade(r, rows, partial=partial, details=details, write=write) for s in self.scorers}
        cost = runs.total(r.costs(x.id for x in rows))
        spent = "" if cost["usd"] is None else f", cost ${cost['usd']:.2f}"
        for name, s in graded.items():
            n = (p := s.get("partial", {})) and p["no_answer"] + p["not_scored"]
            part = f", PARTIAL: {n} unfinished rows counted 0, each a failed line in scores.jsonl" if n else ""
            log.info("%s %s: headline %s; %s, %d failed%s%s", r.name, name, _headline(s),
                     errors.counted(s["rows"], "row"), s["failed"], spent, part)
        return {"run": r.name, "cost": cost, "scorers": graded}

    def run(self, system: adapter.Adapter, out: str | Path, *,
            executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor, **kwargs: Any) -> dict[str, Any]:
        """`plan`, then `predict` (with `executor`), then `grade`. `kwargs` are `plan`'s. Returns the grading. A
        run that stops part way isn't graded: run the same command again to finish it."""
        p = self.plan(system, out, **kwargs)
        log.info("%s", self.describe(p))
        self.predict(p, executor)
        return self.grade(p.run)

    def reread(self, run: runs.Run | str | Path, system: adapter.Adapter | None = None) -> list[str]:
        """Reads every saved answer of `run` again with its system's `read`, with no call. Returns the ids
        whose record changed. Use it after a fix to `read`: the raw answers were kept so that this is free.

        `run` is a `Run` with its system, or a run folder's path with the `system` that made it (the same
        adapter and options; settings.json says which). Grade the run again afterwards."""
        if not isinstance(run, runs.Run): run = runs.Run(Path(run), system)
        elif system is not None: run = runs.Run(run.folder, system)
        if run.system is None:
            raise errors.UsageError(f"{run.folder} needs the system that made it to read its answers again: "
                                    "pass reread(folder, system), the adapter with the options its settings.json "
                                    "names")
        changed = []
        for x in self.rows():
            if (rec := run.record(x.id)) is None: continue
            try:
                new = adapter.reread(run.system, self.task.input(x), rec)
            except errors.Unread as exc:
                raise errors.bug(f"reading the answer of {x.id}", exc.__cause__ or exc, type(run.system)) from None
            if new != rec:
                run.save(x.id, new)
                changed.append(x.id)
        return changed

    def _read_unread(self, r: runs.Run, rows: Sequence[data.Row]) -> None:
        """Reads again, with no call, every answer of `r` that a bug in `read` left unread, so running the same
        command again after the fix picks up where it stopped. A `read` that still raises stops here, before any
        call."""
        for x in rows:
            if not (rec := r.record(x.id)) or not rec.get("unread"): continue
            try:
                r.save(x.id, adapter.reread(r.system, self.task.input(x), rec))
            except errors.Unread as exc:
                raise errors.bug(f"reading the answer of {x.id}", exc.__cause__ or exc, type(r.system)) from None

    def _inputs(self, rows: Sequence[data.Row]) -> None:
        """Refuses rows whose input file is missing (`Task.input` raises), listing them all."""
        missing = []
        for r in rows:
            try:
                self.task.input(r)
            except FileNotFoundError as exc:
                missing.append(f"{r.id}: {exc}")
        if missing:
            raise errors.UsageError(f"input files that don't exist: {errors.counted(len(missing), 'row')}, e.g. "
                                    + "; ".join(missing[:3]))


def _headline(s: dict[str, Any]) -> str:
    """A summary's headline as the log line shows it: the scorer's number over the scored rows, times coverage,
    so a low headline reads as either poor answers or few of them."""
    if not s["coverage"]: return f"{s['headline']:.3f}: no row was scored"
    what = s["headline_is"].removesuffix(" × coverage")
    scored = errors.counted(s["rows"] - s["failed"], "scored row")
    return (f"{s['headline']:.3f} = {s['headline'] / s['coverage']:.3f} ({what}) on {scored} "
            f"× coverage {s['coverage']:.3f}")


def predict_row(t: task.Task, system: adapter.Adapter, row: data.Row, timeout: float) -> dict[str, Any]:
    """One row's record, computed but not saved: its input, then `adapter.predict`.

    This is what an executor runs, as `submit(predict_row, task, system, row, timeout)`. It takes only frozen
    dataclasses, so an executor can send it to another process or machine. A row's paths are this machine's;
    an executor that runs it elsewhere gives the row its own paths first. The machine running the benchmark
    saves what it returns. A row whose input file disappeared after the plan checked it stops the run: it is
    the dataset's problem, not the system's or the adapter's."""
    try:
        x = t.input(row)
    except FileNotFoundError as exc:
        raise errors.Stop(f"the input of row {row.id} disappeared after the run was planned ({exc}); restore it and "
                          "run the same command again") from None
    return adapter.predict(system, x, timeout=timeout)


def _saved(r: runs.Run, x: data.Row, f: cf.Future, board: progress.Board) -> BaseException | None:
    """Saves the record `f` holds for row `x` of run `r`, and returns None; or, when it raised, the failure to
    stop the run with: a bug in the adapter is raised as `errors.Bug`, which names whose code it is."""
    try:
        rec = f.result()
    except errors.Unread as exc:
        r.save(x.id, exc.record)
        return errors.bug(f"reading the answer of {x.id} (the answer is saved, and is read again when you run the "
                          "same command after the fix)", exc.__cause__ or exc, type(r.system))
    except BaseException as exc:
        if isinstance(exc, Exception) and not isinstance(exc, errors.Stop):
            return errors.bug(f"predicting {x.id}", exc, type(r.system))
        return exc
    r.save(x.id, rec)
    board.saved(r.name, x.id, rec)
    return None
