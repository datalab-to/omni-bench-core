"""Benchmarks: a task's dataset and the scorers that grade it, run on predictive systems.

`Benchmark` is the class to use, and its docstring says how. `Plan` is what `Benchmark.plan` checks and
`Benchmark.predict` carries out. `per_adapter` checks the `workers` argument.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses
import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from omni_bench_core import adapter, data, errors, scorer, task
from omni_bench_core import run as runs

log = logging.getLogger(__name__)


class Plan(NamedTuple):
    """One invocation: its runs, the rows they predict, and how. `Benchmark.plan` builds and checks it, so
    `predict` has nothing left to refuse."""

    runs: tuple[runs.Run, ...]
    rows: tuple[data.Row, ...]
    timeout: float
    workers: Mapping[str, int]       # rows in flight per adapter name; one left out takes `Adapter.workers`
    retry_failed: bool


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """A task, its dataset and manifest, and the scorers that grade it.

        oeb = benchmark.Benchmark(Extract(), "hf://datasets/datalab-to/omni_extract_bench", [FieldMatching()])
        oeb.run([Datalab(mode="balanced"), Datalab(mode="accurate")], out="runs", sample=20)
        oeb.grade("mine/")                  # a folder of your own predictions, predicting nothing

    A benchmark is what stays fixed while systems change: one task, one dataset and manifest, and the
    scorers. So every system run on it is graded on the same rows by the same rules, and runs of different
    systems can be compared.

    A run has two steps. Predicting sends each row to a system and saves its answer. Grading compares each
    saved answer with its gold, once per scorer. Predicting costs money and time, and grading is free, so
    the two are separate: an answer is paid for once and graded as often as we like.

    It has six methods:

        plan      the runs, checked before any call: the options, the rows, each scorer's gold, and that no
                  run folder holds another dataset
        predict   sends every row a run has no record of to its system; a stopped run resumes
        grade     each scorer on a run, over the whole manifest
        run       plan, then predict, then grade each run
        reread    reads a run's saved answers again with its adapter's `read`, after a fix to it, with no call
        check     checks the dataset (manifest, input files, golds) without calling anything

    The dataset is a folder or an `hf://` address, never rows made in code, because a run folder records which
    dataset its answers are for and refuses another.
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

    def plan(self, systems: Sequence[adapter.Adapter], out: str | Path, *, suites: Iterable[str] | None = None,
             limit: int = 0, sample: int = 0, only: Iterable[str] | None = None,
             timeout: float = adapter.DEFAULT_TIMEOUT, workers: int | Mapping[str, int] = 0,
             retry_failed: bool = False) -> Plan:
        """A run for each of `systems`, each in its own folder under `out`, all checked before any call.

        Args:
          suites, only, sample, limit: choose the rows to predict (`data.select`). Grading still covers the
            whole manifest, and reports the other rows as pending.
          timeout: the seconds each row may take.
          workers: how many rows are in flight at once per adapter, shared by all its runs, because a
            system's rate limit is per account. One number for every adapter, or a number per adapter name.
          retry_failed: predict again the rows whose recorded failure might not happen again.
        """
        if isinstance(systems, adapter.Adapter):
            raise TypeError(f"systems is a list, not one adapter; pass [{systems!r}]")
        for a in systems:
            if not isinstance(a, adapter.Adapter):
                raise TypeError(f"{a!r} is not an adapter instance; make one, e.g. Datalab(mode='accurate')")
        if not timeout > 0:
            raise errors.UsageError(f"timeout is {timeout}; give the seconds a row may take, a positive number")
        workers = per_adapter(workers, {a.name for a in systems})
        by: dict[str, runs.Run] = {}
        for a in systems:
            r = runs.Run(Path(out) / a.folder_name, a)
            if r.name in by:
                raise errors.UsageError(f"run {r.name} is asked for more than once: the same adapter with the "
                                        "same options (an option set to its default is the same option)")
            r.check_settings(self.dataset, self.manifest, self.task.identity())
            by[r.name] = r
        self.task.check(systems, self.scorers)
        every = self.rows()
        for s in self.scorers: s.check_gold(every)
        return Plan(tuple(by.values()), data.select(every, suites, limit, sample, only), timeout,
                    workers, retry_failed)

    def check(self) -> dict[str, int]:
        """Checks the dataset before it is used or published, and returns how many rows and suites it has.

        It reads every row (`Task.rows` checks the manifest), builds every row's input (`Task.input`, which
        raises when an input file is missing), and reads every gold (each scorer's `check_gold`). It makes no
        call and runs no scorer. It stops at the first problem, except that it lists every missing input."""
        every = self.rows()
        missing = []
        for r in every:
            try:
                self.task.input(r)
            except FileNotFoundError as exc:
                missing.append(f"{r.id}: {exc}")
        if missing:
            raise errors.UsageError(f"input files that don't exist: {errors.counted(len(missing), 'row')}, e.g. "
                                    + "; ".join(missing[:3]))
        for s in self.scorers: s.check_gold(every)
        return {"rows": len(every), "suites": len({r.suite for r in every})}

    def describe(self, p: Plan) -> str:
        """What `predict(p)` would do, without calling anything. Read it before paying for a run."""
        out = [f"{self.task.name} on {self.dataset} ({self.manifest}): {errors.counted(len(p.rows), 'row')}, "
               f"graded by {', '.join(s.name for s in self.scorers)}"]
        out += [f"  {r.folder}: {sum(r.needs(x.id, p.retry_failed) for x in p.rows)} to predict" for r in p.runs]
        return "\n".join(out)

    def predict(self, p: Plan, executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor) -> None:
        """Sends every row each run lacks to its system, and saves each answer as it arrives.

        The runs of one adapter share one pool of workers, and different adapters run at the same time.
        Predicting is mostly waiting on the network, so adapters don't slow each other down. A failure that
        is the same for every row (a `Stop`, or a bug) stops that adapter's runs, and is raised once the
        other adapters finish. An interrupt stops them all, and the calls already in flight are saved.

        `executor` makes the pool, given its number of workers: threads on this machine by default. Another
        executor (processes, a cluster, a GPU container next to a model's server) runs `predict_row` for each
        row wherever it likes; this machine still decides which rows are needed and saves every record."""
        root = data.download_dataset(self.dataset)
        stop = threading.Event()
        by: dict[str, list[runs.Run]] = {}
        for r in p.runs:
            r.prepare(self.dataset, self.manifest, root, self.task.identity(), p.timeout)
            by.setdefault(r.system.name, []).append(r)
        failures: list[Exception] = []
        with cf.ThreadPoolExecutor(max(1, len(by))) as ex:
            fs = [ex.submit(_predict_runs, self.task, p, rs, p.workers.get(name) or type(rs[0].system).workers, stop,
                            executor) for name, rs in by.items()]
            try:
                for f in cf.as_completed(fs):
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001 -- one adapter's failure is its own; the others go on
                        log.error("stopped: %s: %s", type(exc).__name__, str(exc)[:300])
                        failures.append(exc)
            except BaseException:
                stop.set()
                log.warning("interrupted: finishing the calls in flight; rows not yet written come back next run")
                raise
        if failures: raise failures[0]

    def grade(self, run: runs.Run | str | Path, *, details: bool = True, write: bool = True) -> dict[str, Any]:
        """Grades `run`, or the folder of your own predictions at that path, with each scorer, over the whole
        manifest. Returns the run's cost and each scorer's summary by folder name. `write=False` writes no
        results (see `Scorer.grade`).

        A folder of your own is refused when it holds answers for another dataset. Its first successful
        grading writes the dataset into its settings.json, so a later grading on another dataset is refused
        too."""
        r = run if isinstance(run, runs.Run) else runs.Run.at(run)
        r.check_settings(self.dataset, self.manifest, self.task.identity())
        every = self.rows()
        graded = {s.folder_name: s.grade(r, every, details=details, write=write) for s in self.scorers}
        if write and r.brought() and not r.settings().get("dataset"):
            r.write_settings(self.dataset, self.manifest, data.download_dataset(self.dataset), self.task.identity(),
                             **{k: v for k, v in r.settings().items() if k != "task"})
        for name, s in graded.items():
            head = "—" if s["headline"] is None else f"{s['headline']:.3f}"
            log.info("%s %s: headline %s; %d graded, %d failed, %d unscored, %d pending", r.name, name, head,
                     s["graded"], s["failed"], s["unscored"], s["pending"])
        return {"cost": runs.total(r.costs(x.id for x in every)), "scorers": graded}

    def run(self, systems: Sequence[adapter.Adapter], out: str | Path, *,
            executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """`plan`, then `predict` (with `executor`), then `grade` each run. `kwargs` are `plan`'s. Returns each
        run's grading by run name.

        When predicting fails part way, what was predicted is still graded, so a stopped run reports what it
        has. After an interrupt it isn't graded, because whoever stopped the run wants it stopped."""
        p = self.plan(systems, out, **kwargs)
        log.info("%s", self.describe(p))
        try:
            self.predict(p, executor)
        except Exception:
            for r in p.runs: self.grade(r)
            raise
        return {r.name: self.grade(r) for r in p.runs}

    def reread(self, run: runs.Run) -> list[str]:
        """Reads every saved answer of `run` again with its system's `read`, with no call. Returns the ids
        whose record changed. Use it after a fix to `read`: the raw answers were kept so that this is free."""
        if run.system is None:
            raise errors.UsageError(f"{run.folder} has no system to read its answers with; pass "
                                    "runs.Run(folder, the adapter that made it)")
        changed = []
        for x in self.rows():
            if (rec := run.record(x.id)) is None: continue
            if (new := adapter.reread(run.system, self.task.input(x), rec)) != rec:
                run.save(x.id, new)
                changed.append(x.id)
        return changed


def per_adapter(workers: int | Mapping[str, int], names: set[str]) -> dict[str, int]:
    """`workers` as rows in flight per adapter name, checked.

    `workers` is one number for every adapter, or a number per adapter name. 0, or a name left out, means
    the adapter's own `Adapter.workers`, so it is left out of the result."""
    if isinstance(workers, int) and not isinstance(workers, bool):
        workers = dict.fromkeys(names, workers)
    if bad := sorted(set(workers) - names):
        raise errors.UsageError(f"workers names {bad}, which no system of this plan is; they are {sorted(names)}")
    if bad := [k for k, n in workers.items() if isinstance(n, bool) or not isinstance(n, int) or n < 0]:
        raise errors.UsageError(f"workers gives {bad[0]} {workers[bad[0]]!r}; give a whole number, or 0 for its default")
    return {k: n for k, n in workers.items() if n}


def predict_row(t: task.Task, system: adapter.Adapter, row: data.Row, timeout: float) -> dict[str, Any] | None:
    """One row's record, computed but not saved: its input, then `adapter.predict`. None when its input file
    is missing.

    This is what an executor runs, as `submit(predict_row, task, system, row, timeout)`. It takes only frozen
    dataclasses, so an executor can send it to another process or machine. A row's paths are this machine's;
    an executor that runs it elsewhere gives the row its own paths first. The machine running the benchmark
    saves what it returns."""
    try:
        x = t.input(row)
    except FileNotFoundError:
        return None
    return adapter.predict(system, x, timeout=timeout)


def _predict_runs(t: task.Task, p: Plan, rs: Sequence[runs.Run], workers: int, stop: threading.Event,
                  make: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor) -> None:
    """Predicts every row that the runs `rs`, all of one adapter, lack. At most `workers` rows are in flight
    at once, in an executor made by `make`, and each record is saved here as it arrives.

    A new row is sent only when another finishes, not all at once. That way an executor that runs whatever
    it is given, such as a cluster, still keeps an account to `workers` calls at a time. After a failure
    that is the same for every row, or an interrupt (`stop`), nothing more is sent, and the calls already in
    flight are saved, because they are paid for."""
    work = collections.deque((r, x) for r in rs for x in p.rows if r.needs(x.id, p.retry_failed))
    total = {r.name: sum(q is r for q, _ in work) for r in rs}
    done = dict.fromkeys(total, 0)
    for r in rs: log.info("%s: %d rows, %d to predict", r.name, len(p.rows), total[r.name])
    live: dict[cf.Future, tuple[runs.Run, data.Row]] = {}
    failure: BaseException | None = None
    with make(workers) as ex:
        while True:
            while work and len(live) < workers and failure is None and not stop.is_set():
                r, x = work.popleft()
                live[ex.submit(predict_row, t, r.system, x, p.timeout)] = (r, x)
            if not live: break
            finished, _ = cf.wait(live, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                r, x = live.pop(f)
                try:
                    rec = f.result()
                except BaseException as exc:
                    if isinstance(exc, Exception) and not isinstance(exc, errors.Stop):
                        exc.add_note(f"{r.name}: raised while predicting {x.id}, a bug in the {r.system.name} "
                                     "adapter rather than the system's answer, so its runs stop; a run after the "
                                     "fix resumes")
                    failure = failure or exc
                    continue
                if rec is None:
                    # Note: a missing input file is the dataset's, not the system's: nothing is recorded.
                    log.warning("%s: no input file for %s; skipped", r.name, x.id)
                    continue
                r.save(x.id, rec)
                done[r.name] += 1
                log.info("%s: %d/%d %s", r.name, done[r.name], total[r.name], x.id)
    if failure is not None: raise failure
