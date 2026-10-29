"""Benchmarks: a task, its dataset and the scorers that grade it.

- `Benchmark`: plans, predicts, grades and checks;
- `Plan`: one run, checked, before any call; and
- `predict_row`: one row's work, which an executor may run elsewhere.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from omni_bench_core import adapter, data, errors, progress, scorer, task
from omni_bench_core import run as runs

log = logging.getLogger(__name__)


class Plan(NamedTuple):
    """One run before any call, built and checked by `Benchmark.plan`, so `predict` has nothing left to refuse."""

    run: runs.Run
    question: Mapping[str, Any]      # the dataset, manifest, task and choice of rows the run answers
    rows: tuple[data.Row, ...]
    timeout: float
    workers: int                     # rows in flight at once
    retry_failed: bool
    launch: Mapping[str, Any]        # the caller's record of the launch, kept in settings.json
    server: Mapping[str, Any] | None = None   # what a served system's server said, kept in server.json


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """A task, its dataset and manifest, and the scorers that grade it: what stays fixed while systems change.

        oeb = benchmark.Benchmark(Extract(), "hf://datasets/datalab-to/omni_extract_bench", [FieldMatching()])
        oeb.run(Datalab(mode="accurate"), out="runs", sample=20)
        oeb.grade("mine/")                  # a folder of your own predictions

    A run is one system on the benchmark. Predicting pays for each answer once and saves it; grading is free
    and can be repeated. Three rules decide every case:

    1. a run folder answers one exact question: the system and its options, the dataset, manifest and task,
       and the choice of rows (`sample`, `limit`, `only`). Every launch is a new run in a new folder, named
       after its question and a run id; a launch that names a run (`resume`) continues it, and only for the
       same question;
    2. grading reports only a finished run (every row answered or failed, and scored); otherwise it refuses
       and names the rows, and `partial=True` counts them failed; and
    3. a row fails only when code raises `SystemFailed`, and the scorer's `score_failure` scores it. Any other
       exception stops the run and is raised as it is; the answers already paid for are kept.

    Methods:

        plan      a new run, or the run to resume, checked before any call (options, keys, rows, input files
                  and golds);
        predict   sends the rows the run has no record of;
        grade     each scorer on a finished run;
        run       plan, then predict, then grade;
        reread    reads a run's saved answers again after a fix to `read`, with no call; and
        check     checks the dataset with no call.

    `dataset_name`, when given, names the dataset in a run's question instead of `dataset`'s address. Give it
    when one dataset is read from more than one place (a Hugging Face address, a local copy, a copy on another
    machine), so a run of it is the same question, and can be resumed, wherever it is read from.
    """

    task: task.Task
    dataset: str | Path
    scorers: Sequence[scorer.Scorer]
    manifest: str = data.MANIFEST
    dataset_name: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_name is not None and not (isinstance(self.dataset_name, str) and self.dataset_name.strip()):
            raise ValueError(f"dataset_name is {self.dataset_name!r}; give the dataset's name, a non-empty string")
        errors.check_dataclass(type(self.task), task.Task)
        if fields := [f.name for f in dataclasses.fields(self.task)]:
            raise TypeError(f"{type(self.task).__name__} has fields {', '.join(fields)}, but a task has no options: "
                            "what varies between runs is the dataset, the manifest, a scorer or the system")
        scorer.check(self.scorers)

    def rows(self) -> tuple[data.Row, ...]:
        """Every row of the manifest."""
        return self.task.rows(self.dataset, self.manifest)

    def question(self, selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What a run answers besides its system: the dataset, manifest, task and choice of rows."""
        return {"dataset": self.dataset_name or data.dataset_id(self.dataset),
                "manifest": data.manifest_path(self.manifest), "task": self.task.name,
                "rows": dict(selection or {})}

    def plan(self, system: adapter.Adapter, out: str | Path, *, limit: int = 0, sample: int = 0,
             only: Iterable[str] | None = None, timeout: float = adapter.DEFAULT_TIMEOUT, workers: int = 0,
             resume: str | None = None, retry_failed: bool = False,
             launch: Mapping[str, Any] | None = None) -> Plan:
        """A new run of `system` in a new folder under `out`, or the run `resume` names there; checked before any
        call.

        Args:
          only, sample, limit: choose the rows (`data.select`); part of the run's question.
          timeout: the seconds each row may take.
          workers: rows in flight at once; 0 for the system's own `Adapter.workers`.
          resume: the name of a run under `out` to continue; it must answer the same question.
          retry_failed: with `resume`, predict again the rows whose recorded failure may not recur.
          launch: the caller's record of this launch (who, from which commit), plain JSON, kept in a new run's
            settings.json.
        """
        if not isinstance(system, adapter.Adapter):
            raise TypeError(f"{system!r} is not one adapter instance; make one, e.g. Datalab(mode='accurate'). A run "
                            "is one system: to compare systems, run each")
        if not timeout > 0:
            raise ValueError(f"timeout is {timeout}; give the seconds a row may take, a positive number")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError(f"workers is {workers!r}; give a whole number, or 0 for the system's own")
        if retry_failed and resume is None:
            raise ValueError("retry_failed predicts a run's failed rows again; name the run with resume=<its name>")
        launch = dict(launch or {})
        errors.plain(launch, "launch")
        adapter.check_keys(system)
        every = self.rows()
        selection = data.selection(limit, sample, only)
        chosen = data.select(every, **selection)
        # Note: a limit or sample that takes every row asks the same question as no limit, so it is the same run.
        if len(chosen) == len(every) and set(selection) <= {"limit", "sample"}: selection = {}
        q = self.question(selection)
        self._inputs(chosen)
        for s in self.scorers: s.check_gold(every)
        # Note: what the run answers, built for a new run too, so an option that isn't plain JSON is refused here.
        asks = runs.identity(system, q)
        # Note: after the checks that cost nothing, so a wrong dataset is refused without a server.
        said = adapter.check_served(system)
        folder = Path(out) / (self._resumed(asks, Path(out), resume) if resume is not None
                              else runs.folder_name(system, runs.new_id()))
        return Plan(runs.Run(folder, system), q, chosen, timeout, workers or type(system).workers, retry_failed,
                    launch, said)

    def _resumed(self, asks: Mapping[str, Any], out: Path, resume: str) -> str:
        """The name of the run to resume, checked: a folder under `out` that a launch of this same question made."""
        if not isinstance(resume, str) or not resume or "/" in resume or "\\" in resume:
            raise ValueError(f"resume is {resume!r}; give a run's name, the last part of its folder")
        if not (out / resume / "settings.json").is_file():
            raise ValueError(f"no run {resume} under {out}; give the name of a run folder there")
        said = runs.Run(out / resume).settings()
        # Note: through JSON, as settings.json keeps it, so a tuple and its list are one question.
        now = json.loads(json.dumps(asks))
        if differ := [k for k in now if said.get(k) != now[k]]:
            raise ValueError(f"{resume} answers another question than this launch; it differs in {', '.join(differ)}. "
                             "Resume it with the same system, options, dataset and rows, or leave out resume for a "
                             "new run")
        return resume

    def check(self) -> dict[str, int]:
        """Checks the dataset with no call: every row, input file and gold. Returns how many rows it has."""
        every = self.rows()
        self._inputs(every)
        for s in self.scorers: s.check_gold(every)
        return {"rows": len(every)}

    def describe(self, p: Plan) -> str:
        """What `predict(p)` would do, without calling anything."""
        def said(v: Any) -> str:
            if not isinstance(v, list): return str(v)
            return ", ".join(v) if len(v) <= 3 else f"{len(v)} of them"

        chose = "; ".join(f"{k} {said(v)}" for k, v in p.question["rows"].items())
        out = [f"{self.task.name} on {self.dataset} ({self.manifest}): {errors.counted(len(p.rows), 'row')}"
               f"{f' ({chose})' if chose else ''}, graded by {', '.join(s.name for s in self.scorers)}"]
        out.append(f"  {p.run.folder} ({'resumed' if p.run.settings() else 'a new run'}): "
                   f"{sum(p.run.needs(x.id, p.retry_failed) for x in p.rows)} to predict, {p.workers} at a time")
        return "\n".join(out)

    def predict(self, p: Plan, executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor) -> None:
        """Sends every row the run lacks to its system, `p.workers` at a time, and saves each answer as it arrives.
        An exception (an account, a refused connection, a bug) or an interrupt stops the run once the calls in
        flight are saved; `plan(..., resume=<its name>)` continues it.

        `executor` makes the pool from a number of workers: threads by default. Another executor (processes, a
        cluster) runs `predict_row` elsewhere, while this machine chooses the rows and saves every record. A
        factory returning one shared pool lets several runs share a rate limit."""
        r = p.run
        r.prepare(p.question, data.commit(self.dataset, data.download_dataset(self.dataset)), p.timeout, p.launch)
        if p.server is not None: r.served_by(p.server)
        self._read_unread(r, p.rows)
        work = collections.deque(x for x in p.rows if r.needs(x.id, p.retry_failed))
        live: dict[cf.Future, data.Row] = {}
        failure: BaseException | None = None
        if done := len(p.rows) - len(work):
            log.info("%s: %s", r.name, f"resuming: {done} of {errors.counted(len(p.rows), 'row')} done in an earlier "
                     f"run, {len(work)} to predict" if work else f"all {errors.counted(len(p.rows), 'row')} done in an "
                     "earlier run; nothing to predict")
        with progress.Board(r.name, len(work)) as board, executor(p.workers) as ex:
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
                log.warning("interrupted: saving the calls in flight; resume %s to continue", r.name)
                for f in cf.as_completed(live): _saved(r, live[f], f, board)
                raise
        if failure is not None: raise failure

    def grade(self, run: runs.Run | str | Path, *, partial: bool = False, details: bool = True,
              write: bool = True) -> dict[str, Any]:
        """Grades `run` (or a folder of your own predictions) with each scorer, over the rows it answers. Returns
        the run's name, its cost and each scorer's summary by folder name. An unfinished run is refused with a
        `ValueError`; `partial=True` counts its unfinished rows failed. `write=False` writes no results."""
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
            part = f", PARTIAL: {n} unfinished rows counted as failed, each a line in scores.jsonl" if n else ""
            log.info("%s %s: headline %s%s%s", r.name, name, _headline(s), spent, part)
        return {"run": r.name, "cost": cost, "scorers": graded}

    def run(self, system: adapter.Adapter, out: str | Path, *,
            executor: Callable[[int], cf.Executor] = cf.ThreadPoolExecutor, **kwargs: Any) -> dict[str, Any]:
        """`plan`, then `predict` (with `executor`), then `grade`; `kwargs` are `plan`'s, so each call is a new run
        unless it gives `resume`. Returns the grading."""
        p = self.plan(system, out, **kwargs)
        log.info("%s", self.describe(p))
        self.predict(p, executor)
        return self.grade(p.run)

    def reread(self, run: runs.Run | str | Path, system: adapter.Adapter | None = None) -> list[str]:
        """Reads every saved answer of `run` again with its system's `read`, with no call, after a fix to `read`.
        Returns the ids whose record changed. `run` is a `Run` with its system, or a folder with the `system`
        that made it. Grade it again afterwards."""
        if not isinstance(run, runs.Run): run = runs.Run(Path(run), system)
        elif system is not None: run = runs.Run(run.folder, system)
        if run.system is None:
            raise ValueError(f"{run.folder} needs the system that made it to read its answers again: "
                             "pass reread(folder, system), the adapter with the options its settings.json "
                             "names")
        changed = []
        for x in self.rows():
            if (rec := run.record(x.id)) is None: continue
            try:
                new = adapter.reread(run.system, self.task.input(x), rec)
            except Exception as exc:
                errors.noted(exc, type(run.system), f"reading the answer of {x.id}")
                raise
            if new != rec:
                run.save(x.id, new)
                changed.append(x.id)
        return changed

    def _read_unread(self, r: runs.Run, rows: Sequence[data.Row]) -> None:
        """Reads again, with no call, every answer of `r` that a bug in `read` left unread. A `read` that still
        raises stops here, before any call."""
        for x in rows:
            if not (rec := r.record(x.id)) or not rec.get("unread"): continue
            try:
                r.save(x.id, adapter.reread(r.system, self.task.input(x), rec))
            except Exception as exc:
                errors.noted(exc, type(r.system), f"reading the answer of {x.id}")
                raise

    def _inputs(self, rows: Sequence[data.Row]) -> None:
        """Refuses rows whose input file is missing."""
        missing = []
        for r in rows:
            try:
                self.task.input(r)
            except FileNotFoundError as exc:
                missing.append(f"{r.id}: {exc}")
        if missing:
            raise ValueError(f"input files that don't exist: {errors.counted(len(missing), 'row')}, e.g. "
                             + "; ".join(missing[:3]))


def _headline(s: dict[str, Any]) -> str:
    """A summary's headline as the log line shows it, with how many rows were scored and why the rest failed."""
    # Note: scored as a count, since a share rounds 7665/7667 to 1.000.
    failed = ", ".join(f"{n} {c}" for c, n in s["failed"].items() if n) or "none"
    return (f"{s['headline']:.4f} ({s['headline_is']}); {s['rows'] - sum(s['failed'].values())}/{s['rows']} scored, "
            f"failed: {failed}")


def predict_row(t: task.Task, system: adapter.Adapter, row: data.Row, timeout: float) -> dict[str, Any]:
    """One row's record, computed but not saved: what an executor runs. It takes only frozen dataclasses, so it
    can be sent to another process or machine, which gives the row its own paths first."""
    try:
        x = t.input(row)
    except FileNotFoundError as exc:
        exc.add_note(f"the input of row {row.id} disappeared after the run was planned; restore it and resume the "
                     "run")
        raise
    return adapter.predict(system, x, timeout=timeout)


def _saved(r: runs.Run, x: data.Row, f: cf.Future, board: progress.Board) -> BaseException | None:
    """Saves the record `f` holds for row `x`, and returns None; or returns the exception it raised, noted, after
    saving an answer that `read` couldn't read."""
    try:
        rec = f.result()
    except Exception as exc:
        if (unread := getattr(exc, "unread", None)) is None:
            return errors.noted(exc, type(r.system), f"predicting {x.id}")
        r.save(x.id, unread)
        return errors.noted(exc, type(r.system), f"reading the answer of {x.id}; the answer is saved, and is read "
                            f"again when you resume {r.name} after the fix")
    except BaseException as exc:
        return exc
    r.save(x.id, rec)
    board.saved(x.id, rec)
    return None
