"""The core on a toy task, offline: copy a text file's text.

    python tests/test_core.py

A row is a text file and its gold `{"text": ...}`; an adapter answers `{"text": ...}`; `Exact` scores whether
the two are equal, with one detail line per differing character.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from omni_bench_core import adapter, benchmark, data, errors, scorer, task
from omni_bench_core import run as runs

CALLS: list[str] = []
LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class Doc:
    doc_id: str
    doc_path: Path
    gt_path: Path
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.doc_id


@dataclasses.dataclass(frozen=True)
class Copying(task.Task):
    def rows(self, dataset, manifest=data.MANIFEST) -> tuple[Doc, ...]:
        _, rs = data.read_manifest(dataset, manifest, columns=(), id="doc_id", inputs=("doc_path",))
        return tuple(Doc(**r) for r in rs)

    def input(self, row: Doc) -> Path:
        if not row.doc_path.exists(): raise FileNotFoundError(row.doc_path)
        return row.doc_path


@dataclasses.dataclass(frozen=True)
class GoldEcho(adapter.Adapter):
    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        with LOCK: CALLS.append(x.stem)
        return adapter.Call(raw=json.loads((x.parent.parent / "gold" / f"{x.stem}.json").read_text()))


@dataclasses.dataclass(frozen=True)
class Copy(adapter.Adapter):
    """Copies the file's text; `upper` shouts it. A file holding "refuse" is refused with a 400, "garble" is
    answered with what `read` can't read, "account" is a 402, and "flaky" is a 503 on its first try."""

    workers: ClassVar[int] = 2
    backoff: ClassVar[tuple[float, ...]] = ()
    upper: bool = False
    temperature: float = 0.0

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        with LOCK:
            CALLS.append(x.stem)
            first = CALLS.count(x.stem) == 1
        text = x.read_text()
        if "refuse" in text: raise errors.SystemFailed("refused", status=400)
        if "account" in text: raise errors.SystemFailed("out of credits", status=402)
        if "flaky" in text and first: raise errors.SystemFailed("busy", status=503)
        if "garble" in text: return adapter.Call(raw="<garbled>", cost=adapter.Cost(usd=0.5))
        return adapter.Call(raw={"answer": text.upper() if self.upper else text}, cost=adapter.Cost(usd=0.25))

    def read(self, raw: Any, x: Path) -> Any:
        with adapter.reading("the answer"):
            return {"text": raw["answer"]}


@dataclasses.dataclass(frozen=True)
class Exact(scorer.Scorer):
    def gold(self, row: Doc) -> str:
        got = json.loads(row.gt_path.read_text())
        if not isinstance(got.get("text"), str): raise ValueError(f'{row.gt_path} holds {got}, not {{"text": ...}}')
        return got["text"]

    def prediction(self, output: Any) -> str:
        if not (isinstance(output, dict) and isinstance(output.get("text"), str)):
            raise errors.SystemFailed(f"the prediction has no text: {str(output)[:80]}")
        return output["text"]

    def score(self, pred: str, gold: str) -> scorer.Score:
        diffs = [{"at": i, "pred": a, "gold": b} for i, (a, b) in enumerate(zip(pred, gold)) if a != b]
        return scorer.Score({"exact": pred == gold, "length": len(pred)}, diffs or None)

    def summarize(self, lines):
        exact = sum(x["exact"] for x in lines) / len(lines)
        return {"headline": exact, "headline_is": "exact", "exact": exact,
                "length": sum(x["length"] for x in lines) / len(lines)}


@dataclasses.dataclass(frozen=True)
class ExactPooled(Exact):
    workers: ClassVar[int] = 2


JUDGED: list[str] = []
SEEN: set[str] = set()   # what the judge has been asked, never cleared, so "flaky" fails once per test run


@dataclasses.dataclass(frozen=True)
class Judge(adapter.Adapter):
    """A judge that agrees when the two texts match, and doesn't answer the first time it sees "flaky"."""

    retries: ClassVar[int] = 0
    strict: bool = True

    def call(self, x: tuple[str, str], *, timeout: float) -> adapter.Call:
        pred, gold = x
        with LOCK:
            JUDGED.append(gold)
            first, _ = gold not in SEEN, SEEN.add(gold)
        if "flaky" in gold and first: raise errors.SystemTimeout("the judge took too long")
        same = pred == gold if self.strict else pred.casefold() == gold.casefold()
        return adapter.Call(raw={"verdict": "yes" if same else "no"}, cost=adapter.Cost(usd=0.01))


@dataclasses.dataclass(frozen=True)
class Agree(Exact):
    cache: ClassVar[bool] = True
    pool: ClassVar[str] = "thread"
    workers: ClassVar[int] = 4
    judge: adapter.Adapter = Judge()

    def score(self, pred: str, gold: str) -> scorer.Score:
        rec = scorer.ask(self.judge, (pred, gold), timeout=5)
        return scorer.Score({"agree": rec["output"]["verdict"] == "yes", "judge_usd": rec["cost"]["usd"]},
                            [rec["output"]])

    def summarize(self, lines):
        agree = sum(x["agree"] for x in lines) / len(lines)
        return {"headline": agree, "headline_is": "agree", "agree": agree,
                "judge_usd": sum(x["judge_usd"] for x in lines) / len(lines)}


TEXTS = {"a1": ("letters", "hello"), "a2": ("letters", "world"), "b1": ("numbers", "123"),
         "b2": ("numbers", "456"), "c1": ("other", "refuse me")}


def dataset(root: Path, texts=TEXTS, gold=None) -> Path:
    (root / "docs").mkdir(parents=True)
    (root / "gold").mkdir()
    for i, (_, t) in texts.items():
        (root / "docs" / f"{i}.txt").write_text(t)
        (root / "gold" / f"{i}.json").write_text(json.dumps(gold(t) if gold else {"text": t}))
    cols = {"doc_id": list(texts), "doc_path": [f"docs/{i}.txt" for i in texts],
            "gt_path": [f"gold/{i}.json" for i in texts], "lang": ["en"] * len(texts)}
    pq.write_table(pa.table(cols), root / "manifest.parquet")
    return root


def bench(root: Path, *scorers: scorer.Scorer) -> benchmark.Benchmark:
    return benchmark.Benchmark(Copying(), root, scorers or (Exact(),))


def refused(f, *args, match: str = "", **kwargs) -> str:
    try:
        f(*args, **kwargs)
    except (TypeError, ValueError, OSError) as exc:
        said = "\n".join([str(exc), *getattr(exc, "__notes__", [])])     # the message, then the core's notes
        assert match in said, f"{match!r} not in {said}"
        return said
    raise AssertionError(f"{f.__name__} was not refused")


def test_defaults(tmp: Path) -> None:
    assert (GoldEcho.name, ExactPooled.name, Copying.name) == ("gold_echo", "exact_pooled", "copying")
    rows = bench(dataset(tmp / "d")).rows()
    assert rows[0].extra == {"lang": "en"} and rows[0].doc_path.is_absolute()


def test_gold_echo_scores_one(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    g = b.run(GoldEcho(), tmp / "runs")
    name = g["run"]
    s = g["scorers"]["exact"]
    assert s["headline"] == 1.0 and s["scored"]["exact"] == 1.0 and s["scored"]["length"] == 5.0, s
    assert s["scored"]["headline_is"] == "exact" and s["headline_is"] == "scored.headline × coverage", s
    assert g["cost"] == {"usd": None, "credits": None}
    assert (tmp / "runs" / name / "exact" / "scores.jsonl").exists()
    assert not (tmp / "runs" / name / "exact" / "details").exists()


def test_resume_calls_nothing(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "f1": ("other", "flaky")}))
    CALLS.clear()
    name = b.run(Copy(), tmp / "runs")["run"]
    assert CALLS.count("f1") == 2, CALLS          # a 503 is retried within the row
    CALLS.clear()
    said: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: said.append(r.getMessage())
    logging.getLogger("omni_bench_core").addHandler(handler)
    logging.getLogger("omni_bench_core").setLevel(logging.INFO)
    try:
        g = b.run(Copy(), tmp / "runs", resume=name)
    finally:
        logging.getLogger("omni_bench_core").removeHandler(handler)
    assert CALLS == [] and g["run"] == name, CALLS
    assert any("all 6 rows done in an earlier run; nothing to predict" in m for m in said), said
    # the log line splits the headline into the score over the scored rows and coverage
    assert any("headline 0.8333 = 1.0000 (" in m and ") × coverage 5/6; 1 failed" in m for m in said), said
    s = g["scorers"]["exact"]
    assert s["failed"] == 1 and abs(s["coverage"] - 5 / 6) < 1e-9
    assert abs(s["headline"] - 5 / 6) < 1e-9, s     # a refused row counts as 0
    assert g["cost"]["usd"] == 1.25


def test_each_launch_is_a_new_run_and_resume_continues_only_the_same_question(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    one = b.run(GoldEcho(), tmp / "runs", launch={"by": "me", "commit": "abc"})["run"]
    two = b.run(GoldEcho(), tmp / "runs")["run"]
    stem = runs.stem(GoldEcho(), b.question())
    assert one != two and one.startswith(stem + "-") and two.startswith(stem + "-"), (one, two)
    said = json.loads((tmp / "runs" / one / "settings.json").read_text())
    assert said["launch"] == {"by": "me", "commit": "abc"} and said["started"].endswith("+00:00"), said
    CALLS.clear()
    assert b.run(GoldEcho(), tmp / "runs", resume=one, launch={"by": "you"})["run"] == one and CALLS == []
    # a resumed run keeps its first launch's record
    assert json.loads((tmp / "runs" / one / "settings.json").read_text())["launch"]["by"] == "me"
    refused(b.plan, GoldEcho(), tmp / "runs", sample=2, resume=one, match="it differs in rows")
    refused(b.plan, Copy(), tmp / "runs", resume=one, match="it differs in system, options")
    refused(b.plan, GoldEcho(), tmp / "runs", resume="nope", match="no run nope under")
    refused(b.plan, GoldEcho(), tmp / "runs", resume=f"runs/{one}", match="last part of its folder")
    refused(b.plan, GoldEcho(), tmp / "runs", retry_failed=True, match="name the run with resume")
    refused(b.plan, GoldEcho(), tmp / "runs", launch={"at": object()}, match="launch")


def test_a_choice_of_rows_is_its_own_finished_run(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    g = b.run(GoldEcho(), tmp / "runs", limit=2)
    few = g["run"]
    s = g["scorers"]["exact"]
    assert s["rows"] == 2 and s["headline"] == 1.0 and s["scored"]["headline_is"] == "exact", s
    g = b.run(GoldEcho(), tmp / "runs")
    every = g["run"]
    assert every != few and g["scorers"]["exact"]["rows"] == 5, (few, every)     # another question, another folder
    assert json.loads((tmp / "runs" / few / "settings.json").read_text())["rows"] == {"limit": 2}
    refused(b.plan, GoldEcho(), tmp / "runs", limit=2, sample=2, match="not both")


def test_failures_and_details(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "g1": ("other", "garble")}))
    g = b.run(Copy(upper=True), tmp / "runs")
    name = g["run"]
    folder = tmp / "runs" / name
    lines = {x["id"]: x for x in map(json.loads, (folder / "exact" / "scores.jsonl").read_text().splitlines())}
    assert lines["c1"]["status"] == "failed" and "refused" in lines["c1"]["error"]
    assert lines["g1"]["status"] == "failed" and "Unreadable" in lines["g1"]["error"]
    rec = json.loads((folder / "records" / "g1.json").read_text())
    assert rec["raw"] == "<garbled>" and rec["output"] is None and rec["error"]["type"] == "Unreadable"
    assert lines["a1"]["exact"] is False
    assert len((folder / "exact" / "details" / "a1.jsonl").read_text().splitlines()) == 5
    assert abs(g["scorers"]["exact"]["headline"] - 2 / 4 * 4 / 6) < 1e-9    # digits survive upper()


def test_reread_after_a_fix(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "g1": ("other", "garble")}))
    name = b.run(Copy(), tmp / "runs")["run"]

    @dataclasses.dataclass(frozen=True)
    class Fixed(Copy):
        name: ClassVar[str] = "copy"

        def read(self, raw: Any, x: Path) -> Any:
            return {"text": raw} if isinstance(raw, str) else super().read(raw, x)

    CALLS.clear()
    assert b.reread(tmp / "runs" / name, Fixed()) == ["g1"]          # a folder's path and the system
    assert CALLS == []
    assert json.loads((tmp / "runs" / name / "predictions" / "g1.json").read_text()) == {"text": "<garbled>"}
    assert b.reread(runs.Run(tmp / "runs" / name, Fixed())) == []     # a Run with its system; nothing left to fix
    refused(b.reread, tmp / "runs" / name, match="needs the system that made it")


def test_pool_matches_serial(tmp: Path) -> None:
    g = bench(dataset(tmp / "d"), Exact(), ExactPooled()).run(Copy(), tmp / "runs")
    one, two = g["scorers"]["exact"], g["scorers"]["exact_pooled"]
    assert {k: v for k, v in one.items() if k != "scorer"} == {k: v for k, v in two.items() if k != "scorer"}


def test_restore_from_record(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    (tmp / "runs" / name / "predictions" / "a1.json").unlink()
    CALLS.clear()
    g = b.run(GoldEcho(), tmp / "runs", resume=name)
    assert CALLS == [] and g["scorers"]["exact"]["headline"] == 1.0
    assert (tmp / "runs" / name / "predictions" / "a1.json").exists()


def test_brought_predictions(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    mine = tmp / "mine"
    (mine / "predictions").mkdir(parents=True)
    for i in ("a1", "a2", "b1"):
        (mine / "predictions" / f"{i}.json").write_text(json.dumps({"text": TEXTS[i][1]}))
    refused(b.grade, mine, match="2 rows with no file in")
    s = b.grade(mine, partial=True)["scorers"]["exact"]
    assert s["failed"] == 2 and abs(s["headline"] - 0.6) < 1e-9 and s["partial"] == {"no_answer": 2, "not_scored": 0,
                                                                       "e.g.": ["b2", "c1"]}, s
    assert not (mine / "settings.json").exists()       # grading writes only its results into your folder
    stray = tmp / "stray"
    (stray / "predictions").mkdir(parents=True)
    (stray / "predictions" / "zzz.json").write_text("{}")
    refused(b.grade, stray, match="names a row")
    refused(b.grade, tmp / "nowhere", match="no folder")


def test_refused_before_any_call(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", gold=lambda t: {"html": t}))
    CALLS.clear()
    refused(b.run, GoldEcho(), tmp / "runs", match="can't read the gold of row a1")
    assert CALLS == []
    b = bench(dataset(tmp / "ok"))
    refused(adapter.configure, Copy, {"uper": True}, match="did you mean 'upper'")
    refused(adapter.configure, Copy, {"upper": "yes"}, match="takes a bool")
    assert adapter.configure(Copy, {"temperature": 0}) == Copy()
    assert adapter.configure(Exact, {}) == Exact()
    refused(adapter.configure, Exact, {"strict": True}, match="exact has no option 'strict'")
    refused(adapter.configure, Doc, {}, match="an adapter or a scorer")
    q = b.question()
    name = functools.partial(runs.stem, question=q)
    assert name(Copy()) == name(Copy(temperature=0.0)) != name(Copy(upper=True))
    assert runs.stem(Copy(), q) != runs.stem(Copy(), b.question({"sample": 2}))
    refused(b.plan, [Copy()], tmp / "runs", match="not one adapter instance")
    refused(b.plan, GoldEcho(), tmp / "runs", workers=-1, match="workers is -1")
    refused(benchmark.Benchmark, Copying(), tmp / "ok", (), match="at least one scorer")
    refused(benchmark.Benchmark, Copying(), tmp / "ok", (Exact(), Exact()), match="twice")


def test_forgotten_decorator_is_refused(tmp: Path) -> None:
    class Forgot(Copy):
        mode: str = "balanced"

    class Loose(Exact):
        fold_case: bool = True

    refused(lambda a: a.options(), Forgot(), match="decorate it with @dataclasses.dataclass(frozen=True)")
    refused(adapter.configure, Forgot, {}, match="Forgot declares mode")
    refused(benchmark.Benchmark, Copying(), tmp, (Loose(),), match="Loose declares fold_case")


def test_judge_cached_and_unscored(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "f1": ("other", "flaky")}), Agree(), Agree(Judge(strict=False)))
    try:
        b.run(GoldEcho(), tmp / "runs")
    except ValueError as exc:
        assert "1 row not scored" in str(exc) and "f1" in str(exc) and "grade again" in str(exc), exc
    else:
        raise AssertionError("a row the judge didn't answer was graded")
    [name] = [f.name for f in (tmp / "runs").iterdir()]
    strict, loose = (Agree().folder_name, Agree(Judge(strict=False)).folder_name)
    assert strict != loose and strict.startswith("agree-"), (strict, loose)
    JUDGED.clear()
    g = b.grade(tmp / "runs" / name)
    # Note: the first grading stopped at the strict scorer, so grading again asks it the unscored row only,
    # and asks the loose scorer every row for the first time.
    assert JUDGED.count("flaky") == 2 and len(JUDGED) == 7, JUDGED
    assert all(g["scorers"][f]["headline"] == 1.0 for f in (strict, loose)), g
    assert g["scorers"][strict]["options"] == {"judge": {"system": "judge", "strict": True}}
    assert abs(g["scorers"][strict]["scored"]["judge_usd"] - 0.01) < 1e-9
    JUDGED.clear()
    b.grade(tmp / "runs" / name)
    assert JUDGED == []
    assert (tmp / "runs" / name / strict / "details" / "a1.jsonl").exists()
    refused(benchmark.Benchmark, Copying(), tmp / "d", (Agree(), Agree()), match="twice")


@dataclasses.dataclass(frozen=True)
class AgreeOnPaths(Agree):
    name: ClassVar[str] = "agree"

    def gold(self, row: Doc) -> Path:
        return row.gt_path


def test_cache_keys(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), Agree())
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    JUDGED.clear()
    (tmp / "d" / "gold" / "a1.json").write_text(json.dumps({"text": "hullo"}))
    s = b.grade(tmp / "runs" / name)["scorers"][Agree().folder_name]
    assert JUDGED == ["hullo"], JUDGED                    # a changed gold is a new key
    assert s["scored"]["agree"] == 0.8 and s["scored"]["judge_usd"] == 0.01, s
    fresh = json.loads((tmp / "runs" / name / Agree().folder_name / "scores.jsonl").read_text().splitlines()[0])
    JUDGED.clear()
    b.grade(tmp / "runs" / name)
    kept = json.loads((tmp / "runs" / name / Agree().folder_name / "scores.jsonl").read_text().splitlines()[0])
    assert JUDGED == [] and fresh == kept
    assert Agree()._code() != AgreeOnPaths()._code()      # a change to the code is a new key
    refused(bench(dataset(tmp / "d2"), AgreeOnPaths()).run, GoldEcho(), tmp / "runs2", match="must be JSON")


@dataclasses.dataclass(frozen=True)
class BatchAgree(Agree):
    """A judge asked about every row at once, as a batching judge would be."""

    def score_all(self, rows, preds, golds):
        BATCHES.append([r.id for r in rows])
        return super().score_all(rows, preds, golds)


BATCHES: list[list[str]] = []


def test_score_all_override_is_cached(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), BatchAgree())
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    assert BATCHES == [["a1", "a2", "b1", "b2", "c1"]], BATCHES
    BATCHES.clear()
    (tmp / "d" / "gold" / "b2.json").write_text(json.dumps({"text": "457"}))
    s = b.grade(tmp / "runs" / name)["scorers"][BatchAgree().folder_name]
    assert BATCHES == [["b2"]] and s["scored"]["agree"] == 0.8, (BATCHES, s)    # only the row with no kept score


def test_grade_is_not_overridden(tmp: Path) -> None:
    try:
        type("Graded", (Exact,), {"grade": lambda self, run, rows: {}})
    except TypeError as exc:
        assert "override score_all" in str(exc)
    else:
        raise AssertionError("a grade override was not refused")


@dataclasses.dataclass(frozen=True)
class ByLanguage(Exact):
    def row_fields(self, row: Doc) -> dict[str, Any]:
        return {"lang": row.extra["lang"], "pages": 1}

    def summarize(self, lines):
        by: dict[str, int] = {}
        for x in lines: by[x["row"]["lang"]] = by.get(x["row"]["lang"], 0) + 1
        return {**super().summarize(lines), "per_lang": {k: {"n": n} for k, n in by.items()}}


def test_row_fields_and_dry_grading(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), ByLanguage())
    g = b.run(GoldEcho(), tmp / "runs")
    name = g["run"]
    s = g["scorers"]["by_language"]
    assert s["scored"]["per_lang"] == {"en": {"n": 5}} and s["headline"] == 1.0, s
    line = json.loads((tmp / "runs" / name / "by_language" / "scores.jsonl").read_text().splitlines()[0])
    assert line["row"] == {"lang": "en", "pages": 1}
    assert Exact().summarize([line])["exact"] == 1.0 and "pages" not in Exact().summarize([line])   # not averaged
    mine = tmp / "mine"
    (mine / "predictions").mkdir(parents=True)
    (mine / "predictions" / "a1.json").write_text(json.dumps({"text": "hello"}))
    s = bench(tmp / "d").grade(mine, partial=True, write=False)["scorers"]["exact"]
    assert s["failed"] == 4 and abs(s["headline"] - 0.2) < 1e-9, s
    assert sorted(p.name for p in mine.iterdir()) == ["predictions"]       # nothing written


@dataclasses.dataclass(frozen=True)
class Overcounts(Exact):
    def summarize(self, lines):
        return {**super().summarize(lines), "failed": 0}


def test_summary_counts_are_grades(tmp: Path) -> None:
    s = bench(dataset(tmp / "d"), Overcounts()).run(Copy(), tmp / "runs")["scorers"]["overcounts"]
    assert s["failed"] == 1 and s["scored"]["failed"] == 0, s     # a scorer's own numbers never replace grade's


@dataclasses.dataclass(frozen=True)
class ExactInTwos(Exact):
    batch_size: ClassVar[int | None] = 2


@dataclasses.dataclass(frozen=True)
class ExactAtOnce(Exact):
    batch_size: ClassVar[int | None] = None


def test_batches_change_nothing(tmp: Path) -> None:
    g = bench(dataset(tmp / "d"), Exact(), ExactInTwos(), ExactAtOnce()).run(Copy(upper=True), tmp / "runs")
    name = g["run"]
    one, *rest = [{k: v for k, v in s.items() if k != "scorer"} for s in g["scorers"].values()]
    assert all(r == one for r in rest), (one, rest)
    folder = tmp / "runs" / name
    assert sorted(p.name for p in (folder / "exact_in_twos" / "details").iterdir()) == ["a1.jsonl", "a2.jsonl"]


def test_predict_through_another_executor(tmp: Path) -> None:
    import concurrent.futures as cf

    b = bench(dataset(tmp / "d", {**TEXTS, "z1": ("other", "missing")}))
    (tmp / "d" / "docs" / "z1.txt").unlink()
    refused(b.plan, Copy(), tmp / "runs", match="input files that don't exist: 1 row, e.g. z1")
    p = b.plan(Copy(), tmp / "runs", only=list(TEXTS))
    r = p.run
    # Note: processes stand in for a cluster: what is sent must pickle, and only the driver writes files.
    b.predict(p, executor=cf.ProcessPoolExecutor)
    assert sorted(f.stem for f in r.records.iterdir()) == ["a1", "a2", "b1", "b2", "c1"]
    s = b.grade(r)["scorers"]["exact"]
    assert s["rows"] == 5 and s["failed"] == 1 and abs(s["headline"] - 0.8) < 1e-9, s


@dataclasses.dataclass(frozen=True)
class Billing(adapter.Adapter):
    """Fails "busy" rows with a paid 503 once, then answers; refuses "key" rows with a 401 and "quota" rows
    with a message about credits."""

    backoff: ClassVar[tuple[float, ...]] = ()

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        text = x.read_text()
        with LOCK:
            CALLS.append(x.stem)
            first = CALLS.count(x.stem) == 1
        if "key" in text: raise errors.SystemFailed("bad key", status=401)
        if "quota" in text: raise errors.SystemFailed("error: insufficient_quota for this org", status=429)
        if "busy" in text and first:
            raise errors.SystemFailed("busy", status=503, sent={"mode": "fast"}, cost=adapter.Cost(usd=0.5))
        if "never" in text:
            raise errors.SystemFailed("bad request", status=400, sent={"mode": "fast"}, cost=adapter.Cost(usd=0.1),
                                     job_id="job-1")
        return adapter.Call(raw={"text": text}, cost=adapter.Cost(usd=0.25), sent={"mode": "fast"})


def test_failed_calls_keep_cost_and_what_was_sent(tmp: Path) -> None:
    for name, text in (("k", "key"), ("q", "quota")):
        (tmp / f"{name}.txt").write_text(text)
        refused_stop = False
        try:
            adapter.predict(Billing(), tmp / f"{name}.txt")
        except PermissionError:
            refused_stop = True
        assert refused_stop, name
    (tmp / "b.txt").write_text("busy")
    rec = adapter.predict(Billing(), tmp / "b.txt")
    assert rec["error"] is None and rec["cost"]["usd"] == 0.75 and rec["cost"]["attempts"] == 2, rec["cost"]
    (tmp / "n.txt").write_text("never")
    rec = adapter.predict(Billing(), tmp / "n.txt")
    assert rec["error"]["status"] == 400 and rec["sent"] == {"mode": "fast"} and rec["cost"]["usd"] == 0.1, rec
    assert rec["job_id"] == "job-1"


def test_dataset_check(tmp: Path) -> None:
    b = benchmark.Benchmark(Copying(), tmp / "d", [Exact()])
    dataset(tmp / "d")
    assert b.check() == {"rows": 5}
    (tmp / "d" / "docs" / "a1.txt").unlink()
    refused(b.check, match="input files that don't exist: 1 row")


def test_old_records_are_not_restored(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    folder = tmp / "runs" / name
    rec = json.loads((folder / "records" / "a1.json").read_text())
    (folder / "records" / "a1.json").write_text(json.dumps({**{k: v for k, v in rec.items() if k != "output"}, "result": {}}))
    (folder / "predictions" / "a1.json").unlink()
    s = b.grade(folder)["scorers"]["exact"]
    assert s["failed"] == 1, s


def test_reading_marks_the_systems_fault(tmp: Path) -> None:
    try:
        with adapter.reading("the job's status"):
            {}["status"]
    except errors.SystemFailed as exc:
        assert "the job's status" in str(exc)
    else:
        raise AssertionError("a malformed answer was not the system's failure")
    try:
        with adapter.reading("the job's status"):
            raise RuntimeError("a bug of ours")
    except RuntimeError:
        pass


@dataclasses.dataclass(frozen=True)
class CopyingTwice(Copying):
    twice: bool = True


def test_another_task_is_another_folder(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    other = benchmark.Benchmark(CopyingTwice(), tmp / "d", [Exact()])
    assert runs.stem(GoldEcho(), b.question()) != runs.stem(GoldEcho(), other.question())
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    refused(other.grade, tmp / "runs" / name, match="answers task")
    assert json.loads((tmp / "runs" / name / "settings.json").read_text())["task"] == {"name": "copying", "options": {}}


def test_failures_can_be_audited(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    g = b.run(Copy(), tmp / "runs")
    name = g["run"]       # c1 ("refuse me") is refused with a 400
    lines = [json.loads(x) for x in (tmp / "runs" / name / "exact" / "scores.jsonl").read_text().splitlines()]
    [failed] = [x for x in lines if x["status"] == "failed"]
    assert failed["id"] == "c1" and "refused" in failed["error"] and g["scorers"]["exact"]["failed"] == 1
    assert json.loads((tmp / "runs" / name / "records" / "c1.json").read_text())["error"]["status"] == 400


def test_another_dataset_is_another_folder(tmp: Path) -> None:
    one = bench(dataset(tmp / "d1")).run(GoldEcho(), tmp / "runs")["run"]
    two = bench(dataset(tmp / "d2")).run(GoldEcho(), tmp / "runs")["run"]
    assert one != two
    refused(bench(tmp / "d2").grade, tmp / "runs" / one, match="answers dataset")


def test_a_stop_records_nothing_and_grades_nothing(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "z9": ("other", "account")}))
    try:
        b.run(Copy(), tmp / "runs", workers=1)
    except PermissionError:
        pass
    else:
        raise AssertionError("an account failure was not raised")
    [folder] = (tmp / "runs").iterdir()
    assert not (folder / "records" / "z9.json").exists() and not (folder / "exact").exists()
    msg = refused(b.grade, folder, match="1 row never predicted, or answered but not yet read: z9")
    assert "run the command that made this run again" in msg and "--partial" in msg, msg
    s = b.grade(folder, partial=True)["scorers"]["exact"]
    assert s["partial"] == {"no_answer": 1, "not_scored": 0, "e.g.": ["z9"]} and s["failed"] == 2, s


@dataclasses.dataclass(frozen=True)
class Keyed(Copy):
    keys: ClassVar[tuple[str, ...]] = ("OMNI_BENCH_TEST_KEY",)


@dataclasses.dataclass(frozen=True)
class Buggy(Copy):
    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        if x.stem == "b1": return {}["a field the code forgot"]
        return super().call(x, timeout=timeout)


@dataclasses.dataclass(frozen=True)
class Late(Copy):
    """Answers after its budget, ignoring the timeout it is given."""

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        time.sleep(timeout + 0.2)
        return super().call(x, timeout=timeout)


def test_an_answer_after_the_budget_is_a_timeout(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {"a1": ("letters", "hello")}))
    name = b.run(Late(), tmp / "runs", timeout=0.1)["run"]
    rec = json.loads((tmp / "runs" / name / "records" / "a1.json").read_text())
    assert rec["error"]["type"] == "SystemTimeout" and "past the 0.1 s budget" in rec["error"]["message"], rec
    assert rec["raw"] == {"answer": "hello"} and rec["cost"]["usd"] == 0.25      # paid for, so kept


@dataclasses.dataclass(frozen=True)
class Unserved(Copy):
    """Calls a server of its own that isn't running."""

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        import socket

        with socket.socket() as sock:
            sock.connect(("127.0.0.1", 9))   # nothing listens on the discard port
        return super().call(x, timeout=timeout)


@dataclasses.dataclass(frozen=True)
class Crashes(Exact):
    def score(self, pred: str, gold: str) -> scorer.Score:
        if gold == "world": return {}["a field the code forgot"]
        return super().score(pred, gold)


def test_a_refused_connection_stops_the_run(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    try:
        b.run(Unserved(), tmp / "runs")
    except ConnectionRefusedError as exc:     # raised as it is, with a note saying what to do
        assert "connection was refused" in exc.__notes__[0] and "start it" in exc.__notes__[0], exc.__notes__
    else:
        raise AssertionError("a refused connection didn't stop the run")
    [folder] = (tmp / "runs").iterdir()
    assert not any((folder / "records").iterdir())


def test_a_selection_of_every_row_is_every_row(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    every = b.plan(GoldEcho(), tmp / "runs").question
    limited = b.plan(GoldEcho(), tmp / "runs", limit=100).question
    sampled = b.plan(GoldEcho(), tmp / "runs", sample=2).question
    assert every == limited != sampled


def test_a_scorer_bug_names_its_row(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), Crashes())
    try:
        b.run(GoldEcho(), tmp / "runs")
    except KeyError as exc:      # the scorer's own exception, with a note saying where
        assert exc.__notes__ == ["Crashes raised this while scoring a2"], exc.__notes__
    else:
        raise AssertionError("a bug in a scorer was not raised")
    [folder] = (tmp / "runs").iterdir()
    assert not (folder / "crashes" / "summary.json").exists()


@dataclasses.dataclass(frozen=True)
class Polls(Copy):
    """Reads a job id from an answer that hasn't one."""

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        with adapter.reading("the job id"):
            job = {"status": "queued"}["job_id"]
        return adapter.Call(raw={"answer": job})


def test_a_failure_our_code_read_keeps_its_traceback(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {"a1": ("letters", "hello"), "g1": ("other", "garble" * 2000)}))
    p = b.plan(Polls(), tmp / "runs")
    b.predict(p)
    folder = p.run.folder
    err = json.loads((folder / "records" / "a1.json").read_text())["error"]
    assert err["type"] == "SystemFailed" and "couldn't be read (the job id): KeyError" in err["message"], err
    assert "Traceback" in err["traceback"] and "in call" in err["traceback"], err
    name = b.run(Copy(), tmp / "runs")["run"]
    err = json.loads((tmp / "runs" / name / "records" / "g1.json").read_text())["error"]
    assert err["type"] == "Unreadable" and "in read" in err["traceback"], err      # the reader's own line
    assert len(err["message"]) < errors.CAP + 40


@dataclasses.dataclass(frozen=True)
class MisReads(Copy):
    """Answers like Copy, but its read has a bug on one row."""

    name: ClassVar[str] = "mis_reads"

    def read(self, raw: Any, x: Path) -> Any:
        if raw.get("answer") == "world": return {}["a field the code forgot"]
        return super().read(raw, x)


@dataclasses.dataclass(frozen=True)
class ReadsRight(Copy):
    name: ClassVar[str] = "mis_reads"


@dataclasses.dataclass(frozen=True)
class PredictionBug(Exact):
    def prediction(self, output: Any) -> str:
        return output["txet"]


def test_a_bug_in_read_stops_keeps_the_answer_and_a_rerun_reads_it(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {k: v for k, v in TEXTS.items() if k != "c1"}))
    try:
        b.run(MisReads(), tmp / "runs", workers=1)
    except KeyError as exc:
        [note] = exc.__notes__
        assert "MisReads raised this while reading the answer of a2" in note and "is saved" in note, note
    else:
        raise AssertionError("a bug in read was counted instead of stopping the run")
    [folder] = (tmp / "runs").iterdir()
    rec = json.loads((folder / "records" / "a2.json").read_text())
    assert rec["unread"] and rec["raw"] == {"answer": "world"} and rec["error"] is None, rec
    assert not (folder / "predictions" / "a2.json").exists()
    refused(b.grade, folder, match="3 rows never predicted, or answered but not yet read: a2, b1, b2")
    CALLS.clear()
    g = b.run(ReadsRight(), tmp / "runs", resume=folder.name)     # the fix, then resume: no call for a2
    assert g["run"] == folder.name and "a2" not in CALLS and g["scorers"]["exact"]["headline"] == 1.0, (CALLS, g)


def test_a_bug_in_prediction_stops_grading_and_failures_keep_their_traceback(tmp: Path) -> None:
    name = bench(dataset(tmp / "d")).run(Copy(), tmp / "runs")["run"]
    rec = json.loads((tmp / "runs" / name / "records" / "c1.json").read_text())
    assert rec["error"]["type"] == "SystemFailed" and "in call" in rec["error"]["traceback"], rec["error"]
    try:
        bench(tmp / "d", PredictionBug()).grade(tmp / "runs" / name)
    except KeyError as exc:
        assert exc.__notes__ == ["PredictionBug raised this while reading the prediction of a1"], exc.__notes__
    else:
        raise AssertionError("a bug in prediction() was counted instead of stopping grading")


def test_a_missing_key_is_refused_before_any_call(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    CALLS.clear()
    try:
        b.plan(Keyed(), tmp / "runs")
    except ValueError as exc:
        assert "keyed needs OMNI_BENCH_TEST_KEY" in str(exc) and "export OMNI_BENCH_TEST_KEY=" in str(exc), exc
    else:
        raise AssertionError("a missing key was not refused")
    assert CALLS == [] and not (tmp / "runs").exists()
    (tmp / "a.txt").write_text("hello")
    refused(adapter.predict, Keyed(), tmp / "a.txt", match="keyed needs OMNI_BENCH_TEST_KEY")   # called directly too
    assert CALLS == []


def test_a_bug_stops_the_run_and_prints_its_traceback(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    try:
        b.run(Buggy(), tmp / "runs", workers=1)
    except KeyError as exc:      # raised as it is: the adapter's own exception and traceback
        assert exc.__notes__ == ["Buggy raised this while predicting b1"], exc.__notes__
        assert exc.__traceback__ is not None
    else:
        raise AssertionError("a bug in an adapter was not raised")
    [folder] = (tmp / "runs").iterdir()
    assert not (folder / "records" / "b1.json").exists() and sorted(p.name for p in folder.iterdir()) == [
        "predictions", "records", "settings.json"]


def test_sample_spreads_over_the_manifest(tmp: Path) -> None:
    rows = bench(dataset(tmp / "d")).rows()
    assert [r.id for r in data.select(rows, sample=3)] == ["a1", "a2", "b2"]     # rows 0, 1 and 3 of 5
    assert [r.id for r in data.select(rows, sample=9)] == [r.id for r in rows]


@dataclasses.dataclass(frozen=True)
class NoSummary(scorer.Scorer):
    def gold(self, row):
        return None

    def score(self, pred, gold):
        return scorer.Score({"n": 1})


@dataclasses.dataclass(frozen=True)
class NoScore(scorer.Scorer):
    def gold(self, row):
        return None

    def summarize(self, lines):
        return {"headline": 1.0, "headline_is": "n"}


@dataclasses.dataclass(frozen=True)
class BadResult(Exact):
    def score(self, pred, gold):
        return scorer.Score([pred == gold])


@dataclasses.dataclass(frozen=True)
class BadDetails(Exact):
    def score(self, pred, gold):
        return scorer.Score({"exact": pred == gold, "length": 1}, "a diff")


@dataclasses.dataclass(frozen=True)
class BatchJudge(Exact):
    """Scores rows together, and couldn't score b1: it returns that row's Unscored rather than raising it."""

    def score_all(self, rows, preds, golds):
        return [errors.Unscored("the batch judge skipped it") if r.id == "b1" else self.score(p, g)
                for r, p, g in zip(rows, preds, golds, strict=True)]


@dataclasses.dataclass(frozen=True)
class Headline(Exact):
    value: Any = 2.0
    what: Any = "exact"

    def summarize(self, lines):
        return {"headline": self.value, "headline_is": self.what}


@dataclasses.dataclass(frozen=True)
class NoHeadline(Exact):
    def summarize(self, lines):
        return {"exact": 1.0, "headline_is": "exact"}


@dataclasses.dataclass(frozen=True)
class Status(Exact):
    status: str = "ok"

    def score(self, pred, gold):
        return scorer.Score({"status": self.status, "exact": pred == gold, "length": 1})


def test_the_scorer_summarizes_the_scored_rows_and_the_core_checks_its_headline(tmp: Path) -> None:
    refused(NoSummary, match="abstract method 'summarize'")             # Python refuses it: summarize is abstract
    refused(bench, dataset(tmp / "d"), NoScore(), match="NoScore has neither score")
    for value, what, match in ((2.0, "exact", '"headline": 2.0'), (-0.1, "exact", '"headline": -0.1'),
                               (True, "exact", '"headline": True'), (None, "exact", '"headline": None'),
                               (0.5, "", '"headline_is" key')):
        refused(bench(tmp / "d", Headline(value, what)).run, GoldEcho(), tmp / f"runs-{value}-{what}", match=match)
    refused(bench(tmp / "d", NoHeadline()).run, GoldEcho(), tmp / "runs-nohead", match='a "headline" key')
    for status in ("ok", "failed"):     # a result can't say its own status: only the core fails a row
        refused(bench(tmp / "d", Status(status)).run, GoldEcho(), tmp / f"runs-{status}", match="holds ['status']")
    for bad in (BadResult(), BadDetails()):    # what a Score holds is checked as each row is scored
        refused(bench(tmp / "d", bad).run, GoldEcho(), tmp / f"runs-{bad.name}", match=f"{type(bad).__name__} scored row")
    msg = refused(bench(tmp / "d", BatchJudge()).run, GoldEcho(), tmp / "runs-batch", match="1 row not scored")
    assert "the batch judge skipped it" in msg and "b1" in msg, msg
    every = dataset(tmp / "refused", {"r1": ("", "refuse 1"), "r2": ("", "refuse 2")})
    s = bench(every).run(Copy(), tmp / "runs-none")["scorers"]["exact"]
    assert s["scored"] is None and s["headline"] == 0.0 and s["failed"] == 2, s   # summarize is never called


@dataclasses.dataclass(frozen=True)
class WithClient(Copy):
    client: Any = dataclasses.field(default_factory=object)    # an object, not plain data


@dataclasses.dataclass(frozen=True)
class ReadsAnObject(Copy):
    def read(self, raw: Any, x: Path) -> Any:
        return {"text": raw["answer"], "at": datetime.datetime(2026, 1, 1)} if raw["answer"] == "world" \
            else super().read(raw, x)


@dataclasses.dataclass(frozen=True)
class NanScore(Exact):
    def score(self, pred, gold):
        return scorer.Score({"exact": pred == gold, "length": float("nan")})


@dataclasses.dataclass(frozen=True)
class ObjectSummary(Exact):
    def summarize(self, lines):
        return {**super().summarize(lines), "when": datetime.date(2026, 1, 1)}


def test_everything_written_is_plain_json(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {k: v for k, v in TEXTS.items() if k != "c1"}))
    refused(b.plan, WithClient(), tmp / "runs", match="WithClient's options isn't plain JSON")
    try:
        b.run(ReadsAnObject(), tmp / "runs", workers=1)
    except TypeError as exc:       # a bug in read: raised, and the paid answer kept unread
        assert "ReadsAnObject.read's output isn't plain JSON" in str(exc), exc
    else:
        raise AssertionError("an output that isn't plain JSON was written")
    [folder] = (tmp / "runs").iterdir()
    assert json.loads((folder / "records" / "a2.json").read_text())["unread"]
    name = b.run(GoldEcho(), tmp / "runs")["run"]
    refused(bench(tmp / "d", NanScore()).grade, tmp / "runs" / name, match="NanScore's result for row a1 isn't plain")
    refused(bench(tmp / "d", ObjectSummary()).grade, tmp / "runs" / name, match="summarize's summary isn't plain JSON")


def test_progress_lines_say_failed_and_cost(tmp: Path) -> None:
    said: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: said.append(r.getMessage())
    logging.getLogger("omni_bench_core").addHandler(handler)
    logging.getLogger("omni_bench_core").setLevel(logging.INFO)
    try:
        bench(dataset(tmp / "d")).run(Copy(), tmp / "runs", workers=1)     # not a terminal: one line per answer
    finally:
        logging.getLogger("omni_bench_core").removeHandler(handler)
    last = [m for m in said if " 5/5 " in m]
    assert last and last[0].endswith("; 1 failed, $1.00"), said     # c1 is refused; four answers at $0.25


def test_a_dataset_name_makes_one_question_wherever_the_data_is_read(tmp: Path) -> None:
    here, there = dataset(tmp / "here"), dataset(tmp / "there")      # one dataset, read from two places
    named = [benchmark.Benchmark(Copying(), d, [Exact()], dataset_name="texts") for d in (here, there)]
    assert len({runs.stem(Copy(), b.question()) for b in named}) == 1
    assert named[0].question()["dataset"] == "texts"
    assert bench(here).question()["dataset"] != bench(there).question()["dataset"]     # without it, the address
    refused(benchmark.Benchmark, Copying(), here, [Exact()], dataset_name=" ", match="dataset_name is ' '")



@dataclasses.dataclass(frozen=True)
class Inner:
    a: int = 1


@dataclasses.dataclass(frozen=True)
class Stops(Copy):
    name: ClassVar[str] = "stops"
    stops: tuple[str, ...] = ("x",)


@dataclasses.dataclass(frozen=True)
class Nested(Copy):
    name: ClassVar[str] = "nested"
    inner: Inner = dataclasses.field(default_factory=Inner)


def test_options_from_a_config_are_the_types_the_adapter_takes(tmp: Path) -> None:
    """A config's list is a tuple option's tuple, so the adapter is the one its class describes; a nested dataclass
    option, which a config can't say, is refused."""
    assert adapter.configure(Stops, {"stops": ["a", "b"]}) == Stops(stops=("a", "b"))
    refused(adapter.configure, Nested, {"inner": {"a": 2}}, match="option inner is a Inner, a dataclass")


SERVER: dict[str, Any] = {}   # what ServedCopy's server says it serves


@dataclasses.dataclass(frozen=True)
class ServedCopy(adapter.Served):
    """Copies the file's text, as a model served by us would answer; its server says it serves `SERVER`."""

    name: ClassVar[str] = "served_copy"
    stated: ClassVar[tuple[str, ...]] = ("args",)

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        return adapter.Call(raw={"answer": x.read_text()})

    def read(self, raw: Any, x: Path) -> Any:
        with adapter.reading("the answer"):
            return {"text": raw["answer"]}

    def server(self, *, timeout: float) -> dict[str, Any]:
        if not SERVER: raise ConnectionRefusedError("nothing listens")
        return dict(SERVER)


def test_how_a_model_is_served_is_part_of_the_question_and_checked_before_any_call(tmp: Path) -> None:
    """Two servings of one model are two runs; a server that serves something other than `served` says is refused
    before any call, as is one that isn't running; `served` is required."""
    b = bench(dataset(tmp / "d"))
    one = {"weights": "org/m", "revision": "a" * 40, "vllm": "0.20.1", "args": ["--max-model-len 8192"]}
    SERVER.update(weights="org/m", revision="a" * 40, vllm="0.20.1")
    p = b.plan(ServedCopy(served=one), tmp / "runs")
    other = b.plan(ServedCopy(served={**one, "args": ["--max-model-len 4096"]}), tmp / "runs")
    assert p.run.folder.name.split("-")[1] != other.run.folder.name.split("-")[1]
    b.predict(p)
    assert b.grade(p.run)["scorers"]["exact"]["headline"] == 1.0
    assert json.loads((p.run.folder / "settings.json").read_text())["options"]["served"] == one
    SERVER.update(revision="b" * 40)
    refused(b.plan, ServedCopy(served={**one, "revision": "b" * 40}), tmp / "runs", resume=p.run.name,
            match="answers another question")
    said = refused(b.plan, ServedCopy(served=one), tmp / "runs", match="serves something other than its served says")
    assert f"revision is {'b' * 40!r} there and {'a' * 40!r} in served" in said, said
    SERVER.update(revision="a" * 40, quantization="fp8")
    refused(b.plan, ServedCopy(served=one), tmp / "runs", match="quantization is 'fp8' there and '(not given)'")
    SERVER.pop("quantization")
    refused(b.plan, ServedCopy(served={**one, "revison": "a" * 40}), tmp / "runs", match="says revison, which its server")
    SERVER.clear()
    try:
        b.plan(ServedCopy(served=one), tmp / "runs")
        raise AssertionError("a server that isn't running was not refused")
    except ConnectionRefusedError as exc:
        assert "couldn't ask its server what it serves" in exc.__notes__[0], exc.__notes__
    refused(adapter.configure, ServedCopy, {}, match="needs its served")


def test_a_config_is_one_run_written_out_and_its_run_keeps_it_whole(tmp: Path) -> None:
    import os

    import yaml

    from omni_bench_core import config

    def make(name, opts):
        if name != "copy": raise ValueError(f"no system {name!r}; the systems are copy")
        return adapter.configure(Copy, opts)

    good = {"dataset": str(dataset(tmp / "d")), "manifest": "manifest.parquet", "scorers": {"exact": {}},
            "rows": {"sample": 0, "limit": 2, "only": None}, "timeout": 60.0, "workers": 1, "system": {"copy": {}}}
    (path := tmp / "c.yaml").write_text(yaml.safe_dump(good))
    c = config.load(path, ["rows.limit=3"])
    b, s = config.benchmark(c, Copying(), {"exact": Exact}), config.system(c, make)
    p = config.plan(c, b, s, tmp / "runs")
    assert len(p.rows) == 3 and p.timeout == 60.0 and p.workers == 1
    b.predict(p)
    # the run keeps its config resolved, and loading it again asks the same question
    kept = config.load(config.write(c, b, s, p.run.folder))
    assert kept.rows.limit == 3 and kept.system == {"copy": s.options()} and kept.scorers == {"exact": Exact().options()}
    assert config.plan(kept, b, s, tmp / "runs", resume=p.run.name).run.folder == p.run.folder
    # a dataset folder is kept as its full path, so the config reads the same folder from anywhere
    rel = config.parse({**good, "dataset": os.path.relpath(good["dataset"])}, "rel")
    assert config.resolved(rel, b, s)["dataset"] == str(Path(good["dataset"]).resolve())
    # a template writes every option of the system, at its defaults
    t = config.template(make, "copy", dataset="d", manifest="manifest.parquet", scorers={"exact": {}}, timeout=60)
    assert config.parse(t, "t").system == {"copy": Copy().options()}
    for bad, sets, match in (
        ({k: v for k, v in good.items() if k != "rows"}, [], "rows: missing"),
        ({**good, "benchmark": "x"}, [], "benchmark: not a key a config has"),
        ({**good, "timeout": 0}, [], "timeout: Input should be greater than 0; it is 0"),
        ({**good, "system": {"a": {}, "b": {}}}, [], "system is one system"),
        (good, ["rows.limt=3"], "did you mean rows.limit"),
        (good, ["rows.limit"], "write path=value"),
    ):
        path.write_text(yaml.safe_dump(bad))
        refused(config.load, path, sets, match=match)
    refused(config.scorers, config.parse({**good, "scorers": {"nope": {}}}, "x"), {"exact": Exact}, match="no scorer 'nope'")
    refused(config.system, config.parse({**good, "system": {"nope": {}}}, "x"), make, match="no system 'nope'")
    refused(config.apply, {"system": {"m": {True: 1}}}, ["system.m.on=2"], match="system.m has True")
    refused(config.apply, {"system": {"m.x": {"v": 1}, "m": {"x": {"v": 2}}}}, ["system.m.x.v=5"], match="could be read as")
    refused(config.apply, {"scorers": {"e": [{"x": 1}]}}, ["scorers.e.0.x=2"], match="scorers.e is a list")
    refused(config.parse, {**good, "scorers": {"exact": 5}}, "x", match="scorers.exact is 5; give its options")
    refused(config.parse, {**good, "rows": {**good["rows"], "only": [83]}}, "x", match="quote it")
    dotted = {"system": {"org/model-1.5": {"x": {"y": 1}}}}
    assert config.apply(dotted, ["system.org/model-1.5.x.y=2"]) == {"system": {"org/model-1.5": {"x": {"y": 2}}}}


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            d = Path(tempfile.mkdtemp())
            try:
                f(d)
            finally:
                shutil.rmtree(d)
            print("ok", name)
