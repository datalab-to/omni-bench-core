"""The core on a toy task, offline: copy a text file's text.

    python tests/test_core.py

A row is a text file and its gold `{"text": ...}`; an adapter answers `{"text": ...}`; `Exact` scores whether
the two are equal, with one detail line per differing character.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import threading
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
    suite: str
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
        if "refuse" in text: raise errors.VendorError("refused", status=400)
        if "account" in text: raise errors.VendorError("out of credits", status=402)
        if "flaky" in text and first: raise errors.VendorError("busy", status=503)
        if "garble" in text: return adapter.Call(raw="<garbled>", cost=adapter.Cost(usd=0.5))
        return adapter.Call(raw={"answer": text.upper() if self.upper else text}, cost=adapter.Cost(usd=0.25))

    def read(self, raw: Any, x: Path) -> Any:
        return {"text": raw["answer"]}


@dataclasses.dataclass(frozen=True)
class Exact(scorer.Scorer):
    def gold(self, row: Doc) -> str:
        got = json.loads(row.gt_path.read_text())
        if not isinstance(got.get("text"), str): raise ValueError(f'{row.gt_path} holds {got}, not {{"text": ...}}')
        return got["text"]

    def prediction(self, output: Any) -> str:
        if not (isinstance(output, dict) and isinstance(output.get("text"), str)):
            raise ValueError(f"the prediction has no text: {str(output)[:80]}")
        return output["text"]

    def score(self, pred: str, gold: str) -> scorer.Score:
        diffs = [{"at": i, "pred": a, "gold": b} for i, (a, b) in enumerate(zip(pred, gold)) if a != b]
        return scorer.Score({"exact": pred == gold, "length": len(pred)}, diffs or None)


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
        if "flaky" in gold and first: raise errors.VendorTimeout("the judge took too long")
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


TEXTS = {"a1": ("letters", "hello"), "a2": ("letters", "world"), "b1": ("numbers", "123"),
         "b2": ("numbers", "456"), "c1": ("other", "refuse me")}


def dataset(root: Path, texts=TEXTS, gold=None, suites: bool = True) -> Path:
    (root / "docs").mkdir(parents=True)
    (root / "gold").mkdir()
    for i, (_, t) in texts.items():
        (root / "docs" / f"{i}.txt").write_text(t)
        (root / "gold" / f"{i}.json").write_text(json.dumps(gold(t) if gold else {"text": t}))
    cols = {"doc_id": list(texts), "doc_path": [f"docs/{i}.txt" for i in texts],
            "gt_path": [f"gold/{i}.json" for i in texts], "lang": ["en"] * len(texts)}
    if suites: cols["suite"] = [s for s, _ in texts.values()]
    pq.write_table(pa.table(cols), root / "manifest.parquet")
    return root


def bench(root: Path, *scorers: scorer.Scorer) -> benchmark.Benchmark:
    return benchmark.Benchmark(Copying(), root, scorers or (Exact(),))


def refused(f, *args, match: str = "", **kwargs) -> str:
    try:
        f(*args, **kwargs)
    except (errors.UsageError, TypeError) as exc:
        assert match in str(exc), f"{match!r} not in {exc}"
        return str(exc)
    raise AssertionError(f"{f.__name__} was not refused")


def test_defaults(tmp: Path) -> None:
    assert (GoldEcho.name, ExactPooled.name, Copying.name) == ("gold_echo", "exact_pooled", "copying")
    rows = bench(dataset(tmp / "d", suites=False)).rows()
    assert {r.suite for r in rows} == {"other"} and rows[0].extra == {"lang": "en"} and rows[0].doc_path.is_absolute()


def test_gold_echo_scores_one(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    [(name, g)] = b.run([GoldEcho()], tmp / "runs").items()
    s = g["scorers"]["exact"]
    assert s["headline"] == 1.0 and s["exact"] == 1.0 and s["length"] == 5.0, s
    assert s["per_suite"]["numbers"] == {"scored": 2, "exact": 1.0, "length": 3.0}
    assert g["cost"] == {"usd": None, "credits": None}
    assert (tmp / "runs" / name / "exact" / "scores.jsonl").exists()
    assert not (tmp / "runs" / name / "exact" / "details").exists()


def test_resume_calls_nothing(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "f1": ("other", "flaky")}))
    CALLS.clear()
    b.run([Copy()], tmp / "runs")
    assert CALLS.count("f1") == 2, CALLS          # a 503 is retried within the row
    CALLS.clear()
    [g] = b.run([Copy()], tmp / "runs").values()
    assert CALLS == [], CALLS
    s = g["scorers"]["exact"]
    assert s["failed"] == 1 and abs(s["coverage"] - 5 / 6) < 1e-9
    assert abs(s["headline"] - 5 / 6) < 1e-9, s     # a refused row counts as 0
    assert g["cost"]["usd"] == 1.25


def test_pending_nulls_headline(tmp: Path) -> None:
    [g] = bench(dataset(tmp / "d")).run([GoldEcho()], tmp / "runs", limit=2).values()
    s = g["scorers"]["exact"]
    assert s["pending"] == 3 and s["graded"] == 2 and s["headline"] is None, s


def test_failures_and_details(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "g1": ("other", "garble")}))
    [(name, g)] = b.run([Copy(upper=True)], tmp / "runs").items()
    folder = tmp / "runs" / name
    lines = {x["id"]: x for x in map(json.loads, (folder / "exact" / "scores.jsonl").read_text().splitlines())}
    assert lines["c1"]["status"] == "failed" and "refused" in lines["c1"]["error"]
    assert lines["g1"]["status"] == "failed" and "ReadError" in lines["g1"]["error"]
    rec = json.loads((folder / "records" / "g1.json").read_text())
    assert rec["raw"] == "<garbled>" and rec["output"] is None and rec["error"]["type"] == "ReadError"
    assert lines["a1"]["exact"] is False
    assert len((folder / "exact" / "details" / "a1.jsonl").read_text().splitlines()) == 5
    assert abs(g["scorers"]["exact"]["headline"] - 2 / 4 * 4 / 6) < 1e-9    # digits survive upper()


def test_reread_after_a_fix(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "g1": ("other", "garble")}))
    [name] = b.run([Copy()], tmp / "runs")

    @dataclasses.dataclass(frozen=True)
    class Fixed(Copy):
        name: ClassVar[str] = "copy"

        def read(self, raw: Any, x: Path) -> Any:
            return {"text": raw} if isinstance(raw, str) else super().read(raw, x)

    CALLS.clear()
    assert b.reread(runs.Run(tmp / "runs" / name, Fixed())) == ["g1"]
    assert CALLS == []
    assert json.loads((tmp / "runs" / name / "predictions" / "g1.json").read_text()) == {"text": "<garbled>"}
    refused(b.reread, runs.Run(tmp / "runs" / name), match="no system")


def test_pool_matches_serial(tmp: Path) -> None:
    [g] = bench(dataset(tmp / "d"), Exact(), ExactPooled()).run([Copy()], tmp / "runs").values()
    one, two = g["scorers"]["exact"], g["scorers"]["exact_pooled"]
    assert {k: v for k, v in one.items() if k != "scorer"} == {k: v for k, v in two.items() if k != "scorer"}


def test_restore_from_record(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    [name] = b.run([GoldEcho()], tmp / "runs")
    (tmp / "runs" / name / "predictions" / "a1.json").unlink()
    CALLS.clear()
    [g] = b.run([GoldEcho()], tmp / "runs").values()
    assert CALLS == [] and g["scorers"]["exact"]["headline"] == 1.0
    assert (tmp / "runs" / name / "predictions" / "a1.json").exists()


def test_brought_predictions(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    mine = tmp / "mine"
    (mine / "predictions").mkdir(parents=True)
    for i in ("a1", "a2", "b1"):
        (mine / "predictions" / f"{i}.json").write_text(json.dumps({"text": TEXTS[i][1]}))
    s = b.grade(mine)["scorers"]["exact"]
    assert s["pending"] == 0 and s["failed"] == 2 and abs(s["headline"] - 0.6) < 1e-9, s
    assert json.loads((mine / "settings.json").read_text())["dataset"] == data.dataset_id(tmp / "d")
    stray = tmp / "stray"
    (stray / "predictions").mkdir(parents=True)
    (stray / "predictions" / "zzz.json").write_text("{}")
    refused(b.grade, stray, match="names a row")
    refused(b.grade, tmp / "nowhere", match="no folder")


def test_refused_before_any_call(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", gold=lambda t: {"html": t}))
    CALLS.clear()
    refused(b.run, [GoldEcho()], tmp / "runs", match="can't read the gold of row a1")
    assert CALLS == []
    b = bench(dataset(tmp / "ok"))
    refused(adapter.configure, Copy, {"uper": True}, match="did you mean 'upper'")
    refused(adapter.configure, Copy, {"upper": "yes"}, match="takes a bool")
    assert adapter.configure(Copy, {"temperature": 0}) == Copy()
    assert adapter.configure(Exact, {}) == Exact()
    refused(adapter.configure, Exact, {"strict": True}, match="exact has no option 'strict'")
    refused(adapter.configure, Doc, {}, match="an adapter or a scorer")
    assert Copy().folder_name == Copy(temperature=0.0).folder_name != Copy(upper=True).folder_name
    refused(b.plan, [Copy(), Copy()], tmp / "runs", match="more than once")
    refused(b.plan, [Copy()], tmp / "runs", sample=2, suites=["nope"], match="unknown suites")
    refused(b.plan, [GoldEcho()], tmp / "runs", workers={"copy": 3}, match="workers names")
    refused(b.plan, Copy(), tmp / "runs", match="is a list")
    refused(benchmark.Benchmark, Copying(), tmp / "ok", (), match="at least one scorer")
    refused(benchmark.Benchmark, Copying(), tmp / "ok", (Exact(), Exact()), match="twice")


def test_forgotten_decorator_is_refused(tmp: Path) -> None:
    class Forgot(Copy):
        mode: str = "balanced"

    class Loose(Exact):
        fold_case: bool = True

    refused(lambda a: a.folder_name, Forgot(), match="decorate it with @dataclasses.dataclass(frozen=True)")
    refused(adapter.configure, Forgot, {}, match="Forgot declares mode")
    refused(benchmark.Benchmark, Copying(), tmp, (Loose(),), match="Loose declares fold_case")


def test_judge_cached_and_unscored(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "f1": ("other", "flaky")}), Agree(), Agree(Judge(strict=False)))
    [(name, g)] = b.run([GoldEcho()], tmp / "runs").items()
    strict, loose = (Agree().folder_name, Agree(Judge(strict=False)).folder_name)
    assert strict != loose and strict.startswith("agree-"), (strict, loose)
    assert sum(g["scorers"][f]["unscored"] for f in (strict, loose)) == 1
    s = g["scorers"][strict] if g["scorers"][strict]["unscored"] else g["scorers"][loose]
    assert s["unscored"] == 1 and s["failed"] == 0 and s["headline"] is None, s
    assert g["scorers"][strict]["options"] == {"judge": {"provider": "judge", "strict": True}}
    assert abs(s["judge_usd"] - 0.01) < 1e-9
    JUDGED.clear()
    g = b.grade(tmp / "runs" / name)
    # Note: the judge fails only the first time it sees "flaky", which one of the two scorers asked first.
    assert JUDGED == ["flaky"], JUDGED      # only the unscored row is asked again
    assert all(g["scorers"][f]["headline"] == 1.0 and g["scorers"][f]["unscored"] == 0 for f in (strict, loose))
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
    [name] = b.run([GoldEcho()], tmp / "runs")
    JUDGED.clear()
    (tmp / "d" / "gold" / "a1.json").write_text(json.dumps({"text": "hullo"}))
    s = b.grade(tmp / "runs" / name)["scorers"][Agree().folder_name]
    assert JUDGED == ["hullo"], JUDGED                    # a changed gold is a new key
    assert s["agree"] == 0.8 and s["judge_usd"] == 0.01, s
    fresh = json.loads((tmp / "runs" / name / Agree().folder_name / "scores.jsonl").read_text().splitlines()[0])
    JUDGED.clear()
    b.grade(tmp / "runs" / name)
    kept = json.loads((tmp / "runs" / name / Agree().folder_name / "scores.jsonl").read_text().splitlines()[0])
    assert JUDGED == [] and fresh == kept
    assert Agree()._code() != AgreeOnPaths()._code()      # a change to the code is a new key
    refused(bench(dataset(tmp / "d2"), AgreeOnPaths()).run, [GoldEcho()], tmp / "runs2", match="must be JSON")


@dataclasses.dataclass(frozen=True)
class BatchAgree(Agree):
    """A judge asked about every row at once, as a batching judge would be."""

    def score_all(self, rows, preds, golds):
        BATCHES.append([r.id for r in rows])
        return super().score_all(rows, preds, golds)


BATCHES: list[list[str]] = []


def test_score_all_override_is_cached(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), BatchAgree())
    [name] = b.run([GoldEcho()], tmp / "runs")
    assert BATCHES == [["a1", "a2", "b1", "b2", "c1"]], BATCHES
    BATCHES.clear()
    (tmp / "d" / "gold" / "b2.json").write_text(json.dumps({"text": "457"}))
    s = b.grade(tmp / "runs" / name)["scorers"][BatchAgree().folder_name]
    assert BATCHES == [["b2"]] and s["agree"] == 0.8, (BATCHES, s)    # only the row with no kept score


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
        out = scorer.per(lines, "row.lang", lambda xs: {"n": len(xs)})
        return {"headline": super().summarize(lines)["headline"], **out}


def test_row_fields_and_dry_grading(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), ByLanguage())
    [(name, g)] = b.run([GoldEcho()], tmp / "runs").items()
    s = g["scorers"]["by_language"]
    assert s["per_lang"] == {"en": {"n": 5}} and s["headline"] == 1.0, s
    line = json.loads((tmp / "runs" / name / "by_language" / "scores.jsonl").read_text().splitlines()[0])
    assert line["row"] == {"lang": "en", "pages": 1}
    assert Exact().summarize([line])["exact"] == 1.0 and "pages" not in Exact().summarize([line])   # not averaged
    mine = tmp / "mine"
    (mine / "predictions").mkdir(parents=True)
    (mine / "predictions" / "a1.json").write_text(json.dumps({"text": "hello"}))
    s = bench(tmp / "d").grade(mine, write=False)["scorers"]["exact"]
    assert s["failed"] == 4 and abs(s["headline"] - 0.2) < 1e-9, s
    assert sorted(p.name for p in mine.iterdir()) == ["predictions"]       # nothing written


@dataclasses.dataclass(frozen=True)
class Overcounts(Exact):
    def summarize(self, lines):
        return {**super().summarize(lines), "failed": 0}


def test_summary_counts_are_grades(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"), Overcounts())
    refused(b.run, [Copy()], tmp / "runs", match="['failed'], which grade sets")


@dataclasses.dataclass(frozen=True)
class ExactInTwos(Exact):
    batch_size: ClassVar[int | None] = 2


@dataclasses.dataclass(frozen=True)
class ExactAtOnce(Exact):
    batch_size: ClassVar[int | None] = None


def test_batches_change_nothing(tmp: Path) -> None:
    [(name, g)] = bench(dataset(tmp / "d"), Exact(), ExactInTwos(), ExactAtOnce()).run([Copy(upper=True)], tmp / "runs").items()
    one, *rest = [{k: v for k, v in s.items() if k != "scorer"} for s in g["scorers"].values()]
    assert all(r == one for r in rest), (one, rest)
    folder = tmp / "runs" / name
    assert sorted(p.name for p in (folder / "exact_in_twos" / "details").iterdir()) == ["a1.jsonl", "a2.jsonl"]


def test_predict_through_another_executor(tmp: Path) -> None:
    import concurrent.futures as cf

    b = bench(dataset(tmp / "d", {**TEXTS, "z1": ("other", "missing")}))
    (tmp / "d" / "docs" / "z1.txt").unlink()
    p = b.plan([Copy()], tmp / "runs")
    [r] = p.runs
    # Note: processes stand in for a cluster: what is sent must pickle, and only the driver writes files.
    b.predict(p, executor=cf.ProcessPoolExecutor)
    assert sorted(f.stem for f in r.records.iterdir()) == ["a1", "a2", "b1", "b2", "c1"]   # z1 skipped, not failed
    s = b.grade(r)["scorers"]["exact"]
    assert s["pending"] == 1 and s["failed"] == 1 and s["headline"] is None, s


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
        if "key" in text: raise errors.VendorError("bad key", status=401)
        if "quota" in text: raise errors.VendorError("error: insufficient_quota for this org", status=429)
        if "busy" in text and first:
            raise errors.VendorError("busy", status=503, sent={"mode": "fast"}, cost=adapter.Cost(usd=0.5))
        if "never" in text:
            raise errors.VendorError("bad request", status=400, sent={"mode": "fast"}, cost=adapter.Cost(usd=0.1),
                                     job_id="job-1")
        return adapter.Call(raw={"text": text}, cost=adapter.Cost(usd=0.25), sent={"mode": "fast"})


def test_failed_calls_keep_cost_and_what_was_sent(tmp: Path) -> None:
    for name, text in (("k", "key"), ("q", "quota")):
        (tmp / f"{name}.txt").write_text(text)
        refused_stop = False
        try:
            adapter.predict(Billing(), tmp / f"{name}.txt")
        except errors.AccountFailure:
            refused_stop = True
        assert refused_stop, name
    (tmp / "b.txt").write_text("busy")
    rec = adapter.predict(Billing(), tmp / "b.txt")
    assert rec["error"] is None and rec["cost"]["usd"] == 0.75 and rec["cost"]["attempts"] == 2, rec["cost"]
    (tmp / "n.txt").write_text("never")
    rec = adapter.predict(Billing(), tmp / "n.txt")
    assert rec["error"]["status"] == 400 and rec["sent"] == {"mode": "fast"} and rec["cost"]["usd"] == 0.1, rec
    assert rec["job_id"] == "job-1"


@dataclasses.dataclass(frozen=True)
class OnlyCopies(Copying):
    def check(self, systems, scorers):
        if bad := [s.name for s in systems if not isinstance(s, Copy)]:
            raise errors.UsageError(f"{bad} can't copy")


def test_task_check_and_dataset_check(tmp: Path) -> None:
    b = benchmark.Benchmark(OnlyCopies(), tmp / "d", [Exact()])
    dataset(tmp / "d")
    CALLS.clear()
    refused(b.run, [GoldEcho()], tmp / "runs", match="can't copy")
    assert CALLS == []
    assert b.check() == {"rows": 5, "suites": 3}
    (tmp / "d" / "docs" / "a1.txt").unlink()
    refused(b.check, match="input files that don't exist: 1 row")


def test_old_records_are_not_restored(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    [name] = b.run([GoldEcho()], tmp / "runs")
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
    except errors.VendorError as exc:
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


def test_one_task_per_folder(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    b.run([GoldEcho()], tmp / "runs")
    other = benchmark.Benchmark(CopyingTwice(), tmp / "d", [Exact()])
    refused(other.plan, [GoldEcho()], tmp / "runs", match="holds a run on task")
    [folder] = (tmp / "runs").iterdir()
    assert json.loads((folder / "settings.json").read_text())["task"] == {"name": "copying", "options": {}}


def test_failures_can_be_audited(tmp: Path) -> None:
    b = bench(dataset(tmp / "d"))
    [(name, g)] = b.run([Copy()], tmp / "runs").items()       # c1 ("refuse me") is refused with a 400
    lines = [json.loads(x) for x in (tmp / "runs" / name / "exact" / "scores.jsonl").read_text().splitlines()]
    [failed] = [x for x in lines if x["status"] == "failed"]
    assert failed["id"] == "c1" and "refused" in failed["error"] and g["scorers"]["exact"]["failed"] == 1
    assert json.loads((tmp / "runs" / name / "records" / "c1.json").read_text())["error"]["status"] == 400


def test_one_dataset_per_folder(tmp: Path) -> None:
    bench(dataset(tmp / "d1")).run([GoldEcho()], tmp / "runs")
    refused(bench(dataset(tmp / "d2")).plan, [GoldEcho()], tmp / "runs", match="holds a run on dataset")


def test_stop_stops_and_grades_what_landed(tmp: Path) -> None:
    b = bench(dataset(tmp / "d", {**TEXTS, "z9": ("other", "account")}))
    try:
        b.run([Copy()], tmp / "runs", workers=1)
    except errors.AccountFailure:
        pass
    else:
        raise AssertionError("an account failure was not raised")
    [folder] = (tmp / "runs").iterdir()
    assert not (folder / "records" / "z9.json").exists()
    assert json.loads((folder / "exact" / "summary.json").read_text())["headline"] is None


def test_sample_spreads_over_suites(tmp: Path) -> None:
    rows = bench(dataset(tmp / "d")).rows()
    assert {r.suite for r in data.select(rows, sample=3)} == {"letters", "numbers", "other"}


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            d = Path(tempfile.mkdtemp())
            try:
                f(d)
            finally:
                shutil.rmtree(d)
            print("ok", name)
