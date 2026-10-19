# Overview

This document is for someone who has never seen omni-bench-core. It goes through the pieces one at a time, with a
small example for each, and then shows how they fit together.

## What a benchmark does

A benchmark measures a system, such as a parser, an extractor or an LLM, on a dataset. It always does the
same two things:

1. send each row of the dataset to the system, and save what it answers; then
2. compare each saved answer with that row's correct answer (the gold), and summarize the comparisons as
   one number (the headline).

```
   1. predict                                       2. grade

  +---------+  input   +--------+  answer   +------------+  answers   +--------+         +----------+
  | dataset | -------> | system | --------> | run folder | ---------> | scorer | ------> | summary  |
  |  rows   |          +--------+  saved    |  (on disk) |            +--------+         | headline |
  +---------+                               +------------+                ^              +----------+
       |                                                                  |
       +------------------------------- gold -----------------------------+
```

Predicting costs money and time. Grading is cheap. So the two are separate steps, and predicting saves every
answer to disk before anything is graded. You can grade again, with a new scorer or a fixed one, without
calling the system again.

## The pieces

There are five pieces. Three of them you write, and two are given.

| piece | what it is | you write it? |
| --- | --- | --- |
| `Task` | what a row is, and what of a row a system is sent | yes, once per kind of task |
| `Adapter` | how to call one system, and how to read its answer | yes, once per system |
| `Scorer` | how to compare an answer with the gold | yes, once per way of grading |
| `Run` | one system's folder of answers on disk | no |
| `Benchmark` | a task, a dataset and its scorers, and the loop that runs them | no, you only make one |

```
  Benchmark(task, dataset, scorers)
     |
     |  .run([adapter, adapter, ...], out)
     v
  +--------------------------------------------------------------+
  |  for each adapter: a Run (a folder under `out`)              |
  |                                                              |
  |    for each row:  Task.input(row) -> Adapter -> answer       |
  |                   saved in the Run                           |
  |                                                              |
  |    for each scorer:  Scorer.grade(run) -> summary            |
  +--------------------------------------------------------------+
```

Each piece below has a small example from `tests/test_core.py`. That file holds a whole toy task: the
"system" is asked to copy a text file's text, and it is graded on whether the copy is exact.

### The dataset and its rows

A dataset is a folder, or an `hf://` address of a Hugging Face dataset. It holds:

- `manifest.parquet`: one line per row, with its `id`, its `suite` (a group of rows, such as "tables" or
  "handwriting") and the paths of its files;
- the input files, such as PDFs; and
- one gold file per row: the correct answer.

```
  my-dataset/
    manifest.parquet          doc_id | suite   | doc_path      | gt_path
                              a1     | letters | docs/a1.txt   | gold/a1.json
                              b1     | numbers | docs/b1.txt   | gold/b1.json
    docs/a1.txt               hello
    gold/a1.json              {"text": "hello"}
```

A row is one line of the manifest, as a frozen dataclass. It holds paths, never file contents, so a large
dataset fits in memory. It must have an `id` and a `suite`.

```python
@dataclasses.dataclass(frozen=True)
class Doc:
    doc_id: str
    suite: str
    doc_path: Path
    gt_path: Path
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)   # the manifest's other columns

    @property
    def id(self) -> str:
        return self.doc_id
```

A dataset can have more than one manifest. `manifests/sample40.parquet` is a subset of 40 rows, used to
compare systems cheaply (see "A quick look, and a fair comparison" below).

### Task

A task says what the rows are, and what a system is sent. It has two methods:

- `rows(dataset)`: reads the manifest into rows; and
- `input(row)`: the part of the row the system is sent, such as the PDF's path.

`input` exists so that a system never sees the row, which holds the gold.

```python
@dataclasses.dataclass(frozen=True)
class Copying(task.Task):
    def rows(self, dataset, manifest=data.MANIFEST) -> tuple[Doc, ...]:
        _, rs = data.read_manifest(dataset, manifest, columns=(), id="doc_id", inputs=("doc_path",))
        return tuple(Doc(**r) for r in rs)

    def input(self, row: Doc) -> Path:
        return row.doc_path
```

You write a task once per kind of task: omni-parse-bench has `Parse`, and omni-extract-bench has `Extract`.
Most people only use one that exists.

### Adapter

An adapter is how to call one system. It is a frozen dataclass, and its fields are the system's options,
such as a mode or a model. One instance is one configuration of the system: `Datalab(mode="balanced")` and
`Datalab(mode="accurate")` are two instances, and so two runs.

It has two methods:

- `call(input)`: calls the system and returns a `Call`, which holds the answer exactly as received (`raw`)
  and what the system said it cost (`cost`). On a failure it raises a `VendorError`; and
- `read(raw, input)`: turns the raw answer into the output the scorers read. The default returns `raw`
  unchanged.

```python
@dataclasses.dataclass(frozen=True)
class Copy(adapter.Adapter):
    upper: bool = False                                       # an option

    def call(self, x: Path, *, timeout: float) -> adapter.Call:
        text = x.read_text()
        return adapter.Call(raw={"answer": text.upper() if self.upper else text},
                            cost=adapter.Cost(usd=0.25))

    def read(self, raw: Any, x: Path) -> Any:
        return {"text": raw["answer"]}
```

`call` and `read` are separate because the raw answer is saved. When you find a bug in `read`, you fix it and
read every saved answer again (`Benchmark.reread`), without paying for new calls.

You don't write retries, timeouts or error handling. The core wraps every call in the same rules:

```
  adapter.predict(system, input)
    call ----> answer ----> read ----> output                       the row's record
      |
      +-- 429 or 5xx (transient)  -> wait, call again, up to `retries` times
      +-- 400, a refusal, ...     -> recorded as the row's error; the run goes on
      +-- 401, 402 (the account)  -> the whole run stops, since every row would fail
```

### Run

A run is one adapter's folder of answers. You never write one, but you will read its files, so it helps to
know what is in it.

```
  runs/copy-1a2b3c4d/                 <adapter name>-<digest of its options>
    settings.json                     the adapter's options, the dataset, the manifest and the task
    records/a1.json                   everything about one row's call: output, raw answer, error, cost
    predictions/a1.json               just the output; this is what grading reads
    exact/                            one folder per scorer
      scores.jsonl                    one line per row
      summary.json                    the summary, with its headline
      details/a1.jsonl                each row's breakdown, if the scorer gives one
```

The folder's name holds a digest of the adapter's options, so two configurations never share a folder. A
run folder belongs to one dataset, one manifest and one task: pointing it at another is refused, so answers
for one dataset are never graded against another's gold.

A run can be stopped and started again. Before calling the system for a row, the run checks whether that
row already has a record. If it does, the row is skipped. So rerunning the same command only calls the
system for the rows it hasn't answered yet, and no answer is paid for twice.

### Scorer

A scorer is one way of grading. It is a frozen dataclass, and its fields are its options. It has two methods
you write:

- `gold(row)`: reads the row's gold file and checks it. Each scorer reads its own, because what the gold
  holds depends on the scorer; and
- `score(pred, gold)`: compares one prediction with its gold, and returns a `Score`.

A `Score` has two parts:

- `result`: a few numbers for the row, such as `{"exact": True, "length": 5}`; and
- `details`: an optional breakdown, such as which characters differed. It is written to
  `details/<id>.jsonl`, apart from the numbers, so it can be large.

```python
@dataclasses.dataclass(frozen=True)
class Exact(scorer.Scorer):
    def gold(self, row: Doc) -> str:
        return json.loads(row.gt_path.read_text())["text"]

    def prediction(self, output: Any) -> str:                 # optional: what this scorer reads of an output
        return output["text"]

    def score(self, pred: str, gold: str) -> scorer.Score:
        diffs = [{"at": i, "pred": a, "gold": b} for i, (a, b) in enumerate(zip(pred, gold)) if a != b]
        return scorer.Score({"exact": pred == gold, "length": len(pred)}, diffs or None)
```

Grading turns each row into one line of `scores.jsonl`, and all the lines into one summary:

```
  row a1:  prediction "hello", gold "hello"  --score-->  {"exact": true,  "length": 5}
  row a2:  prediction "w0rld", gold "world"  --score-->  {"exact": false, "length": 5}
  row c1:  the call failed                   ---------->  failed
                                                             |
                                                         summarize
                                                             v
           {"headline": 0.333, "exact": 0.5, "coverage": 0.667, "failed": 1, "per_suite": {...}}
```

By default the summary is the mean of each number over the scored rows, overall and per suite. The headline
is the mean of the first number, or of `headline_field` when you set it. A scorer whose rows combine in
another way, such as one rate over every row's counts, writes its own `summarize`.

`Scorer.grade` runs those steps and applies the rules (below). You can't override it, so every scorer
follows the same rules.

### How scoring runs

`grade` reads the predictions and golds in batches of `batch_size` rows (500 by default), and hands each
batch to `score_all`. By default, `score_all` calls `score` on each row of the batch, `workers` rows at a
time, in a pool of processes or threads (`pool`). So you choose how scoring runs with three settings on the
class, and you can take over completely by writing `score_all` yourself:

```
  Scorer.grade(run)
    for each batch of `batch_size` rows:
      read predictions and golds                       on this machine
      score_all(rows, preds, golds)                    how the batch is scored is up to the scorer:

        default, workers = 1          score, score, score, ...           one row at a time, here
        default, pool = "process"     score  score  score  score         N processes, for CPU-bound work
        default, pool = "thread"      score  score  score  score         N threads, for work that waits
        your own score_all            the whole batch, as values         anything: one batched request,
                                                                         a GPU model, a cluster
      write details                                     on this machine
```

The three common cases:

- CPU-bound, such as oeb's field matching. Comparing two JSON documents is pure computation, so it runs in
  processes, one per core:

  ```python
  class FieldMatching(scorer.Scorer):
      workers: ClassVar[int] = min(8, os.cpu_count() or 1)   # pool is "process" by default
  ```

- IO-bound, such as an LLM as a judge. Each row waits on a call to a model, so it runs in many threads.
  The judge is an adapter, so it gets the same retries and time budget as any system. `cache = True` keeps
  each row's score on disk, so grading again doesn't pay for the same judgment twice:

  ```python
  class Agree(scorer.Scorer):
      pool: ClassVar[str] = "thread"
      workers: ClassVar[int] = 16
      cache: ClassVar[bool] = True
      judge: adapter.Adapter = SomeJudge(model="...")        # an option, so it names the scorer's folder

      def score(self, pred, gold):
          rec = scorer.ask(self.judge, (pred, gold))         # the judge's record: output, cost, error
          return scorer.Score({"agree": rec["output"]["verdict"] == "yes"})
  ```

- Your own execution, such as one request per batch to a judge, a model on a GPU, or a cluster. Write
  `score_all`. It gets a batch of values (never files) and returns one `Score` per row, in order.
  `batch_size` sets how many rows a batch holds, and `None` gives it every row at once, for a score that
  needs the whole run:

  ```python
  class BatchJudge(scorer.Scorer):
      batch_size: ClassVar[int | None] = 50

      def score_all(self, rows, preds, golds):
          verdicts = my_cluster.map(compare, preds, golds)  # however you like
          return [scorer.Score({"agree": v}) for v in verdicts]
  ```

Whichever you choose, reading files, the cache, failures and the summary stay in `grade`. So the scorer
only decides how to compute the scores.

### Benchmark

A benchmark is a task, a dataset and its scorers: everything that stays fixed while the systems change.

```python
b = benchmark.Benchmark(Copying(), "my-dataset", [Exact()])
b.run([Copy(), Copy(upper=True)], out="runs")
```

`run` does three steps in order. Each step reads what the one before it wrote to disk, so you can also call
them one at a time:

```
  b.run(systems, out)
    |
    +-- plan(systems, out)    checks everything before any call: the options, the run folders, and that
    |                         every gold can be read. Returns a Plan: the runs, and the rows to predict.
    |
    +-- predict(plan)         for each run, calls the system on each row with no record, `workers` at a
    |                         time, and saves each record as it arrives.
    |
    +-- grade(run)            each scorer grades the run over the whole manifest; returns the summaries
                              and the run's cost.
```

Other things a benchmark does:

- `b.grade("mine/")`: grades a folder of predictions you made yourself, `mine/predictions/<id>.json`;
- `b.reread(run)`: reads every saved raw answer again with the adapter's `read`, after you fix it;
- `b.check()`: checks the dataset (manifest, input files, gold) without calling anything; and
- `b.describe(plan)`: says in words what a plan will do, before you pay for it.

## The rules

These are the rules every run follows. They live in the parts you don't write, so a new adapter or scorer
can't break them.

- A row whose call failed, or whose answer couldn't be read, is failed. It counts as 0 in the headline:
  the headline is the scorer's number times coverage (the share of rows that didn't fail). To find out why a
  row failed, look at its `failed` line in `scores.jsonl`, which holds the error, and at `records/<id>.json`.
- A row the system was never asked about is pending. While any row is pending, the headline is null, because
  a number over part of the rows can't be compared with a number over all of them.
- A row the scorer couldn't score, for a reason that isn't the system's fault (a judge that didn't answer),
  is unscored. It too makes the headline null, until a later grading scores it.
- A run folder holds one configuration of one system, on one dataset, one manifest and one task.
- An answer is never paid for twice: a stopped run resumes where it stopped.

```
  each row of the manifest is one of:

    scored     the scorer's numbers                  counts in the headline
    failed     the call failed, or the answer was    counts as 0 in the headline
               unreadable
    unscored   the scorer couldn't score it          headline is null until it is scored
    pending    never asked                           headline is null until it is asked
```

## A quick look, and a fair comparison

There are two ways to run on fewer rows, for two different purposes:

- to look: `run(..., sample=20)` (or `limit`, or `only`) calls the system on a few rows. The grading shows the
  numbers over those rows, and no headline, because the other rows are pending. Running again without
  `sample` predicts only the rest; and
- to compare: make the subset a manifest of the dataset, such as `manifests/sample40.parquet`, and run on that
  manifest. Every row of the manifest is predicted, so the run has a headline, and its settings.json names
  the manifest, so it can't be mistaken for a run on the whole dataset.

## When you need more

Each of these is optional, has a default, and is described in its class's docstring.

| you want | do this |
| --- | --- |
| a summary broken down by something the row holds (a language, a source) | return it from `Scorer.row_fields`, and group by it in `summarize` with `scorer.per(lines, "row.<field>", over)` |
| a slow scorer, an LLM as a judge, or scoring on your own machines | see "How scoring runs" above |
| to call a system with more or fewer rows at once | `workers=` on `run`, one number or one per adapter name |
| to run each row somewhere else (processes, a cluster) | `executor=` on `run`, a function that makes a `concurrent.futures.Executor` |
| to try a scorer without writing results | `scorer.grade(run, rows, write=False)` |

## Where to go next

- `tests/test_core.py`: the whole toy task above, with tests of every rule. Run it with
  `PYTHONPATH=. python tests/test_core.py`; and
- the class docstrings in `omni_bench_core/`: what each method is for, and when you would write it.
