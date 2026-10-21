# Overview

This document is for someone new to omni-bench-core. It goes through the pieces one at a time, with a small
example for each, then the rules that hold across them.

## What a benchmark does

A benchmark measures a system, such as a parser, an extractor or an LLM, on a dataset. It always does the
same two things:

1. send each row of the dataset to the system, and save what it answers; then
2. compare each saved answer with that row's correct answer (the gold), which gives one score per row,
   written to scores.jsonl; and summarize the scores as one number (the headline).

```
   1. predict                                       2. grade

  +---------+  input   +--------+  answer   +------------+  answers   +--------+  a score   +--------------+
  | dataset | -------> | system | --------> | run folder | ---------> | scorer | ---------> | scores.jsonl |
  |  rows   |          +--------+  saved    |  (on disk) |            +--------+  per row   +--------------+
  +---------+                               +------------+                ^                        |
       |                                                                  |               Scorer.summarize
       +------------------------------- gold -----------------------------+                        |
                                                                                                   v
                                                                                            +--------------+
                                                                                            |   summary    |
                                                                                            |   headline   |
                                                                                            +--------------+
```

Predicting costs money and time; grading is cheap. So predicting saves every answer to disk first, and you can
grade again, with a new or fixed scorer, without calling the system.

## The pieces

There are five pieces. For three of them you choose an existing one (a benchmark package ships its task, its
systems and its scorers) or subclass your own: `Task`, `Adapter` and `Scorer` are abstract base classes, and
Python refuses an instance that's missing a required method. The other two you use as they are.

| piece | what it is | how you use it |
| --- | --- | --- |
| `Task` | what a row is, and what of a row a system is sent | choose an existing one, or subclass your own |
| `Adapter` | how to call one system, and how to read its answer | choose an existing one, or subclass your own |
| `Scorer` | how to compare an answer with the gold | choose an existing one, or subclass your own |
| `Run` | one system's folder of answers on disk | read it; the core makes it |
| `Benchmark` | a task, a dataset and its scorers, and the loop that runs them | make one, `Benchmark(task, dataset, scorers)`; never subclass it |

```
  Benchmark(task, dataset, scorers)
     |
     |  .run(adapter, out)                  one system per run
     v
  +--------------------------------------------------------------+
  |  a Run: a folder under `out`                                 |
  |                                                              |
  |    for each row:  Task.input(row) -> Adapter -> answer       |
  |                   saved in the Run                           |
  |                                                              |
  |    for each scorer:  Scorer.grade(run) -> a score per row    |
  |                      (scores.jsonl) -> summary               |
  +--------------------------------------------------------------+
```

The examples below come from [`tests/test_core.py`](https://github.com/datalab-to/omni-bench-core/blob/main/tests/test_core.py),
a whole toy task: the "system" copies a text file's text, and is graded on whether the copy is exact.

### The dataset and its rows

A dataset is a folder, or an `hf://` address of a Hugging Face dataset. It holds:

- `manifest.parquet`: one line per row, with its `id` and the paths of its files;
- the input files, such as PDFs; and
- one gold file per row: the correct answer.

```
  my-dataset/
    manifest.parquet          doc_id | doc_path      | gt_path
                              a1     | docs/a1.txt   | gold/a1.json
                              b1     | docs/b1.txt   | gold/b1.json
    docs/a1.txt               hello
    gold/a1.json              {"text": "hello"}
```

A row is one line of the manifest, as a frozen dataclass holding paths, never file contents. It must have an
`id`; everything else is the task's.

```python
@dataclasses.dataclass(frozen=True)
class Doc:
    doc_id: str
    doc_path: Path
    gt_path: Path
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)   # the manifest's other columns

    @property
    def id(self) -> str:
        return self.doc_id
```

A dataset can have more than one manifest, such as `manifests/sample40.parquet`, a subset its authors chose.

### Task

A task says what the rows are, and what a system is sent. It has two abstract methods:

- `rows(dataset)`: reads the manifest into rows; and
- `input(row)`: the part of the row the system is sent, such as the PDF's path, so a system never sees the
  gold.

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

### Adapter

An adapter is how to call one system. It is a frozen dataclass whose fields are the system's options, so one
instance is one configuration: `Datalab(mode="balanced")` and `Datalab(mode="accurate")` are two runs. It has:

- `call(input)`, abstract: calls the system and returns a `Call`, the answer as received (`raw`) and what
  the system said it cost (`cost`), or raises `SystemFailed`; and
- `read(raw, input)`: turns the raw answer into the output the scorers read; by default `raw` itself.

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

The raw answer is saved, so after a fix to `read`, `Benchmark.reread` reads every saved answer again with no
call. You don't write retries, timeouts or error handling; the core wraps every call in the same rules:

```
  adapter.predict(system, input)
    call ----> answer ----> read ----> output                       the row's record
      |
      +-- 429 or 5xx (transient)  -> wait, call again, up to `retries` times
      +-- SystemFailed (a 400 ...) -> recorded as the row's error, with its traceback; the run goes on
      +-- past the timeout        -> recorded as a timeout, even if an answer came later
      +-- 401, 402 (the account)  -> PermissionError: the whole run stops, since every row would fail
      +-- connection refused      -> raised as it is: the whole run stops, nothing listens at the address
      +-- any other exception     -> raised as it is, with a note naming the adapter and the row
```

A row counts 0 only when you say the system failed:

- raise `errors.SystemFailed(message, status=...)` (or `SystemTimeout`, `SystemUnreachable`) in `call` or
  `read`; or read the answer inside `with adapter.reading("..."):`, which raises it for you. It is recorded
  with its traceback, and a transient status is retried first; and
- anything else stops the run, raised as it is with a note naming the adapter and the row. Use a native
  exception whose message says what to fix. Declare `keys`, and a missing key is refused before any call. If
  `read` raised it, the answer is kept, and resuming the run after the fix reads it with no call.

### Run

A run is one system's answers to one exact question, kept in a folder. The question is everything that changes
the answers: the system and its options, the dataset, the manifest, the task, and which rows (`sample`, `limit`,
`only`). The folder's name is a digest of it:

```
  runs/copy-1a2b3c4d/                 <system name>-<digest of the question>
    settings.json                     the question: system, options, dataset, manifest, task and rows; the
                                      dataset's commit; and the timeout
    records/a1.json                   everything about one row's call: output, raw answer, error, cost
    predictions/a1.json               just the output; this is what grading reads
    exact/                            one folder per scorer
      scores.jsonl                    one line per row
      summary.json                    the summary, with its headline
      details/a1.jsonl                each row's breakdown, if the scorer gives one
```

Everything in a run folder is plain JSON: dicts, lists, strings, finite numbers, booleans and None. So an
option, a raw answer, `read`'s output, a scorer's result and a summary must be plain JSON too, and the core
refuses anything else (a `TypeError` naming what it was), rather than writing its `str()`.

Every launch is a new run in a new folder, named after its question and a run id:
`<label>-<digest>-<YYYYMMDD-HHMMSS>-<4 hex>`. The digest is of the system, its options and the question, so
runs of one question sort together. `plan(..., resume=<a run's name>)` continues that run instead, skipping
every row that already has a record, so no answer is paid for twice; it refuses a run of another question
(`sample=20` and the full dataset are two questions, as are `sample=6` and `limit=6`). A new run is never mixed
with an old one: a hosted system redeployed since, or two launches at once, each get their own folder.

settings.json keeps the launch: `started` (UTC) and `launch`, whatever the caller records there (who launched
it, the commit of their code). The timeout isn't part of the question: resuming with a longer one (and
`retry_failed`) gives timed-out rows more time in the same folder.

The question names the dataset by its address (a folder, or `hf://...`). When one dataset is read from several
places, such as a Hugging Face address on one machine and a local copy on another, give the `Benchmark` a
`dataset_name`: the question then names the dataset by it, so a run is the same question, and can be resumed,
wherever the data is read from.

### Scorer

A scorer is one way of grading: a frozen dataclass whose fields are its options. You write:

- `gold(row)`, abstract: reads the row's gold and checks it;
- `score(pred, gold)`: compares one prediction with its gold and returns a `Score` (or write `score_all`); and
- `summarize(lines)`, abstract: the scored rows' lines as one summary.

A `Score` holds `result`, a small dict of the row's values, mostly numbers (`{"exact": True, "length": 5}`),
and optional `details`, a larger breakdown written to `details/<id>.jsonl`.

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

    def summarize(self, lines: list[dict]) -> dict:           # the scored rows' lines as one summary
        exact = sum(x["exact"] for x in lines) / len(lines)
        return {"headline": exact, "headline_is": "exact", "exact": exact}
```

Grading turns each row into one line of `scores.jsonl`, and all the lines into one summary:

```
  row a1:  prediction "hello", gold "hello"  --score-->  {"exact": true,  "length": 5}
  row a2:  prediction "w0rld", gold "world"  --score-->  {"exact": false, "length": 5}
  row c1:  the call failed                   ---------->  failed
                                                             |
                                                         summarize
                                                             v
           {"headline": 0.333, "headline_is": "scored.headline × coverage",
            "rows": 3, "failed": 1, "coverage": 0.667,
            "scored": {"headline": 0.5, "headline_is": "exact", "exact": 0.5}}
```

A line is one row's grade. The core writes `id`, `status` and `row` (from `row_fields`), then the row's
numbers, or a failed row's `error`. A result may not use those three keys.

The status comes from types, not from the scorer. A scorer returns a `Score`, or leaves a row unscored with
`errors.Unscored` (a judge that didn't answer; raised from `score`, or returned in the row's place from
`score_all`), which waits for a later grading. Use `Unscored` only when asking again may give a score (the judge
timed out or was down); for anything asking again won't change, it would keep the run from ever finishing. Only
the core fails a row: from its record, a missing prediction, or `prediction` raising `SystemFailed`.

`summarize` is the handoff. It gets only the scored rows' lines, without their status, and returns `headline`,
a number from 0 to 1 where higher is better, and `headline_is`, what it is; anything else is the scorer's. The
core keeps it under `scored` and adds `rows`, `failed`, `coverage`, and the top `headline`, `scored`'s times
coverage, so a failed row counts as 0. When no row was scored, `summarize` isn't called and the headline is 0.
A headline outside 0 to 1 is refused. `Scorer.grade`, which does all this, can't be overridden.

### How scoring runs

`grade` reads predictions and golds in batches of `batch_size` rows (500 by default) and hands each batch to
`score_all`, which by default calls `score` on each row, `workers` at a time, in a pool of processes or threads
(`pool`). Writing `score_all` yourself takes over completely:

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

- Your own execution, such as one request per batch to a judge, or a cluster. Write `score_all`: it gets a
  batch of values (never files) and returns, in order, a `Score` per row or an `errors.Unscored` for a row it
  couldn't score. `batch_size = None` gives it every row at once:

  ```python
  class BatchJudge(scorer.Scorer):
      batch_size: ClassVar[int | None] = 50

      def score_all(self, rows, preds, golds):
          verdicts = my_cluster.map(compare, preds, golds)  # however you like
          return [scorer.Score({"agree": v}) for v in verdicts]
  ```

Whichever you choose, reading files, the cache, failures and the summary stay in `grade`.

### Benchmark

A benchmark is a task, a dataset and its scorers: what stays fixed while systems change. A run is one system on
it; to compare systems, run each with the same choice of rows.

```python
b = benchmark.Benchmark(Copying(), "my-dataset", [Exact()])
b.run(Copy(), out="runs")
b.run(Copy(upper=True), out="runs")          # another configuration is another run, in its own folder
```

`run` does three steps, each reading what the one before wrote to disk, so you can also call them one at a time:

```
  b.run(system, out)
    |
    +-- plan(system, out)     checks everything before any call: the options, the API keys, the rows, their
    |                         input files and golds. Returns a Plan: the run, and the rows to predict.
    |
    +-- predict(plan)         calls the system on each row with no record, `workers` at a
    |                         time, and saves each record as it arrives. A table shows the progress.
    |
    +-- grade(run)            each scorer grades the run over its rows; returns the summaries and the
                              run's cost. It refuses a run that isn't finished.
```

Other things a benchmark does:

- `b.grade("mine/")`: grades a folder of predictions you made yourself, `mine/predictions/<id>.json`, over
  the whole manifest;
- `b.reread(run)`: reads every saved raw answer again with the adapter's `read`, after you fix it;
- `b.check()`: checks the dataset (manifest, input files, gold) without calling anything; and
- `b.describe(plan)`: says in words what a plan will do, before you pay for it.

## The rules

Three rules decide every case. They live in the parts you don't write, so a new adapter or scorer can't break
them:

1. a run folder answers one exact question: the system and its options, the dataset, manifest and task, and
   which rows. Every launch is a new run in its own folder; `resume` continues a run, of the same question only;
2. grading reports only a finished run: every row answered or failed, and scored. Otherwise it refuses and
   names the rows; `partial=True` (`--partial`) counts them failed, and the summary says how many; and
3. a row counts 0 only when code raises `errors.SystemFailed` on purpose. Any other exception stops the run and
   is raised as it is; every answer already paid for is kept, so resuming the run after the fix pays only for the rest.

What that means, case by case:

| what happens | what you see | the row counts | what to do |
| --- | --- | --- | --- |
| the system answers | the run goes on | its score | nothing |
| the system refuses, errors or answers something unusable (the adapter raises `SystemFailed`, or `reading()` does) | the row is recorded as failed, with the traceback of where it was raised, and the run goes on | 0 | nothing; its error is in `scores.jsonl` and `records/<id>.json` |
| the system times out, or keeps answering 429 or 5xx | the row is recorded as failed, and the run goes on; an answer that comes after the timeout counts as a timeout | 0 | optionally resume it with `retry_failed=True` |
| your key is missing or wrong, or your account is out of credits | the run stops and says why; a missing key stops it before any call | not yet | fix it, then resume the run |
| a model server you host isn't running | the run stops and says so | not yet | start it, then resume the run |
| an adapter's or a scorer's code raises anything else (a bug) | the run stops and the exception is raised as it is, with a note naming the class and the row; an answer `read` crashed on is kept, unread | not yet | fix it (or report it to the package's authors); resume the run after the fix, which reads kept answers again with no call |
| you stop the run (Ctrl-C, a closed laptop) | nothing is lost | not yet | resume the run |
| a judge doesn't answer for a row | grading says that row isn't scored | not yet | grade again (kept scores aren't paid for again) |
| you grade while rows are "not yet" | grading refuses and names the rows | — | finish them, or grade with `partial=True` to count them 0 |
| you change the system, options, dataset, manifest or rows | a new run folder is used | — | nothing; the old folder is untouched |
| you grade a folder of your own and a prediction is missing | grading refuses and names the rows | — | add them, or grade with `partial=True` to count them 0 |

Two systems run with the same `sample` answer the same rows, so their headlines can be compared.

## When you need more

Each is optional, and described in its class's docstring.

| you want | do this |
| --- | --- |
| a summary broken down by something the row holds (a language, a source) | return it from `Scorer.row_fields`, and group the lines by `line["row"][<field>]` in `summarize` |
| a slow scorer, an LLM as a judge, or scoring on your own machines | see "How scoring runs" above |
| to call a system with more or fewer rows at once | `workers=` on `run`; 0 means the adapter's own `workers` |
| to run several systems at once, sharing one rate limit | run each in its own thread or process; to share a limit, give each run an `executor` factory that returns one shared pool |
| to run each row somewhere else (processes, a cluster) | `executor=` on `run`, a function that makes a `concurrent.futures.Executor` |
| to try a scorer without writing results | `b.grade(run, write=False)` |
| a number for a run you can't finish | `b.grade(run, partial=True)`: unfinished rows count 0; the summary counts them and names a few, and scores.jsonl has each |

## Where to go next

- [`tests/test_core.py`](https://github.com/datalab-to/omni-bench-core/blob/main/tests/test_core.py): the whole toy task above, with tests of every rule. From a clone of the
  repository, run it with `PYTHONPATH=. python tests/test_core.py`; and
- the class docstrings in `omni_bench_core/`: what each method is for, and when you would write it.
