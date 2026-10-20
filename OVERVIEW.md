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
     |  .run(adapter, out)                  one system per run
     v
  +--------------------------------------------------------------+
  |  a Run: a folder under `out`                                 |
  |                                                              |
  |    for each row:  Task.input(row) -> Adapter -> answer       |
  |                   saved in the Run                           |
  |                                                              |
  |    for each scorer:  Scorer.grade(run) -> summary            |
  +--------------------------------------------------------------+
```

Each piece below has a small example from [`tests/test_core.py`](https://github.com/datalab-to/omni-bench-core/blob/main/tests/test_core.py). That file holds a whole toy task: the
"system" is asked to copy a text file's text, and it is graded on whether the copy is exact.

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

A row is one line of the manifest, as a frozen dataclass. It holds paths, never file contents, so a large
dataset fits in memory. It must have an `id`; everything else is the task's.

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

A dataset can have more than one manifest, such as `manifests/sample40.parquet`, a subset of 40 rows that a
dataset's authors chose.

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
  and what the system said it cost (`cost`). On a failure it raises a `SystemFailed`; and
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
      +-- SystemFailed (a 400 ...) -> recorded as the row's error, with its traceback; the run goes on
      +-- past the timeout        -> recorded as a timeout, even if an answer came later
      +-- 401, 402 (the account)  -> PermissionError: the whole run stops, since every row would fail
      +-- connection refused      -> raised as it is: the whole run stops, nothing listens at the address
      +-- any other exception     -> raised as it is, with a note naming the adapter and the row
```

What your `call` and `read` raise decides which of these happens. One rule covers them: a row counts 0 only
when you say the system failed.

- `errors.SystemFailed(message, status=...)`, or one of its kinds (`SystemTimeout`, `SystemUnreachable`): the
  system failed on this row. It is recorded with its traceback and counts 0, and a transient status is
  retried first. Raise it in `read` for an answer you can't use (an LLM's broken JSON), or wrap the reading
  in `with adapter.reading("..."):`, which raises it for you;
- anything else stops the run, raised as it is, with its traceback and a note naming the adapter and the
  row. Use a native exception with a message that says what to fix: a `ValueError` for a wrong option, say.
  Declaring `keys` makes the core refuse a missing key (a `ValueError`) before any call, and a 401 or 402
  becomes a `PermissionError`, so you rarely raise one for the account yourself. A bug in your adapter
  stops the run the same way; if `read` raised it, the answer is kept, and is read again with no call when
  you run the same command after the fix.

### Run

A run is one system's answers to one exact question, kept in a folder. You never write one, but you will
read its files, so it helps to know what is in it.

The question is everything that changes the answers: the system and its options, the dataset, the manifest,
the task, and which rows (`sample`, `limit`, `only`). The folder's name is a digest of all of it:

```
  runs/copy-1a2b3c4d/                 <system name>-<digest of the question>
    settings.json                     the question: system, options, dataset, manifest, task and rows; and
                                      the timeout
    records/a1.json                   everything about one row's call: output, raw answer, error, cost
    predictions/a1.json               just the output; this is what grading reads
    exact/                            one folder per scorer
      scores.jsonl                    one line per row
      summary.json                    the summary, with its headline
      details/a1.jsonl                each row's breakdown, if the scorer gives one
```

So asking another question uses another folder, and nothing gets mixed: `sample=20` and the full dataset
are two runs, as are two configurations of one system. The rows are named by how they were chosen, so
`sample=6` and `limit=6` are two runs even on a dataset where both pick the same rows. The timeout isn't
part of the question: running again with a longer one (and `retry_failed`) gives the timed-out rows more time
in the same folder, and each record keeps the timeout it had. Asking the same question again uses the same folder
and resumes it: before calling the system for a row, the run checks whether that row already has a record,
and skips it if so. No answer is paid for twice.

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

A line is one row's grade, and scores.jsonl holds every row's. The core writes what the row is (`id`,
`status`, and `row` when the scorer has `row_fields`), then the numbers `score` returned. A failed row's
line holds its `error` instead.

The handoff is `summarize`. Every scorer writes it, because only the scorer knows how its rows combine (a
mean, or one rate over every row's counts). It is given only the scored rows' lines, without their status,
and returns `headline`, a number from 0 to 1 where higher is better, and `headline_is`, what the headline
is; anything else in it is the scorer's own. The core keeps it under `scored`, and adds what it knows:
`rows`, `failed`, `coverage`, and the top `headline`, `scored`'s headline times coverage, so a failed row
counts as 0. When no row was scored, `summarize` isn't called, `scored` is null and the headline is 0. The
core refuses a headline outside 0 to 1, because counting a failure as 0 only makes sense on that scale.

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

A benchmark is a task, a dataset and its scorers: everything that stays fixed while the systems change. A
run is one system on it; to compare systems, run each, with the same choice of rows.

```python
b = benchmark.Benchmark(Copying(), "my-dataset", [Exact()])
b.run(Copy(), out="runs")
b.run(Copy(upper=True), out="runs")          # another configuration is another run, in its own folder
```

`run` does three steps in order. Each step reads what the one before it wrote to disk, so you can also call
them one at a time:

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

Three rules decide what happens in every case. They live in the parts you don't write, so a new adapter or
scorer can't break them.

1. A run folder answers one exact question: the system and its options, the dataset, the manifest, the task
   and which rows. Another question gets another folder; the same question resumes the same one.
2. Grading reports only a finished run: every row it asked about has an answer or a recorded failure, and a
   score. Otherwise grading refuses, names the rows, and says how to finish them. `partial=True` (`--partial`
   on a command line) grades anyway and counts those rows as failed; the summary then says so, with how many
   and a few of their ids.
3. A row counts 0 only when code says, on purpose, that the system failed, by raising `errors.SystemFailed`.
   Any other exception stops the run and is raised as it is. A stop keeps every answer that was paid for,
   so running the same command again after the fix costs nothing for the rows already done.

What that means, case by case:

| what happens | what you see | the row counts | what to do |
| --- | --- | --- | --- |
| the system answers | the run goes on | its score | nothing |
| the system refuses, errors or answers something unusable (the adapter raises `SystemFailed`, or `reading()` does) | the row is recorded as failed, with the traceback of where it was raised, and the run goes on | 0 | nothing; its error is in `scores.jsonl` and `records/<id>.json` |
| the system times out, or keeps answering 429 or 5xx | the row is recorded as failed, and the run goes on; an answer that comes after the timeout counts as a timeout | 0 | optionally run again with `retry_failed=True` (`--retry-failed`) |
| your key is missing or wrong, or your account is out of credits | the run stops and says why; a missing key stops it before any call | not yet | fix it, then run the same command again |
| a model server you host isn't running | the run stops and says so | not yet | start it, then run the same command again |
| an adapter's or a scorer's code raises anything else (a bug) | the run stops and the exception is raised as it is, with a note naming the class and the row; an answer `read` crashed on is kept, unread | not yet | fix it (or report it to the package's authors); run the same command again after the fix, which reads kept answers again with no call |
| you stop the run (Ctrl-C, a closed laptop) | nothing is lost | not yet | run the same command again |
| a judge doesn't answer for a row | grading says that row isn't scored | not yet | grade again (kept scores aren't paid for again) |
| you grade while rows are "not yet" | grading refuses and names the rows | — | finish them, or grade with `partial=True` to count them 0 |
| you change the system, options, dataset, manifest or rows | a new run folder is used | — | nothing; the old folder is untouched |
| you grade a folder of your own and a prediction is missing | grading refuses and names the rows | — | add them, or grade with `partial=True` to count them 0 |

Two systems run with the same `sample` answer the same rows, so their headlines can be compared.

## When you need more

Each of these is optional, has a default, and is described in its class's docstring.

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
