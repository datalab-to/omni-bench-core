# omni-bench-core

The shared core of the omni benchmarks. Measuring a predictive system on documents is always the same
loop: send each row of a dataset to the system, save its answer, and grade the answers against the gold.
A task package (omni-parse-bench, omni-extract-bench, or your own) writes the parts that differ, and this
package runs the loop.

```python
from omni_bench_core import benchmark

b = benchmark.Benchmark(SomeTask(), "hf://datasets/<org>/<name>", [SomeScorer()])
b.run([SomeAdapter(mode="accurate")], out="runs", sample=20)   # plan, predict, grade
b.grade("mine/")                                                # a folder of your own predictions
```

What you write, from least to most:

| to | write |
| --- | --- |
| run a benchmark | `Benchmark(task, dataset, scorers).run([SomeAdapter(...)], out)` |
| add a system | an `Adapter`: a frozen dataclass of its options, and `call(input) -> Call` |
| add a way of grading | a `Scorer`: a frozen dataclass, `gold(row)` and `score(pred, gold) -> Score` |
| add a task | a `Task`: `rows(dataset)` and `input(row)`, a row type, and its adapters and scorers |

Where to read next:

- `OVERVIEW.md`: the pieces one at a time, with diagrams. Start here if you are new;
- `tests/test_core.py`: a whole toy task in about 100 lines (`PYTHONPATH=. python tests/test_core.py`); and
- the class docstrings in `omni_bench_core/`: what each method is for, and when you would write it.

Two limits are deliberate:

- a dataset is a folder or an `hf://` address, never rows made in code, because a run folder records which
  dataset its answers are for and refuses another; and
- an adapter calls one row at a time. A model that batches is served, and its adapter calls the server.
