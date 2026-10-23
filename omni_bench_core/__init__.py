"""omni-bench-core: the shared core of the omni benchmarks.

A task package writes a `task.Task`, `adapter.Adapter`s and `scorer.Scorer`s. `benchmark.Benchmark` plans,
predicts and grades with them, and `run.Run` keeps one system's answers in a folder; `config` says one run as a YAML
file. OVERVIEW.md shows how the pieces fit. Import modules, not names.
"""

from omni_bench_core import adapter, benchmark, config, data, errors, run, scorer, task  # noqa: F401
