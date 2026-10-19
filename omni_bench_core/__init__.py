"""omni-bench-core: the shared core of the omni benchmarks. A task package writes its `Task`, its `Adapter`s and its
`Scorer`s; this package runs them: `Benchmark` plans, predicts and grades, and `Run` keeps one system's answers
in a folder. OVERVIEW.md says how the pieces fit.

Import modules, not names: `adapter.Adapter`, `scorer.Score`, `benchmark.Benchmark`.
"""

from omni_bench_core import adapter, benchmark, data, errors, run, scorer, task  # noqa: F401
