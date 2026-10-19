"""Tasks: what goes into a predictive system and what comes out.

`Task` is the class a task subclasses, and its docstring says how to write one.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from omni_bench_core import data, errors

if TYPE_CHECKING:
    from omni_bench_core import adapter, scorer


@dataclasses.dataclass(frozen=True)
class Task[R: data.Row, I]:
    """What goes into a predictive system, and what comes out. A task is a frozen dataclass with two methods:

        @dataclasses.dataclass(frozen=True)
        class Extract(task.Task):
            def rows(self, dataset, manifest=data.MANIFEST) -> tuple[Doc, ...]: ...   # the dataset's rows
            def input(self, row: Doc) -> Question: ...                               # what a system is sent

    You need a new task only when what goes in or what comes out changes. For example, parsing a page into
    HTML is one task, and extracting a JSON document from a PDF and a schema is another. A new way to grade
    the same output is a new scorer, and a new system is a new adapter. Neither needs a new task.

    `name` is the task's name. By default it is the class's name, in lower case. A task's fields are its
    options, as for an adapter (`options()`). A run records the task's name and options, and refuses a
    different task, so answers to `Segment("base")` and `Segment("hard")` never share a folder.
    """

    name: ClassVar[str] = "task"

    def rows(self, dataset: str | Path, manifest: str = data.MANIFEST) -> tuple[R, ...]:
        """Every row of `dataset`'s `manifest`, as the task's own row type.

        Each task writes its own, because each task's rows hold different things (a page, or a PDF and its
        schema), so each has its own columns and checks. `data.read_manifest` does the checks every task
        shares."""
        raise NotImplementedError(f"{type(self).__name__} has no rows(dataset, manifest)")

    def input(self, row: R) -> I:
        """What a system is sent for `row`. It is the only part of the row an adapter sees.

        It is separate from the row because the row also holds the gold, and anything else the dataset
        carries for scorers (tags, a language). None of that may reach a system. Raise `FileNotFoundError`
        when the row's input file is missing: the dataset is incomplete, so the row is skipped, not failed."""
        raise NotImplementedError(f"{type(self).__name__} has no input(row)")

    def check(self, systems: Sequence[adapter.Adapter], scorers: Sequence[scorer.Scorer]) -> None:
        """Refuses, before any call, a system these scorers can't fairly grade. By default it refuses nothing.

        A task overrides it when its systems differ in what they can produce. For example, a parsing system
        that can't produce layout blocks is refused when a scorer grades blocks, because every one of those
        tests would fail and say nothing about how well it parses. `Benchmark.plan` calls it."""

    def options(self) -> dict[str, Any]:
        """Every option of the task, as a run's settings.json records them."""
        errors.check_dataclass(type(self), Task)
        return dataclasses.asdict(self)

    def identity(self) -> dict[str, Any]:
        """What a run records about its task, and checks every later use of its folder against: the task's
        name and options. It is given as JSON reads it back (a tuple becomes a list), so it compares equal to
        the one a run's settings.json holds."""
        return json.loads(json.dumps({"name": self.name, "options": self.options()}, default=str))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__.lower()
