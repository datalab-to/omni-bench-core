"""Tasks: what goes into a system and what comes out. `Task` is the class to subclass."""

from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import Any, ClassVar

from omni_bench_core import data


@dataclasses.dataclass(frozen=True)
class Task[R: data.Row, I](abc.ABC):
    """What goes into a system, and what comes out. A task is a frozen dataclass with two abstract methods:

        @dataclasses.dataclass(frozen=True)
        class Extract(task.Task):
            def rows(self, dataset, manifest=data.MANIFEST) -> tuple[Doc, ...]: ...   # the dataset's rows
            def input(self, row: Doc) -> Question: ...                               # what a system is sent

    A new task is needed only when the input or the output changes; a new way to grade is a scorer, and a new
    system is an adapter. `name` defaults to the class's name in lower case, and is the task in a run's question.
    A task has no options (no fields): what varies between runs of one task is the dataset, the manifest, the
    scorers or the system, each of which a run already names.
    """

    name: ClassVar[str] = "task"

    @abc.abstractmethod
    def rows(self, dataset: str | Path, manifest: str = data.MANIFEST) -> tuple[R, ...]:
        """Every row of `dataset`'s `manifest`, as the task's own row type. `data.read_manifest` does the checks
        every task shares."""

    @abc.abstractmethod
    def input(self, row: R) -> I:
        """What a system is sent for `row`: the only part of the row an adapter sees, so the gold never reaches a
        system. Raise `FileNotFoundError` when the row's input file is missing."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__.lower()
