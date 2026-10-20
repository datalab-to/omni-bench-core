"""The errors a user can meet, and the one rule that decides what each does:

    A row counts 0 only when code says, on purpose, that the system failed, by raising `SystemFailed`.
    Any other exception stops the run and prints its traceback.

So:

- `SystemFailed` (and its kinds `SystemTimeout`, `SystemUnreachable`, `Unreadable`): the system failed on
  one row. The row's record keeps it, with its traceback, and the row counts 0; the run goes on. A transient
  one (a 429, a 5xx) is retried first;
- `Unscored`: a scorer couldn't score one row for a reason that isn't the system's (a judge that didn't
  answer). Grading isn't finished until a later grading scores it;
- `UsageError`: what was asked is wrong (an option, a dataset, a file). `Unfinished` is one: a run graded
  before every row has an answer and a score;
- `Stop`: the run can't go on for any row (an account, a key, a server, a missing dependency); and
- `Bug`: any other exception, which is a bug in the code that raised it. The message says whose code it is,
  where to report it, and has the whole traceback.

A stop of any kind keeps every answer that was paid for, so running the same command again after the fix
costs nothing for the rows already done.

This module also has helpers: `counted` writes a count in a message, `capped` cuts a message written to a
file, `check_dataclass` refuses an adapter, scorer or task that forgot its dataclass decorator, and `package`
and `bug` say whose code a class is.
"""

from __future__ import annotations

import functools
from typing import Any

# Note: we decide whether a retry can help from the HTTP status, not from the message text. Two spellings
# of the same condition once drifted apart, and 91 predictions were never retried while 18 identical ones
# were.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
CAP = 4000   # the most characters of a message or traceback written to a file


class UsageError(ValueError):
    """What was asked is wrong. The message says what was wrong, the value given, and how to fix it."""


class Unfinished(UsageError):
    """A run graded before it is finished: some of its rows have no answer yet, or no score yet.

    Grading refuses it, because a headline over part of the rows can't be told apart from one over all of
    them. The message lists the rows, and the two ways on: finish them, or grade with `partial`, which counts
    them as failed."""


class Stop(Exception):
    """The run can't go on, for a reason that is the same for every row.

    It is raised, never recorded: recorded, it would look like an answer for one row, and a resumed run
    would never try that row again."""


class Unscored(Exception):
    """A scorer couldn't score a row, for a reason that isn't the system's fault, such as a judge that
    didn't answer.

    The row counts as neither scored nor failed, so grading isn't finished until a later grading scores it."""


class AccountFailure(Stop):
    """The account can't pay.

    It stops the run instead of failing each row. One vendor once marked 174 documents as failed in 60
    seconds after hitting a credit limit, which turned a pause we could recover from into 174 zeros."""


class MissingCredential(Stop):
    """A system's API key is not set in the environment."""


class MissingDependency(ImportError, Stop):
    """An adapter could not import a package it needs."""


class Unread(Exception):
    """`Adapter.read` raised something other than `SystemFailed`: a bug in `read`. It carries the row's record,
    with the system's raw answer and no output, so the answer is saved before the run stops; running the same
    command again reads it again with the fixed `read`, with no call."""

    def __init__(self, record: dict[str, Any]):
        super().__init__(record)   # Note: the record is its one argument, so it survives being sent between processes
        self.record = record

    def __str__(self) -> str:
        return "read raised; the answer is kept unread"


class Bug(Stop):
    """An adapter or a scorer raised something that is neither an answer nor a failure we know of: a bug in
    its code, not a failure of the system. The message says whose code it is and where its traceback was
    saved (`bug`)."""


class SystemFailed(RuntimeError):
    """The system failed on this row: raise it, on purpose, to record the row's failure, which counts 0.

    It is the only exception that counts a row as 0. Raise it in `Adapter.call` for a refusal, an error
    status, or an answer you can see is unusable; in `Adapter.read` for an answer that can't be read; and in
    `Scorer.prediction` for an output a scorer can't use. `adapter.reading` raises it for you. The row's
    record keeps its message and traceback, so it is always clear which line decided the system failed.

    It carries what the row's record should keep about the failed call:

    - `status`: the HTTP status, when there was one. It alone decides whether asking again is worth it;
    - `body`: what the system sent back;
    - `sent`: what was sent besides the input, as in `Call.sent`, so the record says what was asked; and
    - `cost`: what the failed call cost, an `adapter.Cost`, when the system charged for it. It is added to
      the run's cost, so failures that were paid for are counted; and
    - `job_id`: the system's id for the job, when it made one, so a job that failed or outlived its call
      can be found and asked about."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None, sent: Any = None,
                 cost: Any = None, job_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.sent = sent
        self.cost = cost
        self.job_id = job_id

    @property
    def transient(self) -> bool:
        """Whether asking again may get an answer: a rate limit or a server fault is, a refusal is not."""
        return self.status in TRANSIENT_STATUSES


class SystemTimeout(SystemFailed):
    """The row's time ran out while the system was still working.

    It is not retried within the same call, because a retry would spend a second budget on a row we know
    takes longer than one. A later run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class SystemUnreachable(SystemFailed):
    """The system couldn't be reached, or its answer never came back.

    It is not retried within the same call. A failed poll can leave the job running at the system, and
    asking again from the start would pay for it twice. A later run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class Unreadable(SystemFailed):
    """The system answered, but its answer can't be read: `Adapter.read` raised `SystemFailed`. The raw answer
    is kept in the record, so a fix to `read` reads it again without a new call (`Benchmark.reread`)."""

    @property
    def transient(self) -> bool:
        return False


def capped(text: str) -> str:
    """`text`, cut to `CAP` characters when it is written to a file (a record, a line of scores.jsonl). It is the
    one place a message is cut: a vendor's error can quote a whole web page, which the record keeps in `raw`
    anyway, and a message is never cut where it is raised."""
    return text if len(text) <= CAP else text[:CAP] + f"... ({len(text) - CAP} more characters)"


def counted(n: int, noun: str) -> str:
    """`1 row` or `2 rows`."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def check_dataclass(cls: type, base: type) -> None:
    """Refuses a class that declares fields but is not a dataclass.

    Adapters, scorers and tasks are frozen dataclasses because their fields are their options. An adapter's
    options are written to its run and are part of its run folder's name. A subclass that forgets
    `@dataclasses.dataclass(frozen=True)` still runs, but its fields are not options. They are not written
    down, and changing one changes the answers without changing the run's name."""
    import dataclasses
    import inspect

    fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    for c in cls.__mro__:
        if c is base or not issubclass(c, base): continue
        try:
            notes = inspect.get_annotations(c)
        except Exception:  # noqa: BLE001 -- an annotation that doesn't resolve is checked by the dataclass itself
            continue
        loose = [k for k, v in notes.items() if "ClassVar" not in str(v) and k not in fields]
        if loose:
            raise TypeError(f"{c.__name__} declares {', '.join(loose)} but is not a dataclass, so "
                            f"{'they are' if len(loose) > 1 else 'it is'} not an option: decorate it with "
                            "@dataclasses.dataclass(frozen=True)")


@functools.cache
def package(cls: type) -> tuple[str, str, str | None] | None:
    """The installed package `cls` comes from: its name, its version, and its GitHub page when it names one.
    None for a class in no installed package, such as one defined in a script or a notebook."""
    import importlib.metadata as md

    for dist in md.packages_distributions().get(cls.__module__.split(".")[0], []):
        meta = md.metadata(dist)
        urls = [u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL") or []] + [meta.get("Home-page") or ""]
        return meta["Name"], meta["Version"], next((u.rstrip("/") for u in urls if "github.com/" in u), None)
    return None


def refused(exc: BaseException) -> bool:
    """Whether `exc`, or an exception it was raised from, is a refused connection: nothing listens at the
    address. That is the same for every row, so it stops the run, rather than failing each row or passing for
    a bug in the adapter."""
    import errno

    # Note: by type, never by message: a vendor's own error text can say "connection refused" about its servers.
    while exc is not None:
        if isinstance(exc, ConnectionRefusedError) or (isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def bug(what: str, exc: BaseException, cls: type) -> Bug:
    """The `Bug` to raise when `cls`'s code raised `exc` while doing `what` ("predicting doc_17"). Its message
    says whose code it is (an installed package's, with where to report it, or your own class's, with its
    file), the versions a report needs, and the whole traceback."""
    import importlib.metadata
    import inspect
    import platform
    import traceback

    head = f"{cls.__name__} raised {type(exc).__name__} while {what}: {exc}"
    if (p := package(cls)) is not None:
        name, version, home = p
        where = f"report it at {home}/issues" if home else f"report it to {name}'s authors"
        body = f"this is a bug in {name} {version}, not a failure of the system; {where}, with this message"
    else:
        try:
            file = inspect.getfile(cls)
        except TypeError:
            file = cls.__module__
        body = f"this is a bug in your {cls.__name__} ({file}), not a failure of the system"
    versions = (f"omni-bench-core {importlib.metadata.version('omni-bench-core')}, "
                f"Python {platform.python_version()}")
    tb = "".join(traceback.format_exception(exc)).rstrip()
    return Bug(f"{head}\n{body}. Run the same command again after a fix.\n({versions})\n{tb}")
