"""The errors a user can meet, kept apart from our own bugs.

There are four kinds:

- `UsageError`: what was asked is wrong, such as a bad option, dataset, manifest or file. Fix it and ask
  again;
- `Stop`: the run can't go on, and the reason is the same for every row, such as an account that can't
  pay, a missing key or a missing dependency. Fix it and run again;
- `VendorError`: one call to a system failed. It is recorded as that row's answer, so a resumed run knows
  the row was tried. It is retried first when it is `transient`; and
- `Unscored`: a scorer couldn't score one row, for a reason that isn't the system's fault. The row waits
  for a later grading.

Anything else that is raised is a bug, either ours or an adapter's.

This module also has two helpers: `counted` writes a count in a message, and `check_dataclass` refuses an
adapter, scorer or task that forgot its dataclass decorator.
"""

from __future__ import annotations

from typing import Any

# Note: we decide whether a retry can help from the HTTP status, not from the message text. Two spellings
# of the same condition once drifted apart, and 91 predictions were never retried while 18 identical ones
# were.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524})


class UsageError(ValueError):
    """What was asked is wrong. The message says what was wrong, the value given, and how to fix it."""


class Stop(Exception):
    """The run can't go on, for a reason that is the same for every row.

    It is raised, never recorded: recorded, it would look like an answer for one row, and a resumed run
    would never try that row again."""


class Unscored(Exception):
    """A scorer couldn't score a row, for a reason that isn't the system's fault, such as a judge that
    didn't answer.

    The row counts as neither scored nor failed. The headline stays empty until a later grading scores it."""


class AccountFailure(Stop):
    """The account can't pay.

    It stops the run instead of failing each row. One vendor once marked 174 documents as failed in 60
    seconds after hitting a credit limit, which turned a pause we could recover from into 174 zeros."""


class MissingCredential(Stop):
    """A system's API key is not set in the environment."""


class MissingDependency(ImportError, Stop):
    """An adapter could not import a package it needs."""


class VendorError(RuntimeError):
    """A call to a system was made and gave no answer.

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


class VendorTimeout(VendorError):
    """The row's time ran out while the system was still working.

    It is not retried within the same call, because a retry would spend a second budget on a row we know
    takes longer than one. A later run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class VendorUnreachable(VendorError):
    """The system couldn't be reached, or its answer never came back.

    It is not retried within the same call. A failed poll can leave the job running at the system, and
    asking again from the start would pay for it twice. A later run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class ReadError(VendorError):
    """The system answered, but `Adapter.read` couldn't read the answer.

    The raw answer is kept in the record. After the reader is fixed, the answer is read again without a
    new call."""

    @property
    def transient(self) -> bool:
        return False


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
