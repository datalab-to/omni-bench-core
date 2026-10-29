"""The exceptions the core acts on, and helpers for messages.

The one rule: a row fails only when code raises `SystemFailed` on purpose. Any other exception stops the run
and is raised as it is. So:

- `SystemFailed` (and its kinds `SystemTimeout`, `SystemUnreachable` and `Unreadable`): the system failed on one
  row, which is recorded and fails;
- `Unscored`: a scorer couldn't score one row (a judge that didn't answer), which waits for a later grading; and
- anything else is a native exception (`ValueError` for wrong input, `PermissionError` for an account that
  can't pay, and so on), raised as it is, with a note from `noted` saying which class and which row.

Helpers: `capped`, `counted`, `check_dataclass`, `plain`, `package`, `refused` and `noted`.
"""

from __future__ import annotations

import functools
import json
from typing import Any

# Note: whether a retry can help is decided from the HTTP status, never from the message, whose wording drifts.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
CAP = 4000   # the most characters of a message or traceback written to a file


class Unscored(Exception):
    """A scorer couldn't score a row this time, and asking again may give a score: the judge timed out, was
    rate-limited, or was down. `scorer.ask` raises it once the judge's call has used up its retries.

    Raise it from `Scorer.score`, or return it in the row's place from `Scorer.score_all`. The row counts as
    neither scored nor failed, so grading refuses to report until a later grading scores it. Don't use it for
    what asking again won't change, or the run never finishes:

    - a prediction the scorer can't use: raise `SystemFailed` in `Scorer.prediction`, and the row fails;
    - a gold the scorer can't read: let it raise, and grading stops and names the row; and
    - a bug: let it raise."""


class SystemFailed(RuntimeError):
    """The system failed on this row. It is the only exception that counts a row as 0.

    Raise it in `Adapter.call` (a refusal, an error status), in `Adapter.read` (an answer that can't be read)
    or in `Scorer.prediction` (an output the scorer can't use); `adapter.reading` raises it for you. The row's
    record keeps its message and traceback. It carries what the record keeps about the call:

    - `status`: the HTTP status, if any. It alone decides whether a retry may help (`transient`);
    - `body`: what the system sent back;
    - `sent`: what was sent besides the input, as in `Call.sent`;
    - `cost`: what the failed call cost, an `adapter.Cost`, added to the run's cost; and
    - `job_id`: the system's id for the job, if it made one."""

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
        """Whether a retry may help: a rate limit or a server fault, not a refusal."""
        return self.status in TRANSIENT_STATUSES


class SystemTimeout(SystemFailed):
    """The row's time ran out. Not retried in the same run, since a retry would need a second budget; a later
    run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class SystemUnreachable(SystemFailed):
    """The system couldn't be reached, or its answer never came back. Not retried in the same run, since the job
    may still be running and a new call would pay for it twice; a later run with `retry_failed` tries it again."""

    @property
    def transient(self) -> bool:
        return False


class Unreadable(SystemFailed):
    """The system answered, but `Adapter.read` raised `SystemFailed` on its answer. The raw answer is kept, so a
    fix to `read` reads it again with no call (`Benchmark.reread`)."""

    @property
    def transient(self) -> bool:
        return False


def capped(text: str) -> str:
    """`text` cut to `CAP` characters, for writing to a file (a record, a line of scores.jsonl). Messages are cut
    only here, never where they are raised."""
    return text if len(text) <= CAP else text[:CAP] + f"... ({len(text) - CAP} more characters)"


def counted(n: int, noun: str) -> str:
    """`1 row` or `2 rows`."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def check_dataclass(cls: type, base: type) -> None:
    """Refuses a subclass of `base` that declares fields but isn't a dataclass. Its fields would not be options,
    so changing one would change the answers without changing the run's folder."""
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
    """The installed package `cls` comes from: its name, version and GitHub page (if it names one). None for a
    class in no installed package, such as one in a script."""
    import importlib.metadata as md

    for dist in md.packages_distributions().get(cls.__module__.split(".")[0], []):
        meta = md.metadata(dist)
        urls = [u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL") or []] + [meta.get("Home-page") or ""]
        return meta["Name"], meta["Version"], next((u.rstrip("/") for u in urls if "github.com/" in u), None)
    return None


def refused(exc: BaseException) -> bool:
    """Whether `exc`, or an exception in its chain, is a refused connection: nothing listens at the address."""
    import errno

    # Note: by type, never by message: a vendor's own error text can say "connection refused" about its servers.
    while exc is not None:
        if isinstance(exc, ConnectionRefusedError) or (isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def noted[E: BaseException](exc: E, cls: type, what: str) -> E:
    """`exc` with a note saying which class raised it and while doing `what` ("predicting doc_17")."""
    exc.add_note(f"{cls.__name__} raised this while {what}")
    return exc


def plain(x: Any, what: str) -> None:
    """Refuses `x` unless it is plain JSON, naming `what` it is.

    Everything the core writes or digests is plain JSON: dicts, lists, strings, finite numbers, booleans and None.
    Anything else would be written as its `str()` and read back as a string, and an option's `str()` (an object's
    address) would name a new run folder every time."""
    try:
        json.dumps(x, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} isn't plain JSON: {exc}. Give dicts, lists, strings, finite numbers, booleans and "
                        "None (None for a number that has no value)") from None
