"""Adapters: how predictive systems are reached, and the rules every call follows.

`Adapter` is the class you subclass to add a system, and its docstring says how. The rest of the module:

- `Call` and `Cost`: what `Adapter.call` returns;
- `Budget` and `PollRetry`: helpers for a call that polls a job until it is done;
- `predict`: sends one input to one adapter under the shared rules, and returns the row's record;
- `record` and `reread`: build a record, and read a saved answer again;
- `configure`: makes an adapter from a config's options, checking them; and
- `retryable`: says whether a recorded failure is worth trying again in a later run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import hashlib
import json
import re
import time
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, NamedTuple

from omni_bench_core import errors

DEFAULT_TIMEOUT = 1800.0
# Phrases systems use for an account that can't pay, when the HTTP status doesn't say so.
ACCOUNT_MARKERS = ("exceeded the maximum number of credits", "subscription has expired", "insufficient_quota",
                   "insufficient credits")


class Cost(NamedTuple):
    """What the system said a call cost. It is kept so that a comparison can weigh accuracy against price.

    `usd` is None when the system doesn't say. `source` names the field of the response the figure was read
    from, so it can be checked. Credits are never converted to dollars, because the rate depends on a
    contract, not on the system."""

    usd: float | None = None
    credits: float | None = None
    source: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    @classmethod
    def total(cls, costs: Sequence[Cost]) -> Cost:
        """Several calls' costs added up, field by field. A field is None when no call reported it."""
        def add(f: str) -> Any:
            xs = [getattr(c, f) for c in costs if getattr(c, f) is not None]
            return round(sum(xs), 6) if xs else None
        sources = sorted({c.source for c in costs if c.source})
        return cls(add("usd"), add("credits"), " + ".join(sources) or None, add("tokens_in"), add("tokens_out"))

    @classmethod
    def reported(cls, value: Any, source: str, *, cents: bool = False, **extra: Any) -> Cost:
        """A cost the system stated, or an empty `Cost` when `value` is not a number.

        It is the one place that decides what counts as a figure. This matters because `True` is an int in
        Python, and a JSON `true` in a cost field is not a price. `cents=True` converts cents to dollars."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return cls(**extra)
        return cls(usd=round(float(value) / (100.0 if cents else 1.0), 6), source=source, **extra)


class Call(NamedTuple):
    """One row's answer as the system sent it, before any reading.

    It is kept whole in the row's record. So after a fix to `read`, the answer is read again instead of
    paid for again. It has four fields:

    - `raw`: the answer as received;
    - `cost`: what the system said it cost;
    - `job_id`: the system's own id for the call, for asking the system about one row; and
    - `sent`: what was sent besides the input (a schema as reshaped, a prompt), so the record says what
      the system was actually asked."""

    raw: Any
    cost: Cost = Cost()
    job_id: str | None = None
    sent: Any = None


@dataclasses.dataclass(frozen=True)
class Adapter[I]:
    """One predictive system. An instance is one configuration of it.

    The smallest adapter is a frozen dataclass with one method:

        @dataclasses.dataclass(frozen=True)
        class Tesseract(adapter.Adapter):
            lang: str = "eng"

            def call(self, page, *, timeout):
                return adapter.Call(raw=subprocess.run([...], capture_output=True, text=True).stdout)

    Its fields are the system's options, so an instance is one configuration, such as
    `Datalab(mode="accurate")`. Every field is written to the run's settings.json and is part of its folder's
    name (`folder_name`). So one folder holds one configuration, and a run always says what it asked.

    `call` is given the task's input (`Task.input`), which is what the system is sent. It is never given the
    row, because the row holds the gold.

    Everything else is optional and has a default:

        name      the name configs and run folders use; by default the class's name, in snake case
        read      turns a raw answer into the output; by default the output is the raw answer
        workers   how many rows are in flight at once for this adapter; 5
        retries   how many more attempts after a transient failure; 3, waiting `backoff` seconds between
        label     the run folder's name before the digest; `name`

    `account_failure(e)` decides whether a failure is the account's rather than the row's. By default it is
    HTTP 401 or 402, or a known phrase (`ACCOUNT_MARKERS`) in the system's error. An adapter overrides it
    when its system says so differently, or when one of those statuses can be about the row.

    `options()` and `folder_name` are what a run records and is named after. They are not overridden.

    `predict` holds the rules every call follows, so no adapter writes them again: one time budget per row,
    transient failures retried, a failure that is an answer recorded, and a failure that is the same for
    every row (`errors.Stop`) raised.

    An adapter is called for one row at a time. To use a model that runs many rows at once, serve it, and
    have its adapter call the server. The run and the server can share one machine.
    """

    name: ClassVar[str]
    workers: ClassVar[int] = 5
    retries: ClassVar[int] = 3
    backoff: ClassVar[tuple[float, ...]] = (20.0, 60.0, 120.0)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    def call(self, x: I, *, timeout: float) -> Call:
        """Asks the system about one input, and returns its answer as received.

        This is the one method every adapter writes, because how to reach a system is what differs between
        systems. On a failure, raise it (`errors.VendorError` or one of its kinds) rather than returning an
        empty answer. That way the row is recorded as failed, not as a system that answered nothing.
        `timeout` is the seconds left in the row's budget. A call that polls spends it with a `Budget`."""
        raise NotImplementedError(f"{type(self).__name__} has no call(x, *, timeout) -> Call")

    def read(self, raw: Any, x: I) -> Any:
        """The output in a raw answer. It is what predictions/<id>.json holds, and what every scorer reads.

        It is separate from `call` because a bug in reading is cheap to fix and a call is not: after a fix,
        `Benchmark.reread` reads every saved answer again, with no call. Raise when `raw` holds no output."""
        return raw

    @property
    def label(self) -> str:
        """The run folder's name before the digest. It is `name`, unless one adapter stands for many systems
        (one class for every model behind one API). Then the model should name the folder."""
        return self.name

    def account_failure(self, e: errors.VendorError) -> bool:
        """Whether failure `e` is the account's: a key the system refuses (401), an account that can't pay
        (402), or a known phrase saying so. Such a failure is the same for every row, so the run stops.

        The phrases are looked for in the system's error only, never in a successful answer's body, which
        can hold the document's own text."""
        body = (e.body or "") if (e.status or 0) >= 400 else ""
        return e.status in (401, 402) or any(m in f"{e} {body}".lower() for m in ACCOUNT_MARKERS)

    def options(self) -> dict[str, Any]:
        """Every option of the adapter. This is what its run's settings.json and records say it was asked."""
        errors.check_dataclass(type(self), Adapter)
        return dataclasses.asdict(self)

    @property
    def folder_name(self) -> str:
        """The name of its run's folder: `label`, then a digest of its name and every option.

        The digest covers every option, not only those that differ from the defaults. Defaults change, and
        then two configurations would share a folder and be averaged into one number."""
        spelled = json.dumps({"provider": self.name, **self.options()}, sort_keys=True, default=str)
        return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', self.label)}-{hashlib.sha256(spelled.encode()).hexdigest()[:8]}"


class Budget:
    """One row's deadline, shared by every call, poll and retry made for that row.

    Every system gets the same number of seconds per row, with uploading, polling and retrying all counted.
    A setting of ours must never decide that one system answers more rows than another. It uses a monotonic
    clock, because a wall clock can jump back."""

    __slots__ = ("total", "_deadline")

    def __init__(self, seconds: float):
        self.total = seconds
        self._deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def check(self, what: str) -> None:
        """Raises `VendorTimeout` if the time is up. `what` says what was still being waited for."""
        if self.expired():
            raise errors.VendorTimeout(f"timed out at the {self.total:.0f}s limit; {what}")


class PollRetry:
    """Keeps polling a job after a poll fails.

    A failed poll is not a failed job: the job is still running, and is being paid for. If the failure were
    raised out of `call`, the row would be retried from the upload, and the answer bought twice. A poll is
    tried again only after a transient status, or after no status at all (the request never got that far)."""

    __slots__ = ("budget", "limit", "count")

    def __init__(self, budget: Budget, limit: int = 60):
        self.budget, self.limit, self.count = budget, limit, 0

    def reset(self) -> None:
        """Called after a poll succeeds, so the count of failures in a row starts again."""
        self.count = 0

    def again(self, status: int | None = None) -> bool:
        """Called after a poll fails. Waits, within the budget, and returns whether to poll again."""
        if status is not None and status not in errors.TRANSIENT_STATUSES: return False
        self.count += 1
        if self.count > self.limit: return False
        time.sleep(min(30.0, 2.0 ** min(self.count, 5), self.budget.remaining()))
        return True


@contextlib.contextmanager
def reading(what: str) -> Iterator[None]:
    """Marks code that reads a system's answer. A `ValueError`, `KeyError`, `IndexError`, `TypeError` or
    `AttributeError` raised inside (a JSON array where an object was expected) is the system's fault, not a bug: it becomes a `VendorError` saying what couldn't be read, so the
    row fails and the run goes on.

        with adapter.reading("the job's status"):
            status = resp.json()["status"]

    It exists so the line between a system's failure and our bug is exact. Inside `reading`, a malformed answer
    fails one row; anywhere else in `call`, an exception is a bug in the adapter, and stops the run. Reading
    the output itself belongs in `read`, where any failure is already the row's (`errors.ReadError`)."""
    try:
        yield
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise errors.VendorError(f"the system's answer couldn't be read ({what}): {type(exc).__name__}: {exc}"[:400]
                                 ) from None


def configure[A: Adapter](cls: type[A], options: Mapping[str, Any] | None = None) -> A:
    """An instance of `cls` with `options` from a config, checked.

    It checks three things:

    - an unknown option is refused by name, because a typo that went through would run the defaults and
      report them as what was asked;
    - a whole number given for a float option becomes a float, so `0` and `0.0` name the same run; and
    - a value of the wrong simple type is refused."""
    options = dict(options or {})
    errors.check_dataclass(cls, Adapter)
    fields = [f.name for f in dataclasses.fields(cls)]
    if unknown := sorted(set(options) - set(fields)):
        near = difflib.get_close_matches(unknown[0], fields, n=1)
        raise errors.UsageError(f"{cls.name} has no option {unknown[0]!r}"
                                + (f" (did you mean {near[0]!r}?)" if near else "")
                                + f"; its options are: {', '.join(fields) or '(none)'}")
    hints = typing.get_type_hints(cls)
    for k, v in options.items():
        try:
            json.dumps(v, allow_nan=False, default=str)
        except ValueError:
            raise errors.UsageError(f"{cls.name}: option {k} is {v!r}; give a finite number") from None
        t = hints.get(k)
        if t not in (str, int, float, bool): continue
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            options[k] = float(v)
        elif not isinstance(v, t) or (t is not bool and isinstance(v, bool)):
            raise errors.UsageError(f"{cls.name}: option {k} is {v!r}, a {type(v).__name__}; it takes a {t.__name__}")
    return cls(**options)


def predict[I](a: Adapter[I], x: I, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Sends one input to `a`, and returns the row's record: its answer and the evidence for it.

    Failures are handled in three ways:

    - a transient failure (a 429, a 5xx) is retried, up to `a.retries` times, within the row's budget;
    - any other failure is the row's answer. It is recorded as `error`, not raised, because it is a fact
      about this row; and
    - a `Stop`, or an account failure, is raised. It is the same for every row, and if it were recorded it
      would look like a settled answer that no resumed run tries again.

    The record holds `provider`, `settings`, `output` (None on a failure), `raw`, `sent`, `error` (its type,
    status, whether it was transient, and its message), `cost` (with `wall_s` and `attempts`), `job_id`,
    `timeout_s` and `captured_at`.
    """
    budget = Budget(timeout)
    started = time.time()
    got, error, attempts = None, None, 0
    spent: list[Cost] = []   # what each failed attempt cost, so retries that were paid for are counted
    for attempt in range(a.retries + 1):
        attempts = attempt + 1
        try:
            got = a.call(x, timeout=budget.remaining())
            if not isinstance(got, Call):
                raise TypeError(f"{type(a).__name__}.call must return an adapter.Call, not a {type(got).__name__}")
            error = None
            break
        except errors.VendorError as exc:
            if a.account_failure(exc):
                raise errors.AccountFailure(f"{a.name}: {str(exc)[:200]}") from None
            if isinstance(exc.cost, Cost): spent.append(exc.cost)
            error = exc
            if not exc.transient or attempt == a.retries: break
            time.sleep(min(a.backoff[min(attempt, len(a.backoff) - 1)] if a.backoff else 0.0, budget.remaining()))
            if budget.expired():
                error = errors.VendorTimeout(f"the {timeout:.0f}s budget was spent over {attempts} attempts; "
                                             f"last failure: {exc}"[:400])
                break
    rec = record(a, got, error, attempts=attempts, timeout=timeout, wall_s=round(time.time() - started, 1),
                 at=time.strftime("%Y-%m-%dT%H:%M:%S"), spent=spent)
    return rec if error is not None else reread(a, x, rec)


def record(a: Adapter, got: Call | None, error: errors.VendorError | None, *, attempts: int, timeout: float,
           wall_s: float, at: str, spent: Sequence[Cost] = ()) -> dict[str, Any]:
    """A row's record, built from what the calls returned: either the answer as received, or the failure.

    Its cost is every attempt's cost added up: the failed attempts that were paid for (`spent`), and the
    answer. Its `output` is filled in afterwards by `reread`, which `predict` calls when there is an answer."""
    costs = [*spent, *([got.cost] if got is not None else [])]
    rec = {"provider": a.name, "settings": a.options(), "output": None, "raw": None, "sent": None, "error": None,
           "cost": {**Cost.total(costs)._asdict(), "wall_s": wall_s, "attempts": attempts}, "job_id": None,
           "timeout_s": timeout, "captured_at": at}
    if got is not None:
        rec |= {"raw": got.raw, "sent": got.sent, "job_id": got.job_id}
    if error is not None:
        rec |= {"raw": error.body, "sent": error.sent, "job_id": error.job_id, "error": _failure(error)}
    return rec


def reread[I](a: Adapter[I], x: I, rec: dict[str, Any]) -> dict[str, Any]:
    """`rec` with its output read again from its raw answer by `a.read`.

    The result is either an answer, or a `ReadError` that keeps the raw answer for a later fix. A record
    whose call failed is returned unchanged, because it has no answer to read."""
    if rec["error"] is not None and rec["error"]["type"] != "ReadError":
        return rec
    try:
        return rec | {"output": a.read(rec["raw"], x), "error": None}
    except errors.Stop:
        raise
    except Exception as exc:  # noqa: BLE001 -- any reader failure is this row's, and its raw answer is kept
        return rec | {"output": None, "error": _failure(errors.ReadError(f"{type(exc).__name__}: {exc}"[:400]))}


def _failure(e: errors.VendorError) -> dict[str, Any]:
    """A failure, as a record states it."""
    return {"type": type(e).__name__, "status": e.status, "transient": e.transient, "message": str(e)[:600]}


def retryable(error: Mapping[str, Any] | None) -> bool:
    """Whether a recorded failure might succeed in a later run with `retry_failed`.

    A timeout, an outage or a transient status might. A failure that is an answer (a 400, a refusal, an
    answer that couldn't be read) won't."""
    if not error: return False
    return (error.get("type") in ("VendorTimeout", "VendorUnreachable")
            or error.get("status") in errors.TRANSIENT_STATUSES)
