"""Adapters: how systems are reached, and the rules every call follows.

- `Adapter`: the class to subclass for a system, and `Served`, for one whose answers come from a server we run;
- `Call` and `Cost`: what `Adapter.call` returns;
- `Budget`, `PollRetry` and `reading`: helpers for writing `call` and `read`;
- `predict`, `record` and `reread`: one row's call under the rules, and its record;
- `configure`: an adapter or a scorer from a config's options, checked;
- `check_served`: a served system's server, checked against what its `served` says; and
- `retryable`: whether a recorded failure may succeed in a later run.
"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import difflib
import os
import re
import time
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, NamedTuple, final

from omni_bench_core import errors

DEFAULT_TIMEOUT = 1800.0
SERVER_TIMEOUT = 60.0  # seconds a server has to say what it serves
# Phrases systems use for an account that can't pay, when the HTTP status doesn't say so.
ACCOUNT_MARKERS = ("exceeded the maximum number of credits", "subscription has expired", "insufficient_quota",
                   "insufficient credits")


class Cost(NamedTuple):
    """What the system said a call cost. `usd` is None when it doesn't say, and `source` names the response field
    the figure came from. Credits are never converted to dollars: the rate is a contract's, not the system's."""

    usd: float | None = None
    credits: float | None = None
    source: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    @classmethod
    def total(cls, costs: Sequence[Cost]) -> Cost:
        """Several calls' costs added up, field by field; None where no call reported one."""
        def add(f: str) -> Any:
            xs = [getattr(c, f) for c in costs if getattr(c, f) is not None]
            return round(sum(xs), 6) if xs else None
        sources = sorted({c.source for c in costs if c.source})
        return cls(add("usd"), add("credits"), " + ".join(sources) or None, add("tokens_in"), add("tokens_out"))

    @classmethod
    def reported(cls, value: Any, source: str, *, cents: bool = False, **extra: Any) -> Cost:
        """The cost the system stated, or an empty `Cost` when `value` isn't a number (a JSON `true` isn't a
        price, though `True` is an int). `cents=True` converts cents to dollars."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return cls(**extra)
        return cls(usd=round(float(value) / (100.0 if cents else 1.0), 6), source=source, **extra)


class Call(NamedTuple):
    """One row's answer as the system sent it, kept whole in the row's record so a fix to `read` costs no call:

    - `raw`: the answer as received;
    - `cost`: what the system said it cost;
    - `job_id`: the system's id for the call, if any; and
    - `sent`: what was sent besides the input (a reshaped schema, a prompt)."""

    raw: Any
    cost: Cost = Cost()
    job_id: str | None = None
    sent: Any = None


@dataclasses.dataclass(frozen=True)
class Adapter[I](abc.ABC):
    """One system. An instance is one configuration of it, such as `Datalab(mode="accurate")`.

    An adapter is a frozen dataclass with one abstract method:

        @dataclasses.dataclass(frozen=True)
        class Tesseract(adapter.Adapter):
            lang: str = "eng"

            def call(self, page, *, timeout):
                return adapter.Call(raw=subprocess.run([...], capture_output=True, text=True).stdout)

    Its fields are its options: they are part of the run's question, which its settings.json keeps and a resume
    must match, so one folder holds one configuration. `call` gets the task's input, never the row (which holds the
    gold), and is called from `workers` threads at once, so anything it shares must be thread-safe. A system whose
    answers come from a server we run subclasses `Served` instead, which adds what the server is to its options.

    Optional, with defaults:

        name             the name configs and run folders use; the class's name in snake case
        read             the output in a raw answer; the raw answer itself
        keys             the environment variables it needs, checked before any call; none
        located          the options that say where the system is, not what it is (an address): recorded with the
                         run but not part of its question (`asked`); none
        workers          rows in flight at once; 5
        retries          further attempts after a transient failure, `backoff` seconds apart; 3
        label            the run folder's name before its run id; `name`
        account_failure  whether a failure is the account's (stops the run); 401, 402 or `ACCOUNT_MARKERS`
        missing_keys     the `keys` not set; override when an option can stand in for a key

    `predict` applies the rules to every call, so no adapter writes them: one time budget per row, transient
    failures retried, a `SystemFailed` recorded, and an account failure or refused connection raised. To run
    many rows at once on one model, serve the model and have the adapter call the server.
    """

    name: ClassVar[str]
    keys: ClassVar[tuple[str, ...]] = ()
    located: ClassVar[tuple[str, ...]] = ()
    workers: ClassVar[int] = 5
    retries: ClassVar[int] = 3
    backoff: ClassVar[tuple[float, ...]] = (20.0, 60.0, 120.0)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    @abc.abstractmethod
    def call(self, x: I, *, timeout: float) -> Call:
        """Asks the system about one input, and returns its answer as received. Raise `errors.SystemFailed` on a
        failure, never return an empty answer. `timeout` is the seconds left in the row's budget; a call that
        polls spends it with a `Budget`."""

    def read(self, raw: Any, x: I) -> Any:
        """The output in a raw answer: what predictions/<id>.json holds and every scorer reads. Raise
        `errors.SystemFailed` for an unusable answer (the row fails). Anything else is a bug: the run stops,
        the answer is kept, and resuming the run after the fix reads it again with no call."""
        return raw

    @property
    def label(self) -> str:
        """The run folder's name before its run id: `name`, or the model when one class stands for many models."""
        return self.name

    def account_failure(self, e: errors.SystemFailed) -> bool:
        """Whether failure `e` is the account's: a refused key (401), an account that can't pay (402), or a known
        phrase in the system's error (never in a successful body, which can hold the document's own text)."""
        body = (e.body or "") if (e.status or 0) >= 400 else ""
        return e.status in (401, 402) or any(m in f"{e} {body}".lower() for m in ACCOUNT_MARKERS)

    def missing_keys(self) -> list[str]:
        """The variables of `keys` that aren't set."""
        return [k for k in self.keys if not os.environ.get(k)]

    @final
    def options(self) -> dict[str, Any]:
        """The adapter's options, as its records state them."""
        errors.check_dataclass(type(self), Adapter)
        out = dataclasses.asdict(self)
        errors.plain(out, f"{type(self).__name__}'s options")
        return out

    @final
    def asked(self) -> dict[str, Any]:
        """The options that are part of a run's question: all but those that say where the system is (`located`)."""
        return {k: v for k, v in self.options().items() if k not in self.located}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Served[I](Adapter[I]):
    """A system whose answers come from a server we run (a model served by vLLM, or a pipeline's server), not a
    vendor's. How the model is served decides its answers as much as how it is asked, so it is an option:

        served   what the server is: its weights, their revision, its software and its flags, such as
                 {weights: org/model, revision: <commit>, vllm: 0.20.1, args: [--max-model-len 131072]}; its keys
                 are the server's own

    and part of the run's question, so two servings of a model are two runs, and a run is never resumed against
    another one. Where the server is, `base_url`, is not part of it (`located`): the same model served beside a run or
    at an endpoint asks the same question. `server` asks the server what it serves; before any call, a run refuses a server that says
    something other than `served` about any key it reports, and a key of `served` the server doesn't report
    unless the class names it in `stated` (`check_served`). A stated key (vLLM's flags, which vLLM can't report)
    is true when the server was started from `served`, and is the config's word otherwise."""

    stated: ClassVar[tuple[str, ...]] = ()
    located: ClassVar[tuple[str, ...]] = ("base_url",)
    served: dict[str, Any]
    base_url: str

    @abc.abstractmethod
    def server(self, *, timeout: float) -> dict[str, Any]:
        """What the server this adapter calls says it serves, such as {weights, revision, vllm}: the keys it can
        report, each as `served` states it. Compared with `served` before any call."""

    def details(self, *, timeout: float) -> dict[str, Any]:
        """Anything else the server says about itself, such as vLLM's whole engine config: recorded with the run, in its
        server.json, and never compared. Nothing, unless the adapter says how to ask."""
        return {}


def check_served(a: Adapter, *, timeout: float = SERVER_TIMEOUT) -> dict[str, Any] | None:
    """Refuses a served system whose server says something other than its `served` about any key the server
    reports, before any call. Returns what the server said, `{"server": ..., "details": ...}`, for the run to record;
    None for a system that isn't `Served`."""
    if not isinstance(a, Served): return None
    try:
        got = a.server(timeout=timeout)
    except Exception as exc:
        if errors.refused(exc) and not getattr(exc, "__notes__", None):
            exc.add_note(f"{a.label} couldn't ask its server what it serves: the connection was refused. Start the "
                         "server (or give its base_url), then run again")
        raise
    if not isinstance(got, dict):
        raise TypeError(f"{type(a).__name__}.server must return a dict, not a {type(got).__name__}")
    errors.plain(got, f"what {a.label}'s server says it serves")
    if wrong := [k for k, v in got.items() if k not in a.served or a.served[k] != v]:
        raise ValueError(f"{a.label}'s server serves something other than its served says: "
                         + "; ".join(f"{k} is {got[k]!r} there and {a.served.get(k, '(not given)')!r} in served"
                                     for k in wrong)
                         + ". Serve what served says, or say in served what the server serves")
    if unknown := sorted(set(a.served) - set(got) - set(a.stated)):
        raise ValueError(f"{a.label}'s served says {', '.join(unknown)}, which its server doesn't report and "
                         f"{type(a).__name__} doesn't take as stated; its server reports {', '.join(got) or 'nothing'}"
                         + (f", and it takes {', '.join(a.stated)} as stated" if a.stated else ""))
    details = a.details(timeout=timeout)
    if not isinstance(details, dict):
        raise TypeError(f"{type(a).__name__}.details must return a dict, not a {type(details).__name__}")
    errors.plain(details, f"what {a.label}'s server says about itself")
    return {"server": got, "details": details}


class Budget:
    """One row's deadline, shared by every call, poll and retry for it, so every system gets the same time."""

    __slots__ = ("total", "_deadline")

    def __init__(self, seconds: float):
        self.total = seconds
        self._deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def check(self, what: str) -> None:
        """Raises `SystemTimeout` if the time is up. `what` says what was still being waited for."""
        if self.expired():
            raise errors.SystemTimeout(f"timed out at the {self.total:g} s limit; {what}")


class PollRetry:
    """Keeps polling a job after a poll fails: the job may still be running, and a retry from the upload would
    pay for it twice. A poll is tried again after a transient status, or no status at all, up to `limit` times."""

    __slots__ = ("budget", "limit", "count")

    def __init__(self, budget: Budget, limit: int = 60):
        self.budget, self.limit, self.count = budget, limit, 0

    def reset(self) -> None:
        """Call after a poll succeeds."""
        self.count = 0

    def again(self, status: int | None = None) -> bool:
        """Call after a poll fails: waits, within the budget, and returns whether to poll again."""
        if status is not None and status not in errors.TRANSIENT_STATUSES: return False
        self.count += 1
        if self.count > self.limit: return False
        time.sleep(min(30.0, 2.0 ** min(self.count, 5), self.budget.remaining()))
        return True


@contextlib.contextmanager
def reading(what: str) -> Iterator[None]:
    """Turns a `ValueError`, `KeyError`, `IndexError`, `TypeError` or `AttributeError` raised inside the block into a
    `SystemFailed` saying what couldn't be read, so a malformed answer fails its row instead of stopping the run:

        with adapter.reading("the job's status"):
            status = resp.json()["status"]
    """
    try:
        yield
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise errors.SystemFailed(f"the system's answer couldn't be read ({what}): {type(exc).__name__}: "
                                  f"{exc}") from exc


def configure[A](cls: type[A], options: Mapping[str, Any] | None = None) -> A:
    """An instance of `cls`, an adapter or a scorer, with `options` from a config. An unknown option or a value of
    the wrong simple type is refused, and a whole number for a float option becomes a float, so `0` and `0.0`
    name the same run."""
    from . import scorer  # deferred: scorer imports this module

    options = dict(options or {})
    if not issubclass(cls, (Adapter, scorer.Scorer)):
        raise TypeError(f"configure makes an adapter or a scorer, not a {cls.__name__}")
    errors.check_dataclass(cls, Adapter if issubclass(cls, Adapter) else scorer.Scorer)
    fields = [f.name for f in dataclasses.fields(cls)]
    if unknown := sorted(set(options) - set(fields)):
        near = difflib.get_close_matches(unknown[0], fields, n=1)
        raise ValueError(f"{cls.name} has no option {unknown[0]!r}"
                         + (f" (did you mean {near[0]!r}?)" if near else "")
                         + f"; its options are: {', '.join(fields) or '(none)'}")
    if needed := [f.name for f in dataclasses.fields(cls) if f.name not in options and f.init
                  and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]:
        raise ValueError(f"{cls.name} needs its {', '.join(needed)}, an option with no default; give it")
    hints = typing.get_type_hints(cls)
    for k, v in options.items():
        errors.plain(v, f"{cls.name}'s option {k} ({v!r})")
        t = hints.get(k)
        # Note: a config is JSON or YAML, which have lists and no tuples, and a nested option has no class to rebuild.
        if typing.get_origin(t) is tuple and isinstance(v, list):
            options[k] = tuple(v)
        if dataclasses.is_dataclass(t):
            raise TypeError(f"{cls.name}: option {k} is a {t.__name__}, a dataclass; an option is a plain value "
                            "(str, int, float, bool, a list or a dict), so a config can say it")
        if t not in (str, int, float, bool): continue
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            options[k] = float(v)
        elif not isinstance(v, t) or (t is not bool and isinstance(v, bool)):
            raise ValueError(f"{cls.name}: option {k} is {v!r}, a {type(v).__name__}; it takes a {t.__name__}")
    return cls(**options)


def check_keys(a: Adapter) -> None:
    """Refuses `a` when an env var of its `keys` isn't set, before any call."""
    if missing := a.missing_keys():
        raise ValueError(f"{a.label} needs {', '.join(missing)}: " + " and ".join(f"export {k}=..." for k in missing)
                         + ", then run again")


def predict[I](a: Adapter[I], x: I, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Sends one input to `a`, and returns the row's record. Exceptions follow the one rule:

    - `SystemFailed` is recorded as the row's `error` and fails the row; a transient one (429, 5xx) is retried first,
      within the row's budget, and an answer after the budget is a `SystemTimeout`;
    - an account failure is raised as a `PermissionError`, and a refused connection as it is, with a note; and
    - anything else is raised as it is. When `read` raised it, it carries the record as `unread`, so the paid
      answer can be saved.

    A key of `a.keys` that isn't set is refused first (`check_keys`), as `Benchmark.plan` does.

    The record holds `system`, `options`, `package` (the adapter's package and version), `output`, `raw`, `sent`,
    `error` (type, status, transient, message and traceback), `cost` (with `wall_s` and `attempts`), `job_id`,
    `timeout_s` and `captured_at`.
    """
    check_keys(a)
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
        except Exception as exc:
            if errors.refused(exc):
                # Note: an adapter that knows its server says how to start it; ours is only the fallback.
                if not getattr(exc, "__notes__", None):
                    exc.add_note(f"{a.label} couldn't connect: the connection was refused. If it calls a server you "
                                 "run, start it (or pass its base_url), then resume the run")
                raise
            if not isinstance(exc, errors.SystemFailed): raise
            if a.account_failure(exc):
                raise PermissionError(f"{a.label}: {exc}; fix the account (its key, its credits), then resume the "
                                      "run") from exc
            if isinstance(exc.cost, Cost): spent.append(exc.cost)
            error = exc
            if not exc.transient or attempt == a.retries: break
            time.sleep(min(a.backoff[min(attempt, len(a.backoff) - 1)] if a.backoff else 0.0, budget.remaining()))
            if budget.expired():
                error = errors.SystemTimeout(f"the {timeout:g} s budget was spent over {attempts} attempts; "
                                             f"last failure: {exc}")
                break
    # Note: an adapter is asked to keep to its budget, but nothing makes it; an answer that came after the budget
    # is a timeout like any other, so every system gets the same time. The answer is kept: it was paid for.
    if error is None and (wall := time.time() - started) > timeout:
        error = errors.SystemTimeout(f"answered after {wall:.1f} s, past the {timeout:g} s budget")
    rec = record(a, got, error, attempts=attempts, timeout=timeout, wall_s=round(time.time() - started, 1),
                 at=time.strftime("%Y-%m-%dT%H:%M:%S"), spent=spent)
    return rec if error is not None else reread(a, x, rec)


def record(a: Adapter, got: Call | None, error: errors.SystemFailed | None, *, attempts: int, timeout: float,
           wall_s: float, at: str, spent: Sequence[Cost] = ()) -> dict[str, Any]:
    """A row's record from what the calls returned, the answer or the failure, with every paid attempt's cost.
    `reread` fills in its `output`."""
    costs = [*spent, *([got.cost] if got is not None else [])]
    p = errors.package(type(a))
    rec = {"system": a.name, "options": a.options(), "package": f"{p[0]} {p[1]}" if p else None,
           "output": None, "raw": None, "sent": None, "error": None,
           "cost": {**Cost.total(costs)._asdict(), "wall_s": wall_s, "attempts": attempts}, "job_id": None,
           "timeout_s": timeout, "captured_at": at}
    if got is not None:
        errors.plain(got.raw, f"{type(a).__name__}.call's raw answer")
        errors.plain(got.sent, f"{type(a).__name__}.call's sent")
        rec |= {"raw": got.raw, "sent": got.sent, "job_id": got.job_id}
    if error is not None:
        rec |= {"error": _failure(error)}
        if got is None:
            errors.plain(error.sent, f"{type(a).__name__}'s SystemFailed sent")
            rec |= {"raw": error.body, "sent": error.sent, "job_id": error.job_id}
    return rec


def reread[I](a: Adapter[I], x: I, rec: dict[str, Any]) -> dict[str, Any]:
    """`rec` with its output read again from its raw answer by `a.read`; unchanged when its call failed. `read`
    raising `SystemFailed` records the answer as `Unreadable`. Anything else is raised as it is, carrying as
    `unread` the record with no output, so the answer can be saved."""
    if rec["error"] is not None and rec["error"]["type"] != "Unreadable":
        return rec
    rec = {k: v for k, v in rec.items() if k != "unread"}
    try:
        output = a.read(rec["raw"], x)
        errors.plain(output, f"{type(a).__name__}.read's output")
        return rec | {"output": output, "error": None}
    except errors.SystemFailed as exc:
        e = errors.Unreadable(f"{type(exc).__name__}: {exc}" if type(exc) is not errors.SystemFailed else str(exc))
        e.__cause__ = exc
        return rec | {"output": None, "error": _failure(e)}
    except Exception as exc:
        exc.unread = rec | {"output": None, "error": None, "unread": True}
        raise


def _failure(e: errors.SystemFailed) -> dict[str, Any]:
    """A failure as a record states it, with the traceback of the line that raised it."""
    import traceback

    return {"type": type(e).__name__, "status": e.status, "transient": e.transient, "message": errors.capped(str(e)),
            "traceback": errors.capped("".join(traceback.format_exception(e)))}


def retryable(error: Mapping[str, Any] | None) -> bool:
    """Whether a recorded failure may succeed in a later run with `retry_failed`: a timeout, an outage or a
    transient status may; a 400, a refusal or an unreadable answer won't."""
    if not error: return False
    return (error.get("type") in ("SystemTimeout", "SystemUnreachable")
            or error.get("status") in errors.TRANSIENT_STATUSES)
