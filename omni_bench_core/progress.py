"""Progress: how predicting shows while it runs. `Board` is the one class."""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

log = logging.getLogger(__name__)


class Board:
    """Every run being predicted, one line each: rows done, failed and still to go, and what they cost so far.

    On a terminal it is a table drawn live, redrawn as each answer arrives, with a line above it for each row
    that failed. Anywhere else (a log file, CI, a pipe) a table can't be redrawn, so it writes one log line per
    answer instead.

    Runs of different systems are predicted at once, from different threads, so every update takes a lock."""

    def __init__(self, live: bool | None = None):
        self.live = sys.stderr.isatty() if live is None else live
        self.lock = threading.Lock()
        self.runs: dict[str, dict[str, Any]] = {}
        self._view: Any = None
        self._streams: list[tuple[logging.StreamHandler, Any]] = []

    def __enter__(self) -> Board:
        if self.live:
            from rich.console import Console
            from rich.live import Live

            was = sys.stderr
            self._view = Live(self._table(), console=Console(stderr=True), refresh_per_second=4, transient=False)
            self._view.__enter__()
            # Note: log handlers keep the stderr they were made with, so their lines would tear the table; while
            # it is drawn they write to the stderr the table put in place, which prints them above it.
            for h in logging.getLogger().handlers:
                if isinstance(h, logging.StreamHandler) and h.stream in (sys.__stderr__, was):
                    self._streams.append((h, h.stream))
                    h.setStream(sys.stderr)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._view is not None:
            for h, stream in self._streams: h.setStream(stream)
            self._view.update(self._table())
            self._view.__exit__(*exc)

    def start(self, run: str, rows: int, todo: int) -> None:
        """Adds a run: how many rows it answers, and how many of them still need a call."""
        with self.lock:
            self.runs[run] = {"rows": rows, "todo": todo, "done": 0, "failed": 0, "usd": None}
            self._redraw()

    def saved(self, run: str, id: str, rec: dict[str, Any]) -> None:
        """Counts one row's saved record."""
        with self.lock:
            r = self.runs[run]
            r["done"] += 1
            if rec["error"] is not None: r["failed"] += 1
            if (usd := (rec.get("cost") or {}).get("usd")) is not None: r["usd"] = (r["usd"] or 0.0) + usd
            self._redraw()
            done, todo = r["done"], r["todo"]
        if rec["error"] is not None:
            e = rec["error"]
            log.info("%s: %s failed: %s: %s", run, id, e["type"], str(e["message"])[:160])
        elif not self.live:
            log.info("%s: %d/%d %s", run, done, todo, id)

    def _redraw(self) -> None:
        if self._view is not None: self._view.update(self._table())

    def _table(self) -> Any:
        from rich.table import Table

        t = Table(title="predicting", title_justify="left", box=None, pad_edge=False)
        for col in ("run", "done", "failed", "to go", "cost"):
            t.add_column(col, justify="left" if col == "run" else "right")
        for name, r in self.runs.items():
            t.add_row(name, f"{r['done']}/{r['todo']}", str(r["failed"]), str(r["todo"] - r["done"]),
                      "—" if r["usd"] is None else f"${r['usd']:.2f}")
        return t
