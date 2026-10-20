"""Progress: how predicting shows while it runs. `Board` is the one class."""

from __future__ import annotations

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)


class Board:
    """One run being predicted: rows done, failed and still to go, and their cost so far.

    On a terminal it is a table redrawn as each answer is saved, with a line above it for each failed row.
    Elsewhere (a log file, CI) it writes one log line per answer."""

    def __init__(self, run: str, todo: int, live: bool | None = None):
        self.run, self.todo = run, todo
        self.done, self.failed, self.usd = 0, 0, None
        self.live = sys.stderr.isatty() if live is None else live
        self._view: Any = None
        self._streams: list[tuple[logging.StreamHandler, Any]] = []

    def __enter__(self) -> Board:
        # Note: no table when there is nothing to predict, so a finished run prints none.
        if self.live and self.todo: self._show()
        return self

    def _show(self) -> None:
        from rich.console import Console
        from rich.live import Live

        was = sys.stderr
        self._view = Live(self._table(), console=Console(stderr=True), refresh_per_second=4, transient=False)
        self._view.__enter__()
        # Note: log handlers keep the stderr they were made with, so their lines would tear the table; while it
        # is drawn they write to the stderr the table put in place, which prints them above it.
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and h.stream in (sys.__stderr__, was):
                self._streams.append((h, h.stream))
                h.setStream(sys.stderr)

    def __exit__(self, *exc: Any) -> None:
        if self._view is not None:
            for h, stream in self._streams: h.setStream(stream)
            self._view.update(self._table())
            self._view.__exit__(*exc)

    def saved(self, id: str, rec: dict[str, Any]) -> None:
        """Counts one row's saved record."""
        self.done += 1
        if rec["error"] is not None: self.failed += 1
        if (usd := (rec.get("cost") or {}).get("usd")) is not None: self.usd = (self.usd or 0.0) + usd
        if self._view is not None: self._view.update(self._table())
        if rec["error"] is not None:
            e = rec["error"]
            log.info("%s: %s failed: %s: %s", self.run, id, e["type"], str(e["message"])[:160])
        elif not self.live:
            log.info("%s: %d/%d %s", self.run, self.done, self.todo, id)

    def _table(self) -> Any:
        from rich.table import Table

        t = Table(title="predicting", title_justify="left", box=None, pad_edge=False)
        for col in ("run", "done", "failed", "to go", "cost"):
            t.add_column(col, justify="left" if col == "run" else "right")
        t.add_row(self.run, f"{self.done}/{self.todo}", str(self.failed), str(self.todo - self.done),
                  "—" if self.usd is None else f"${self.usd:.2f}")
        return t
