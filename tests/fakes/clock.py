"""A deterministic clock: each call is one second after the last, so ordering tests never tie."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta


class TickingClock:
    def __init__(self, start: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> None:
        self._next = start

    def __call__(self) -> datetime:
        current, self._next = self._next, self._next + timedelta(seconds=1)
        return current
