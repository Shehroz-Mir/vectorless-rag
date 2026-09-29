"""Shared plumbing for the SQL repositories: sessions, transactions, time."""
from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; everything leaving db/ is timezone-aware UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class SqlRepository(ABC):
    """An ABC under our rule: every SQL repository shares this session and clock handling."""

    def __init__(self, sessions: sessionmaker[Session], clock: Clock = utc_now) -> None:
        self._sessions = sessions
        self._clock = clock

    def _transaction(self) -> AbstractContextManager[Session]:
        """One short transaction per call: commits on success, rolls back on any error."""
        return self._sessions.begin()

    def _now(self) -> datetime:
        return self._clock()
