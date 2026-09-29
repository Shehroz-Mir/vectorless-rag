"""Per-user write locks (spec 5.4; implements ports.UserLocks).

In-process only. If the service runs as several processes, this must become a file lock or a
database advisory lock (spec 15, still open).
"""
from __future__ import annotations

import threading


class PerUserLocks:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def for_user(self, user_key: str) -> threading.Lock:
        """The same lock every time for one user; created on first use and kept (one per user)."""
        with self._guard:
            return self._locks.setdefault(user_key, threading.Lock())
