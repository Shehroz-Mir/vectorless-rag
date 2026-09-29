"""UserLocks that record who holds a lock (see operations/ports.py), so tests can check what ran under it."""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager


class RecordingLocks:
    def __init__(self) -> None:
        self.held: set[str] = set()
        self.acquired: list[str] = []

    @contextmanager
    def for_user(self, user_key: str) -> Generator[None]:
        if user_key in self.held:
            raise AssertionError(f"lock for {user_key} taken twice")  # the real lock would deadlock
        self.acquired.append(user_key)
        self.held.add(user_key)
        try:
            yield
        finally:
            self.held.discard(user_key)
