"""One PageIndex client per user (implements ports.UserIndexProvider; spec 5.5).

Each user's library lives in its own storage_path, {data_root}/users/{user_key}/pageindex, so one
user's tools, context and citations never reach another user's documents (Spike C). Clients are
created on first use and kept. One client may serve the API's reads while the worker indexes
(Spike C: 108 reads during an index write, no errors); writes are serialised by UserLocks.
"""
from __future__ import annotations

import threading
from pathlib import Path

from pageindex import PageIndexClient

from vectorless_rag.indexing.user_index import PageIndexUserIndex
from vectorless_rag.operations import checked_user_key


class PageIndexClientPool:
    def __init__(self, data_root: Path, *, api_key: str, model: str, summary_concurrency: int) -> None:
        self._users_root = data_root.resolve() / "users"
        self._api_key = api_key
        self._model = model
        self._summary_concurrency = summary_concurrency
        self._indexes: dict[str, PageIndexUserIndex] = {}
        self._guard = threading.Lock()

    def for_user(self, user_key: str) -> PageIndexUserIndex:
        with self._guard:
            index = self._indexes.get(user_key)
            if index is None:
                index = self._indexes[user_key] = PageIndexUserIndex(self._new_client(user_key))
            return index

    def storage_path(self, user_key: str) -> Path:
        return self._users_root / checked_user_key(user_key) / "pageindex"

    def _new_client(self, user_key: str) -> PageIndexClient:
        # No chat side: LangChain's model answers (spec 5.5).
        return PageIndexClient(index={
            "model": self._model,
            "storage_path": str(self.storage_path(user_key)),
            "summary_concurrency": self._summary_concurrency,
            "backend": {"api_key": self._api_key},  # LiteLLM connection params for the indexing calls
        })
