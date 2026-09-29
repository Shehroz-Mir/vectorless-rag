"""In-memory FileStore (see operations/ports.py). Paths are synthetic; nothing touches disk."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID


class InMemoryFileStore:
    def __init__(self, root: Path = Path("mem")) -> None:
        self.root = root
        self.files: dict[Path, bytes] = {}

    def save_original(self, user_key: str, document_id: UUID, filename: str, data: bytes) -> Path:
        path = self._folder(user_key, document_id) / "original" / filename
        self.files[path] = data
        return path

    def enriched_path(self, user_key: str, document_id: UUID, filename: str) -> Path:
        return self._folder(user_key, document_id) / "enriched" / filename

    def delete_document_files(self, user_key: str, document_id: UUID) -> None:
        folder = self._folder(user_key, document_id)
        self.files = {path: data for path, data in self.files.items() if not path.is_relative_to(folder)}

    def _folder(self, user_key: str, document_id: UUID) -> Path:
        return self.root / "users" / user_key / "documents" / str(document_id)
