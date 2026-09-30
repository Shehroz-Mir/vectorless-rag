"""Original and enriched PDFs on disk (implements ports.FileStore).

Layout: {data_root}/users/{user_key}/documents/{document_id}/original/{name}
                                                            /enriched/{name}
The enriched copy keeps the original's name: PageIndex shows the file's base name to the agent.
"""
from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path, PureWindowsPath
from uuid import UUID

from vectorless_rag.operations import checked_user_key

MAX_NAME_LENGTH = 150
DEFAULT_NAME = "document.pdf"
_UNSAFE_CHARS = re.compile(r"[^\w.\- ()]+")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(filename: str) -> str:
    """A plain, portable file name ending in .pdf (PageIndex local mode only accepts .pdf names)."""
    name = PureWindowsPath(unicodedata.normalize("NFKC", filename)).name  # drops any folders, / or \
    name = _UNSAFE_CHARS.sub("_", name).strip()
    stem, suffix = (name[:-4], name[-4:]) if name.lower().endswith(".pdf") else (name, ".pdf")
    stem = stem.strip(" .")[: MAX_NAME_LENGTH - len(suffix)]
    if not stem:
        return DEFAULT_NAME
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    return stem + suffix.lower()


class LocalFileStore:
    def __init__(self, data_root: Path) -> None:
        self._root = data_root.resolve()

    def save_original(self, user_key: str, document_id: UUID, filename: str, data: bytes) -> Path:
        path = self._document_dir(user_key, document_id) / "original" / safe_filename(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        os.replace(partial, path)  # never leaves a half-written original behind
        return path

    def enriched_path(self, user_key: str, document_id: UUID, filename: str) -> Path:
        return self._document_dir(user_key, document_id) / "enriched" / safe_filename(filename)

    def delete_document_files(self, user_key: str, document_id: UUID) -> None:
        folder = self._document_dir(user_key, document_id)
        if folder.exists():
            shutil.rmtree(folder)

    def _document_dir(self, user_key: str, document_id: UUID) -> Path:
        return self._root / "users" / checked_user_key(user_key) / "documents" / str(document_id)
