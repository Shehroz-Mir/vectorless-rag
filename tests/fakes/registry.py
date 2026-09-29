"""In-memory DocumentRepository and FigureRepository (see operations/ports.py)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from vectorless_rag.models.documents import Document, DocumentChanges, DocumentStatus, NewDocument
from vectorless_rag.models.figures import FigureDescription, NewFigureDescription
from vectorless_rag.operations.errors import DocumentNotFound, DuplicateDocument


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.rows: dict[UUID, Document] = {}  # insertion order = creation order

    def add(self, document: NewDocument) -> Document:
        if self.find_by_hash(document.user_id, document.file_sha256) is not None:
            raise DuplicateDocument(f"{document.user_id} already has {document.file_sha256}")
        now = _now()
        stored = Document(**document.model_dump(), status=DocumentStatus.QUEUED, created_at=now, updated_at=now)
        self.rows[stored.id] = stored
        return stored

    def get_for_user(self, user_id: str, document_id: UUID) -> Document | None:
        document = self.rows.get(document_id)
        return document if document is not None and document.user_id == user_id else None

    def find_by_hash(self, user_id: str, file_sha256: str) -> Document | None:
        return self._first(lambda d: d.user_id == user_id and d.file_sha256 == file_sha256)

    def find_by_pageindex_name(self, user_id: str, name: str) -> Document | None:
        return self._first(lambda d: d.user_id == user_id and d.pageindex_name == name)

    def find_by_pageindex_doc_id(self, user_id: str, doc_id: str) -> Document | None:
        return self._first(lambda d: d.user_id == user_id and d.pageindex_doc_id == doc_id)

    def list_for_user(self, user_id: str) -> list[Document]:
        return [d for d in reversed(self.rows.values()) if d.user_id == user_id]

    def list_by_status(self, status: DocumentStatus) -> list[Document]:
        return [d for d in self.rows.values() if d.status is status]

    def update(self, document_id: UUID, changes: DocumentChanges) -> Document:
        current = self.rows.get(document_id)
        if current is None:
            raise DocumentNotFound(str(document_id))
        updated = current.model_copy(update={**changes.as_update(), "updated_at": _now()})
        self.rows[document_id] = updated
        return updated

    def delete(self, document_id: UUID) -> None:
        if self.rows.pop(document_id, None) is None:
            raise DocumentNotFound(str(document_id))

    def _first(self, matches: Callable[[Document], bool]) -> Document | None:
        return next((d for d in self.rows.values() if matches(d)), None)


class InMemoryFigureRepository:
    def __init__(self) -> None:
        self.rows: list[FigureDescription] = []

    def add_many(self, figures: Sequence[NewFigureDescription]) -> list[FigureDescription]:
        now = _now()
        stored = [FigureDescription(**figure.model_dump(), id=uuid4(), created_at=now) for figure in figures]
        self.rows.extend(stored)
        return stored

    def list_for_document(self, document_id: UUID) -> list[FigureDescription]:
        found = [row for row in self.rows if row.document_id == document_id]
        return sorted(found, key=lambda row: (row.page, row.figure_index))

    def delete_for_document(self, document_id: UUID) -> None:
        self.rows = [row for row in self.rows if row.document_id != document_id]
