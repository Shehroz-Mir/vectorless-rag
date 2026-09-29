"""The `documents` table behind ports.DocumentRepository."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError

from vectorless_rag.db.base import SqlRepository, as_utc
from vectorless_rag.db.tables import DocumentRow
from vectorless_rag.models import Document, DocumentChanges, DocumentStatus, NewDocument
from vectorless_rag.operations import DocumentNotFound, DuplicateDocument


class SqlDocumentRepository(SqlRepository):
    def add(self, document: NewDocument) -> Document:
        now = self._now()
        row = DocumentRow(
            id=document.id,
            user_id=document.user_id,
            filename=document.filename,
            original_path=str(document.original_path),
            file_sha256=document.file_sha256,
            page_count=document.page_count,
            status=DocumentStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        try:
            with self._transaction() as session:
                session.add(row)
        except IntegrityError as error:
            raise DuplicateDocument(f"{document.user_id} already has {document.file_sha256}") from error
        return _to_document(row)

    def get_for_user(self, user_id: str, document_id: UUID) -> Document | None:
        with self._transaction() as session:
            row = session.get(DocumentRow, document_id)
            return _to_document(row) if row is not None and row.user_id == user_id else None

    def find_by_hash(self, user_id: str, file_sha256: str) -> Document | None:
        return self._first(select(DocumentRow).where(DocumentRow.user_id == user_id, DocumentRow.file_sha256 == file_sha256))

    def find_by_pageindex_name(self, user_id: str, name: str) -> Document | None:
        return self._first(select(DocumentRow).where(DocumentRow.user_id == user_id, DocumentRow.pageindex_name == name))

    def find_by_pageindex_doc_id(self, user_id: str, doc_id: str) -> Document | None:
        return self._first(select(DocumentRow).where(DocumentRow.user_id == user_id, DocumentRow.pageindex_doc_id == doc_id))

    def list_for_user(self, user_id: str) -> list[Document]:
        statement = select(DocumentRow).where(DocumentRow.user_id == user_id)
        return self._all(statement.order_by(DocumentRow.created_at.desc(), DocumentRow.id))

    def list_by_status(self, status: DocumentStatus) -> list[Document]:
        statement = select(DocumentRow).where(DocumentRow.status == status)
        return self._all(statement.order_by(DocumentRow.created_at, DocumentRow.id))

    def update(self, document_id: UUID, changes: DocumentChanges) -> Document:
        with self._transaction() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                raise DocumentNotFound(str(document_id))
            for field, value in changes.as_update().items():
                setattr(row, field, str(value) if isinstance(value, Path) else value)
            row.updated_at = self._now()
        return _to_document(row)

    def delete(self, document_id: UUID) -> None:
        """Its figure descriptions go with it (ON DELETE CASCADE)."""
        with self._transaction() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                raise DocumentNotFound(str(document_id))
            session.delete(row)

    def _first(self, statement: Select[DocumentRow]) -> Document | None:
        with self._transaction() as session:
            row = session.scalars(statement.limit(1)).first()
            return _to_document(row) if row is not None else None

    def _all(self, statement: Select[DocumentRow]) -> list[Document]:
        with self._transaction() as session:
            return [_to_document(row) for row in session.scalars(statement)]


def _to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        user_id=row.user_id,
        filename=row.filename,
        original_path=Path(row.original_path),
        file_sha256=row.file_sha256,
        page_count=row.page_count,
        status=row.status,
        created_at=as_utc(row.created_at),
        updated_at=as_utc(row.updated_at),
        enriched_path=Path(row.enriched_path) if row.enriched_path is not None else None,
        figure_page_count=row.figure_page_count,
        pageindex_doc_id=row.pageindex_doc_id,
        pageindex_name=row.pageindex_name,
        error=row.error,
    )
