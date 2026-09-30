"""Document use cases (spec 7, Section 8 "Ingest" steps 1-3 and "Delete"): upload, list, read, delete.

An upload is only checked, saved and queued here; the ingestion worker does the slow part.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PureWindowsPath
from uuid import UUID, uuid4

from vectorless_rag.models import Document, FigureDescription, NewDocument
from vectorless_rag.operations.errors import DocumentNotFound, DuplicateDocument, FileTooLarge, TooManyPages, UnsupportedFile
from vectorless_rag.operations.ports import (
    CountPages,
    DocumentRepository,
    FigureRepository,
    FileStore,
    UserIndexProvider,
    UserLocks,
)
from vectorless_rag.operations.users import user_key_for

PDF_SIGNATURE = b"%PDF-"
SIGNATURE_WINDOW = 1024  # the PDF format allows a little junk before the signature
MAX_FILENAME_LENGTH = 255  # the registry column
DEFAULT_FILENAME = "document.pdf"


@dataclass(frozen=True)
class Upload:
    document: Document
    created: bool  # False: this user already had the same file, and this is that document


@dataclass(frozen=True, kw_only=True)
class DocumentLibrary:
    documents: DocumentRepository
    figures: FigureRepository
    files: FileStore
    indexes: UserIndexProvider
    locks: UserLocks
    count_pages: CountPages
    on_queued: Callable[[], None]  # wakes the ingestion worker
    max_upload_bytes: int
    max_pages: int

    def upload(self, user_id: str, filename: str, data: bytes) -> Upload:
        """Check, deduplicate, save and queue one PDF. Raises FileTooLarge, UnsupportedFile, TooManyPages."""
        if len(data) > self.max_upload_bytes:
            raise FileTooLarge(f"the file is too large; the limit is {self.max_upload_bytes / 1_048_576:.3g} MB")
        if PDF_SIGNATURE not in data[:SIGNATURE_WINDOW]:
            raise UnsupportedFile("the file is not a PDF")
        file_sha256 = hashlib.sha256(data).hexdigest()
        existing = self.documents.find_by_hash(user_id, file_sha256)
        if existing is not None:
            return Upload(document=existing, created=False)
        page_count = self.count_pages(data)
        if page_count > self.max_pages:
            raise TooManyPages(f"the PDF has {page_count} pages; the limit is {self.max_pages}")
        return self._store(user_id, display_name(filename), data, file_sha256, page_count)

    def list_documents(self, user_id: str) -> list[Document]:
        return self.documents.list_for_user(user_id)

    def get(self, user_id: str, document_id: UUID) -> Document:
        """Raises DocumentNotFound, also for another user's document."""
        document = self.documents.get_for_user(user_id, document_id)
        if document is None:
            raise DocumentNotFound(str(document_id))
        return document

    def figures_of(self, user_id: str, document_id: UUID) -> list[FigureDescription]:
        return self.figures.list_for_document(self.get(user_id, document_id).id)

    def delete(self, user_id: str, document_id: UUID) -> None:
        """Remove the document from PageIndex, the registry and disk (spec 8 "Delete"). Everything happens
        under the user's lock, so an ingestion that re-checks the row under it cannot index a deleted
        document. While one of the user's documents is being indexed, this waits for it."""
        user_key = user_key_for(user_id)
        with self.locks.for_user(user_key):
            document = self.get(user_id, document_id)
            if document.pageindex_doc_id is not None:
                self.indexes.for_user(user_key).delete(document.pageindex_doc_id)
            self.figures.delete_for_document(document.id)
            self.documents.delete(document.id)
            self.files.delete_document_files(user_key, document.id)

    def _store(self, user_id: str, filename: str, data: bytes, file_sha256: str, page_count: int) -> Upload:
        user_key, document_id = user_key_for(user_id), uuid4()
        original = self.files.save_original(user_key, document_id, filename, data)
        try:
            document = self.documents.add(NewDocument(
                id=document_id, user_id=user_id, filename=filename, original_path=original,
                file_sha256=file_sha256, page_count=page_count,
            ))
        except DuplicateDocument:  # the same file, uploaded at the same moment, got in first
            self.files.delete_document_files(user_key, document_id)
            existing = self.documents.find_by_hash(user_id, file_sha256)
            if existing is None:
                raise
            return Upload(document=existing, created=False)
        self.on_queued()
        return Upload(document=document, created=True)


def display_name(filename: str) -> str:
    """The upload's own name without any folders, as the user will see it. The file store makes its
    own safe name for disk."""
    name = PureWindowsPath(filename).name.strip()  # drops folders written with / or \
    return name[:MAX_FILENAME_LENGTH] or DEFAULT_FILENAME
