"""Contracts between the use cases and the outside world (spec Section 18).

`operations/` depends only on these. Adapters in `db/`, `storage/`, `pdf/`, `vision/`, `indexing/`
and `agent/` implement them, and the composition roots (`api/`, `worker/`) wire them together.
All are Protocols under our rule: each wraps a third-party SDK or is replaced by a fake in tests.
Stateless PDF steps are plain functions, typed here as Callable aliases; their thresholds are bound
with functools.partial at the composition root.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol
from uuid import UUID

from vectorless_rag.models import (
    ChatMessage,
    Document,
    DocumentChanges,
    DocumentStatus,
    FigureDescription,
    FigureNote,
    FigurePage,
    IndexedDocument,
    NewDocument,
    NewFigureDescription,
    PageDescription,
    PageImage,
    ResolvedAnswer,
)

# ── registry and files ──


class DocumentRepository(Protocol):
    """The `documents` table. Lookups scoped by user return None for other users' documents."""

    def add(self, document: NewDocument) -> Document:
        """Store as `queued`. Raises DuplicateDocument if (user_id, file_sha256) exists."""
        ...

    def get_for_user(self, user_id: str, document_id: UUID) -> Document | None: ...

    def find_by_hash(self, user_id: str, file_sha256: str) -> Document | None: ...

    def find_by_pageindex_name(self, user_id: str, name: str) -> Document | None: ...

    def find_by_pageindex_doc_id(self, user_id: str, doc_id: str) -> Document | None: ...

    def list_for_user(self, user_id: str) -> list[Document]:
        """Newest first."""
        ...

    def list_by_status(self, status: DocumentStatus) -> list[Document]:
        """Oldest first, for the worker."""
        ...

    def update(self, document_id: UUID, changes: DocumentChanges) -> Document:
        """Apply the fields set on `changes`. Raises DocumentNotFound."""
        ...

    def delete(self, document_id: UUID) -> None:
        """Raises DocumentNotFound."""
        ...


class FigureRepository(Protocol):
    """The `figure_descriptions` table."""

    def add_many(self, figures: Sequence[NewFigureDescription]) -> list[FigureDescription]: ...

    def list_for_document(self, document_id: UUID) -> list[FigureDescription]:
        """Ordered by page, then figure_index."""
        ...

    def delete_for_document(self, document_id: UUID) -> None: ...


class FileStore(Protocol):
    """Original and enriched PDFs under DATA_ROOT/users/{user_key}/documents/{document_id}/."""

    def save_original(self, user_key: str, document_id: UUID, filename: str, data: bytes) -> Path: ...

    def enriched_path(self, user_key: str, document_id: UUID, filename: str) -> Path:
        """Where the enriched copy goes: same file name as the original, in its own folder (spec 5.3e)."""
        ...

    def delete_document_files(self, user_key: str, document_id: UUID) -> None: ...


# ── PDF steps, rendering and vision ──

ReadPageTexts = Callable[[Path], list[str]]
"""Text layer of every page, in page order (scanned check, describer context)."""

DetectFigurePages = Callable[[Path], list[FigurePage]]
"""Pages with figures (spec 5.3a)."""

WriteInvisibleNotes = Callable[[Path, Path, Sequence[FigureNote]], None]
"""(original, target, notes): copy the PDF and write each note as invisible text (spec 5.3d)."""


class PageRenderer(Protocol):
    def render_png(self, pdf_path: Path, pages: Sequence[int]) -> list[bytes]:
        """One PNG per requested 1-based page, at the configured DPI."""
        ...


class FigureDescriber(Protocol):
    def describe(self, page_png: bytes, page_text: str) -> PageDescription: ...


# ── PageIndex ──


class UserIndex(Protocol):
    """One user's PageIndex library (one storage_path)."""

    def submit(self, pdf_path: Path) -> IndexedDocument:
        """Index synchronously. The PDF's file name becomes the stored document name."""
        ...

    def delete(self, doc_id: str) -> None:
        """Deleting an id that is not there is not an error, so a retried delete is safe."""
        ...

    def document_context(self, doc_ids: Sequence[str]) -> str:
        """Targeting text for the first user message (spec 5.6). Raises DocumentNotFound for unknown ids."""
        ...

    def agent_instructions(self) -> str:
        """PageIndex's agent instructions plus its citation prompt."""
        ...

    def agent_tools(self) -> list[Callable[..., str]]:
        """The read-only PageIndex tools as plain functions returning JSON strings."""
        ...

    def resolve_citations(self, answer: str) -> ResolvedAnswer: ...


class UserIndexProvider(Protocol):
    def for_user(self, user_key: str) -> UserIndex: ...


class UserLocks(Protocol):
    """One writer per user library (spec 5.4): PageIndex's own lock does nothing on Windows."""

    def for_user(self, user_key: str) -> AbstractContextManager[object]:
        """Hold around every change to the user's PageIndex library (index, delete). Deletion also
        removes the registry row while holding it, so ingestion can re-check the row under the lock."""
        ...


# ── the answering agent ──

PageViewer = Callable[[str, str], list[PageImage]]
"""(doc_name, pages) → rendered pages of the user's original PDF. Raises ViewPagesRejected."""


class AnswerAgent(Protocol):
    def answer(
        self,
        instructions: str,
        tools: Sequence[Callable[..., str]],
        view_pages: PageViewer,
        messages: Sequence[ChatMessage],
    ) -> str:
        """Run the agent over `messages` (context, history, question) with the user's PageIndex
        `instructions` and read-only `tools` plus view_pages; return the answer with <cite> tags."""
        ...
