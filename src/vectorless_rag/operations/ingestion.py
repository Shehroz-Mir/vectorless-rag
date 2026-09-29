"""Ingestion use case (spec 5.3 a-e, 5.4, Section 8): make a queued upload searchable.

queued → enriching → indexing → completed, or failed with an error its owner can read.
Enriching: scanned check, detect figure pages, describe them (stored as they arrive, reused on a
retry), write them into a copy of the PDF as invisible text. Indexing: submit that copy to the
user's PageIndex library while holding the user's lock.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from vectorless_rag.models import (
    Document,
    DocumentChanges,
    DocumentStatus,
    FigureNote,
    FigurePage,
    NewFigureDescription,
    PageDescription,
)
from vectorless_rag.operations.errors import DocumentNotFound, ScannedDocument, VectorlessRagError
from vectorless_rag.operations.ports import (
    DetectFigurePages,
    DocumentRepository,
    FigureDescriber,
    FigureRepository,
    FileStore,
    PageRenderer,
    ReadPageTexts,
    UserIndexProvider,
    UserLocks,
    WriteInvisibleNotes,
)
from vectorless_rag.operations.users import user_key_for

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 2_000  # a longer reply is cut, so it can never fail enrichment (spec 5.3b)
CUT_MARK = " ..."
FIGURE_INDEX = 1  # one description per figure page; it covers every figure on the page


@dataclass(frozen=True)
class IngestionRules:
    max_figure_pages: int
    vision_concurrency: int
    scanned_max_text_chars: int
    scanned_page_share: float


@dataclass(frozen=True, kw_only=True)
class DocumentIngestion:
    documents: DocumentRepository
    figures: FigureRepository
    files: FileStore
    read_page_texts: ReadPageTexts
    detect_figure_pages: DetectFigurePages
    renderer: PageRenderer
    describer: FigureDescriber
    write_invisible_notes: WriteInvisibleNotes
    indexes: UserIndexProvider
    locks: UserLocks
    rules: IngestionRules

    def ingest(self, document: Document) -> Document | None:
        """Run every step for one queued document. Returns it `completed` or `failed`, or None if it
        was deleted meanwhile. A failure is stored on the document, not raised; only a failing
        registry makes this raise. A document that is no longer queued is returned untouched, so a
        stale read of the queue never processes a document twice."""
        current = self.documents.get_for_user(document.user_id, document.id)
        if current is None or current.status is not DocumentStatus.QUEUED:
            return current
        user_key = user_key_for(current.user_id)
        stage = "enrichment"
        try:
            enriched_pdf = self._enrich(current, user_key)
            stage = "indexing"
            return self._index(current, enriched_pdf, user_key)
        except DocumentNotFound:
            return self._discard(current, user_key)
        except VectorlessRagError as error:
            return self._fail(current, user_key, str(error))
        except Exception as error:  # any adapter failure becomes the document's status, not a crash
            logger.exception("%s of document %s failed", stage, current.id)
            return self._fail(current, user_key, f"{stage} failed ({type(error).__name__}); see the service log")

    def _enrich(self, document: Document, user_key: str) -> Path:
        """Steps a-d. Returns the enriched copy; the document is `indexing` from here on."""
        self.documents.update(document.id, DocumentChanges(status=DocumentStatus.ENRICHING, error=None))
        page_texts = self.read_page_texts(document.original_path)
        check_not_scanned(page_texts, self.rules.scanned_max_text_chars, self.rules.scanned_page_share)
        figure_pages = self._figure_pages(document)
        descriptions = self._describe(document, figure_pages, page_texts)
        enriched_pdf = self.files.enriched_path(user_key, document.id, document.filename)
        self.write_invisible_notes(document.original_path, enriched_pdf, figure_notes(figure_pages, descriptions))
        self.documents.update(document.id, DocumentChanges(
            status=DocumentStatus.INDEXING, enriched_path=enriched_pdf, figure_page_count=len(figure_pages),
        ))
        return enriched_pdf

    def _figure_pages(self, document: Document) -> list[FigurePage]:
        pages = self.detect_figure_pages(document.original_path)
        limit = self.rules.max_figure_pages
        if len(pages) > limit:
            logger.warning("document %s has %d figure pages; enriching the first %d", document.id, len(pages), limit)
        return pages[:limit]

    def _describe(self, document: Document, pages: Sequence[FigurePage], page_texts: Sequence[str]) -> dict[int, str]:
        """Descriptions by page number. Stored ones are reused, so a retry repeats no vision calls; new
        ones are stored as they arrive, so a failure part-way keeps what was already paid for."""
        descriptions = {row.page: row.description for row in self.figures.list_for_document(document.id)}
        missing = [page for page in pages if page.page not in descriptions]
        if not missing:
            return descriptions
        pool = ThreadPoolExecutor(max_workers=self.rules.vision_concurrency, thread_name_prefix="vision")
        try:
            futures = {
                pool.submit(self._describe_page, document.original_path, page.page, page_texts[page.page - 1]): page
                for page in missing
            }
            for future in as_completed(futures):
                page = futures[future]
                (stored,) = self.figures.add_many([new_figure(document, page, future.result())])
                descriptions[page.page] = stored.description
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # after a failure, calls not yet started are skipped
        return descriptions

    def _describe_page(self, pdf: Path, page: int, page_text: str) -> PageDescription:
        """Runs on a vision thread: render the original page, then ask the vision model about it."""
        (png,) = self.renderer.render_png(pdf, [page])
        return self.describer.describe(png, page_text)

    def _index(self, document: Document, enriched_pdf: Path, user_key: str) -> Document:
        """Step e, with one writer per user library."""
        index = self.indexes.for_user(user_key)
        with self.locks.for_user(user_key):
            # Deletion removes the row under this lock, so a row still here stays until we are done.
            if self.documents.get_for_user(document.user_id, document.id) is None:
                raise DocumentNotFound(str(document.id))
            indexed = index.submit(enriched_pdf)
            try:
                return self.documents.update(document.id, DocumentChanges(
                    status=DocumentStatus.COMPLETED, pageindex_doc_id=indexed.doc_id, pageindex_name=indexed.name,
                ))
            except Exception:
                index.delete(indexed.doc_id)  # an index entry without a registry row would be an orphan
                raise

    def _fail(self, document: Document, user_key: str, error: str) -> Document | None:
        try:
            return self.documents.update(document.id, DocumentChanges(status=DocumentStatus.FAILED, error=error))
        except DocumentNotFound:
            return self._discard(document, user_key)

    def _discard(self, document: Document, user_key: str) -> None:
        """The document was deleted while we worked on it: remove files written after the deletion."""
        logger.info("document %s was deleted during ingestion", document.id)
        self.files.delete_document_files(user_key, document.id)


def requeue_interrupted(documents: DocumentRepository) -> list[Document]:
    """At start-up, put documents that a stopped process left `enriching` or `indexing` back in the queue.

    Safe while one process does all ingestion (spec 5.4, v1). Enrichment reuses stored descriptions,
    so the rerun repeats no vision calls. If the process stopped after PageIndex finished but before the
    registry update, the rerun indexes the file again under a suffixed name (open item, spec 15).
    """
    interrupted = documents.list_by_status(DocumentStatus.ENRICHING) + documents.list_by_status(DocumentStatus.INDEXING)
    if interrupted:
        logger.warning("re-queueing %d documents interrupted by a restart", len(interrupted))
    return [documents.update(document.id, DocumentChanges(status=DocumentStatus.QUEUED)) for document in interrupted]


def check_not_scanned(page_texts: Sequence[str], max_text_chars: int, page_share: float) -> None:
    """Spec 9.5: fail when more than `page_share` of the pages have almost no text layer."""
    blank = sum(1 for text in page_texts if len(text.strip()) < max_text_chars)
    if blank > page_share * len(page_texts):
        raise ScannedDocument(
            f"{blank} of {len(page_texts)} pages have no text layer; scanned PDFs are not supported yet"
        )


def new_figure(document: Document, page: FigurePage, description: PageDescription) -> NewFigureDescription:
    return NewFigureDescription(
        document_id=document.id,
        page=page.page,
        figure_index=FIGURE_INDEX,
        kind=page.main_figure.kind,
        description=cap_description(description.text),
        vision_model=description.model,
        input_tokens=description.input_tokens,
        output_tokens=description.output_tokens,
    )


def cap_description(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Cut an overlong description at a word boundary and mark the cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - len(CUT_MARK)].rsplit(maxsplit=1)[0] + CUT_MARK


def figure_notes(pages: Sequence[FigurePage], descriptions: Mapping[int, str]) -> list[FigureNote]:
    """One invisible note per figure page, inside its largest figure, marked as generated text."""
    return [
        FigureNote(page=page.page, box=page.main_figure.box, text=f"{figure_marker(page.page)} {descriptions[page.page]}")
        for page in pages
    ]


def figure_marker(page: int) -> str:
    """Tells the agent that the text after it was generated from a figure (spec 5.3d, 5.6)."""
    return f"[FIGURE DESCRIPTION p{page} fig{FIGURE_INDEX}]"
