"""The ingestion use case on in-memory fakes: status changes, descriptions, notes, indexing, failures."""
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.fakes import (
    FakeFigureDescriber,
    FakePageRenderer,
    FakeUserIndex,
    FakeUserIndexProvider,
    InMemoryDocumentRepository,
    InMemoryFigureRepository,
    InMemoryFileStore,
    RecordingLocks,
    TickingClock,
)
from vectorless_rag.models import (
    DetectedFigure,
    Document,
    DocumentChanges,
    DocumentStatus,
    FigureKind,
    FigureNote,
    FigurePage,
    IndexedDocument,
    NewDocument,
    PageDescription,
)
from vectorless_rag.operations import DocumentIngestion, IngestionRules, ScannedDocument, requeue_interrupted, user_key_for
from vectorless_rag.operations.ingestion import cap_description, check_not_scanned  # internal helpers

TEXT = "Body text of a manual page, long enough to count as a real text layer."
BOX = (10.0, 10.0, 200.0, 200.0)


def figure_page(page: int) -> FigurePage:
    """A photo plus a small drawing; the photo is the larger, so the note goes in its box."""
    return FigurePage(page=page, figures=(
        DetectedFigure(kind=FigureKind.VECTOR, box=(0, 0, 5, 5)),
        DetectedFigure(kind=FigureKind.RASTER, box=BOX),
    ))


class StatusLog(InMemoryDocumentRepository):
    """Remembers every status it was given; can fail one update, or react to each."""

    def __init__(self, clock: Callable[[], datetime]) -> None:
        super().__init__(clock)
        self.statuses: list[DocumentStatus] = []
        self.fail_on: DocumentStatus | None = None
        self.after_update: Callable[[Document], None] = lambda _document: None

    def update(self, document_id: UUID, changes: DocumentChanges) -> Document:
        if changes.status is not None and changes.status is self.fail_on:
            raise RuntimeError("registry write failed")
        if changes.status is not None:
            self.statuses.append(changes.status)
        updated = super().update(document_id, changes)
        self.after_update(updated)
        return updated


class Harness:
    def __init__(
        self,
        *,
        page_texts: Sequence[str] = (TEXT,) * 4,
        figure_pages: Sequence[FigurePage] = (figure_page(2), figure_page(4)),
        max_figure_pages: int = 200,
        vision_concurrency: int = 2,
        describer: FakeFigureDescriber | None = None,
        indexes: FakeUserIndexProvider | None = None,
    ) -> None:
        clock = TickingClock()
        self.documents = StatusLog(clock)
        self.figures = InMemoryFigureRepository(clock)
        self.files = InMemoryFileStore()
        self.renderer = FakePageRenderer()
        self.describer = describer or FakeFigureDescriber()
        self.indexes = indexes or FakeUserIndexProvider()
        self.locks = RecordingLocks()
        self.written: list[tuple[Path, Path, list[FigureNote]]] = []
        self.ingestion = DocumentIngestion(
            documents=self.documents,
            figures=self.figures,
            files=self.files,
            read_page_texts=lambda _pdf: list(page_texts),
            detect_figure_pages=lambda _pdf: list(figure_pages),
            renderer=self.renderer,
            describer=self.describer,
            write_invisible_notes=self.write,
            indexes=self.indexes,
            locks=self.locks,
            rules=IngestionRules(
                max_figure_pages=max_figure_pages, vision_concurrency=vision_concurrency,
                scanned_max_text_chars=50, scanned_page_share=0.5,
            ),
        )

    def write(self, original: Path, target: Path, notes: Sequence[FigureNote]) -> None:
        self.written.append((original, target, list(notes)))
        self.files.files[target] = b"%PDF enriched"

    def upload(self, user_id: str = "alice", filename: str = "manual.pdf") -> Document:
        document_id = uuid4()
        path = self.files.save_original(user_key_for(user_id), document_id, filename, b"%PDF original")
        return self.documents.add(NewDocument(
            id=document_id, user_id=user_id, filename=filename, original_path=path,
            file_sha256=uuid4().hex * 2, page_count=4,
        ))

    def enriched_path(self, document: Document) -> Path:
        return self.files.enriched_path(user_key_for(document.user_id), document.id, document.filename)

    def files_of(self, document: Document) -> list[Path]:
        return [path for path in self.files.files if str(document.id) in path.parts]


def test_a_queued_document_ends_completed_and_indexed() -> None:
    harness = Harness()
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and result == harness.documents.get_for_user("alice", document.id)
    assert harness.documents.statuses == [DocumentStatus.ENRICHING, DocumentStatus.INDEXING, DocumentStatus.COMPLETED]
    assert (result.enriched_path, result.figure_page_count, result.error) == (harness.enriched_path(document), 2, None)
    index = harness.indexes.for_user(user_key_for("alice"))
    assert index.submitted == [harness.enriched_path(document)]
    assert (result.pageindex_name, index.names[result.pageindex_doc_id or ""]) == ("manual.pdf", "manual.pdf")


def test_each_figure_page_is_described_stored_and_written_as_a_marked_note() -> None:
    harness = Harness()
    document = harness.upload()

    harness.ingestion.ingest(document)

    assert sorted(harness.renderer.calls) == [(document.original_path, (2,)), (document.original_path, (4,))]
    assert sorted(harness.describer.calls) == [(b"png:manual.pdf:2", TEXT), (b"png:manual.pdf:4", TEXT)]
    stored = harness.figures.list_for_document(document.id)
    assert [(f.page, f.figure_index, f.kind, f.vision_model) for f in stored] == [
        (2, 1, FigureKind.RASTER, "fake-vision"), (4, 1, FigureKind.RASTER, "fake-vision"),
    ]
    (original, target, notes) = harness.written[0]
    assert (original, target) == (document.original_path, harness.enriched_path(document))
    assert notes == [
        FigureNote(page=2, box=BOX, text="[FIGURE DESCRIPTION p2 fig1] A figure seen in png:manual.pdf:2."),
        FigureNote(page=4, box=BOX, text="[FIGURE DESCRIPTION p4 fig1] A figure seen in png:manual.pdf:4."),
    ]


@pytest.mark.parametrize("status", [DocumentStatus.ENRICHING, DocumentStatus.COMPLETED, DocumentStatus.FAILED])
def test_only_a_queued_document_is_processed(status: DocumentStatus) -> None:
    harness = Harness()
    stale = harness.upload()  # the worker read it while it was still queued
    current = harness.documents.update(stale.id, DocumentChanges(status=status))
    harness.documents.statuses.clear()

    result = harness.ingestion.ingest(stale)

    assert result == current
    assert harness.documents.statuses == [] and harness.describer.calls == []


def test_a_document_without_figures_is_still_indexed() -> None:
    harness = Harness(figure_pages=())
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and (result.status, result.figure_page_count) == (DocumentStatus.COMPLETED, 0)
    assert harness.describer.calls == [] and harness.written[0][2] == []


def test_vision_calls_run_concurrently() -> None:
    both_in_flight = threading.Barrier(2, timeout=5)

    class WaitsForTheOther(FakeFigureDescriber):
        def describe(self, page_png: bytes, page_text: str) -> PageDescription:
            both_in_flight.wait()  # breaks, and fails the document, unless both calls run at once
            return super().describe(page_png, page_text)

    harness = Harness(describer=WaitsForTheOther(), vision_concurrency=2)

    result = harness.ingestion.ingest(harness.upload())

    assert result is not None and result.status is DocumentStatus.COMPLETED


def test_an_overlong_description_is_cut_before_it_is_stored_and_written() -> None:
    class Rambling(FakeFigureDescriber):
        def describe(self, page_png: bytes, page_text: str) -> PageDescription:
            return PageDescription(text="word " * 1_000, model="fake-vision", input_tokens=1, output_tokens=1)

    harness = Harness(describer=Rambling(), figure_pages=(figure_page(2),))
    document = harness.upload()

    harness.ingestion.ingest(document)

    (stored,) = harness.figures.list_for_document(document.id)
    assert len(stored.description) <= 2_000 and stored.description.endswith("word ...")
    assert harness.written[0][2][0].text == "[FIGURE DESCRIPTION p2 fig1] " + stored.description


def test_a_scanned_document_fails_with_a_clear_message() -> None:
    harness = Harness(page_texts=("", "12", TEXT, " "))
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and result.status is DocumentStatus.FAILED
    assert result.error == "3 of 4 pages have no text layer; scanned PDFs are not supported yet"
    assert harness.describer.calls == [] and harness.written == []
    assert harness.indexes.for_user(user_key_for("alice")).submitted == []


@pytest.mark.parametrize("texts", [
    (TEXT, TEXT, "", ""),  # exactly half without text is allowed ("more than" the share)
    (TEXT,),
])
def test_mostly_text_pages_pass_the_scanned_check(texts: tuple[str, ...]) -> None:
    check_not_scanned(texts, max_text_chars=50, page_share=0.5)


def test_mostly_blank_pages_are_a_scanned_document() -> None:
    with pytest.raises(ScannedDocument, match="3 of 4 pages"):
        check_not_scanned((TEXT, "", "", ""), max_text_chars=50, page_share=0.5)


def test_figure_pages_are_capped() -> None:
    harness = Harness(max_figure_pages=1)
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and result.figure_page_count == 1
    assert [f.page for f in harness.figures.list_for_document(document.id)] == [2]


def test_indexing_runs_under_the_users_lock() -> None:
    locks_seen: list[set[str]] = []

    class LockCheckingIndex(FakeUserIndex):
        def submit(self, pdf_path: Path) -> IndexedDocument:
            locks_seen.append(set(harness.locks.held))
            return super().submit(pdf_path)

    class Provider(FakeUserIndexProvider):
        def for_user(self, user_key: str) -> FakeUserIndex:
            return self.indexes.setdefault(user_key, LockCheckingIndex())

    harness = Harness(indexes=Provider())

    harness.ingestion.ingest(harness.upload())

    assert locks_seen == [{user_key_for("alice")}]
    assert harness.locks.held == set()


def test_a_failed_vision_call_fails_the_document_but_keeps_finished_descriptions() -> None:
    class DownOnPageFour(FakeFigureDescriber):
        def describe(self, page_png: bytes, page_text: str) -> PageDescription:
            if page_png.endswith(b":4"):
                raise ConnectionError("vision service unreachable")
            return super().describe(page_png, page_text)

    harness = Harness(describer=DownOnPageFour(), vision_concurrency=1)
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and result.status is DocumentStatus.FAILED
    assert result.error == "enrichment failed (ConnectionError); see the service log"
    assert [f.page for f in harness.figures.list_for_document(document.id)] == [2]


def test_a_retry_reuses_stored_descriptions_and_clears_the_error() -> None:
    class FailsFirstSubmit(FakeUserIndex):
        def submit(self, pdf_path: Path) -> IndexedDocument:
            if not self.submitted:
                self.submitted.append(pdf_path)
                raise RuntimeError("index busy")
            return super().submit(pdf_path)

    class Provider(FakeUserIndexProvider):
        def for_user(self, user_key: str) -> FakeUserIndex:
            return self.indexes.setdefault(user_key, FailsFirstSubmit())

    harness = Harness(indexes=Provider())
    document = harness.upload()

    failed = harness.ingestion.ingest(document)
    requeued = harness.documents.update(document.id, DocumentChanges(status=DocumentStatus.QUEUED))
    retried = harness.ingestion.ingest(requeued)

    assert failed is not None and failed.error == "indexing failed (RuntimeError); see the service log"
    assert retried is not None and (retried.status, retried.error) == (DocumentStatus.COMPLETED, None)
    assert len(harness.describer.calls) == 2  # both pages described once, in the first run only
    assert len(harness.figures.list_for_document(document.id)) == 2
    assert len(harness.written[1][2]) == 2  # the retry still writes both notes


def test_a_registry_failure_after_indexing_removes_the_index_entry() -> None:
    harness = Harness()
    harness.documents.fail_on = DocumentStatus.COMPLETED
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is not None and result.status is DocumentStatus.FAILED
    index = harness.indexes.for_user(user_key_for("alice"))
    assert len(index.submitted) == 1 and index.names == {}


def test_a_document_deleted_during_enrichment_is_dropped() -> None:
    class DeletesTheDocument(FakeFigureDescriber):
        def describe(self, page_png: bytes, page_text: str) -> PageDescription:
            if document.id in harness.documents.rows:
                harness.documents.delete(document.id)
            return super().describe(page_png, page_text)

    harness = Harness(describer=DeletesTheDocument())
    document = harness.upload()

    result = harness.ingestion.ingest(document)

    assert result is None
    assert harness.files_of(document) == []  # the enriched copy written after the delete is gone too
    assert harness.indexes.for_user(user_key_for("alice")).submitted == []


def test_a_document_deleted_while_waiting_for_the_lock_is_not_indexed() -> None:
    harness = Harness()
    document = harness.upload()

    def delete_once_indexing(updated: Document) -> None:
        if updated.status is DocumentStatus.INDEXING:
            harness.documents.delete(updated.id)  # as a delete that got the lock first would

    harness.documents.after_update = delete_once_indexing

    result = harness.ingestion.ingest(document)

    assert result is None
    assert harness.indexes.for_user(user_key_for("alice")).submitted == []
    assert harness.files_of(document) == []


def test_interrupted_documents_are_requeued() -> None:
    harness = Harness()
    enriching, indexing, completed, failed = (harness.upload(filename=f"{n}.pdf") for n in range(4))
    for document, status in [(enriching, DocumentStatus.ENRICHING), (indexing, DocumentStatus.INDEXING),
                             (completed, DocumentStatus.COMPLETED), (failed, DocumentStatus.FAILED)]:
        harness.documents.update(document.id, DocumentChanges(status=status))

    requeued = requeue_interrupted(harness.documents)

    assert sorted(d.id for d in requeued) == sorted([enriching.id, indexing.id])
    assert [d.id for d in harness.documents.list_by_status(DocumentStatus.QUEUED)] == [enriching.id, indexing.id]


@pytest.mark.parametrize("text, expected", [
    ("  short  ", "short"),
    ("a" * 2_000, "a" * 2_000),
])
def test_cap_description_keeps_short_text(text: str, expected: str) -> None:
    assert cap_description(text) == expected


def test_cap_description_cuts_long_text_at_a_word() -> None:
    capped = cap_description("alpha beta gamma delta", limit=16)

    assert capped == "alpha beta ..." and len(capped) <= 16
