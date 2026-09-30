"""The document use cases on in-memory fakes: upload checks, dedup, ownership, delete."""
from pathlib import Path
from uuid import uuid4

import pytest

from tests.fakes import (
    FakeUserIndex,
    FakeUserIndexProvider,
    InMemoryDocumentRepository,
    InMemoryFigureRepository,
    InMemoryFileStore,
    RecordingLocks,
)
from vectorless_rag.models import Document, DocumentChanges, DocumentStatus, FigureKind, NewFigureDescription
from vectorless_rag.operations import (
    DocumentLibrary,
    DocumentNotFound,
    FileTooLarge,
    TooManyPages,
    UnsupportedFile,
    user_key_for,
)
from vectorless_rag.operations.documents import display_name  # internal helper

PDF = b"%PDF-1.7 fake body"


class MissesOnce(InMemoryDocumentRepository):
    """The next hash lookup misses, as if a parallel upload of the same file had not committed yet."""

    def __init__(self) -> None:
        super().__init__()
        self.miss_next = False

    def find_by_hash(self, user_id: str, file_sha256: str) -> Document | None:
        if self.miss_next:
            self.miss_next = False
            return None
        return super().find_by_hash(user_id, file_sha256)


class Harness:
    def __init__(self, pages: int = 10) -> None:
        self.documents = MissesOnce()
        self.figures = InMemoryFigureRepository()
        self.files = InMemoryFileStore()
        self.indexes = FakeUserIndexProvider()
        self.locks = RecordingLocks()
        self.wakes = 0
        self.library = DocumentLibrary(
            documents=self.documents, figures=self.figures, files=self.files, indexes=self.indexes,
            locks=self.locks, count_pages=lambda _data: pages, on_queued=self.wake,
            max_upload_bytes=1_000, max_pages=50,
        )

    def wake(self) -> None:
        self.wakes += 1


def test_an_upload_is_saved_queued_and_wakes_the_worker() -> None:
    harness = Harness()

    upload = harness.library.upload("alice", "C:\\scans\\Manual v2.pdf", PDF)

    document = upload.document
    assert upload.created and harness.wakes == 1
    assert (document.status, document.filename, document.page_count) == (DocumentStatus.QUEUED, "Manual v2.pdf", 10)
    assert harness.files.files[document.original_path] == PDF


def test_the_same_file_again_returns_the_existing_document() -> None:
    harness = Harness()
    first = harness.library.upload("alice", "a.pdf", PDF)

    again = harness.library.upload("alice", "renamed.pdf", PDF)
    other_user = harness.library.upload("bob", "a.pdf", PDF)

    assert (again.created, again.document) == (False, first.document)
    assert other_user.created and other_user.document.id != first.document.id
    assert harness.wakes == 2


def test_a_simultaneous_duplicate_keeps_one_document_and_drops_its_files() -> None:
    harness = Harness()
    winner = harness.library.upload("alice", "a.pdf", PDF).document
    harness.documents.miss_next = True  # this upload does not see the winner until the insert fails

    result = harness.library.upload("alice", "a.pdf", PDF)

    assert (result.created, result.document) == (False, winner)
    assert list(harness.files.files) == [winner.original_path]  # the loser's copy is gone
    assert harness.wakes == 1


@pytest.mark.parametrize("data, error", [
    (b"x" * 1_001, FileTooLarge),
    (b"<html>not a pdf</html>", UnsupportedFile),
    (b"", UnsupportedFile),
])
def test_bad_uploads_are_rejected_before_anything_is_saved(data: bytes, error: type[Exception]) -> None:
    harness = Harness()

    with pytest.raises(error):
        harness.library.upload("alice", "a.pdf", data)
    assert harness.files.files == {} and harness.documents.rows == {} and harness.wakes == 0


def test_too_many_pages_is_rejected() -> None:
    harness = Harness(pages=51)

    with pytest.raises(TooManyPages, match="51 pages"):
        harness.library.upload("alice", "a.pdf", PDF)
    assert harness.documents.rows == {}


def test_documents_are_only_visible_to_their_owner() -> None:
    harness = Harness()
    mine = harness.library.upload("alice", "a.pdf", PDF).document
    harness.library.upload("bob", "b.pdf", PDF + b"b")

    assert harness.library.list_documents("alice") == [mine]
    assert harness.library.get("alice", mine.id) == mine
    with pytest.raises(DocumentNotFound):
        harness.library.get("bob", mine.id)
    with pytest.raises(DocumentNotFound):
        harness.library.figures_of("bob", mine.id)


def test_delete_removes_the_document_everywhere_under_the_users_lock() -> None:
    harness = Harness()
    document = harness.library.upload("alice", "a.pdf", PDF).document
    index = harness.indexes.for_user(user_key_for("alice"))
    indexed = index.submit(Path("a.pdf"))
    harness.documents.update(document.id, DocumentChanges(
        status=DocumentStatus.COMPLETED, pageindex_doc_id=indexed.doc_id, pageindex_name=indexed.name,
    ))
    harness.figures.add_many([NewFigureDescription(
        document_id=document.id, page=1, figure_index=1, kind=FigureKind.RASTER, description="d",
        vision_model="m", input_tokens=1, output_tokens=1,
    )])
    seen_under_lock: list[bool] = []

    class CheckingIndex(FakeUserIndex):
        def delete(self, doc_id: str) -> None:
            seen_under_lock.append(user_key_for("alice") in harness.locks.held)
            super().delete(doc_id)

    checking = CheckingIndex()
    checking.names = dict(index.names)
    harness.indexes.indexes[user_key_for("alice")] = checking

    harness.library.delete("alice", document.id)

    assert seen_under_lock == [True] and checking.names == {}
    assert harness.documents.rows == {} and harness.figures.rows == [] and harness.files.files == {}
    with pytest.raises(DocumentNotFound):
        harness.library.delete("alice", document.id)


def test_another_users_document_cannot_be_deleted() -> None:
    harness = Harness()
    document = harness.library.upload("alice", "a.pdf", PDF).document

    with pytest.raises(DocumentNotFound):
        harness.library.delete("bob", document.id)
    assert document.id in harness.documents.rows


@pytest.mark.parametrize("raw, shown", [
    ("report.pdf", "report.pdf"),
    ("../../etc/report.pdf", "report.pdf"),
    ("C:\\Users\\me\\report.pdf", "report.pdf"),
    ("   ", "document.pdf"),
    ("a" * 300 + ".pdf", "a" * 255),
])
def test_display_names_drop_folders_and_stay_short(raw: str, shown: str) -> None:
    assert display_name(raw) == shown


def test_uuid_of_a_missing_document_is_not_found() -> None:
    with pytest.raises(DocumentNotFound):
        Harness().library.get("alice", uuid4())
