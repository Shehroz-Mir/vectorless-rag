"""Behaviour every DocumentRepository / FigureRepository must have.

Each test runs against the in-memory fakes and against the SQL repositories on SQLite, so the fakes
the operation tests rely on and the real registry are held to the same contract.
"""
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tests.fakes import InMemoryDocumentRepository, InMemoryFigureRepository, TickingClock
from vectorless_rag.db import (
    SqlDocumentRepository,
    SqlFigureRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from vectorless_rag.models import DocumentChanges, DocumentStatus, FigureKind, NewDocument, NewFigureDescription
from vectorless_rag.operations import DocumentNotFound, DocumentRepository, DuplicateDocument, FigureRepository

Registry = tuple[DocumentRepository, FigureRepository]


@pytest.fixture(params=["in-memory", "sqlite"])
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> Registry:
    clock = TickingClock()
    if request.param == "in-memory":
        return InMemoryDocumentRepository(clock), InMemoryFigureRepository(clock)
    engine = create_database_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    create_schema(engine)
    sessions = create_session_factory(engine)
    return SqlDocumentRepository(sessions, clock), SqlFigureRepository(sessions, clock)


@pytest.fixture
def documents(registry: Registry) -> DocumentRepository:
    return registry[0]


@pytest.fixture
def figures(registry: Registry) -> FigureRepository:
    return registry[1]


def new_document(user_id: str = "alice", sha: str = "a", filename: str = "manual.pdf") -> NewDocument:
    return NewDocument(
        id=uuid4(), user_id=user_id, filename=filename, original_path=Path(filename),
        file_sha256=sha * 64, page_count=10,
    )


def test_added_document_is_queued_and_visible_to_its_owner_only(documents: DocumentRepository) -> None:
    stored = documents.add(new_document())

    assert stored.status is DocumentStatus.QUEUED
    assert documents.get_for_user("alice", stored.id) == stored
    assert documents.get_for_user("bob", stored.id) is None


def test_same_file_twice_for_one_user_is_a_duplicate(documents: DocumentRepository) -> None:
    documents.add(new_document(sha="a"))

    with pytest.raises(DuplicateDocument):
        documents.add(new_document(sha="a"))
    documents.add(new_document(user_id="bob", sha="a"))  # another user may upload the same file


def test_find_by_hash_is_scoped_to_the_user(documents: DocumentRepository) -> None:
    stored = documents.add(new_document(sha="b"))

    assert documents.find_by_hash("alice", "b" * 64) == stored
    assert documents.find_by_hash("bob", "b" * 64) is None


def test_update_applies_only_set_fields_and_can_clear_errors(documents: DocumentRepository) -> None:
    stored = documents.add(new_document())

    failed = documents.update(stored.id, DocumentChanges(status=DocumentStatus.FAILED, error="boom"))
    retried = documents.update(stored.id, DocumentChanges(status=DocumentStatus.QUEUED, error=None))

    assert (failed.status, failed.error) == (DocumentStatus.FAILED, "boom")
    assert (retried.status, retried.error, retried.filename) == (DocumentStatus.QUEUED, None, "manual.pdf")
    assert retried.updated_at > stored.updated_at


def test_paths_and_index_results_round_trip(documents: DocumentRepository) -> None:
    stored = documents.add(new_document())

    updated = documents.update(stored.id, DocumentChanges(
        enriched_path=Path("var/enriched/manual.pdf"), figure_page_count=7,
        pageindex_doc_id="pi-1", pageindex_name="manual.pdf",
    ))

    assert updated == documents.get_for_user("alice", stored.id)
    assert (updated.enriched_path, updated.figure_page_count) == (Path("var/enriched/manual.pdf"), 7)


def test_pageindex_lookups_are_scoped_to_the_user(documents: DocumentRepository) -> None:
    stored = documents.add(new_document())
    documents.update(stored.id, DocumentChanges(pageindex_doc_id="pi-1", pageindex_name="manual.pdf"))

    assert documents.find_by_pageindex_name("alice", "manual.pdf") is not None
    assert documents.find_by_pageindex_doc_id("alice", "pi-1") is not None
    assert documents.find_by_pageindex_name("bob", "manual.pdf") is None
    assert documents.find_by_pageindex_doc_id("bob", "pi-1") is None


def test_listing_order(documents: DocumentRepository) -> None:
    first = documents.add(new_document(sha="a"))
    second = documents.add(new_document(sha="b"))
    documents.add(new_document(user_id="bob", sha="c"))

    assert [d.id for d in documents.list_for_user("alice")] == [second.id, first.id]  # newest first
    assert [d.id for d in documents.list_by_status(DocumentStatus.QUEUED)][:2] == [first.id, second.id]  # oldest first


def test_missing_documents_raise_on_update_and_delete(documents: DocumentRepository) -> None:
    with pytest.raises(DocumentNotFound):
        documents.update(uuid4(), DocumentChanges(status=DocumentStatus.FAILED))
    with pytest.raises(DocumentNotFound):
        documents.delete(uuid4())


def test_delete_removes_the_document(documents: DocumentRepository) -> None:
    stored = documents.add(new_document())

    documents.delete(stored.id)

    assert documents.get_for_user("alice", stored.id) is None


def figure(document_id: UUID, page: int, index: int = 1) -> NewFigureDescription:
    return NewFigureDescription(
        document_id=document_id, page=page, figure_index=index, kind=FigureKind.RASTER,
        description=f"figure on page {page}", vision_model="fake", input_tokens=1, output_tokens=1,
    )


def test_figures_are_listed_in_page_order_and_deleted_per_document(
    documents: DocumentRepository, figures: FigureRepository,
) -> None:
    doc_a = documents.add(new_document(sha="a")).id
    doc_b = documents.add(new_document(sha="b")).id
    figures.add_many([figure(doc_a, 14), figure(doc_a, 3, 2), figure(doc_a, 3, 1), figure(doc_b, 1)])

    assert [(f.page, f.figure_index) for f in figures.list_for_document(doc_a)] == [(3, 1), (3, 2), (14, 1)]

    figures.delete_for_document(doc_a)

    assert figures.list_for_document(doc_a) == []
    assert len(figures.list_for_document(doc_b)) == 1
