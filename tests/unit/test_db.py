"""What only the real database does: UTC timestamps, cascade delete, SQLite settings, schema setup."""
from datetime import UTC
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Engine, text

from vectorless_rag.db import (
    SqlDocumentRepository,
    SqlFigureRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from vectorless_rag.models import FigureKind, NewDocument, NewFigureDescription


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'nested' / 'app.db'}")
    create_schema(engine)
    return engine


def add_document(repository: SqlDocumentRepository) -> NewDocument:
    document = NewDocument(
        id=uuid4(), user_id="alice", filename="m.pdf", original_path=Path("m.pdf"), file_sha256="a" * 64, page_count=3,
    )
    repository.add(document)
    return document


def test_sqlite_folder_is_created_and_safety_settings_are_on(engine: Engine, tmp_path: Path) -> None:
    with engine.connect() as connection:
        foreign_keys = connection.execute(text("PRAGMA foreign_keys")).scalar()
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar()

    assert (tmp_path / "nested" / "app.db").is_file()
    assert (foreign_keys, journal_mode) == (1, "wal")


def test_schema_creation_can_run_again(engine: Engine) -> None:
    create_schema(engine)


def test_timestamps_come_back_timezone_aware_utc(engine: Engine) -> None:
    documents = SqlDocumentRepository(create_session_factory(engine))

    document = add_document(documents)
    stored = documents.get_for_user("alice", document.id)

    assert stored is not None and stored.created_at.tzinfo is UTC and stored.updated_at.tzinfo is UTC


def test_deleting_a_document_deletes_its_figure_descriptions(engine: Engine) -> None:
    sessions = create_session_factory(engine)
    documents, figures = SqlDocumentRepository(sessions), SqlFigureRepository(sessions)
    document = add_document(documents)
    figures.add_many([NewFigureDescription(
        document_id=document.id, page=1, figure_index=1, kind=FigureKind.VECTOR, description="drawing",
        vision_model="fake", input_tokens=1, output_tokens=1,
    )])

    documents.delete(document.id)

    assert figures.list_for_document(document.id) == []
