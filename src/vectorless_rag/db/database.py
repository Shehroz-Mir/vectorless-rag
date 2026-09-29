"""Engine, sessions and schema for the registry DB (spec 5.2): SQLite in dev, Postgres in prod.

Called once by the composition roots at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, make_url
from sqlalchemy.orm import Session, sessionmaker

from vectorless_rag.db.tables import Base

SQLITE_BUSY_TIMEOUT_MS = 5_000  # wait this long for another thread's write instead of failing


def create_database_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return create_engine(url, pool_pre_ping=True)
    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    # API request threads and worker threads share the engine.
    engine = create_engine(url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _configure_sqlite)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # Rows stay readable after commit; repositories convert them to models right away.
    return sessionmaker(engine, expire_on_commit=False)


def create_schema(engine: Engine) -> None:
    """Create missing tables. v1 has no migrations; add Alembic once the schema changes after release."""
    Base.metadata.create_all(engine)


def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # enables ON DELETE CASCADE
    cursor.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
    cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    cursor.close()
