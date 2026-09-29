"""SQLAlchemy registry: implements DocumentRepository and FigureRepository (spec 5.2, Section 6)."""
from vectorless_rag.db.base import Clock, utc_now
from vectorless_rag.db.database import create_database_engine, create_schema, create_session_factory
from vectorless_rag.db.documents import SqlDocumentRepository
from vectorless_rag.db.figures import SqlFigureRepository

__all__ = [
    "Clock",
    "SqlDocumentRepository",
    "SqlFigureRepository",
    "create_database_engine",
    "create_schema",
    "create_session_factory",
    "utc_now",
]
