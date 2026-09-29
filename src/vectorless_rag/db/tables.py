"""Registry schema (spec Section 6). Rows stay inside db/; repositories hand out Pydantic models."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from vectorless_rag.models import DocumentStatus, FigureKind


class Base(DeclarativeBase):
    pass


def _enum_column(enum: type[StrEnum]) -> Enum:
    """Store the enum's values ("queued"), not its names, as plain strings: portable across databases."""
    return Enum(enum, native_enum=False, length=16, values_callable=lambda members: [m.value for m in members])


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("user_id", "file_sha256", name="uq_documents_user_hash"),
        UniqueConstraint("user_id", "pageindex_name", name="uq_documents_user_pageindex_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str] = mapped_column(String(1024))
    enriched_path: Mapped[str | None] = mapped_column(String(1024))
    file_sha256: Mapped[str] = mapped_column(String(64))
    pageindex_doc_id: Mapped[str | None] = mapped_column(String(64))
    pageindex_name: Mapped[str | None] = mapped_column(String(255))
    page_count: Mapped[int] = mapped_column(Integer)
    figure_page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(_enum_column(DocumentStatus), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FigureRow(Base):
    __tablename__ = "figure_descriptions"
    __table_args__ = (UniqueConstraint("document_id", "page", "figure_index", name="uq_figures_document_page_index"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    # Deleting a document deletes its figure descriptions (needs SQLite's foreign_keys pragma).
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    figure_index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[FigureKind] = mapped_column(_enum_column(FigureKind))
    description: Mapped[str] = mapped_column(Text)
    vision_model: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
