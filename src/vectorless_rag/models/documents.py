"""Document registry records (spec Section 6, `documents` table)."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    ENRICHING = "enriching"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class NewDocument(BaseModel):
    """What an upload creates. The upload operation assigns the id (the file store needs it first)."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    original_path: Path
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)


class Document(BaseModel):
    """One registry row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    user_id: str
    filename: str
    original_path: Path
    file_sha256: str
    page_count: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    enriched_path: Path | None = None
    figure_page_count: int | None = None
    pageindex_doc_id: str | None = None
    pageindex_name: str | None = None  # the name the agent, view_pages and citations use
    error: str | None = None


class DocumentChanges(BaseModel):
    """A partial update. Only fields that were set are applied, so `error=None` clears an error."""

    model_config = ConfigDict(frozen=True)

    status: DocumentStatus | None = None
    error: str | None = None
    enriched_path: Path | None = None
    figure_page_count: int | None = Field(default=None, ge=0)
    pageindex_doc_id: str | None = None
    pageindex_name: str | None = None

    def as_update(self) -> dict[str, object]:
        return self.model_dump(exclude_unset=True)
