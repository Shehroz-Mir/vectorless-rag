"""What the API shows about documents and figures (spec 7). File paths and PageIndex ids stay inside."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from vectorless_rag.models.documents import Document, DocumentStatus
from vectorless_rag.models.figures import FigureDescription, FigureKind


class DocumentOut(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: UUID
    filename: str
    status: DocumentStatus
    page_count: int
    figure_page_count: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, document: Document) -> DocumentOut:
        return cls.model_validate(document)


class FigureOut(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    page: int
    figure_index: int
    kind: FigureKind
    description: str
    vision_model: str
    input_tokens: int
    output_tokens: int

    @classmethod
    def of(cls, figure: FigureDescription) -> FigureOut:
        return cls.model_validate(figure)
