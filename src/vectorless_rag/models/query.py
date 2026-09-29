"""Questions, answers and page images (spec 5.6–5.8, Section 7)."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    document_ids: tuple[UUID, ...] = ()  # empty: search the user's whole library
    history: tuple[ChatMessage, ...] = ()


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    document_id: UUID
    filename: str
    page: int = Field(ge=1)
    from_figure: bool


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[Citation, ...] = ()
    trace_id: str


class PageImage(BaseModel):
    """One rendered page of an original PDF, for the agent to look at (view_pages)."""

    model_config = ConfigDict(frozen=True)

    doc_name: str
    page: int = Field(ge=1)
    png: bytes
