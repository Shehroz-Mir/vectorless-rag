"""PageIndex results as plain records, so operations never see SDK types (spec 5.5, 5.8)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IndexedDocument(BaseModel):
    """What indexing one PDF returned."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)  # "pi-…"
    name: str = Field(min_length=1)  # stored name; a clash gets a _1 … _99 suffix


class IndexCitation(BaseModel):
    """One citation parsed from an answer's <cite doc=… page=…/> tags."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)  # the number shown in the answer text
    document: str
    doc_id: str | None  # None: the name is not in this user's library, so the citation is dropped
    page: int = Field(ge=1)


class ResolvedAnswer(BaseModel):
    """The answer with its tags rewritten to numbered markers, plus the citations behind them."""

    model_config = ConfigDict(frozen=True)

    text: str
    citations: tuple[IndexCitation, ...] = ()
