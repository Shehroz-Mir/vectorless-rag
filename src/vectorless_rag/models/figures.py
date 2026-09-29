"""Figure detection, descriptions and enrichment notes (spec 5.3, `figure_descriptions` table)."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

BoundingBox = tuple[float, float, float, float]
"""x0, y0, x1, y1 in PDF points, origin top-left (PyMuPDF's convention)."""


class FigureKind(StrEnum):
    RASTER = "raster"
    VECTOR = "vector"


class DetectedFigure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FigureKind
    box: BoundingBox

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class FigurePage(BaseModel):
    """A page that detection flagged, with the figures found on it."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)  # 1-based
    figures: tuple[DetectedFigure, ...] = Field(min_length=1)

    @property
    def main_figure(self) -> DetectedFigure:
        """The largest figure: its kind is recorded, and the description is written inside its box."""
        return max(self.figures, key=lambda figure: figure.area)


class PageDescription(BaseModel):
    """What the vision model wrote about one figure page."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    model: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class FigureNote(BaseModel):
    """Invisible text to write onto one page of the enriched copy."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    box: BoundingBox
    text: str = Field(min_length=1)  # already carries the [FIGURE DESCRIPTION ...] marker


class NewFigureDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    page: int = Field(ge=1)
    figure_index: int = Field(ge=1)
    kind: FigureKind
    description: str = Field(min_length=1)
    vision_model: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class FigureDescription(NewFigureDescription):
    """One stored row."""

    id: UUID
    created_at: datetime
