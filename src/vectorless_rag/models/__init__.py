"""Typed records shared by every layer: documents, figures, index results, queries."""
from vectorless_rag.models.documents import Document, DocumentChanges, DocumentStatus, NewDocument
from vectorless_rag.models.figures import (
    BoundingBox,
    DetectedFigure,
    FigureDescription,
    FigureKind,
    FigureNote,
    FigurePage,
    NewFigureDescription,
    PageDescription,
)
from vectorless_rag.models.index import IndexCitation, IndexedDocument, ResolvedAnswer
from vectorless_rag.models.query import ChatMessage, Citation, PageImage, QueryRequest, QueryResponse

__all__ = [
    "BoundingBox",
    "ChatMessage",
    "Citation",
    "DetectedFigure",
    "Document",
    "DocumentChanges",
    "DocumentStatus",
    "FigureDescription",
    "FigureKind",
    "FigureNote",
    "FigurePage",
    "IndexCitation",
    "IndexedDocument",
    "NewDocument",
    "NewFigureDescription",
    "PageDescription",
    "PageImage",
    "QueryRequest",
    "QueryResponse",
    "ResolvedAnswer",
]
