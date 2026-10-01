"""Typed records shared by every layer: documents, figures, index results, queries, agent runs, API views."""
from vectorless_rag.models.api import DocumentOut, FigureOut
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
from vectorless_rag.models.runs import (
    VIEW_PAGES_TOOL,
    AgentRun,
    ModelStep,
    RunLabels,
    RunStep,
    RunTotals,
    ToolOutcome,
    ToolStep,
)

__all__ = [
    "VIEW_PAGES_TOOL",
    "AgentRun",
    "BoundingBox",
    "ChatMessage",
    "Citation",
    "DetectedFigure",
    "Document",
    "DocumentChanges",
    "DocumentOut",
    "DocumentStatus",
    "FigureDescription",
    "FigureKind",
    "FigureNote",
    "FigureOut",
    "FigurePage",
    "IndexCitation",
    "IndexedDocument",
    "ModelStep",
    "NewDocument",
    "NewFigureDescription",
    "PageDescription",
    "PageImage",
    "QueryRequest",
    "QueryResponse",
    "ResolvedAnswer",
    "RunLabels",
    "RunStep",
    "RunTotals",
    "ToolOutcome",
    "ToolStep",
]
