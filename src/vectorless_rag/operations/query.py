"""Question-answering use case (spec 5.6-5.8, Section 8 "Query").

Checks the selected documents, builds the messages (document context, history, question), runs the
answering agent with the user's read-only PageIndex tools and a view_pages bound to the user, then
maps the answer's citations to our document IDs.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from vectorless_rag.models import (
    ChatMessage,
    Citation,
    DocumentStatus,
    IndexCitation,
    PageImage,
    QueryRequest,
    QueryResponse,
)
from vectorless_rag.operations.errors import DocumentNotFound, DocumentNotReady, ViewPagesRejected
from vectorless_rag.operations.ports import (
    AnswerAgent,
    DocumentRepository,
    FigureRepository,
    PageRenderer,
    PageViewer,
    UserIndex,
    UserIndexProvider,
)
from vectorless_rag.operations.users import user_key_for

_PAGE_PART = re.compile(r"(\d+)(?:-(\d+))?")


@dataclass(frozen=True, kw_only=True)
class QuestionAnswering:
    documents: DocumentRepository
    figures: FigureRepository
    indexes: UserIndexProvider
    renderer: PageRenderer
    agent: AnswerAgent
    view_pages_max_pages: int

    def answer(self, user_id: str, request: QueryRequest) -> QueryResponse:
        """Raises DocumentNotFound (also for other users' documents), DocumentNotReady, AnswerIncomplete."""
        index_ids = self._selected_index_ids(user_id, request.document_ids)
        index = self.indexes.for_user(user_key_for(user_id))
        reply = self.agent.answer(
            index.agent_instructions(), index.agent_tools(), self.page_viewer(user_id), _messages(index, index_ids, request),
        )
        resolved = index.resolve_citations(reply)
        return QueryResponse(answer=resolved.text, citations=self._citations(user_id, resolved.citations), trace_id=uuid4().hex)

    def page_viewer(self, user_id: str) -> PageViewer:
        """view_pages for one user (spec 5.7): the name is looked up among that user's documents only,
        and the pages come from the original PDF, never the enriched copy."""

        def view_pages(doc_name: str, pages: str) -> list[PageImage]:
            document = self.documents.find_by_pageindex_name(user_id, doc_name)
            if document is None:
                raise ViewPagesRejected(f"There is no document named {doc_name!r}. Use the name the other tools show.")
            numbers = parse_pages(pages, document.page_count, self.view_pages_max_pages)
            pngs = self.renderer.render_png(document.original_path, numbers)
            return [PageImage(doc_name=doc_name, page=page, png=png) for page, png in zip(numbers, pngs, strict=True)]

        return view_pages

    def _selected_index_ids(self, user_id: str, document_ids: Sequence[UUID]) -> list[str]:
        """PageIndex ids of the selected documents, which must be the user's and completed."""
        index_ids: list[str] = []
        for document_id in dict.fromkeys(document_ids):
            document = self.documents.get_for_user(user_id, document_id)
            if document is None:
                raise DocumentNotFound(str(document_id))
            if document.status is not DocumentStatus.COMPLETED or document.pageindex_doc_id is None:
                raise DocumentNotReady(f"{document.filename} is {document.status}")
            index_ids.append(document.pageindex_doc_id)
        return index_ids

    def _citations(self, user_id: str, citations: Sequence[IndexCitation]) -> tuple[Citation, ...]:
        """Our IDs for each citation. A citation naming a document outside this user's library is dropped."""
        mapped: list[Citation] = []
        figure_pages: dict[UUID, set[int]] = {}
        for citation in citations:
            document = self.documents.find_by_pageindex_doc_id(user_id, citation.doc_id) if citation.doc_id else None
            if document is None:
                continue
            if document.id not in figure_pages:
                figure_pages[document.id] = {figure.page for figure in self.figures.list_for_document(document.id)}
            mapped.append(Citation(
                index=citation.index, document_id=document.id, filename=document.filename, page=citation.page,
                from_figure=citation.page in figure_pages[document.id],
            ))
        return tuple(mapped)


def _messages(index: UserIndex, index_ids: Sequence[str], request: QueryRequest) -> list[ChatMessage]:
    """Spec 5.6: the document context first (only when documents are selected), then history, then the question."""
    context = [ChatMessage(role="user", content=index.document_context(index_ids))] if index_ids else []
    return [*context, *request.history, ChatMessage(role="user", content=request.question)]


def parse_pages(spec: str, page_count: int, max_pages: int) -> list[int]:
    """Page numbers from "12", "12,13" or "12-13", in the order asked and without repeats.
    Anything else raises ViewPagesRejected with a message the agent can act on."""
    pages: list[int] = []
    for part in spec.replace(" ", "").split(","):
        match = _PAGE_PART.fullmatch(part)
        if match is None:
            raise ViewPagesRejected(f"Pages must look like '12', '12,13' or '12-13', not {spec!r}.")
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= page_count:
            raise ViewPagesRejected(f"This document has pages 1-{page_count}; {part!r} is outside that.")
        if last - first >= max_pages:
            raise ViewPagesRejected(f"At most {max_pages} pages per call.")
        pages.extend(page for page in range(first, last + 1) if page not in pages)
        if len(pages) > max_pages:
            raise ViewPagesRejected(f"At most {max_pages} pages per call.")
    return pages
