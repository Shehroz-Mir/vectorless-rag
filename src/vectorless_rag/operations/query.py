"""Question-answering use case (spec 5.6-5.8, Section 8 "Query").

Checks the selected documents, builds the messages (document context, history, question), runs the
answering agent with the user's read-only PageIndex tools and a view_pages bound to the user, then
maps the answer's citations to our document IDs. Each run is logged by its trace id and totals.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from vectorless_rag.models import (
    AgentRun,
    ChatMessage,
    Citation,
    DocumentStatus,
    IndexCitation,
    PageImage,
    QueryRequest,
    QueryResponse,
    RunLabels,
)
from vectorless_rag.operations.errors import AnswerIncomplete, DocumentNotFound, DocumentNotReady, ViewPagesRejected
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

logger = logging.getLogger(__name__)

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
        user_key = user_key_for(user_id)
        index = self.indexes.for_user(user_key)
        labels = RunLabels(trace_id=uuid4().hex, user_key=user_key)  # before the run, so every record shares it
        try:
            run = self.agent.answer(
                index.agent_instructions(), index.agent_tools(), self.page_viewer(user_id),
                _messages(index, index_ids, request), labels,
            )
        except AnswerIncomplete as error:
            _log_run(labels, error.run, outcome=f"incomplete ({error})")
            raise
        _log_run(labels, run, outcome="answered")
        resolved = index.resolve_citations(run.answer)
        return QueryResponse(answer=resolved.text, citations=self._citations(user_id, resolved.citations), trace_id=labels.trace_id)

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


def _log_run(labels: RunLabels, run: AgentRun | None, outcome: str) -> None:
    """One line per question (agent-runs spec 2.6): the trace id and the totals, never the question or document text."""
    totals = " ".join(f"{name}={value}" for name, value in run.totals.model_dump().items()) if run else "no run"
    logger.info("question %s %s: %s", labels.trace_id, outcome, totals)


def _messages(index: UserIndex, index_ids: Sequence[str], request: QueryRequest) -> list[ChatMessage]:
    """Spec 5.6: the document context first (only when documents are selected), then history, then the question."""
    context = [ChatMessage(role="user", content=index.document_context(index_ids))] if index_ids else []
    return [*context, *request.history, ChatMessage(role="user", content=request.question)]


def page_ranges(spec: str) -> list[tuple[int, int]]:
    """The (first, last) ranges of a page spec: "12", "12,13", "12-13" or a mix such as "1-3,7".
    Only the syntax is checked; anything else raises ViewPagesRejected."""
    ranges: list[tuple[int, int]] = []
    for part in spec.replace(" ", "").split(","):
        match = _PAGE_PART.fullmatch(part)
        if match is None:
            raise ViewPagesRejected(f"Pages must look like '12', '12,13' or '12-13', not {spec!r}.")
        ranges.append((int(match[1]), int(match[2] or match[1])))
    return ranges


def parse_pages(spec: str, page_count: int, max_pages: int) -> list[int]:
    """Page numbers from a page spec, in the order asked and without repeats. A bad spec, or more
    than the document or the limit allows, raises ViewPagesRejected with a message the agent can act on."""
    pages: list[int] = []
    for first, last in page_ranges(spec):
        if not 1 <= first <= last <= page_count:
            part = str(first) if first == last else f"{first}-{last}"
            raise ViewPagesRejected(f"This document has pages 1-{page_count}; {part!r} is outside that.")
        if last - first >= max_pages:
            raise ViewPagesRejected(f"At most {max_pages} pages per call.")
        pages.extend(page for page in range(first, last + 1) if page not in pages)
        if len(pages) > max_pages:
            raise ViewPagesRejected(f"At most {max_pages} pages per call.")
    return pages
