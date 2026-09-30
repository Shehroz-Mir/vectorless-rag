"""One user's PageIndex library (implements ports.UserIndex; spec 5.5, 5.6, 5.8).

Wraps a local-mode PageIndexClient bound to that user's storage_path, so its tools, document
context and citations only ever see that user's documents (Spike C).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pageindex import PageIndexAPIError, PageIndexClient

from vectorless_rag.models import IndexCitation, IndexedDocument, ResolvedAnswer
from vectorless_rag.operations import DocumentNotFound

# resolve_citations() writes [[1]](#pageindex-citation-01); the API shows a plain [1] (spec Section 7).
_CITATION_LINK = re.compile(r"\[\[(\d+)\]\]\(#pageindex-citation-\d+\)")


class PageIndexUserIndex:
    def __init__(self, client: PageIndexClient) -> None:
        self._client = client

    def submit(self, pdf_path: Path) -> IndexedDocument:
        """Local mode indexes synchronously: the document is ready when this returns (spec 5.4)."""
        result = self._client.submit_document(str(pdf_path))
        return IndexedDocument(doc_id=result["doc_id"], name=result["name"])

    def delete(self, doc_id: str) -> None:
        try:
            self._client.delete_document(doc_id)
        except PageIndexAPIError:
            if self._has(doc_id):
                raise  # a real failure, not "already gone"

    def document_context(self, doc_ids: Sequence[str]) -> str:
        try:
            return self._client.document_context(list(doc_ids))
        except PageIndexAPIError as error:
            raise DocumentNotFound(str(error)) from error

    def agent_instructions(self) -> str:
        return self._client.agent_instructions() + "\n\n" + self._client.citation_prompt()

    def agent_tools(self) -> list[Callable[..., str]]:
        return self._client.agent_tools(include_management=False)  # read-only: the agent never changes the library

    def resolve_citations(self, answer: str) -> ResolvedAnswer:
        resolved = self._client.resolve_citations(answer)
        return ResolvedAnswer(
            text=_CITATION_LINK.sub(r"[\1]", resolved["answer"]),
            citations=tuple(
                IndexCitation(index=entry["index"], document=entry["document"], doc_id=entry["doc_id"], page=entry["page"])
                for entry in resolved["citations"]
            ),
        )

    def _has(self, doc_id: str) -> bool:
        try:
            self._client.get_document(doc_id)
        except PageIndexAPIError:
            return False
        return True
