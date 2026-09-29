"""Fakes for the rendering, vision, PageIndex and agent ports (see operations/ports.py)."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

from vectorless_rag.models.figures import PageDescription
from vectorless_rag.models.index import IndexCitation, IndexedDocument, ResolvedAnswer
from vectorless_rag.models.query import ChatMessage
from vectorless_rag.operations.errors import DocumentNotFound
from vectorless_rag.operations.ports import PageViewer

_CITE_TAG = re.compile(r'<cite doc="(?P<doc>[^"]+)" page="(?P<page>\d+)"\s*/>')


class FakePageRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[int, ...]]] = []

    def render_png(self, pdf_path: Path, pages: Sequence[int]) -> list[bytes]:
        self.calls.append((pdf_path, tuple(pages)))
        return [f"png:{pdf_path.name}:{page}".encode() for page in pages]


class FakeFigureDescriber:
    def __init__(self, model: str = "fake-vision") -> None:
        self.model = model
        self.calls: list[tuple[bytes, str]] = []

    def describe(self, page_png: bytes, page_text: str) -> PageDescription:
        self.calls.append((page_png, page_text))
        return PageDescription(text=f"A figure seen in {page_png.decode()}.", model=self.model, input_tokens=100, output_tokens=10)


class FakeUserIndex:
    """One user's library: names are made unique the way PageIndex does it (_1, _2, …)."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}  # doc_id -> stored name
        self.submitted: list[Path] = []

    def submit(self, pdf_path: Path) -> IndexedDocument:
        self.submitted.append(pdf_path)
        name, number = pdf_path.name, 1
        while name in self.names.values():
            name = f"{pdf_path.stem}_{number}{pdf_path.suffix}"
            number += 1
        doc_id = f"pi-{uuid4().hex}"
        self.names[doc_id] = name
        return IndexedDocument(doc_id=doc_id, name=name)

    def delete(self, doc_id: str) -> None:
        self.names.pop(doc_id, None)

    def document_context(self, doc_ids: Sequence[str]) -> str:
        missing = [doc_id for doc_id in doc_ids if doc_id not in self.names]
        if missing:
            raise DocumentNotFound(", ".join(missing))
        return "The user has specified documents: " + ", ".join(self.names[doc_id] for doc_id in doc_ids)

    def agent_instructions(self) -> str:
        return "FAKE PAGEINDEX INSTRUCTIONS"

    def agent_tools(self) -> list[Callable[..., str]]:
        return []

    def resolve_citations(self, answer: str) -> ResolvedAnswer:
        ids_by_name = {name: doc_id for doc_id, name in self.names.items()}
        numbers: dict[tuple[str, int], int] = {}

        def number_for(match: re.Match[str]) -> str:
            key = (match["doc"], int(match["page"]))
            numbers.setdefault(key, len(numbers) + 1)
            return f"[{numbers[key]}]"

        text = _CITE_TAG.sub(number_for, answer)
        citations = tuple(
            IndexCitation(index=index, document=doc, doc_id=ids_by_name.get(doc), page=page)
            for (doc, page), index in numbers.items()
        )
        return ResolvedAnswer(text=text, citations=citations)


class FakeUserIndexProvider:
    def __init__(self) -> None:
        self.indexes: dict[str, FakeUserIndex] = {}

    def for_user(self, user_key: str) -> FakeUserIndex:
        return self.indexes.setdefault(user_key, FakeUserIndex())


class FakeAnswerAgent:
    """Returns a scripted reply and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[Callable[..., str]], PageViewer, list[ChatMessage]]] = []

    def answer(
        self,
        instructions: str,
        tools: Sequence[Callable[..., str]],
        view_pages: PageViewer,
        messages: Sequence[ChatMessage],
    ) -> str:
        self.calls.append((instructions, list(tools), view_pages, list(messages)))
        return self.reply
