"""The question-answering use case on in-memory fakes: document checks, messages, view_pages, citations, run labels."""
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from tests.fakes import (
    FakeAnswerAgent,
    FakePageRenderer,
    FakeUserIndexProvider,
    InMemoryDocumentRepository,
    InMemoryFigureRepository,
)
from vectorless_rag.models import (
    AgentRun,
    ChatMessage,
    Citation,
    Document,
    DocumentChanges,
    DocumentStatus,
    FigureKind,
    ModelStep,
    NewDocument,
    NewFigureDescription,
    QueryRequest,
    RunLabels,
)
from vectorless_rag.operations import (
    AnswerIncomplete,
    DocumentNotFound,
    DocumentNotReady,
    PageViewer,
    QuestionAnswering,
    ViewPagesRejected,
    page_ranges,
    parse_pages,
    user_key_for,
)


class Harness:
    def __init__(self, reply: str = "An answer.") -> None:
        self.documents = InMemoryDocumentRepository()
        self.figures = InMemoryFigureRepository()
        self.indexes = FakeUserIndexProvider()
        self.renderer = FakePageRenderer()
        self.agent = FakeAnswerAgent(reply)
        self.qa = QuestionAnswering(
            documents=self.documents, figures=self.figures, indexes=self.indexes,
            renderer=self.renderer, agent=self.agent, view_pages_max_pages=3,
        )

    def upload(self, user_id: str = "alice", name: str = "manual.pdf") -> Document:
        return self.documents.add(NewDocument(
            id=uuid4(), user_id=user_id, filename=name, original_path=Path("originals") / name,
            file_sha256=uuid4().hex * 2, page_count=40,
        ))

    def indexed(self, user_id: str = "alice", name: str = "manual.pdf") -> Document:
        document = self.upload(user_id, name)
        result = self.indexes.for_user(user_key_for(user_id)).submit(Path(name))
        return self.documents.update(document.id, DocumentChanges(
            status=DocumentStatus.COMPLETED, pageindex_doc_id=result.doc_id, pageindex_name=result.name,
        ))


def test_the_agent_gets_context_history_and_question_in_that_order() -> None:
    harness = Harness()
    manual = harness.indexed()
    history = (ChatMessage(role="user", content="Earlier question"), ChatMessage(role="assistant", content="Earlier answer"))

    harness.qa.answer("alice", QueryRequest(question="How hot?", document_ids=(manual.id,), history=history))

    ((instructions, tools, _view_pages, messages, _labels),) = harness.agent.calls
    assert (instructions, tools) == ("FAKE PAGEINDEX INSTRUCTIONS", [])
    assert messages == [
        ChatMessage(role="user", content="The user has specified documents: manual.pdf"),
        *history,
        ChatMessage(role="user", content="How hot?"),
    ]


def test_without_selected_documents_the_whole_library_is_searched() -> None:
    harness = Harness()
    harness.indexed()

    harness.qa.answer("alice", QueryRequest(question="How hot?"))

    assert harness.agent.calls[0][3] == [ChatMessage(role="user", content="How hot?")]


@pytest.mark.parametrize("owner, status, error", [
    ("bob", DocumentStatus.COMPLETED, DocumentNotFound),  # someone else's: same answer as a missing one
    ("alice", DocumentStatus.QUEUED, DocumentNotReady),
    ("alice", DocumentStatus.FAILED, DocumentNotReady),
])
def test_selected_documents_must_be_the_users_and_completed(owner: str, status: DocumentStatus, error: type[Exception]) -> None:
    harness = Harness()
    document = harness.indexed(user_id=owner)
    harness.documents.update(document.id, DocumentChanges(status=status))

    with pytest.raises(error):
        harness.qa.answer("alice", QueryRequest(question="?", document_ids=(document.id,)))
    with pytest.raises(DocumentNotFound):
        harness.qa.answer("alice", QueryRequest(question="?", document_ids=(uuid4(),)))
    assert harness.agent.calls == []


def test_citations_are_mapped_to_our_documents_and_unknown_ones_dropped() -> None:
    harness = Harness(reply=(
        'Hot <cite doc="manual.pdf" page="3"/>, cool <cite doc="manual.pdf" page="9"/>, '
        'made up <cite doc="missing.pdf" page="1"/>, orphan <cite doc="orphan.pdf" page="2"/>.'
    ))
    manual = harness.indexed()
    harness.indexes.for_user(user_key_for("alice")).submit(Path("orphan.pdf"))  # in the index, not in our registry
    harness.figures.add_many([NewFigureDescription(
        document_id=manual.id, page=3, figure_index=1, kind=FigureKind.RASTER, description="A photo.",
        vision_model="fake", input_tokens=1, output_tokens=1,
    )])

    response = harness.qa.answer("alice", QueryRequest(question="?"))

    assert response.answer == "Hot [1], cool [2], made up [3], orphan [4]."
    assert response.citations == (
        Citation(index=1, document_id=manual.id, filename="manual.pdf", page=3, from_figure=True),
        Citation(index=2, document_id=manual.id, filename="manual.pdf", page=9, from_figure=False),
    )
    assert response.trace_id


def test_the_run_is_labelled_before_it_starts_and_the_response_shares_the_trace_id() -> None:
    harness = Harness()
    harness.indexed()

    response = harness.qa.answer("alice", QueryRequest(question="?"))

    labels = harness.agent.calls[0][4]
    assert (labels.trace_id, labels.user_key) == (response.trace_id, user_key_for("alice"))


def test_each_question_logs_its_trace_id_and_totals_but_not_its_text(caplog: pytest.LogCaptureFixture) -> None:
    harness = Harness(reply="The secret answer.")
    harness.indexed()

    with caplog.at_level(logging.INFO, logger="vectorless_rag.operations.query"):
        response = harness.qa.answer("alice", QueryRequest(question="A private question?"))

    (line,) = caplog.messages
    assert line.startswith(f"question {response.trace_id} answered: model_calls=1 tool_calls=0")
    assert "private" not in line and "secret" not in line


class GivesUp:
    def answer(
        self, instructions: str, tools: Sequence[Callable[..., str]], view_pages: PageViewer,
        messages: Sequence[ChatMessage], labels: RunLabels,
    ) -> AgentRun:
        partial = AgentRun.of("", [ModelStep(index=1, duration_ms=1, tool_calls=("view_pages",))], duration_ms=1)
        raise AnswerIncomplete("no answer within 20 steps", run=partial)


def test_an_incomplete_run_is_logged_and_its_error_keeps_the_partial_run(caplog: pytest.LogCaptureFixture) -> None:
    harness = Harness()
    qa = replace(harness.qa, agent=GivesUp())

    with caplog.at_level(logging.INFO, logger="vectorless_rag.operations.query"), pytest.raises(AnswerIncomplete) as raised:
        qa.answer("alice", QueryRequest(question="?"))

    assert raised.value.run is not None and raised.value.run.totals.model_calls == 1
    (line,) = caplog.messages
    assert "incomplete (no answer within 20 steps): model_calls=1" in line


def test_view_pages_renders_the_original_pdf_of_the_users_document() -> None:
    harness = Harness()
    manual = harness.indexed()

    images = harness.qa.page_viewer("alice")("manual.pdf", "3-4")

    assert harness.renderer.calls == [(manual.original_path, (3, 4))]
    assert [(image.doc_name, image.page, image.png) for image in images] == [
        ("manual.pdf", 3, b"png:manual.pdf:3"), ("manual.pdf", 4, b"png:manual.pdf:4"),
    ]


def test_view_pages_cannot_reach_another_users_document() -> None:
    harness = Harness()
    harness.indexed(user_id="bob", name="report.pdf")

    with pytest.raises(ViewPagesRejected, match="no document named 'report.pdf'"):
        harness.qa.page_viewer("alice")("report.pdf", "1")
    assert harness.renderer.calls == []


def test_the_agent_gets_a_view_pages_bound_to_the_asking_user() -> None:
    harness = Harness()
    harness.indexed(user_id="alice", name="manual.pdf")
    harness.indexed(user_id="bob", name="report.pdf")

    harness.qa.answer("alice", QueryRequest(question="?"))
    view_pages = harness.agent.calls[0][2]

    assert [image.page for image in view_pages("manual.pdf", "2")] == [2]
    with pytest.raises(ViewPagesRejected):
        view_pages("report.pdf", "1")


@pytest.mark.parametrize("spec, expected", [
    ("12", [12]),
    ("12,13", [12, 13]),
    (" 13 , 12 ", [13, 12]),
    ("12-14", [12, 13, 14]),
    ("12,12-13", [12, 13]),
])
def test_page_specs(spec: str, expected: list[int]) -> None:
    assert parse_pages(spec, page_count=40, max_pages=3) == expected


@pytest.mark.parametrize("spec, message", [
    ("", "must look like"),
    ("twelve", "must look like"),
    ("12,", "must look like"),
    ("0", "pages 1-40"),
    ("41", "pages 1-40"),
    ("14-12", "pages 1-40"),
    ("1-4", "At most 3"),
    ("1,2,3,4", "At most 3"),
    ("1-100000", "pages 1-40"),
])
def test_bad_page_specs_tell_the_agent_what_is_allowed(spec: str, message: str) -> None:
    with pytest.raises(ViewPagesRejected, match=message):
        parse_pages(spec, page_count=40, max_pages=3)


def test_page_ranges_check_only_the_syntax() -> None:
    assert page_ranges("1-3, 7,90-99") == [(1, 3), (7, 7), (90, 99)]
    with pytest.raises(ViewPagesRejected, match="must look like"):
        page_ranges("1-3,")
