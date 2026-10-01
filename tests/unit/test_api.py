"""The HTTP API over the use cases, with in-memory fakes behind them (no database, no network)."""
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from uuid import UUID, uuid4

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.fakes import (
    FakeAnswerAgent,
    FakePageRenderer,
    FakeUserIndexProvider,
    InMemoryDocumentRepository,
    InMemoryFigureRepository,
    InMemoryFileStore,
    RecordingLocks,
)
from vectorless_rag.api import Services, create_app
from vectorless_rag.models import (
    AgentRun,
    ChatMessage,
    DocumentChanges,
    DocumentStatus,
    FigureKind,
    NewFigureDescription,
    RunLabels,
)
from vectorless_rag.operations import (
    AnswerAgent,
    AnswerIncomplete,
    DocumentLibrary,
    PageViewer,
    QuestionAnswering,
    user_key_for,
)

PDF = b"%PDF-1.7 fake body"
ALICE = {"X-User-Id": "alice"}
BOB = {"X-User-Id": "bob"}


class GivesUp:
    def answer(
        self, instructions: str, tools: Sequence[Callable[..., str]], view_pages: PageViewer,
        messages: Sequence[ChatMessage], labels: RunLabels,
    ) -> AgentRun:
        raise AnswerIncomplete("no answer within 20 steps")


class Api:
    """The app wired to fakes, plus direct access to them for setting up and checking."""

    def __init__(self, reply: str = "Answer.") -> None:
        self.documents, self.figures = InMemoryDocumentRepository(), InMemoryFigureRepository()
        self.indexes = FakeUserIndexProvider()
        self.agent = FakeAnswerAgent(reply)
        self.wakes = 0
        self.library = DocumentLibrary(
            documents=self.documents, figures=self.figures, files=InMemoryFileStore(), indexes=self.indexes,
            locks=RecordingLocks(), count_pages=lambda _data: 3, on_queued=self.wake,
            max_upload_bytes=1_000, max_pages=50,
        )

    def wake(self) -> None:
        self.wakes += 1

    def client(self, agent: AnswerAgent | None = None) -> TestClient:
        questions = QuestionAnswering(
            documents=self.documents, figures=self.figures, indexes=self.indexes, renderer=FakePageRenderer(),
            agent=agent or self.agent, view_pages_max_pages=3,
        )
        return TestClient(create_app(lambda: Services(library=self.library, questions=questions)))

    def upload(self, client: TestClient, headers: dict[str, str] = ALICE, data: bytes = PDF, name: str = "manual.pdf") -> httpx2.Response:
        return client.post("/documents", headers=headers, files={"file": (name, data, "application/pdf")})

    def complete(self, document_id: str, user_id: str = "alice", name: str = "manual.pdf") -> None:
        indexed = self.indexes.for_user(user_key_for(user_id)).submit(Path(name))
        self.documents.update(UUID(document_id), DocumentChanges(
            status=DocumentStatus.COMPLETED, pageindex_doc_id=indexed.doc_id, pageindex_name=indexed.name,
        ))


@pytest.fixture
def api() -> Api:
    return Api()


@pytest.fixture
def client(api: Api) -> Iterator[TestClient]:
    with api.client() as client:
        yield client


def test_upload_returns_202_then_200_for_the_same_file(api: Api, client: TestClient) -> None:
    first = api.upload(client)
    again = api.upload(client, name="renamed.pdf")

    assert first.status_code == 202 and again.status_code == 200
    assert again.json()["id"] == first.json()["id"]
    assert sorted(first.json()) == ["created_at", "error", "figure_page_count", "filename", "id", "page_count", "status", "updated_at"]
    assert (first.json()["status"], first.json()["filename"], api.wakes) == ("queued", "manual.pdf", 1)


@pytest.mark.parametrize("data, status, detail", [
    (b"<html>no</html>", 415, "the file is not a PDF"),
    (b"%PDF-" + b"x" * 1_000, 413, "the file is too large; the limit is 0.000954 MB"),
])
def test_bad_uploads_get_a_clear_status(api: Api, client: TestClient, data: bytes, status: int, detail: str) -> None:
    response = api.upload(client, data=data)

    assert (response.status_code, response.json()) == (status, {"detail": detail})


def test_every_route_needs_the_user_header(client: TestClient) -> None:
    for method, path in [("GET", "/documents"), ("GET", f"/documents/{uuid4()}"), ("POST", "/query")]:
        response = client.request(method, path, json={"question": "?"})
        assert (response.status_code, response.json()) == (401, {"detail": "missing X-User-Id header"})


def test_a_user_id_longer_than_the_registry_column_is_rejected(client: TestClient) -> None:
    assert client.get("/documents", headers={"X-User-Id": "u" * 256}).status_code == 422
    assert client.get("/documents", headers={"X-User-Id": "u" * 255}).status_code == 200


def test_users_only_see_their_own_documents(api: Api, client: TestClient) -> None:
    mine = api.upload(client).json()
    api.upload(client, headers=BOB, data=PDF + b"bob")

    assert [d["id"] for d in client.get("/documents", headers=ALICE).json()] == [mine["id"]]
    assert client.get(f"/documents/{mine['id']}", headers=ALICE).json()["id"] == mine["id"]
    for path in (f"/documents/{mine['id']}", f"/documents/{mine['id']}/figures"):
        response = client.get(path, headers=BOB)
        assert (response.status_code, response.json()) == (404, {"detail": "document not found"})
    assert client.delete(f"/documents/{mine['id']}", headers=BOB).status_code == 404


def test_figures_are_listed(api: Api, client: TestClient) -> None:
    document = api.upload(client).json()
    api.figures.add_many([NewFigureDescription(
        document_id=UUID(document["id"]), page=2, figure_index=1, kind=FigureKind.VECTOR,
        description="A line drawing.", vision_model="gpt-5.6-luna", input_tokens=3200, output_tokens=210,
    )])

    figures = client.get(f"/documents/{document['id']}/figures", headers=ALICE).json()

    assert figures == [{
        "page": 2, "figure_index": 1, "kind": "vector", "description": "A line drawing.",
        "vision_model": "gpt-5.6-luna", "input_tokens": 3200, "output_tokens": 210,
    }]


def test_delete_returns_204_and_the_document_is_gone(api: Api, client: TestClient) -> None:
    document = api.upload(client).json()

    assert client.delete(f"/documents/{document['id']}", headers=ALICE).status_code == 204
    assert client.get(f"/documents/{document['id']}", headers=ALICE).status_code == 404


def test_a_question_gets_an_answer_with_citations() -> None:
    api = Api(reply='It is 60 °C <cite doc="manual.pdf" page="27"/>.')
    with api.client() as client:
        document = api.upload(client).json()
        api.complete(document["id"])

        response = client.post("/query", headers=ALICE, json={
            "question": "How hot?", "document_ids": [document["id"]],
            "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        })

    body = response.json()
    assert response.status_code == 200 and body["answer"] == "It is 60 °C [1]."
    assert body["citations"] == [{"index": 1, "document_id": document["id"], "filename": "manual.pdf", "page": 27, "from_figure": False}]
    assert body["trace_id"]


def test_a_brief_answer_has_stats_but_no_steps(client: TestClient) -> None:
    body = client.post("/query", headers=ALICE, json={"question": "How hot?"}).json()

    assert set(body) == {"answer", "citations", "trace_id", "stats"}
    assert body["stats"] == {
        "model_calls": 2, "tool_calls": 1, "pages_read": 1, "images_viewed": 0,
        "input_tokens": 300, "output_tokens": 30, "reasoning_tokens": 0, "duration_ms": 12,
    }


def test_a_full_answer_also_lists_the_steps(client: TestClient) -> None:
    body = client.post("/query", headers=ALICE, json={"question": "How hot?", "detail": "full"}).json()

    assert set(body) == {"answer", "citations", "trace_id", "stats", "steps"}
    assert [(step["index"], step["kind"]) for step in body["steps"]] == [(1, "model"), (2, "tool"), (3, "model")]
    assert body["steps"][1] == {
        "kind": "tool", "index": 2, "tool": "get_page_content", "arguments": {"doc_name": "manual.pdf", "pages": "27"},
        "document": "manual.pdf", "pages": [27], "outcome": "ok", "result_preview": '{"success": true}', "duration_ms": 1,
    }


def test_an_unknown_detail_is_422(client: TestClient) -> None:
    response = client.post("/query", headers=ALICE, json={"question": "?", "detail": "everything"})

    assert response.status_code == 422


def test_a_question_about_a_document_still_being_indexed_is_409(api: Api, client: TestClient) -> None:
    document = api.upload(client).json()

    response = client.post("/query", headers=ALICE, json={"question": "?", "document_ids": [document["id"]]})

    assert response.status_code == 409


def test_an_agent_that_runs_out_of_steps_is_504(api: Api) -> None:
    with api.client(agent=GivesUp()) as client:
        response = client.post("/query", headers=ALICE, json={"question": "?"})

    assert (response.status_code, response.json()) == (504, {"detail": "no answer within 20 steps"})


def test_the_openapi_schema_builds(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/documents", "/documents/{document_id}", "/documents/{document_id}/figures", "/query"}
