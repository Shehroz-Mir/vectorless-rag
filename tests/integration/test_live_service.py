"""The whole service over HTTP with every real part: SQLite, files on disk, the worker, OpenAI vision,
PageIndex and the answering agent. Upload two TDI-110 pages, wait until they are indexed, ask the
shut-off temperature (60 °C / 140 °F, p27 = page 2 of the cut), and check another user sees nothing.

About 10 cents. Opt-in: set RUN_LIVE_TESTS=1.
"""
import os
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.live import TDI_110, live_only
from tests.sample_pdfs import cut_pages
from vectorless_rag.api import build_services, create_app
from vectorless_rag.config import Settings

pytestmark = live_only(TDI_110)

ALICE = {"X-User-Id": "alice@example.com"}
BOB = {"X-User-Id": "bob@example.com"}
INDEXING_TIMEOUT_S = 300


def wait_until_done(client: TestClient, document_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + INDEXING_TIMEOUT_S
    document = client.get(f"/documents/{document_id}", headers=ALICE).json()
    while document["status"] not in ("completed", "failed"):
        if time.monotonic() > deadline:
            raise AssertionError(f"still {document['status']} after {INDEXING_TIMEOUT_S} s")
        time.sleep(1)
        document = client.get(f"/documents/{document_id}", headers=ALICE).json()
    return document


def test_upload_index_and_ask_over_http(live_settings: Settings, tmp_path: Path) -> None:
    settings = live_settings.model_copy(update={
        "data_root": tmp_path / "var", "database_url": f"sqlite:///{tmp_path / 'app.db'}",
    })
    pdf = cut_pages(TDI_110, 26, 27, tmp_path / "upload" / "TDI-110 safety.pdf")

    with TestClient(create_app(lambda: build_services(settings))) as client:
        upload = client.post("/documents", headers=ALICE, files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
        assert upload.status_code == 202, upload.text
        document = wait_until_done(client, upload.json()["id"])
        assert document["status"] == "completed", document["error"]

        answer = client.post("/query", headers=ALICE, json={
            "question": "At what temperature does the TD I-110 shut itself off to avoid harm? Give °C and °F.",
            "document_ids": [document["id"]],
        })
        body = answer.json()
        assert answer.status_code == 200, answer.text
        assert "60" in body["answer"] and "140" in body["answer"], body["answer"]
        assert any(c["document_id"] == document["id"] and c["page"] == 2 for c in body["citations"]), body

        assert client.get(f"/documents/{document['id']}", headers=BOB).status_code == 404
        assert client.get("/documents", headers=BOB).json() == []
        bob_answer = client.post("/query", headers=BOB, json={"question": "What temperature shuts the TD I-110 off?"})
        assert bob_answer.status_code == 200 and bob_answer.json()["citations"] == []

    key_in_environment = "OPENAI_API_KEY" in os.environ  # a bool, so a failure never prints the environment
    assert not key_in_environment
