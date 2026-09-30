"""The PageIndex adapter for real, on two pages cut from TDI-110 (the index model's calls cost about a cent).

Opt-in: set RUN_LIVE_TESTS=1. The key is removed from the process environment first, so a pass also
shows it reaches PageIndex only through the adapter.
"""
import json
import os
from pathlib import Path

import pytest

from tests.live import TDI_110, live_only
from tests.sample_pdfs import cut_pages
from vectorless_rag.config import Settings
from vectorless_rag.indexing import PageIndexClientPool
from vectorless_rag.operations import DocumentNotFound, user_key_for

pytestmark = live_only(TDI_110)


def test_index_read_cite_isolate_and_delete(live_settings: Settings, tmp_path: Path) -> None:
    """Page 2 of the cut is TDI-110 p27: it shuts itself off at 60 °C (140 °F) (Spike C)."""
    pdf = cut_pages(TDI_110, 26, 27, tmp_path / "upload" / "TDI-110 safety.pdf")
    pool = PageIndexClientPool(
        tmp_path / "var", api_key=live_settings.openai_api_key.get_secret_value(),
        model=live_settings.index_model, summary_concurrency=4,
    )
    alice, bob = pool.for_user(user_key_for("alice")), pool.for_user(user_key_for("bob"))

    indexed = alice.submit(pdf)

    assert indexed.name == "TDI-110 safety.pdf" and indexed.doc_id.startswith("pi-")
    alice_tools = {tool.__name__: tool for tool in alice.agent_tools()}
    page_two = alice_tools["get_page_content"](doc_name=indexed.name, pages="2")
    assert "60" in page_two and "140" in page_two
    assert indexed.name in alice.document_context([indexed.doc_id])
    answer = f'It shuts off at 60 °C <cite doc="{indexed.name}" page="2"/>.'
    resolved = alice.resolve_citations(answer)
    assert resolved.text == "It shuts off at 60 °C [1]."
    assert [(c.doc_id, c.page) for c in resolved.citations] == [(indexed.doc_id, 2)]

    # Bob's library knows nothing of Alice's document.
    bob_tools = {tool.__name__: tool for tool in bob.agent_tools()}
    assert json.loads(bob_tools["browse_documents"]())["documents"] == []
    assert "error" in json.loads(bob_tools["get_page_content"](doc_name=indexed.name, pages="2"))
    assert [c.doc_id for c in bob.resolve_citations(answer).citations] == [None]
    with pytest.raises(DocumentNotFound):
        bob.document_context([indexed.doc_id])

    alice.delete(indexed.doc_id)
    alice.delete(indexed.doc_id)  # already gone: not an error
    with pytest.raises(DocumentNotFound):
        alice.document_context([indexed.doc_id])
    key_in_environment = "OPENAI_API_KEY" in os.environ  # a bool, so a failure never prints the environment
    assert not key_in_environment
