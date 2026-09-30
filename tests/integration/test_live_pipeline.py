"""Upload to indexed with every real part: SQLite registry, file store, PDF steps, OpenAI vision,
PageIndex and the worker wiring (spec 13, "enrichment check"). Two figure pages cut from TDI-110,
about a cent in all.

Opt-in: set RUN_LIVE_TESTS=1. The key is removed from the process environment first.
"""
import hashlib
import os
from pathlib import Path
from uuid import uuid4

from tests.live import TDI_110, live_only
from tests.sample_pdfs import cut_pages
from vectorless_rag.config import Settings
from vectorless_rag.db import (
    SqlDocumentRepository,
    SqlFigureRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from vectorless_rag.indexing import PageIndexClientPool
from vectorless_rag.models import DocumentStatus, FigureKind, NewDocument
from vectorless_rag.operations import user_key_for
from vectorless_rag.pdf import PyMuPdfPageRenderer
from vectorless_rag.storage import LocalFileStore
from vectorless_rag.worker import PerUserLocks, build_ingestion_worker

pytestmark = live_only(TDI_110)


def test_an_upload_is_enriched_and_indexed_with_its_figure_descriptions(live_settings: Settings, tmp_path: Path) -> None:
    """TDI-110 p13 (a photo) and p14 (a line drawing) become pages 1 and 2 of the cut (Spike D)."""
    settings = live_settings
    upload = cut_pages(TDI_110, 13, 14, tmp_path / "upload" / "TDI-110 views.pdf")
    engine = create_database_engine(f"sqlite:///{tmp_path / 'app.db'}")
    create_schema(engine)
    sessions = create_session_factory(engine)
    documents, figures = SqlDocumentRepository(sessions), SqlFigureRepository(sessions)
    files = LocalFileStore(tmp_path / "var")
    api_key = settings.openai_api_key.get_secret_value()
    indexes = PageIndexClientPool(tmp_path / "var", api_key=api_key, model=settings.index_model, summary_concurrency=4)
    worker = build_ingestion_worker(
        settings, documents=documents, figures=figures, files=files,
        renderer=PyMuPdfPageRenderer(settings.render_dpi), indexes=indexes, locks=PerUserLocks(),
    )
    user_key, document_id, data = user_key_for("alice"), uuid4(), upload.read_bytes()
    original = files.save_original(user_key, document_id, upload.name, data)
    documents.add(NewDocument(
        id=document_id, user_id="alice", filename=upload.name, original_path=original,
        file_sha256=hashlib.sha256(data).hexdigest(), page_count=2,
    ))

    assert worker.dispatch_queued() == 1
    worker.stop(wait=True)

    document = documents.get_for_user("alice", document_id)
    assert document is not None
    assert document.status is DocumentStatus.COMPLETED, document.error
    assert (document.figure_page_count, document.pageindex_name) == (2, "TDI-110 views.pdf")
    assert [(f.page, f.kind) for f in figures.list_for_document(document_id)] == [(1, FigureKind.RASTER), (2, FigureKind.VECTOR)]
    tools = {tool.__name__: tool for tool in indexes.for_user(user_key).agent_tools()}
    content = tools["get_page_content"](doc_name=document.pageindex_name, pages="1-2")
    assert "[FIGURE DESCRIPTION p1 fig1]" in content and "[FIGURE DESCRIPTION p2 fig1]" in content
    key_in_environment = "OPENAI_API_KEY" in os.environ  # a bool, so a failure never prints the environment
    assert not key_in_environment
