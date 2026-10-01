"""Composition root of the service (spec 18): every adapter is built here, once, at start-up.

The ingestion worker runs in this process and shares the registry, file store, PageIndex pool and
per-user locks with the API (spec 5.4). Routes get the use cases through the Depends providers below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from vectorless_rag.agent import AgentRules, LangChainAnswerAgent, create_chat_model
from vectorless_rag.config import Settings, load_settings
from vectorless_rag.db import (
    SqlDocumentRepository,
    SqlFigureRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from vectorless_rag.indexing import PageIndexClientPool
from vectorless_rag.operations import DocumentLibrary, QuestionAnswering
from vectorless_rag.pdf import PyMuPdfPageRenderer, count_pages
from vectorless_rag.storage import LocalFileStore
from vectorless_rag.worker import IngestionWorker, PerUserLocks, build_ingestion_worker


@dataclass(frozen=True)
class Services:
    library: DocumentLibrary
    questions: QuestionAnswering
    worker: IngestionWorker | None = None  # None when a test drives the use cases without a worker


def build_services(settings: Settings) -> Services:
    engine = create_database_engine(settings.database_url)
    create_schema(engine)
    sessions = create_session_factory(engine)
    documents, figures = SqlDocumentRepository(sessions), SqlFigureRepository(sessions)
    files = LocalFileStore(settings.data_root)
    renderer = PyMuPdfPageRenderer(settings.render_dpi)
    api_key = settings.openai_api_key.get_secret_value()
    indexes = PageIndexClientPool(
        settings.data_root, api_key=api_key, model=settings.index_model,
        summary_concurrency=settings.index_summary_concurrency,
    )
    locks = PerUserLocks()
    worker = build_ingestion_worker(
        settings, documents=documents, figures=figures, files=files, renderer=renderer, indexes=indexes, locks=locks,
    )
    agent = LangChainAnswerAgent(
        create_chat_model(
            api_key, settings.chat_model, settings.agent_timeout_s, reasoning_summary=settings.agent_reasoning_summary,
        ),
        AgentRules(
            max_steps=settings.agent_max_steps,
            view_pages_max_calls=settings.view_pages_max_calls,
            max_image_sets=settings.max_image_sets_in_context,
            image_detail=settings.view_pages_image_detail,
            timeout_s=settings.agent_timeout_s,
        ),
    )
    return Services(
        library=DocumentLibrary(
            documents=documents, figures=figures, files=files, indexes=indexes, locks=locks,
            count_pages=count_pages, on_queued=worker.wake,
            max_upload_bytes=settings.max_upload_bytes, max_pages=settings.max_pages,
        ),
        questions=QuestionAnswering(
            documents=documents, figures=figures, indexes=indexes, renderer=renderer, agent=agent,
            view_pages_max_pages=settings.view_pages_max_pages,
        ),
        worker=worker,
    )


def services_from_environment() -> Services:
    """The default for the app: settings from the environment and .env."""
    return build_services(load_settings())


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_library(services: Annotated[Services, Depends(get_services)]) -> DocumentLibrary:
    return services.library


def get_questions(services: Annotated[Services, Depends(get_services)]) -> QuestionAnswering:
    return services.questions


MAX_USER_ID_LENGTH = 255  # the registry's user_id column


def current_user(x_user_id: Annotated[str | None, Header(max_length=MAX_USER_ID_LENGTH)] = None) -> str:
    """The user id the upstream auth layer puts in the X-User-Id header; it is trusted as it is (spec 5.1)."""
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(status_code=401, detail="missing X-User-Id header")
    return x_user_id


CurrentUser = Annotated[str, Depends(current_user)]
Library = Annotated[DocumentLibrary, Depends(get_library)]
Questions = Annotated[QuestionAnswering, Depends(get_questions)]
