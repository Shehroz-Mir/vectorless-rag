"""The FastAPI app (spec 5.1, Section 7).

Run: `uvicorn vectorless_rag.api.main:app` or `python -m vectorless_rag.api`. The services are built
when the app starts, not on import, so importing this module needs no settings.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectorless_rag.api.dependencies import Services, services_from_environment
from vectorless_rag.api.errors import register_error_handlers
from vectorless_rag.api.routers import documents_router, query_router


def create_app(build_services: Callable[[], Services] = services_from_environment) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        services = build_services()
        app.state.services = services
        if services.worker is not None:
            services.worker.start()
        try:
            yield
        finally:
            if services.worker is not None:
                services.worker.stop()  # lets running ingestions finish; queued ones stay queued

    app = FastAPI(title="Vectorless RAG", version="0.1.0", lifespan=lifespan)
    app.include_router(documents_router)
    app.include_router(query_router)
    register_error_handlers(app)
    return app


app = create_app()
