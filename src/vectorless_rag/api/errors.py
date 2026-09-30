"""Domain errors as HTTP responses (spec 18). Another user's document is a 404, the same as a missing one."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vectorless_rag.operations import (
    AnswerIncomplete,
    DocumentNotFound,
    DocumentNotReady,
    FileTooLarge,
    TooManyPages,
    UnsupportedFile,
    VectorlessRagError,
)

logger = logging.getLogger(__name__)

STATUS_BY_ERROR: dict[type[VectorlessRagError], int] = {
    DocumentNotFound: 404,
    DocumentNotReady: 409,
    FileTooLarge: 413,
    TooManyPages: 413,
    UnsupportedFile: 415,
    AnswerIncomplete: 504,
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(VectorlessRagError)
    async def domain_error(request: Request, error: VectorlessRagError) -> JSONResponse:
        status = next((STATUS_BY_ERROR[kind] for kind in type(error).__mro__ if kind in STATUS_BY_ERROR), None)
        if status is None:  # a domain error that should never reach a route (e.g. ScannedDocument)
            logger.error("unmapped domain error on %s %s: %r", request.method, request.url.path, error)
            return JSONResponse(status_code=500, content={"detail": "internal error"})
        detail = "document not found" if isinstance(error, DocumentNotFound) else str(error)
        return JSONResponse(status_code=status, content={"detail": detail})
