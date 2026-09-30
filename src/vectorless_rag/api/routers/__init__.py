"""HTTP routes; each calls one use case."""
from vectorless_rag.api.routers.documents import router as documents_router
from vectorless_rag.api.routers.query import router as query_router

__all__ = ["documents_router", "query_router"]
