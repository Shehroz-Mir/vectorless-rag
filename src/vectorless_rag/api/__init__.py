"""Entry point: FastAPI app and routers; a composition root (spec 5.1, Section 7)."""
from vectorless_rag.api.dependencies import Services, build_services
from vectorless_rag.api.main import create_app

__all__ = ["Services", "build_services", "create_app"]
