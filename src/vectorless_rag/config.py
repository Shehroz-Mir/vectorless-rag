"""Service settings, read from the environment and `.env` (spec Section 10).

Only the composition roots (`api/`, `worker/`) create `Settings` and hand values to the adapters.
Reading `.env` here does not put the OpenAI key into `os.environ`, so adapters get it passed in
explicitly instead of the SDKs finding it on their own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Ratio = Annotated[float, Field(ge=0, le=1)]
ModelName = Annotated[str, Field(min_length=1)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore", frozen=True)

    openai_api_key: Annotated[SecretStr, Field(min_length=1)]

    # Models (spec 10): tree building, ingestion-time figure descriptions, the answering agent.
    index_model: ModelName = "gpt-5.6-luna"
    vision_model: ModelName = "gpt-5.6-luna"
    chat_model: ModelName = "gpt-5.6-sol"  # must read images and call tools via the Responses API
    index_summary_concurrency: PositiveInt = 8

    # Page images, for enrichment and view_pages.
    render_dpi: Annotated[int, Field(ge=50, le=400)] = 170
    view_pages_image_detail: Literal["low", "high", "auto"] = "high"

    # Figure-page detection (spec 5.3a).
    min_image_area_ratio: Ratio = 0.03
    min_graphic_cluster_ratio: Ratio = 0.01
    max_cluster_text_density: PositiveFloat = 5.0
    min_vector_figure_area: Ratio = 0.02
    max_figure_pages: PositiveInt = 200
    vision_concurrency: PositiveInt = 4

    # Scanned-document check (spec 9.5).
    scanned_max_text_chars: NonNegativeInt = 50
    scanned_page_share: Ratio = 0.5

    # Ingestion worker (spec 5.4): documents enriched and indexed at the same time.
    ingestion_workers: PositiveInt = 2

    # view_pages and agent limits.
    view_pages_max_pages: PositiveInt = 3
    view_pages_max_calls: PositiveInt = 4
    max_image_sets_in_context: PositiveInt = 2
    agent_max_steps: PositiveInt = 20
    agent_timeout_s: PositiveFloat = 120

    # Storage and uploads. Data/ holds read-only sample PDFs; app files live under data_root.
    data_root: Path = Path("var")
    database_url: str = "sqlite:///./var/app.db"
    max_upload_mb: PositiveInt = 50
    max_pages: PositiveInt = 500

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Settings from the environment, plus `env_file` when given (environment wins)."""
    # Values come from the environment, which pyright cannot see; this is the one place that relies on it.
    return Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
