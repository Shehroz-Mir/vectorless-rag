"""Fixtures for the integration tests."""
import pytest

from tests.live import REPO
from vectorless_rag.config import Settings, load_settings


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings from the repo's .env. The key is then removed from the process environment, so a
    passing live test shows each SDK got it only as an argument."""
    settings = load_settings(REPO / ".env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return settings
