"""The OpenAI describer against the real API, on the page the vision spike used (about $0.001).

Opt-in: set RUN_LIVE_TESTS=1. The key comes from .env or the environment and is never printed.
"""
import os
from pathlib import Path

import pytest

from vectorless_rag.config import load_settings
from vectorless_rag.pdf import PyMuPdfPageRenderer, read_page_texts
from vectorless_rag.vision import OpenAiFigureDescriber

REPO = Path(__file__).resolve().parents[2]
I_SERIES = REPO / "Data" / "TD_I-Series I-13_I-16_UsersManual_en-US_1000280.pdf"

pytestmark = [
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="calls the OpenAI API; set RUN_LIVE_TESTS=1"),
    pytest.mark.skipif(not I_SERIES.is_file(), reason="sample PDFs in Data/ not present"),
]


def test_describes_the_calibration_screenshot() -> None:
    """p30: nine calibration points, 3 "Great", 5 "Good", 1 "No data" (top centre); labels only in the image."""
    settings = load_settings(REPO / ".env")
    (png,) = PyMuPdfPageRenderer(settings.render_dpi).render_png(I_SERIES, [30])
    describer = OpenAiFigureDescriber(settings.openai_api_key.get_secret_value(), settings.vision_model)

    description = describer.describe(png, read_page_texts(I_SERIES)[29])

    text = description.text.lower()
    assert "great" in text and "no data" in text
    assert description.input_tokens > 1_000  # the image was sent, not just the text
