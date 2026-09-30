"""The OpenAI describer against the real API, on the page the vision spike used (about $0.001).

Opt-in: set RUN_LIVE_TESTS=1. The key comes from .env and is never printed.
"""
from tests.live import I_SERIES, live_only
from vectorless_rag.config import Settings
from vectorless_rag.pdf import PyMuPdfPageRenderer, read_page_texts
from vectorless_rag.vision import OpenAiFigureDescriber

pytestmark = live_only(I_SERIES)


def test_describes_the_calibration_screenshot(live_settings: Settings) -> None:
    """p30: nine calibration points, 3 "Great", 5 "Good", 1 "No data" (top centre); labels only in the image."""
    settings = live_settings
    (png,) = PyMuPdfPageRenderer(settings.render_dpi).render_png(I_SERIES, [30])
    describer = OpenAiFigureDescriber(settings.openai_api_key.get_secret_value(), settings.vision_model)

    description = describer.describe(png, read_page_texts(I_SERIES)[29])

    text = description.text.lower()
    assert "great" in text and "no data" in text
    assert description.input_tokens > 1_000  # the image was sent, not just the text
