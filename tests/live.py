"""Shared setup for the opt-in live tests, which call the OpenAI API (set RUN_LIVE_TESTS=1)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "Data"  # read-only input
I_SERIES = SAMPLES / "TD_I-Series I-13_I-16_UsersManual_en-US_1000280.pdf"
TDI_110 = SAMPLES / "TDI-110_UsersManual_en-US_WEB_1000958-01.pdf"


def live_only(*samples: Path) -> list[pytest.MarkDecorator]:
    """Marks for a live test module: run only with RUN_LIVE_TESTS=1 and when its sample PDFs are present."""
    return [
        pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="calls the OpenAI API; set RUN_LIVE_TESTS=1"),
        pytest.mark.skipif(not all(sample.is_file() for sample in samples), reason="sample PDFs in Data/ not present"),
    ]
