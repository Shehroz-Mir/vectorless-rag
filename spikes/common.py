"""Shared paths and settings for the throwaway spikes.

OPENAI_API_KEY is read from the project .env and never printed.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "Data"  # read-only input PDFs; never written
OUT_DIR = Path(__file__).resolve().parent / "out"

load_dotenv(ROOT / ".env")
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is missing from .env")

INDEX_MODEL = os.environ.get("INDEX_MODEL", "gpt-5.6-luna")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5.6-sol")

TDI_110 = SAMPLES_DIR / "TDI-110_UsersManual_en-US_WEB_1000958-01.pdf"
I_SERIES = SAMPLES_DIR / "TD_I-Series I-13_I-16_UsersManual_en-US_1000280.pdf"
NAVIO = SAMPLES_DIR / "TD_Navio_UsersManual_en-US_1000965-01.pdf"
TD_PILOT = SAMPLES_DIR / "TobiiDynavox_TDPilot_UsersManual_en-GB_1001335-20.pdf"


def out_path(*parts: str) -> Path:
    """A path under spikes/out/, creating parent folders."""
    path = OUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
