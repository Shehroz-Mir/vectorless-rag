"""Write figure descriptions into a copy of a PDF as invisible text (spec 5.3d; implements
ports.WriteInvisibleNotes). Verified in Spike A: pages render unchanged, and both of PageIndex's
extractors (PyPDF2 for page content, pdfium for the tree) read the text.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path

import pymupdf

from vectorless_rag.models import FigureNote

INVISIBLE = 3  # PDF text render mode 3: neither filled nor stroked
FONT = "helv"  # a base-14 font: nothing to embed, but it covers Latin-1 only
START_FONT_SIZE = 6.0
MIN_FONT_SIZE = 2.0
FONT_STEP = 0.5
PAGE_MARGIN = 36.0

# Common characters outside Latin-1 that vision models write; others fall back to NFKD or are dropped.
_LATIN1_REPLACEMENTS = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "−": "-", "•": "-", "…": "...",
    "→": "->", "←": "<-", "↑": "^", "↓": "v", "≥": ">=", "≤": "<=", "≈": "~",
    " ": " ", " ": " ",
})


def write_invisible_notes(original: Path, target: Path, notes: Sequence[FigureNote]) -> None:
    """Copy `original` to `target`, writing each note invisibly inside its figure box."""
    if target.resolve() == original.resolve():
        raise ValueError(f"refusing to write over the original PDF: {original}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(original) as doc:
        for note in notes:
            if not 1 <= note.page <= doc.page_count:
                raise ValueError(f"note for page {note.page}, but {original.name} has {doc.page_count} pages")
            # The leading newline keeps PyPDF2 from gluing the marker to the page's last line.
            _write_invisible(doc[note.page - 1], note.page, pymupdf.Rect(note.box), "\n" + to_latin1(note.text))
        doc.save(target, garbage=3, deflate=True)


def to_latin1(text: str) -> str:
    """What the base-14 font can show; anything else would be written as '?'."""
    kept: list[str] = []
    for char in text.translate(_LATIN1_REPLACEMENTS):
        if ord(char) <= 0xFF:
            kept.append(char)
            continue
        decomposed = unicodedata.normalize("NFKD", char)
        kept.append("".join(c for c in decomposed if ord(c) <= 0xFF and not unicodedata.combining(c)))
    return "".join(kept)


def _write_invisible(page: pymupdf.Page, page_number: int, box: pymupdf.Rect, text: str) -> None:
    """Inside the figure box if it fits at a readable size, else anywhere on the page."""
    page_area = page.rect + (PAGE_MARGIN, PAGE_MARGIN, -PAGE_MARGIN, -PAGE_MARGIN)
    for area in (box & page.rect, page_area):
        if _insert_at_largest_fitting_size(page, area, text):
            return
    raise ValueError(f"description too long for page {page_number}: {len(text)} chars")


def _insert_at_largest_fitting_size(page: pymupdf.Page, area: pymupdf.Rect, text: str) -> bool:
    size = START_FONT_SIZE
    while size >= MIN_FONT_SIZE:
        # insert_textbox writes nothing and returns a negative number when the text does not fit.
        if page.insert_textbox(area, text, fontsize=size, fontname=FONT, render_mode=INVISIBLE) >= 0:
            return True
        size -= FONT_STEP
    return False
