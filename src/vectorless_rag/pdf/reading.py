"""Text layer of a PDF (implements ports.ReadPageTexts)."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from vectorless_rag.pdf.mupdf_lock import with_mupdf_lock


def page_text(page: pymupdf.Page, clip: pymupdf.Rect | None = None) -> str:
    """Plain text of a page, or of the part inside `clip`."""
    # get_text() is typed as a union of all its output formats; "text" always returns str.
    return str(page.get_text("text", clip=clip)).strip()


@with_mupdf_lock
def read_page_texts(pdf_path: Path) -> list[str]:
    """Text of every page, in page order."""
    with pymupdf.open(pdf_path) as doc:
        return [page_text(page) for page in doc.pages()]
