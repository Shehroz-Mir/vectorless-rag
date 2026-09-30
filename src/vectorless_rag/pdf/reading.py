"""Text layer and page count of a PDF (implements ports.ReadPageTexts and ports.CountPages)."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from vectorless_rag.operations import UnsupportedFile
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


@with_mupdf_lock
def count_pages(data: bytes) -> int:
    """Pages of an uploaded file. Raises UnsupportedFile unless it is a readable PDF without a password."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as error:  # also covers an empty file
        raise UnsupportedFile("the file is not a readable PDF") from error
    with doc:
        if doc.needs_pass:
            raise UnsupportedFile("password-protected PDFs are not supported")
        if doc.page_count < 1:
            raise UnsupportedFile("the PDF has no pages")
        return doc.page_count
