"""Page images for the vision model and for view_pages (implements ports.PageRenderer)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pymupdf


class PyMuPdfPageRenderer:
    def __init__(self, dpi: int) -> None:
        self._dpi = dpi

    def render_png(self, pdf_path: Path, pages: Sequence[int]) -> list[bytes]:
        """One PNG per requested 1-based page, in the order asked."""
        with pymupdf.open(pdf_path) as doc:
            missing = [page for page in pages if not 1 <= page <= doc.page_count]
            if missing:
                raise ValueError(f"{pdf_path.name} has {doc.page_count} pages; cannot render {missing}")
            return [doc[page - 1].get_pixmap(dpi=self._dpi).tobytes("png") for page in pages]
