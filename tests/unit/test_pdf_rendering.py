from pathlib import Path

import pymupdf
import pytest

from tests.sample_pdfs import build_pdf, drawing, text
from vectorless_rag.operations.ports import PageRenderer
from vectorless_rag.pdf.rendering import PyMuPdfPageRenderer


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "manual.pdf", [[text("page one")], [drawing((100, 100, 400, 400))], [text("page three")]])


def test_renders_requested_pages_as_png_in_order(pdf: Path) -> None:
    renderer: PageRenderer = PyMuPdfPageRenderer(dpi=72)

    images = renderer.render_png(pdf, [3, 1])

    assert len(images) == 2 and all(png.startswith(b"\x89PNG") for png in images)
    assert images[0] != images[1]


def test_resolution_follows_the_dpi(pdf: Path) -> None:
    [png] = PyMuPdfPageRenderer(dpi=144).render_png(pdf, [1])

    pixmap = pymupdf.Pixmap(png)
    assert (pixmap.width, pixmap.height) == (1190, 1684)  # A4 595 × 842 pt at 2× (144 / 72)


def test_pages_outside_the_document_are_rejected(pdf: Path) -> None:
    with pytest.raises(ValueError, match=r"has 3 pages; cannot render \[0, 4\]"):
        PyMuPdfPageRenderer(dpi=72).render_png(pdf, [0, 1, 4])
