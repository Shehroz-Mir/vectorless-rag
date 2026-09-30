"""pyright proves the pdf/ adapters fit the ports, wired the way the composition root will wire them."""
from functools import partial

from tests.sample_pdfs import DEFAULT_DETECTION_RULES
from vectorless_rag.operations import CountPages, DetectFigurePages, PageRenderer, ReadPageTexts, WriteInvisibleNotes
from vectorless_rag.pdf import (
    count_pages,
    detect_figure_pages,
    PyMuPdfPageRenderer,
    read_page_texts,
    write_invisible_notes,
)


def test_pdf_adapters_satisfy_the_ports() -> None:
    count: CountPages = count_pages
    read: ReadPageTexts = read_page_texts
    detect: DetectFigurePages = partial(detect_figure_pages, rules=DEFAULT_DETECTION_RULES)
    write: WriteInvisibleNotes = write_invisible_notes
    renderer: PageRenderer = PyMuPdfPageRenderer(dpi=170)

    assert all(callable(step) for step in (count, read, detect, write)) and renderer is not None
