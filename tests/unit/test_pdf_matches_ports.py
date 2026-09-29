"""pyright proves the pdf/ adapters fit the ports, wired the way the composition root will wire them."""
from functools import partial

from vectorless_rag.operations.ports import DetectFigurePages, PageRenderer, ReadPageTexts, WriteInvisibleNotes
from vectorless_rag.pdf.detection import DetectionRules, detect_figure_pages
from vectorless_rag.pdf.enrichment import write_invisible_notes
from vectorless_rag.pdf.reading import read_page_texts
from vectorless_rag.pdf.rendering import PyMuPdfPageRenderer


def test_pdf_adapters_satisfy_the_ports() -> None:
    rules = DetectionRules(min_image_area_ratio=0.03, min_graphic_cluster_ratio=0.01, max_cluster_text_density=5.0, min_vector_figure_area=0.02)

    read: ReadPageTexts = read_page_texts
    detect: DetectFigurePages = partial(detect_figure_pages, rules=rules)
    write: WriteInvisibleNotes = write_invisible_notes
    renderer: PageRenderer = PyMuPdfPageRenderer(dpi=170)

    assert all(callable(step) for step in (read, detect, write)) and renderer is not None
