"""PyMuPDF adapter: page counts and texts, figure detection, invisible notes, page rendering (spec 5.3)."""
from vectorless_rag.pdf.detection import DetectionRules, detect_figure_pages
from vectorless_rag.pdf.enrichment import write_invisible_notes
from vectorless_rag.pdf.reading import count_pages, read_page_texts
from vectorless_rag.pdf.rendering import PyMuPdfPageRenderer

__all__ = [
    "DetectionRules",
    "PyMuPdfPageRenderer",
    "count_pages",
    "detect_figure_pages",
    "read_page_texts",
    "write_invisible_notes",
]
