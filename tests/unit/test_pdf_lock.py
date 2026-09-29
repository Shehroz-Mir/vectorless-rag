"""Every pdf/ entry point waits while another thread is inside PyMuPDF (see pdf/mupdf_lock.py)."""
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.sample_pdfs import DEFAULT_DETECTION_RULES, build_pdf, image, text
from vectorless_rag.models import FigureNote
from vectorless_rag.pdf import PyMuPdfPageRenderer, detect_figure_pages, read_page_texts, write_invisible_notes
from vectorless_rag.pdf.mupdf_lock import MUPDF_LOCK  # internal: the tests hold it themselves

CALLS: dict[str, Callable[[Path, Path], object]] = {
    "read_page_texts": lambda pdf, _out: read_page_texts(pdf),
    "detect_figure_pages": lambda pdf, _out: detect_figure_pages(pdf, DEFAULT_DETECTION_RULES),
    "write_invisible_notes": lambda pdf, out: write_invisible_notes(pdf, out / "copy.pdf", [FigureNote(page=1, box=(100, 100, 300, 300), text="note")]),
    "render_png": lambda pdf, _out: PyMuPdfPageRenderer(dpi=72).render_png(pdf, [1]),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_entry_point_waits_for_the_lock(name: str, tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "manual.pdf", [[text("page one"), image((100, 100, 300, 300))]])
    finished = threading.Event()

    def call() -> None:
        CALLS[name](pdf, tmp_path)
        finished.set()

    with MUPDF_LOCK:
        worker = threading.Thread(target=call)
        worker.start()
        assert not finished.wait(0.2), f"{name} ran while another thread held PyMuPDF"
    assert finished.wait(10)
    worker.join()
