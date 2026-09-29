"""The enriched copy must look identical and carry the notes for both of PageIndex's extractors.

PyPDF2 and pypdfium2 are what PageIndex itself uses (page content and tree/summaries); they are
installed as pageindex dependencies, so checking with them checks what PageIndex will read.
"""
import hashlib
from pathlib import Path

import pymupdf
import pypdfium2
import pytest
from PyPDF2 import PdfReader

from tests.sample_pdfs import build_pdf, drawing, text
from vectorless_rag.models import FigureNote
from vectorless_rag.pdf import write_invisible_notes
from vectorless_rag.pdf.enrichment import to_latin1  # internal helper


BOX = (100.0, 150.0, 450.0, 450.0)
NOTE = FigureNote(page=1, box=BOX, text="[FIGURE DESCRIPTION p1 fig1] Line drawing of the device. Canary: ZEBRA-7731.")


@pytest.fixture
def original(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "manual.pdf", [[text("7 Microphone"), drawing(BOX)], [text("second page")]])


def render(pdf: Path, page: int) -> bytes:
    with pymupdf.open(pdf) as doc:
        return doc[page - 1].get_pixmap(dpi=72).samples


def test_page_looks_identical_after_enrichment(original: Path, tmp_path: Path) -> None:
    target = tmp_path / "enriched" / "manual.pdf"

    write_invisible_notes(original, target, [NOTE])

    assert render(target, 1) == render(original, 1)


def test_both_pageindex_extractors_read_the_note_on_its_own_line(original: Path, tmp_path: Path) -> None:
    target = tmp_path / "enriched" / "manual.pdf"

    write_invisible_notes(original, target, [NOTE])

    pypdf2_text = PdfReader(str(target)).pages[0].extract_text()
    pdfium_text = pypdfium2.PdfDocument(str(target))[0].get_textpage().get_text_range()
    assert "\n[FIGURE DESCRIPTION p1 fig1]" in pypdf2_text
    assert "ZEBRA-7731" in pypdf2_text and "ZEBRA-7731" in pdfium_text


def test_original_is_never_modified(original: Path, tmp_path: Path) -> None:
    before = hashlib.sha256(original.read_bytes()).hexdigest()

    write_invisible_notes(original, tmp_path / "copy.pdf", [NOTE])

    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="original"):
        write_invisible_notes(original, original, [NOTE])


def test_note_for_a_missing_page_is_rejected(original: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page 5"):
        write_invisible_notes(original, tmp_path / "copy.pdf", [FigureNote(page=5, box=BOX, text="x")])


def test_long_note_in_a_small_box_still_gets_written(original: Path, tmp_path: Path) -> None:
    target = tmp_path / "copy.pdf"
    long_note = FigureNote(page=1, box=(100, 150, 130, 170), text="[FIGURE DESCRIPTION p1 fig1] " + "detail " * 150 + "END-MARK")

    write_invisible_notes(original, target, [long_note])

    assert "END-MARK" in PdfReader(str(target)).pages[0].extract_text()


@pytest.mark.parametrize("raw, expected", [
    ("“Great”", '"Great"'),
    ("60 °C – 140 °F", "60 °C - 140 °F"),
    ("arrow → right, 3 × 2", "arrow -> right, 3 × 2"),
    ("ﬁle ™ naïve", "file TM naïve"),
    ("label 中文 end", "label  end"),
])
def test_text_is_reduced_to_what_the_font_can_show(raw: str, expected: str) -> None:
    assert to_latin1(raw) == expected
