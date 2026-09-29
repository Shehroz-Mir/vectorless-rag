"""Ingestion with the real pdf/ steps and file store on a generated PDF; only vision and PageIndex are fakes.

PyPDF2 is the reader PageIndex uses for page content, so the check below is what the agent will see.
"""
from functools import partial
from pathlib import Path
from uuid import uuid4

from PyPDF2 import PdfReader

from tests.fakes import FakeUserIndexProvider, InMemoryDocumentRepository, InMemoryFigureRepository, RecordingLocks
from tests.sample_pdfs import DEFAULT_DETECTION_RULES, build_pdf, image, text
from vectorless_rag.models import DocumentStatus, NewDocument, PageDescription
from vectorless_rag.operations import DocumentIngestion, IngestionRules, user_key_for
from vectorless_rag.pdf import PyMuPdfPageRenderer, detect_figure_pages, read_page_texts, write_invisible_notes
from vectorless_rag.storage import LocalFileStore

TEXT = "Body text of a manual page, long enough to count as a real text layer."


class CanaryDescriber:
    def __init__(self) -> None:
        self.pngs: list[bytes] = []

    def describe(self, page_png: bytes, page_text: str) -> PageDescription:
        self.pngs.append(page_png)
        return PageDescription(text="Photo of the device, front view. Canary: ZEBRA-7731.", model="fake", input_tokens=1, output_tokens=1)


def test_the_enriched_copy_carries_the_description_under_the_original_name(tmp_path: Path) -> None:
    upload = build_pdf(tmp_path / "upload.pdf", [[text(TEXT)], [text(TEXT), image((100, 150, 450, 450))]])
    documents, figures, files = InMemoryDocumentRepository(), InMemoryFigureRepository(), LocalFileStore(tmp_path / "var")
    indexes, describer = FakeUserIndexProvider(), CanaryDescriber()
    document_id, user_key = uuid4(), user_key_for("alice")
    original = files.save_original(user_key, document_id, "Manual v2.pdf", upload.read_bytes())
    document = documents.add(NewDocument(
        id=document_id, user_id="alice", filename="Manual v2.pdf", original_path=original,
        file_sha256="a" * 64, page_count=2,
    ))
    ingestion = DocumentIngestion(
        documents=documents, figures=figures, files=files,
        read_page_texts=read_page_texts,
        detect_figure_pages=partial(detect_figure_pages, rules=DEFAULT_DETECTION_RULES),
        renderer=PyMuPdfPageRenderer(dpi=72),
        describer=describer,
        write_invisible_notes=write_invisible_notes,
        indexes=indexes, locks=RecordingLocks(),
        rules=IngestionRules(max_figure_pages=200, vision_concurrency=2, scanned_max_text_chars=50, scanned_page_share=0.5),
    )

    result = ingestion.ingest(document)

    assert result is not None and result.status is DocumentStatus.COMPLETED and result.enriched_path is not None
    assert result.enriched_path.name == original.name == "Manual v2.pdf"
    assert result.enriched_path.parent.name == "enriched"
    assert [png[:4] for png in describer.pngs] == [b"\x89PNG"]  # one real render: page 2 only
    pages = PdfReader(str(result.enriched_path)).pages
    assert "FIGURE DESCRIPTION" not in pages[0].extract_text()
    assert "[FIGURE DESCRIPTION p2 fig1] Photo of the device" in pages[1].extract_text()
    assert "ZEBRA-7731" in pages[1].extract_text()
    assert "FIGURE DESCRIPTION" not in PdfReader(str(original)).pages[1].extract_text()
    assert result.pageindex_name == "Manual v2.pdf"
    assert indexes.for_user(user_key).submitted == [result.enriched_path]
