"""Spec 13, image delivery check (required): page images must reach the main agent.

I-Series p30 is a calibration-result screenshot: 3 points "Great", "No data" at the top centre, and
those labels are not in the text layer (Spike B). Pages 29-31 are indexed as they are, without figure
descriptions, so the agent can only answer by looking at the page image. A silent failure makes the
model invent an answer, so the control shows it a blank image and it must say it cannot see the answer.

Real PageIndex and the real answering model; about 10-20 cents. Opt-in: set RUN_LIVE_TESTS=1.
"""
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest

from tests.fakes import InMemoryDocumentRepository, InMemoryFigureRepository
from tests.live import I_SERIES, live_only
from tests.sample_pdfs import cut_pages
from vectorless_rag.agent import AgentRules, LangChainAnswerAgent, create_chat_model
from vectorless_rag.config import Settings
from vectorless_rag.indexing import PageIndexClientPool
from vectorless_rag.models import DocumentChanges, DocumentStatus, NewDocument, QueryRequest, QueryResponse
from vectorless_rag.operations import PageRenderer, QuestionAnswering, user_key_for
from vectorless_rag.pdf import PyMuPdfPageRenderer

pytestmark = live_only(I_SERIES)

QUESTION = (
    "Use view_pages to look at the calibration result screen on page 2. How many calibration points are "
    "rated 'Great', and where on the screen is the point that shows 'No data'?"
)
CANNOT_SEE = (
    "can't", "cannot", "unable", "not able", "not visible", "blank", "empty",
    "does not show", "doesn't show", "no visible", "could not", "couldn't",
)


def plain(answer: str) -> str:
    """Lower case, with the model's curly apostrophes (U+2019, as in "can’t") made straight."""
    return answer.lower().replace("’", "'")


class BlankPages:
    """The control: every page image is plain white."""

    def render_png(self, pdf_path: Path, pages: Sequence[int]) -> list[bytes]:
        blank = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 600, 800), False)
        blank.clear_with(255)
        return [blank.tobytes("png") for _ in pages]


class CalibrationLibrary:
    """Alice's library with the three original pages, indexed for real, and a way to ask about it."""

    def __init__(self, settings: Settings, tmp_path: Path) -> None:
        api_key = settings.openai_api_key.get_secret_value()
        pdf = cut_pages(I_SERIES, 29, 31, tmp_path / "upload" / "I-Series calibration.pdf")
        self.documents, self.figures = InMemoryDocumentRepository(), InMemoryFigureRepository()
        self.indexes = PageIndexClientPool(tmp_path / "var", api_key=api_key, model=settings.index_model, summary_concurrency=4)
        indexed = self.indexes.for_user(user_key_for("alice")).submit(pdf)  # the original: no figure descriptions
        document = self.documents.add(NewDocument(
            id=uuid4(), user_id="alice", filename=pdf.name, original_path=pdf, file_sha256="c" * 64, page_count=3,
        ))
        self.document = self.documents.update(document.id, DocumentChanges(
            status=DocumentStatus.COMPLETED, pageindex_doc_id=indexed.doc_id, pageindex_name=indexed.name,
        ))
        self.settings = settings
        self.agent = LangChainAnswerAgent(
            create_chat_model(api_key, settings.chat_model, settings.agent_timeout_s),
            AgentRules(
                max_steps=settings.agent_max_steps, view_pages_max_calls=settings.view_pages_max_calls,
                max_image_sets=settings.max_image_sets_in_context, image_detail=settings.view_pages_image_detail,
                timeout_s=settings.agent_timeout_s,
            ),
        )

    def ask(self, renderer: PageRenderer) -> QueryResponse:
        qa = QuestionAnswering(
            documents=self.documents, figures=self.figures, indexes=self.indexes, renderer=renderer,
            agent=self.agent, view_pages_max_pages=self.settings.view_pages_max_pages,
        )
        return qa.answer("alice", QueryRequest(question=QUESTION, document_ids=(self.document.id,)))


@pytest.fixture
def library(live_settings: Settings, tmp_path: Path) -> CalibrationLibrary:
    return CalibrationLibrary(live_settings, tmp_path)


def test_the_agent_reads_the_answer_from_the_page_image(library: CalibrationLibrary) -> None:
    response = library.ask(PyMuPdfPageRenderer(library.settings.render_dpi))

    text = plain(response.answer)
    assert ("3" in text or "three" in text) and "great" in text, response.answer
    assert "top" in text and any(word in text for word in ("centre", "center", "middle")), response.answer
    assert any(c.document_id == library.document.id and c.page == 2 and not c.from_figure for c in response.citations)
    key_in_environment = "OPENAI_API_KEY" in os.environ  # a bool, so a failure never prints the environment
    assert not key_in_environment


def test_control_with_a_blank_image_the_agent_says_it_cannot_see(library: CalibrationLibrary) -> None:
    response = library.ask(BlankPages())

    text = plain(response.answer)
    assert any(phrase in text for phrase in CANNOT_SEE), response.answer
    assert not ("great" in text and "top" in text and ("3" in text or "three" in text)), response.answer
