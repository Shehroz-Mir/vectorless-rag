"""The eval runner and its report on in-memory fakes: the real QuestionAnswering with a scripted agent, a
fake judge and fake page files (no network)."""
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from evals import (
    EvalQuestion,
    Evidence,
    GoldPages,
    Judgement,
    PageText,
    QuestionResult,
    QuestionType,
    RunInfo,
    Verdict,
    passed,
)
from evals.run import EVAL_USER, EvidenceReader, eval_settings, evaluate, results_folder, run_settings, write_results
from tests.fakes import FakeAnswerAgent, FakePageRenderer, FakeUserIndexProvider, InMemoryDocumentRepository, InMemoryFigureRepository
from vectorless_rag.config import load_settings
from vectorless_rag.models import (
    AgentRun,
    ChatMessage,
    Document,
    DocumentChanges,
    DocumentStatus,
    ModelStep,
    NewDocument,
    RunLabels,
    ToolOutcome,
    ToolStep,
)
from vectorless_rag.operations import AnswerIncomplete, PageViewer, QuestionAnswering, user_key_for

QUESTION = EvalQuestion(
    id="tdi110-shutoff-temperature", question="At what temperature does it shut off?", type=QuestionType.TEXT,
    gold=(GoldPages(document="manual.pdf", pages=(27,)),), answer="At 60 °C (140 °F).", key_facts=("60", "140"), evidence="p27",
)
VERDICT = Verdict(
    correctness="correct", correctness_reason="Same facts.", claims_total=2, claims_supported=2,
    faithfulness_reason="Both are on p27.", relevance="yes", relevance_reason="Answers it.",
    refused=False, invents_answer=False, refusal_reason="It answers.",
)
PARTIAL = AgentRun.of("", [ModelStep(index=1, duration_ms=3, tool_calls=("view_pages",))], duration_ms=3)
INFO = RunInfo(
    started_at=datetime(2026, 10, 1, 12, 0), duration_s=90, repeat=1, chat_model="gpt-5.6-sol", judge_model="gpt-5.6-sol",
    index_model="gpt-5.6-luna", vision_model="gpt-5.6-luna", settings={"AGENT_MAX_STEPS": "20"},
)


class FakeJudge:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[tuple[str, tuple[tuple[str, int], ...], Evidence]] = []

    def judge(self, question: EvalQuestion, answer: str, citations: Sequence[tuple[str, int]], evidence: Evidence) -> Judgement:
        self.calls.append((answer, tuple(citations), evidence))
        if self.fails:
            raise RuntimeError("judge down")
        return Judgement(verdict=VERDICT, model="gpt-5.6-sol", input_tokens=1000, output_tokens=100)


class Raises:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def answer(
        self, instructions: str, tools: Sequence[Callable[..., str]], view_pages: PageViewer,
        messages: Sequence[ChatMessage], labels: RunLabels,
    ) -> AgentRun:
        raise self.error


class Library:
    """The eval user's library in memory: manual.pdf indexed, and a page-text reader that records its calls."""

    def __init__(self, reply: str = 'It is 60 °C (140 °F) <cite doc="manual.pdf" page="27"/>.') -> None:
        self.documents, self.figures = InMemoryDocumentRepository(), InMemoryFigureRepository()
        self.indexes, self.renderer = FakeUserIndexProvider(), FakePageRenderer()
        self.agent = FakeAnswerAgent(reply)
        self.manual = self._indexed("manual.pdf")
        self.questions = QuestionAnswering(
            documents=self.documents, figures=self.figures, indexes=self.indexes, renderer=self.renderer,
            agent=self.agent, view_pages_max_pages=3,
        )
        self.text_reads: list[Path] = []
        self.judge = FakeJudge()
        self.evidence = EvidenceReader({"manual.pdf": self.manual}, self.renderer, self._read_texts)

    def _indexed(self, name: str) -> Document:
        document = self.documents.add(NewDocument(
            id=uuid4(), user_id=EVAL_USER, filename=name, original_path=Path("originals") / name,
            file_sha256=uuid4().hex * 2, page_count=40,
        ))
        indexed = self.indexes.for_user(user_key_for(EVAL_USER)).submit(Path(name))
        return self.documents.update(document.id, DocumentChanges(
            status=DocumentStatus.COMPLETED, pageindex_doc_id=indexed.doc_id, pageindex_name=indexed.name,
            enriched_path=Path("enriched") / name,
        ))

    def _read_texts(self, path: Path) -> list[str]:
        self.text_reads.append(path)
        return [f"{path.name} p{page}" for page in range(1, 41)]

    def evaluate(self, questions: QuestionAnswering | None = None, judge: FakeJudge | None = None) -> QuestionResult:
        return evaluate(
            QUESTION, 0, questions=questions or self.questions, judge=judge or self.judge, evidence=self.evidence,
            chat_model="gpt-5.6-sol",
        )


@pytest.fixture
def library() -> Library:
    return Library()


def test_a_question_is_asked_like_a_user_scored_and_judged_on_what_the_agent_read(library: Library) -> None:
    result = library.evaluate()

    assert library.agent.calls[0][3] == [ChatMessage(role="user", content=QUESTION.question)]  # no document_ids
    assert (result.answer, result.citations, result.error) == ("It is 60 °C (140 °F) [1].", (("manual.pdf", 27),), None)
    assert (result.retrieval.page_recall, result.retrieval.hit_at_3, result.generation.key_facts) == (1.0, True, True)
    assert (result.generation.correctness, result.generation.citation_recall) == (1.0, 1.0)
    ((answer, citations, evidence),) = library.judge.calls
    assert (answer, citations) == (result.answer, result.citations)
    assert evidence == Evidence(texts=[PageText("manual.pdf", 27, "manual.pdf p27")], images=[])
    assert library.text_reads == [Path("enriched") / "manual.pdf"]  # what PageIndex indexed
    assert result.judge_cost_usd == pytest.approx(0.006)
    assert passed(result)


def test_an_incomplete_run_keeps_its_steps_and_is_not_judged(library: Library) -> None:
    gives_up = replace(library.questions, agent=Raises(AnswerIncomplete("no answer within 20 steps", run=PARTIAL)))

    result = library.evaluate(questions=gives_up)

    assert (result.error, result.answer, result.run) == ("incomplete: no answer within 20 steps", "", PARTIAL)
    assert library.judge.calls == [] and result.verdict is None
    assert (result.generation.correctness, result.retrieval.page_recall) == (0.0, 0.0)
    assert not passed(result)


def test_an_unexpected_failure_is_recorded_and_the_run_goes_on(library: Library) -> None:
    broken = replace(library.questions, agent=Raises(RuntimeError("model gone")))

    result = library.evaluate(questions=broken)

    assert result.error == "RuntimeError: model gone" and result.run.steps == ()


def test_a_judge_failure_keeps_the_other_scores(library: Library) -> None:
    result = library.evaluate(judge=FakeJudge(fails=True))

    assert (result.judge_error, result.verdict, result.generation.correctness) == ("RuntimeError: judge down", None, None)
    assert (result.generation.key_facts, result.retrieval.page_recall) == (True, 1.0)


def test_evidence_has_images_of_viewed_pages_from_the_original_pdf(library: Library) -> None:
    run = AgentRun.of("", [
        ToolStep(index=1, tool="view_pages", arguments={}, document="manual.pdf", pages=(3, 4), outcome=ToolOutcome.OK, duration_ms=1),
        ToolStep(index=2, tool="view_pages", arguments={}, document="manual.pdf", outcome=ToolOutcome.ERROR, duration_ms=1),
        ToolStep(index=3, tool="get_page_content", arguments={}, document="other.pdf", pages=(1,), outcome=ToolOutcome.OK, duration_ms=1),
    ], duration_ms=3)

    evidence = library.evidence.evidence(run)

    assert [(image.document, image.page, image.png) for image in evidence.images] == [
        ("manual.pdf", 3, b"png:manual.pdf:3"), ("manual.pdf", 4, b"png:manual.pdf:4"),
    ]
    assert library.renderer.calls == [(Path("originals") / "manual.pdf", (3, 4))]
    assert evidence.texts == []  # other.pdf is not in the eval library


def test_a_run_never_overwrites_an_earlier_one(tmp_path: Path) -> None:
    (tmp_path / "2026-10-01").mkdir()
    (tmp_path / "2026-10-01-2").mkdir()

    assert results_folder(tmp_path, datetime(2026, 10, 1).date()) == tmp_path / "2026-10-01-3"
    assert results_folder(tmp_path, datetime(2026, 10, 2).date()) == tmp_path / "2026-10-02"


def test_results_and_summary_are_written(library: Library, tmp_path: Path) -> None:
    good = library.evaluate()
    bad = library.evaluate(questions=replace(library.questions, agent=Raises(AnswerIncomplete("no answer within 20 steps", run=PARTIAL))))

    folder = write_results([good, bad], INFO, root=tmp_path)

    lines = (folder / "results.jsonl").read_bytes().decode("utf-8").splitlines()
    assert [QuestionResult.model_validate_json(line) for line in lines] == [good, bad]
    text = (folder / "summary.md").read_bytes().decode("utf-8")
    assert "\r\n" not in text
    assert text.startswith("# Eval run 2026-10-01 12:00\n\n2 answers (1 questions × 1); 1 passed")
    assert "| Metric | All | text (2) |" in text and "| Correctness (judge) | 50% | 50% |" in text
    assert "| tdi110-shutoff-temperature | text | pass | correct | yes | 100% | 100% | 1 | 1 |" in text
    assert "**fail** | no answer |" in text and "- Error: incomplete: no answer within 20 steps" in text
    assert "## Spot-check these" in text and "- Correctness (correct): Same facts." in text
    assert "| Judging | $0.006 | $0.003 |" in text  # only the answered question was judged


def test_the_eval_library_has_its_own_root_and_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = eval_settings(load_settings(env_file=None), root=tmp_path / "var")

    assert settings.data_root == tmp_path / "var"
    assert settings.database_url == f"sqlite:///{(tmp_path / 'var' / 'eval.db').resolve().as_posix()}"
    assert run_settings(settings)["AGENT_MAX_STEPS"] == str(settings.agent_max_steps)
