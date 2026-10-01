"""The eval's metric functions on hand-made runs (agent-runs spec 4.3, 4.4), and the eval set file itself."""
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals import (
    EvalQuestion,
    GoldPages,
    QuestionType,
    Verdict,
    citation_scores,
    cost_usd,
    generation_scores,
    key_facts_present,
    load_questions,
    navigation_cost,
    normalise,
    pages_read,
    retrieval_scores,
    select,
)
from vectorless_rag.models import AgentRun, ModelStep, ToolOutcome, ToolStep

SAMPLES = Path(__file__).resolve().parents[2] / "Data"


def question(
    kind: QuestionType = QuestionType.TEXT,
    gold: dict[str, tuple[int, ...]] | None = None,
    figure_pages: tuple[int, ...] = (),
    key_facts: tuple[str, ...] = (),
) -> EvalQuestion:
    gold = {"a.pdf": (3, 4)} if gold is None else gold
    return EvalQuestion(
        id="q", question="?", type=kind, gold=tuple(GoldPages(document=d, pages=p) for d, p in gold.items()),
        figure_pages=figure_pages, answer="Reference.", key_facts=key_facts, evidence="e",
    )


UNANSWERABLE = question(QuestionType.UNANSWERABLE, gold={})


def call(index: int, document: str | None, *pages: int, tool: str = "get_page_content", outcome: ToolOutcome = ToolOutcome.OK) -> ToolStep:
    return ToolStep(
        index=index, tool=tool, arguments={}, document=document,
        pages=pages if outcome is ToolOutcome.OK else (), outcome=outcome, duration_ms=1,
    )


def run(*steps: ToolStep) -> AgentRun:
    return AgentRun.of("Answer.", [ModelStep(index=len(steps) + 1, duration_ms=1, input_tokens=1000, output_tokens=100), *steps], 50)


def verdict(**changes: object) -> Verdict:
    fields: dict[str, object] = {
        "correctness": "correct", "correctness_reason": "Same facts.", "claims_total": 4, "claims_supported": 3,
        "faithfulness_reason": "One claim is not on the pages.", "relevance": "yes", "relevance_reason": "On topic.",
        "refused": False, "invents_answer": False, "refusal_reason": "It answers.",
    }
    return Verdict.model_validate(fields | changes)


# ── retrieval ──

def test_recall_precision_hit_and_mrr_follow_the_reading_order() -> None:
    scores = retrieval_scores(question(), run(call(1, "b.pdf", 1), call(2, "a.pdf", 2), call(3, "a.pdf", 3)))

    assert (scores.page_recall, scores.page_precision, scores.hit_at_3, scores.mrr) == (0.5, pytest.approx(1 / 3), True, pytest.approx(1 / 3))
    assert (scores.document_hit, scores.first_document_hit) == (True, False)
    assert scores.dead_ends == 2  # two reads without a gold page


def test_a_gold_page_read_fourth_misses_hit_at_3() -> None:
    scores = retrieval_scores(question(), run(call(1, "a.pdf", 1, 2), call(2, "a.pdf", 5), call(3, "a.pdf", 4)))

    assert (scores.hit_at_3, scores.mrr, scores.page_precision) == (False, 0.25, 0.25)


def test_pages_read_twice_or_as_text_and_image_count_once() -> None:
    steps = run(call(1, "a.pdf", 3), call(2, "a.pdf", 3, tool="view_pages"), call(3, "a.pdf", 3, 4))

    assert pages_read(steps) == [("a.pdf", 3), ("a.pdf", 4)]
    assert retrieval_scores(question(), steps).page_precision == 1.0


def test_failed_calls_are_dead_ends_and_read_nothing() -> None:
    steps = run(
        call(1, None, tool="browse_documents"),
        call(2, "a.pdf", tool="get_page_content", outcome=ToolOutcome.ERROR),
        call(3, "a.pdf", tool="view_pages", outcome=ToolOutcome.BLOCKED),
    )

    scores = retrieval_scores(question(), steps)

    assert (scores.page_recall, scores.page_precision, scores.hit_at_3, scores.mrr) == (0.0, None, False, 0.0)
    assert (scores.document_hit, scores.first_document_hit, scores.dead_ends) == (True, True, 2)  # targeted counts as opened


def test_only_browsing_opens_no_document() -> None:
    scores = retrieval_scores(question(), run(call(1, None, tool="browse_documents")))

    assert (scores.document_hit, scores.first_document_hit) == (False, False)


def test_a_figure_hit_needs_the_figure_page_viewed_as_an_image() -> None:
    figure = question(QuestionType.FIGURE, gold={"a.pdf": (30,)}, figure_pages=(30,))

    as_text = retrieval_scores(figure, run(call(1, "a.pdf", 30)))
    as_image = retrieval_scores(figure, run(call(1, "a.pdf", 30, tool="view_pages")))

    assert (as_text.figure_hit, as_image.figure_hit) == (False, True)
    assert retrieval_scores(question(), run(call(1, "a.pdf", 3, tool="view_pages"))).figure_hit is None


def test_unanswerable_questions_count_only_failed_calls() -> None:
    scores = retrieval_scores(UNANSWERABLE, run(call(1, "a.pdf", 1), call(2, "a.pdf", outcome=ToolOutcome.ERROR)))

    assert scores.model_dump() == {
        "page_recall": None, "page_precision": None, "hit_at_3": None, "mrr": None, "document_hit": None,
        "first_document_hit": None, "figure_hit": None, "dead_ends": 1,
    }


def test_navigation_cost_prices_the_run() -> None:
    cost = navigation_cost(run(call(1, "a.pdf", 3)), "gpt-5.6-sol")

    assert (cost.tool_calls, cost.pages_read, cost.model_calls, cost.input_tokens, cost.output_tokens) == (1, 1, 1, 1000, 100)
    assert cost.cost_usd == pytest.approx(1000 * 4 / 1e6 + 100 * 20 / 1e6)
    assert navigation_cost(run(), "some-other-model").cost_usd is None
    assert cost_usd("gpt-5.6-luna", 1_000_000, 1_000_000) == pytest.approx(1.40)


# ── generation ──

def test_key_facts_ignore_case_spaces_degrees_and_curly_quotes() -> None:
    facts = question(key_facts=("60°C", "140", "it’s on"))

    assert normalise("It's  60 °C\nON") == "it's60con"
    assert key_facts_present(facts, "IT'S ON at 60 C (140 °F).") is True
    assert key_facts_present(facts, "It’s on at 60 °C.") is False  # 140 is missing


def test_a_key_fact_can_have_alternatives() -> None:
    facts = question(key_facts=("3|three", "top"))

    assert key_facts_present(facts, "Three are Great; No data is at the top.") is True
    assert key_facts_present(facts, "Two are Great; No data is at the top.") is False
    assert key_facts_present(question(), "anything") is None


def test_citation_precision_and_recall() -> None:
    assert citation_scores(question(), [("a.pdf", 3), ("a.pdf", 3), ("b.pdf", 1)]) == (0.5, 0.5)
    assert citation_scores(question(), []) == (None, 0.0)
    assert citation_scores(UNANSWERABLE, [("a.pdf", 1)]) == (None, None)


def test_the_judges_verdict_becomes_scores() -> None:
    scores = generation_scores(question(key_facts=("60",)), "It is 60 °C [1].", [("a.pdf", 3)], verdict())

    assert (scores.key_facts, scores.correctness, scores.faithfulness, scores.relevance) == (True, 1.0, 0.75, 1.0)
    assert (scores.citation_precision, scores.citation_recall) == (1.0, 0.5)
    assert (scores.false_refusal, scores.refusal_correct) == (False, None)


@pytest.mark.parametrize("correctness, relevance, expected", [("partly", "partly", (0.5, 0.5)), ("wrong", "no", (0.0, 0.0))])
def test_partial_and_wrong_grades(correctness: str, relevance: str, expected: tuple[float, float]) -> None:
    scores = generation_scores(question(), "Answer.", [], verdict(correctness=correctness, relevance=relevance))

    assert (scores.correctness, scores.relevance) == expected


def test_faithfulness_needs_claims_and_never_exceeds_one() -> None:
    assert generation_scores(question(), "Hello.", [], verdict(claims_total=0, claims_supported=0)).faithfulness is None
    assert generation_scores(question(), "A.", [], verdict(claims_total=2, claims_supported=5)).faithfulness == 1.0


@pytest.mark.parametrize("refused, invents, expected", [(True, False, True), (True, True, False), (False, True, False)])
def test_refusals_of_unanswerable_questions(refused: bool, invents: bool, expected: bool) -> None:
    scores = generation_scores(UNANSWERABLE, "Not in the manual.", [], verdict(refused=refused, invents_answer=invents))

    assert (scores.refusal_correct, scores.false_refusal, scores.citation_recall) == (expected, None, None)


def test_refusing_an_answerable_question_is_a_false_refusal() -> None:
    assert generation_scores(question(), "I could not find it.", [], verdict(refused=True)).false_refusal is True


def test_a_run_without_an_answer_scores_zero() -> None:
    answerable = generation_scores(question(key_facts=("60",)), "", [], None)
    unanswerable = generation_scores(UNANSWERABLE, "", [], None)

    assert (answerable.correctness, answerable.relevance, answerable.key_facts, answerable.false_refusal) == (0.0, 0.0, False, False)
    assert unanswerable.refusal_correct is False


def test_an_answer_the_judge_could_not_grade_has_no_judged_scores() -> None:
    scores = generation_scores(question(key_facts=("60",)), "It is 60.", [("a.pdf", 3)], None)

    assert (scores.correctness, scores.faithfulness, scores.relevance, scores.false_refusal) == (None, None, None, None)
    assert (scores.key_facts, scores.citation_recall) == (True, 0.5)


# ── the eval set ──

def test_the_eval_set_has_the_agreed_mix_over_the_sample_manuals() -> None:
    questions = load_questions()

    assert len(questions) == 24
    assert Counter(q.type for q in questions) == {
        QuestionType.TEXT: 8, QuestionType.FIGURE: 8, QuestionType.MULTI_PAGE: 3,
        QuestionType.CROSS_DOCUMENT: 2, QuestionType.UNANSWERABLE: 3,
    }
    documents = {gold.document for q in questions for gold in q.gold}
    assert documents == {pdf.name for pdf in SAMPLES.glob("*.pdf")}
    assert all(q.gold_figure_pages for q in questions if q.type is QuestionType.FIGURE)
    assert all(len(q.gold_documents) == 2 for q in questions if q.type is QuestionType.CROSS_DOCUMENT)
    assert all(key_facts_present(q, q.answer) is not False for q in questions)  # each reference passes its own key facts


def test_gold_pages_must_fit_the_question_type() -> None:
    with pytest.raises(ValidationError, match="no gold pages"):
        question(QuestionType.UNANSWERABLE)
    with pytest.raises(ValidationError, match="no gold pages"):
        question(gold={})
    with pytest.raises(ValidationError, match="figure pages must be gold"):
        question(QuestionType.FIGURE, figure_pages=(9,))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    entry = "- {id: q, question: '?', type: text, gold: [{document: a.pdf, pages: [1]}], answer: A, evidence: e}\n"
    path = tmp_path / "questions.yaml"
    path.write_text(entry * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate question ids: q"):
        load_questions(path)


def test_select_keeps_the_file_order_and_rejects_unknown_ids() -> None:
    questions = load_questions()

    assert [q.id for q in select(questions, ["pilot-dwell-time", "tdi110-shutoff-temperature"])] == [
        "tdi110-shutoff-temperature", "pilot-dwell-time",
    ]
    assert select(questions, []) == questions
    with pytest.raises(ValueError, match="unknown question ids: nope"):
        select(questions, ["nope"])
