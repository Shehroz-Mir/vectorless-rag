"""Retrieval and generation metrics (agent-runs spec 4.3, 4.4): pure functions of a question and its run.

"Read" a page: get_page_content returned its text, or view_pages showed its image (decision 6: the pages
the agent actually got). "Opened" a document: any tool call targeted it. Unanswerable questions have no
gold pages, so they count only for navigation cost and refusals.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.config import cost_usd
from evals.questions import EvalQuestion, PageRef, QuestionType
from vectorless_rag.models import VIEW_PAGES_TOOL, AgentRun, ToolOutcome, ToolStep

GRADES = {"correct": 1.0, "partly": 0.5, "wrong": 0.0}
RELEVANCE = {"yes": 1.0, "partly": 0.5, "no": 0.0}
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class Verdict(BaseModel):
    """What the judge decides about one answer (evals/judge.py asks for exactly these fields)."""

    model_config = ConfigDict(frozen=True)

    correctness: Literal["correct", "partly", "wrong"]
    correctness_reason: str
    claims_total: int  # factual claims in the answer
    claims_supported: int  # of those, supported by what the agent saw
    faithfulness_reason: str
    relevance: Literal["yes", "partly", "no"]
    relevance_reason: str
    refused: bool  # the answer says the documents do not contain the answer
    invents_answer: bool  # the answer states an answer the evidence does not support
    refusal_reason: str


class RetrievalScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_recall: float | None = None  # None: unanswerable
    page_precision: float | None = None  # None: also when no page was read
    hit_at_3: bool | None = None
    mrr: float | None = None
    document_hit: bool | None = None
    first_document_hit: bool | None = None
    figure_hit: bool | None = None  # figure questions only
    dead_ends: int  # tool calls that ended error or blocked, plus page reads without a gold page


class NavigationCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_calls: int
    pages_read: int
    images_viewed: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    duration_ms: int
    cost_usd: float | None  # None: the chat model is not in the price table


class GenerationScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    key_facts: bool | None  # None: the question has no key facts
    correctness: float | None  # 1 / 0.5 / 0; None: the judge failed
    faithfulness: float | None  # share of claims supported; None: no claims, or not judged
    relevance: float | None
    citation_precision: float | None  # None: nothing cited, or unanswerable
    citation_recall: float | None  # None: unanswerable
    refusal_correct: bool | None  # unanswerable questions only
    false_refusal: bool | None  # answerable questions only


class QuestionResult(BaseModel):
    """One question asked once: a line of results.jsonl."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: QuestionType
    repeat: int  # 0 for the first run of this question
    question: str
    reference: str  # the eval set's reference answer
    answer: str  # with numbered citation markers; empty when the run ended without one
    citations: tuple[PageRef, ...]
    error: str | None  # why there is no answer (AnswerIncomplete or another failure)
    run: AgentRun
    retrieval: RetrievalScores
    navigation: NavigationCost
    generation: GenerationScores
    verdict: Verdict | None
    judge_error: str | None
    judge_cost_usd: float | None


def tool_steps(run: AgentRun) -> list[ToolStep]:
    return [step for step in run.steps if isinstance(step, ToolStep)]


def pages_read(run: AgentRun) -> list[PageRef]:
    """Unique (document, page) pairs the agent read as text or saw as an image, in reading order."""
    seen: dict[PageRef, None] = {}
    for step in tool_steps(run):
        if step.outcome is ToolOutcome.OK and step.document is not None:
            for page in step.pages:
                seen.setdefault((step.document, page))
    return list(seen)


def pages_viewed(run: AgentRun) -> set[PageRef]:
    return {
        (step.document, page)
        for step in tool_steps(run)
        if step.tool == VIEW_PAGES_TOOL and step.outcome is ToolOutcome.OK and step.document is not None
        for page in step.pages
    }


def documents_opened(run: AgentRun) -> list[str]:
    """Unique documents any tool call targeted, in the order first targeted."""
    return list(dict.fromkeys(step.document for step in tool_steps(run) if step.document is not None))


def retrieval_scores(question: EvalQuestion, run: AgentRun) -> RetrievalScores:
    failed_calls = sum(1 for step in tool_steps(run) if step.outcome is not ToolOutcome.OK)
    if not question.answerable:
        return RetrievalScores(dead_ends=failed_calls)
    gold = question.gold_pages
    read = pages_read(run)
    first_gold = next((position for position, ref in enumerate(read, start=1) if ref in gold), None)
    opened = documents_opened(run)
    reads_without_gold = sum(
        1 for step in tool_steps(run)
        if step.outcome is ToolOutcome.OK and step.pages and not any((step.document, page) in gold for page in step.pages)
    )
    return RetrievalScores(
        page_recall=len(gold.intersection(read)) / len(gold),
        page_precision=len(gold.intersection(read)) / len(read) if read else None,
        hit_at_3=any(ref in gold for ref in read[:3]),
        mrr=1 / first_gold if first_gold else 0.0,
        document_hit=any(document in question.gold_documents for document in opened),
        first_document_hit=bool(opened) and opened[0] in question.gold_documents,
        figure_hit=bool(pages_viewed(run) & question.gold_figure_pages) if question.type is QuestionType.FIGURE else None,
        dead_ends=failed_calls + reads_without_gold,
    )


def navigation_cost(run: AgentRun, chat_model: str) -> NavigationCost:
    totals = run.totals
    return NavigationCost(**totals.model_dump(), cost_usd=cost_usd(chat_model, totals.input_tokens, totals.output_tokens))


def normalise(text: str) -> str:
    """Case, spaces, ° and curly quotes do not matter when looking for key facts: "60 °C", "60°C" and
    "60 c" all become "60c"."""
    return re.sub(r"\s+", "", text.lower().replace("°", "").translate(_QUOTES))


def key_facts_present(question: EvalQuestion, answer: str) -> bool | None:
    """All key facts are in the answer; "a|b" means either a or b counts."""
    if not question.key_facts:
        return None
    text = normalise(answer)
    return all(any(normalise(option) in text for option in fact.split("|")) for fact in question.key_facts)


def citation_scores(question: EvalQuestion, cited: Sequence[PageRef]) -> tuple[float | None, float | None]:
    """(precision, recall): cited pages that are gold ÷ cited pages, gold pages cited ÷ gold pages."""
    if not question.answerable:
        return None, None
    unique, gold = set(cited), question.gold_pages
    precision = len(unique & gold) / len(unique) if unique else None
    return precision, len(unique & gold) / len(gold)


def generation_scores(
    question: EvalQuestion, answer: str, cited: Sequence[PageRef], verdict: Verdict | None,
) -> GenerationScores:
    """Scores from the answer itself and, when it was judged, from the judge's verdict. A run that ended
    without an answer scores 0 for correctness and relevance, and is not a correct refusal."""
    precision, recall = citation_scores(question, cited)
    key_facts = key_facts_present(question, answer)
    if not answer.strip():
        return GenerationScores(
            key_facts=key_facts, correctness=0.0, faithfulness=None, relevance=0.0,
            citation_precision=precision, citation_recall=recall,
            refusal_correct=None if question.answerable else False,
            false_refusal=False if question.answerable else None,
        )
    if verdict is None:
        return GenerationScores(
            key_facts=key_facts, correctness=None, faithfulness=None, relevance=None,
            citation_precision=precision, citation_recall=recall, refusal_correct=None, false_refusal=None,
        )
    supported = min(max(verdict.claims_supported, 0), verdict.claims_total)
    return GenerationScores(
        key_facts=key_facts,
        correctness=GRADES[verdict.correctness],
        faithfulness=supported / verdict.claims_total if verdict.claims_total > 0 else None,
        relevance=RELEVANCE[verdict.relevance],
        citation_precision=precision,
        citation_recall=recall,
        refusal_correct=None if question.answerable else verdict.refused and not verdict.invents_answer,
        false_refusal=verdict.refused if question.answerable else None,
    )
