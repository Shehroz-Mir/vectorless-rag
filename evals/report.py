"""summary.md (agent-runs spec 4.5): every metric overall and per question type, a row per question, the
failures with the judge's reasons, a few passes to spot-check the judge, and cost and time."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from evals.metrics import QuestionResult
from evals.questions import QuestionType

Value = float | bool | None
SPOT_CHECKS = 5
ANSWER_CHARS = 600


class RunInfo(BaseModel):
    """What the run was, so two summaries can be compared."""

    model_config = ConfigDict(frozen=True)

    started_at: datetime
    duration_s: float
    repeat: int
    chat_model: str
    judge_model: str
    index_model: str
    vision_model: str
    settings: dict[str, str]  # the limits that shape a run, e.g. AGENT_MAX_STEPS


def _rate(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _decimal(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def _one_place(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _whole(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def _usd(value: float | None) -> str:
    return "–" if value is None else f"${value:.3f}"


# (label, value of one result, how to show the average)
METRICS: list[tuple[str, Callable[[QuestionResult], Value], Callable[[float | None], str]]] = [
    ("Page recall", lambda r: r.retrieval.page_recall, _rate),
    ("Page precision", lambda r: r.retrieval.page_precision, _rate),
    ("Hit@3", lambda r: r.retrieval.hit_at_3, _rate),
    ("MRR", lambda r: r.retrieval.mrr, _decimal),
    ("Document hit rate", lambda r: r.retrieval.document_hit, _rate),
    ("First document was gold", lambda r: r.retrieval.first_document_hit, _rate),
    ("Figure hit rate", lambda r: r.retrieval.figure_hit, _rate),
    ("Dead ends per question", lambda r: r.retrieval.dead_ends, _one_place),
    ("Key facts present", lambda r: r.generation.key_facts, _rate),
    ("Correctness (judge)", lambda r: r.generation.correctness, _rate),
    ("Faithfulness", lambda r: r.generation.faithfulness, _rate),
    ("Answer relevance", lambda r: r.generation.relevance, _rate),
    ("Citation precision", lambda r: r.generation.citation_precision, _rate),
    ("Citation recall", lambda r: r.generation.citation_recall, _rate),
    ("Refusal accuracy (unanswerable)", lambda r: r.generation.refusal_correct, _rate),
    ("False refusals (answerable)", lambda r: r.generation.false_refusal, _rate),
    ("No answer (run incomplete)", lambda r: r.error is not None, _rate),
    ("Tool calls", lambda r: r.navigation.tool_calls, _one_place),
    ("Pages read", lambda r: r.navigation.pages_read, _one_place),
    ("Images viewed", lambda r: r.navigation.images_viewed, _one_place),
    ("Model calls", lambda r: r.navigation.model_calls, _one_place),
    ("Input tokens", lambda r: r.navigation.input_tokens, _whole),
    ("Output tokens", lambda r: r.navigation.output_tokens, _whole),
    ("Time (s)", lambda r: r.navigation.duration_ms / 1000, _one_place),
    ("Answering cost", lambda r: r.navigation.cost_usd, _usd),
]


def mean(values: Sequence[Value]) -> float | None:
    """The average of the values that apply; booleans count as 1 and 0."""
    known = [float(value) for value in values if value is not None]
    return sum(known) / len(known) if known else None


def passed(result: QuestionResult) -> bool:
    generation = result.generation
    return (
        result.error is None and generation.correctness == 1.0
        and generation.key_facts is not False and generation.refusal_correct is not False
    )


def summary(results: Sequence[QuestionResult], info: RunInfo) -> str:
    sections = [
        _header(results, info),
        _metrics_table(results),
        _question_table(results),
        _failures(results),
        _spot_checks(results),
        _cost_and_time(results, info),
    ]
    return "\n\n".join(sections) + "\n"


def _header(results: Sequence[QuestionResult], info: RunInfo) -> str:
    settings = ", ".join(f"`{name}={value}`" for name, value in info.settings.items())
    return "\n".join([
        f"# Eval run {info.started_at:%Y-%m-%d %H:%M}",
        "",
        f"{len(results)} answers ({len({r.id for r in results})} questions × {info.repeat}); "
        f"{sum(passed(r) for r in results)} passed (correct, key facts present, no wrong refusal).",
        "",
        f"Models: chat `{info.chat_model}`, judge `{info.judge_model}`, index `{info.index_model}`, "
        f"vision `{info.vision_model}`. Settings: {settings}.",
    ])


def _metrics_table(results: Sequence[QuestionResult]) -> str:
    types = [kind for kind in QuestionType if any(r.type is kind for r in results)]
    rows = [
        "## Metrics",
        "",
        "Averages over the answers each metric applies to; – means it applies to none.",
        "",
        "| Metric | All | " + " | ".join(f"{kind.value} ({sum(r.type is kind for r in results)})" for kind in types) + " |",
        "|---|---|" + "---|" * len(types),
    ]
    for label, value_of, show in METRICS:
        cells = [show(mean([value_of(r) for r in results]))]
        cells += [show(mean([value_of(r) for r in results if r.type is kind])) for kind in types]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _mark(value: bool | None) -> str:
    return "–" if value is None else "yes" if value else "no"


def _grade(result: QuestionResult) -> str:
    if result.error is not None:
        return "no answer"
    if result.verdict is None:
        return "not judged"
    if result.type is QuestionType.UNANSWERABLE:
        return "refused" if result.generation.refusal_correct else f"answered ({result.verdict.correctness})"
    return result.verdict.correctness


def _label(result: QuestionResult) -> str:
    return result.id if result.repeat == 0 else f"{result.id} #{result.repeat + 1}"


def _question_table(results: Sequence[QuestionResult]) -> str:
    rows = [
        "## Questions",
        "",
        "| Question | Type | Pass | Answer | Key facts | Page recall | Citation recall | Tools | Pages | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rows.append(
            f"| {_label(r)} | {r.type.value} | {'pass' if passed(r) else '**fail**'} | {_grade(r)} | "
            f"{_mark(r.generation.key_facts)} | {_rate(r.retrieval.page_recall)} | {_rate(r.generation.citation_recall)} | "
            f"{r.navigation.tool_calls} | {r.navigation.pages_read} | {_usd(r.navigation.cost_usd)} |"
        )
    return "\n".join(rows)


def _quote(text: str) -> str:
    shown = text if len(text) <= ANSWER_CHARS else text[:ANSWER_CHARS] + " …"
    return "\n".join("> " + line for line in (shown or "(none)").splitlines())


def _reasons(result: QuestionResult) -> list[str]:
    verdict = result.verdict
    if verdict is None:
        return [f"- Judge: {result.judge_error or 'not judged'}"]
    return [
        f"- Correctness ({verdict.correctness}): {verdict.correctness_reason}",
        f"- Faithfulness ({verdict.claims_supported}/{verdict.claims_total} claims supported): {verdict.faithfulness_reason}",
        f"- Relevance ({verdict.relevance}): {verdict.relevance_reason}",
        f"- Refusal (refused: {_mark(verdict.refused)}, invents an answer: {_mark(verdict.invents_answer)}): {verdict.refusal_reason}",
    ]


def _failures(results: Sequence[QuestionResult]) -> str:
    failed = [r for r in results if not passed(r)]
    rows = ["## Failures", "", "None." if not failed else f"{len(failed)} answers did not pass."]
    for r in failed:
        rows += ["", f"### {_label(r)} ({r.type.value})", "", f"**Question:** {r.question}", "", "**Answer:**", _quote(r.answer)]
        rows += ["", f"**Reference:** {r.reference}", ""]
        if r.error is not None:
            rows.append(f"- Error: {r.error}")
        rows += [f"- Key facts present: {_mark(r.generation.key_facts)}", *_reasons(r)]
    return "\n".join(rows)


def _spot_checks(results: Sequence[QuestionResult]) -> str:
    """Up to SPOT_CHECKS passes, one per question type in turn: a judge can be wrong in both directions."""
    by_type = {kind: [r for r in results if r.type is kind and passed(r)] for kind in QuestionType}
    picked: list[QuestionResult] = []
    while len(picked) < SPOT_CHECKS and any(by_type.values()):
        for kind in QuestionType:
            if by_type[kind] and len(picked) < SPOT_CHECKS:
                picked.append(by_type[kind].pop(0))
    rows = ["## Spot-check these", "", "Passes the judge accepted; read the answer and its reasons."]
    for r in picked:
        rows += ["", f"### {_label(r)} ({r.type.value})", "", f"**Question:** {r.question}", "", _quote(r.answer), "", *_reasons(r)]
    return "\n".join(rows)


def _cost_and_time(results: Sequence[QuestionResult], info: RunInfo) -> str:
    answering = sum(r.navigation.cost_usd or 0.0 for r in results)
    judging = sum(r.judge_cost_usd or 0.0 for r in results)
    count = max(len(results), 1)
    answer_time = sum(r.navigation.duration_ms for r in results) / 1000
    return "\n".join([
        "## Cost and time",
        "",
        "| | Total | Per answer |",
        "|---|---|---|",
        f"| Answering | {_usd(answering)} | {_usd(answering / count)} |",
        f"| Judging | {_usd(judging)} | {_usd(judging / count)} |",
        f"| All | {_usd(answering + judging)} | {_usd((answering + judging) / count)} |",
        f"| Answering time | {answer_time:.0f} s | {answer_time / count:.1f} s |",
        f"| Whole run | {info.duration_s / 60:.1f} min | |",
        "",
        "Costs use the full input price for every token (cached input is cheaper), so they are upper bounds.",
        "Building the eval library is a one-off and not included.",
    ])
