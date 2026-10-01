"""The eval (agent-runs spec Part C): a development tool next to the service, not part of it.

`python -m evals.run` asks the questions in questions.yaml through the real service and writes the
results; the modules below hold the eval set, the metrics, the judge and the report. run.py is the entry
point and composition root, so it is not re-exported here.
"""
from evals.config import PRICES_PER_MILLION, EvalSettings, cost_usd, load_eval_settings
from evals.judge import Evidence, Judge, Judgement, NoVerdict, OpenAiJudge, PageImageEvidence, PageText
from evals.metrics import (
    GenerationScores,
    NavigationCost,
    QuestionResult,
    RetrievalScores,
    Verdict,
    citation_scores,
    documents_opened,
    generation_scores,
    key_facts_present,
    navigation_cost,
    normalise,
    pages_read,
    pages_viewed,
    retrieval_scores,
)
from evals.questions import QUESTIONS_FILE, EvalQuestion, GoldPages, PageRef, QuestionType, load_questions, select
from evals.report import RunInfo, mean, passed, summary

__all__ = [
    "PRICES_PER_MILLION",
    "QUESTIONS_FILE",
    "EvalQuestion",
    "EvalSettings",
    "Evidence",
    "GenerationScores",
    "GoldPages",
    "Judge",
    "Judgement",
    "NavigationCost",
    "NoVerdict",
    "OpenAiJudge",
    "PageImageEvidence",
    "PageRef",
    "PageText",
    "QuestionResult",
    "QuestionType",
    "RetrievalScores",
    "RunInfo",
    "Verdict",
    "citation_scores",
    "cost_usd",
    "documents_opened",
    "generation_scores",
    "key_facts_present",
    "load_eval_settings",
    "load_questions",
    "mean",
    "navigation_cost",
    "normalise",
    "pages_read",
    "pages_viewed",
    "passed",
    "retrieval_scores",
    "select",
    "summary",
]
