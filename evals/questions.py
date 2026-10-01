"""The eval set (agent-runs spec 4.1): evals/questions.yaml as typed, checked records."""
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"

PageRef = tuple[str, int]  # (document name, 1-based page of the original PDF)


class QuestionType(StrEnum):
    TEXT = "text"
    FIGURE = "figure"
    MULTI_PAGE = "multi_page"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"


class GoldPages(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: str = Field(min_length=1)  # the file name, which is also the name the agent sees
    pages: tuple[int, ...] = Field(min_length=1)


class EvalQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str = Field(min_length=1)
    type: QuestionType
    gold: tuple[GoldPages, ...] = ()
    figure_pages: tuple[int, ...] = ()
    answer: str = Field(min_length=1)  # the reference answer
    key_facts: tuple[str, ...] = ()  # each must be in a correct answer; "a|b": either counts
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def _gold_fits_the_type(self) -> Self:
        if (self.type is QuestionType.UNANSWERABLE) != (not self.gold):
            raise ValueError(f"{self.id}: unanswerable questions, and only they, have no gold pages")
        if not {page for _, page in self.gold_pages} >= set(self.figure_pages):
            raise ValueError(f"{self.id}: figure pages must be gold pages")
        return self

    @property
    def answerable(self) -> bool:
        return self.type is not QuestionType.UNANSWERABLE

    @property
    def gold_pages(self) -> frozenset[PageRef]:
        return frozenset((gold.document, page) for gold in self.gold for page in gold.pages)

    @property
    def gold_documents(self) -> frozenset[str]:
        return frozenset(gold.document for gold in self.gold)

    @property
    def gold_figure_pages(self) -> frozenset[PageRef]:
        return frozenset((document, page) for document, page in self.gold_pages if page in self.figure_pages)


def load_questions(path: Path = QUESTIONS_FILE) -> list[EvalQuestion]:
    questions = [EvalQuestion.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]
    duplicates = sorted({q.id for q in questions if [p.id for p in questions].count(q.id) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ids: {', '.join(duplicates)}")
    return questions


def select(questions: Sequence[EvalQuestion], only: Sequence[str]) -> list[EvalQuestion]:
    """The questions named in `only`, in file order; all of them when `only` is empty."""
    unknown = sorted(set(only) - {q.id for q in questions})
    if unknown:
        raise ValueError(f"unknown question ids: {', '.join(unknown)}")
    return [q for q in questions if not only or q.id in only]
