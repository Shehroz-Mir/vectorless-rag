"""The record of how the agent answered one question (agent-runs spec 2.1, 2.4).

The agent adapter writes it, the use case reads the answer from it, and the API and the eval show it.
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VIEW_PAGES_TOOL = "view_pages"  # the one tool that returns page images


class ToolOutcome(StrEnum):
    OK = "ok"
    ERROR = "error"  # the tool reported an error, e.g. an unknown document name or a bad page spec
    BLOCKED = "blocked"  # a tool-call limit stopped the call before it ran


class ModelStep(BaseModel):
    """One call of the answering model."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["model"] = "model"
    index: int = Field(ge=1)  # position in the run
    duration_ms: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    reasoning_summary: tuple[str, ...] = ()  # empty unless the model was asked for summaries
    text: str = ""  # the final answer is the text of the last model step
    tool_calls: tuple[str, ...] = ()  # names of the tools it asked for


class ToolStep(BaseModel):
    """One tool call the model asked for."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool"] = "tool"
    index: int = Field(ge=1)
    tool: str
    arguments: dict[str, Any]  # as the model sent them
    document: str | None = None  # the document name it targets, if any
    pages: tuple[int, ...] = ()  # pages it read (get_page_content, view_pages); empty unless the call succeeded
    outcome: ToolOutcome
    result_preview: str = ""  # the start of the result; page images appear as "[page image: <name> p12]"
    duration_ms: int = Field(ge=0)  # 0 for a blocked call


RunStep = Annotated[ModelStep | ToolStep, Field(discriminator="kind")]


class RunTotals(BaseModel):
    """The run in numbers: the `stats` of a brief /query response (agent-runs spec 2.5)."""

    model_config = ConfigDict(frozen=True)

    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)  # every call the model asked for, blocked ones too
    pages_read: int = Field(ge=0)  # unique (document, page) pairs read as text or viewed as images
    images_viewed: int = Field(ge=0)  # page images view_pages returned, repeats included
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    duration_ms: int = Field(ge=0)

    @classmethod
    def of(cls, steps: Sequence[ModelStep | ToolStep], duration_ms: int) -> RunTotals:
        models = [step for step in steps if isinstance(step, ModelStep)]
        tools = [step for step in steps if isinstance(step, ToolStep)]
        return cls(
            model_calls=len(models),
            tool_calls=len(tools),
            pages_read=len({(step.document, page) for step in tools for page in step.pages}),
            images_viewed=sum(len(step.pages) for step in tools if step.tool == VIEW_PAGES_TOOL),
            input_tokens=sum(step.input_tokens for step in models),
            output_tokens=sum(step.output_tokens for step in models),
            reasoning_tokens=sum(step.reasoning_tokens for step in models),
            duration_ms=duration_ms,
        )


class AgentRun(BaseModel):
    """Everything the agent did for one question, in order, and its answer with <cite> tags."""

    model_config = ConfigDict(frozen=True)

    answer: str  # empty when the run ended without one (AnswerIncomplete)
    steps: tuple[RunStep, ...] = ()
    totals: RunTotals

    @classmethod
    def of(cls, answer: str, steps: Sequence[ModelStep | ToolStep], duration_ms: int) -> AgentRun:
        return cls(answer=answer, steps=tuple(steps), totals=RunTotals.of(steps, duration_ms))


class RunLabels(BaseModel):
    """What ties one run to the rest of the system: logs, the API response and, later, Langfuse."""

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(min_length=1)  # made before the run; later the seed of the Langfuse trace id
    user_key: str = Field(min_length=1)  # the internal key, so raw user ids never leave the service
