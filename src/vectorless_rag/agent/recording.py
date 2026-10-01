"""Records how the agent answers one question (agent-runs spec 2.1, 2.2).

A middleware made per question: it times every model call and tool call and keeps each as a step of
the run. Tokens come from each AIMessage's usage metadata; the pages a tool read come from page specs.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from vectorless_rag.agent.middleware import IMAGE_BLOCK_TYPES
from vectorless_rag.models import VIEW_PAGES_TOOL, AgentRun, ModelStep, ToolOutcome, ToolStep
from vectorless_rag.operations import AnswerIncomplete, ViewPagesRejected, page_ranges

RESULT_PREVIEW_CHARS = 2_000
PAGE_CONTENT_TOOL = "get_page_content"


@dataclass(frozen=True)
class _Asked:
    """A tool call the model asked for that has not run (yet)."""

    call: ToolCall


class RunRecorder(AgentMiddleware):
    """The steps of one question's run.

    A tool call takes its place in the run when the model asks for it, so the steps keep the model's
    order even when the calls of one turn run in parallel. A call that never runs was stopped by a
    tool-call limit (it gets an error result without running) and is recorded as blocked.
    """

    def __init__(self) -> None:
        super().__init__()
        self._started = time.monotonic()
        self._lock = threading.Lock()  # the tool calls of one model turn can run on several threads
        self._steps: list[ModelStep | ToolStep | _Asked] = []
        self._slots: dict[str, int] = {}  # tool call id -> its place in _steps

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        started = time.monotonic()
        response = handler(request)
        message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        if message is not None:
            self._model_called(message, _ms_since(started))
        return response

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        started = time.monotonic()
        result = handler(request)
        self._tool_called(request.tool_call, result, _ms_since(started))
        return result

    def run(self, answer: str) -> AgentRun:
        """The run so far; calls asked for but never run count as blocked."""
        with self._lock:
            steps = [
                _blocked(index, step.call) if isinstance(step, _Asked) else step
                for index, step in enumerate(self._steps, start=1)
            ]
        return AgentRun.of(answer, steps, _ms_since(self._started))

    def incomplete(self, reason: str) -> AnswerIncomplete:
        """The error for a run that ended without an answer, carrying the run so far."""
        return AnswerIncomplete(reason, run=self.run(""))

    def _model_called(self, message: AIMessage, duration_ms: int) -> None:
        with self._lock:
            self._steps.append(_model_step(len(self._steps) + 1, message, duration_ms))
            for call in message.tool_calls:
                self._slots[call["id"] or ""] = len(self._steps)
                self._steps.append(_Asked(call))

    def _tool_called(self, call: ToolCall, result: ToolMessage | Command[Any], duration_ms: int) -> None:
        with self._lock:
            slot = self._slots.pop(call["id"] or "", None)
            if slot is None:  # every call should come from a recorded model call; keep it anyway
                slot = len(self._steps)
                self._steps.append(_Asked(call))
            self._steps[slot] = _tool_step(slot + 1, call, result, duration_ms)


def _model_step(index: int, message: AIMessage, duration_ms: int) -> ModelStep:
    usage = message.usage_metadata
    details = usage.get("output_token_details", {}) if usage else {}
    return ModelStep(
        index=index,
        duration_ms=duration_ms,
        input_tokens=usage["input_tokens"] if usage else 0,
        output_tokens=usage["output_tokens"] if usage else 0,
        reasoning_tokens=details.get("reasoning", 0),
        reasoning_summary=_reasoning_summary(message),
        text=message.text,
        tool_calls=tuple(call["name"] for call in message.tool_calls),
    )


def _reasoning_summary(message: AIMessage) -> tuple[str, ...]:
    """The reasoning summaries in the message, as LangChain's standard `reasoning` blocks."""
    summaries: list[str] = []
    for block in message.content_blocks:
        if block["type"] == "reasoning" and (text := block.get("reasoning")):
            summaries.append(text)
    return tuple(summaries)


def _tool_step(index: int, call: ToolCall, result: ToolMessage | Command[Any], duration_ms: int) -> ToolStep:
    arguments = dict(call["args"])
    document = _document(arguments)
    content = result.content if isinstance(result, ToolMessage) else ""
    payload = _json_object(content)
    failed = (isinstance(result, ToolMessage) and result.status == "error") or (payload is not None and "error" in payload)
    pages = () if failed else _pages_read(call["name"], arguments, payload)
    return ToolStep(
        index=index,
        tool=call["name"],
        arguments=arguments,
        document=document,
        pages=pages,
        outcome=ToolOutcome.ERROR if failed else ToolOutcome.OK,
        result_preview=_preview(content, document, pages),
        duration_ms=duration_ms,
    )


def _blocked(index: int, call: ToolCall) -> ToolStep:
    arguments = dict(call["args"])
    return ToolStep(
        index=index, tool=call["name"], arguments=arguments, document=_document(arguments),
        outcome=ToolOutcome.BLOCKED, duration_ms=0,
    )


def _document(arguments: dict[str, Any]) -> str | None:
    name = arguments.get("doc_name")
    return name if isinstance(name, str) else None


def _pages_read(tool: str, arguments: dict[str, Any], payload: dict[str, Any] | None) -> tuple[int, ...]:
    """view_pages: the pages it was asked for (it rejects the whole call otherwise). get_page_content:
    the pages PageIndex says it returned, which leave out pages past the end or over its size budget."""
    if tool == VIEW_PAGES_TOOL:
        return _pages(arguments.get("pages"))
    if tool == PAGE_CONTENT_TOOL and payload is not None:
        return _pages(payload.get("returned_pages"))
    return ()


def _pages(spec: object) -> tuple[int, ...]:
    """The pages of a spec the tool accepted, in order and without repeats."""
    if not isinstance(spec, str) or not spec:
        return ()
    try:
        ranges = page_ranges(spec)
    except ViewPagesRejected:
        return ()
    return tuple(dict.fromkeys(page for first, last in ranges for page in range(first, last + 1)))


def _json_object(content: str | list[str | dict[Any, Any]]) -> dict[str, Any] | None:
    """PageIndex tools answer with a JSON object: {"success": true, ...} or {"error": ...}."""
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _preview(content: str | list[str | dict[Any, Any]], document: str | None, pages: Sequence[int]) -> str:
    """The start of a result as text; image blocks become "[page image: <name> p12]", never bytes."""
    if isinstance(content, str):
        return content[:RESULT_PREVIEW_CHARS]
    viewed = iter(pages)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif block.get("type") in IMAGE_BLOCK_TYPES:
            parts.append(f"[page image: {document} p{next(viewed, '?')}]")
        else:
            parts.append(f"[{block.get('type')}]")
    return "\n".join(parts)[:RESULT_PREVIEW_CHARS]


def _ms_since(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
