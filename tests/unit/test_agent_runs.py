"""The run record of the LangChain agent, with a scripted chat model (no network): steps, totals, outcomes."""
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from tests.scripted_model import ScriptedModel, call, script, usage
from vectorless_rag.agent import AgentRules, LangChainAnswerAgent
from vectorless_rag.models import AgentRun, ChatMessage, ModelStep, PageImage, RunLabels, ToolOutcome, ToolStep
from vectorless_rag.operations import AnswerIncomplete, PageViewer, ViewPagesRejected

RULES = AgentRules(max_steps=20, view_pages_max_calls=4, max_image_sets=2, image_detail="high", timeout_s=60)
QUESTION = [ChatMessage(role="user", content="How hot does it get?")]
LABELS = RunLabels(trace_id="trace-1", user_key="u_alice")


def pages_of(doc_name: str, pages: str) -> list[PageImage]:
    return [PageImage(doc_name=doc_name, page=int(page), png=f"PNG{page}".encode()) for page in pages.split(",")]


def get_page_content(doc_name: str, pages: str) -> str:
    """Read the text of some pages, as PageIndex does: only pages 1-40 exist.

    Args:
        doc_name: The document name.
        pages: Page numbers.
    """
    first, _, last = pages.partition("-")
    returned = f"{first}-{min(int(last or first), 40)}" if last else first
    return json.dumps({"success": True, "doc_name": doc_name, "returned_pages": returned, "content": [{"page": 27, "text": "60 °C"}]})


def get_document(doc_name: str) -> str:
    """Look up a document.

    Args:
        doc_name: The document name.
    """
    return json.dumps({"error": f"Document '{doc_name}' not found"})


def run(
    model: ScriptedModel,
    rules: AgentRules = RULES,
    *,
    tools: Sequence[Callable[..., str]] = (get_page_content, get_document),
    view_pages: PageViewer = pages_of,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> AgentRun:
    return LangChainAnswerAgent(model, rules, callbacks).answer("PAGEINDEX RULES", tools, view_pages, QUESTION, LABELS)


def tool_steps(agent_run: AgentRun) -> list[ToolStep]:
    return [step for step in agent_run.steps if isinstance(step, ToolStep)]


def test_model_and_tool_calls_are_steps_in_order_with_tokens_and_totals() -> None:
    model = script(
        AIMessage(content="", tool_calls=call("get_page_content", doc_name="manual.pdf", pages="27").tool_calls, usage_metadata=usage(1000, 50, 30)),
        AIMessage(content='It is 60 °C <cite doc="manual.pdf" page="27"/>.', usage_metadata=usage(1500, 20)),
    )

    agent_run = run(model)

    first, read, last = agent_run.steps
    assert isinstance(first, ModelStep) and isinstance(read, ToolStep) and isinstance(last, ModelStep)
    assert (first.index, first.tool_calls, first.input_tokens, first.output_tokens, first.reasoning_tokens) == (1, ("get_page_content",), 1000, 50, 30)
    assert (read.index, read.tool, read.arguments, read.document, read.pages, read.outcome) == (
        2, "get_page_content", {"doc_name": "manual.pdf", "pages": "27"}, "manual.pdf", (27,), ToolOutcome.OK,
    )
    assert json.loads(read.result_preview)["content"] == [{"page": 27, "text": "60 °C"}]
    assert (last.index, last.text, last.tool_calls) == (3, 'It is 60 °C <cite doc="manual.pdf" page="27"/>.', ())
    assert agent_run.answer == last.text
    totals = agent_run.totals
    assert (totals.model_calls, totals.tool_calls, totals.pages_read, totals.images_viewed) == (2, 1, 1, 0)
    assert (totals.input_tokens, totals.output_tokens, totals.reasoning_tokens) == (2500, 70, 30)
    assert totals.duration_ms >= max(step.duration_ms for step in agent_run.steps)


def test_page_content_pages_are_the_pages_pageindex_returned() -> None:
    agent_run = run(script(call("get_page_content", doc_name="manual.pdf", pages="38-42"), "Done."))

    (read,) = tool_steps(agent_run)
    assert read.pages == (38, 39, 40)  # 41 and 42 are past the end


def test_view_pages_counts_images_and_the_preview_names_them_without_bytes() -> None:
    agent_run = run(script(call("view_pages", doc_name="manual.pdf", pages="30,31"), "Seen."))

    (view,) = tool_steps(agent_run)
    assert (view.pages, view.outcome) == ((30, 31), ToolOutcome.OK)
    assert view.result_preview == (
        "Document: manual.pdf, page 30:\n[page image: manual.pdf p30]\n"
        "Document: manual.pdf, page 31:\n[page image: manual.pdf p31]"
    )
    assert (agent_run.totals.pages_read, agent_run.totals.images_viewed) == (2, 2)


def test_the_same_page_as_text_and_as_image_is_one_page_read() -> None:
    model = script(
        call("get_page_content", "call_1", doc_name="manual.pdf", pages="27"),
        call("view_pages", "call_2", doc_name="manual.pdf", pages="27"),
        "Done.",
    )

    totals = run(model).totals

    assert (totals.tool_calls, totals.pages_read, totals.images_viewed) == (2, 1, 1)


def test_tool_errors_are_recorded_and_read_no_pages() -> None:
    def refuse(doc_name: str, pages: str) -> list[PageImage]:
        raise ViewPagesRejected("At most 3 pages per call.")

    model = script(
        call("get_document", "call_1", doc_name="nope.pdf"),
        call("view_pages", "call_2", doc_name="manual.pdf", pages="1-9"),
        "Sorry.",
    )

    unknown, rejected = tool_steps(run(model, view_pages=refuse))

    assert (unknown.outcome, unknown.document, unknown.pages) == (ToolOutcome.ERROR, "nope.pdf", ())
    assert "not found" in unknown.result_preview
    assert (rejected.outcome, rejected.pages, rejected.result_preview) == (ToolOutcome.ERROR, (), "At most 3 pages per call.")


def test_calls_over_a_tool_limit_are_blocked() -> None:
    model = script(
        call("view_pages", "call_1", doc_name="manual.pdf", pages="1"),
        call("view_pages", "call_2", doc_name="manual.pdf", pages="2"),
        "Answer from page 1.",
    )

    shown, blocked = tool_steps(run(model, rules=replace(RULES, view_pages_max_calls=1)))

    assert (shown.outcome, shown.pages) == (ToolOutcome.OK, (1,))
    assert (blocked.index, blocked.outcome, blocked.pages, blocked.result_preview, blocked.duration_ms) == (
        4, ToolOutcome.BLOCKED, (), "", 0,
    )


def test_parallel_tool_calls_keep_the_order_the_model_asked_in() -> None:
    both = AIMessage(content="", tool_calls=[
        *call("view_pages", "call_1", doc_name="manual.pdf", pages="5").tool_calls,
        *call("get_page_content", "call_2", doc_name="manual.pdf", pages="6").tool_calls,
    ])

    agent_run = run(script(both, "Done."))

    assert [(step.index, step.kind) for step in agent_run.steps] == [(1, "model"), (2, "tool"), (3, "tool"), (4, "model")]
    assert [(step.tool, step.pages) for step in tool_steps(agent_run)] == [("view_pages", (5,)), ("get_page_content", (6,))]


def test_reasoning_summaries_are_kept_apart_from_the_text() -> None:
    thought = AIMessage(content=[
        {"type": "reasoning", "reasoning": "The safety section has the limit."},
        {"type": "text", "text": "It is 60 °C."},
    ])

    (step,) = run(script(thought)).steps

    assert isinstance(step, ModelStep)
    assert (step.reasoning_summary, step.text) == (("The safety section has the limit.",), "It is 60 °C.")


def test_long_results_are_cut_to_the_preview_length() -> None:
    def get_document_structure(doc_name: str) -> str:
        """The outline.

        Args:
            doc_name: The document name.
        """
        return json.dumps({"success": True, "structure": "x" * 5_000})

    agent_run = run(script(call("get_document_structure", doc_name="manual.pdf"), "Done."), tools=[get_document_structure])

    (outline,) = tool_steps(agent_run)
    assert len(outline.result_preview) == 2_000


def endless_view_pages() -> Iterator[AIMessage]:
    number = 0
    while True:
        number += 1
        yield call("view_pages", f"call_{number}", doc_name="manual.pdf", pages="1")


def test_a_run_that_hits_the_step_limit_carries_its_steps() -> None:
    with pytest.raises(AnswerIncomplete, match="2 steps") as raised:
        run(ScriptedModel(messages=endless_view_pages()), rules=replace(RULES, max_steps=2))

    partial = raised.value.run
    assert partial is not None and partial.answer == ""
    assert [step.outcome for step in tool_steps(partial)] == [ToolOutcome.OK, ToolOutcome.OK, ToolOutcome.BLOCKED, ToolOutcome.BLOCKED]
    assert (partial.totals.model_calls, partial.totals.tool_calls) == (4, 4)


def test_a_run_past_its_deadline_carries_an_empty_run() -> None:
    with pytest.raises(AnswerIncomplete, match="no answer within") as raised:
        run(script("Too late."), rules=replace(RULES, timeout_s=1e-9))

    assert raised.value.run is not None and raised.value.run.steps == ()


def test_a_run_that_ends_without_text_carries_its_model_call() -> None:
    with pytest.raises(AnswerIncomplete, match="without an answer") as raised:
        run(script(AIMessage(content="")))

    assert raised.value.run is not None and raised.value.run.totals.model_calls == 1


class SeesMetadata(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, **kwargs: Any) -> None:
        self.metadata.append(kwargs.get("metadata") or {})


def test_callbacks_from_the_composition_root_see_the_runs_labels() -> None:
    handler = SeesMetadata()

    run(script("Done."), callbacks=[handler])

    assert handler.metadata and {"trace_id": "trace-1", "user_key": "u_alice"}.items() <= handler.metadata[0].items()
