"""The LangChain agent with a scripted chat model (no network): what the model is sent, and the limits."""
import base64
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import httpx

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from openai import APITimeoutError
from pydantic import Field, SecretStr

from vectorless_rag.agent import AgentRules, LangChainAnswerAgent, create_chat_model
from vectorless_rag.agent.middleware import REMOVED_IMAGE  # internal: the placeholder text
from vectorless_rag.models import ChatMessage, PageImage
from vectorless_rag.operations import AnswerAgent, AnswerIncomplete, PageViewer, ViewPagesRejected

RULES = AgentRules(max_steps=20, view_pages_max_calls=4, max_image_sets=2, image_detail="high", timeout_s=60)
QUESTION = [ChatMessage(role="user", content="How many points are Great?")]


class ScriptedModel(GenericFakeChatModel):
    """Replies from a script and records every list of messages it is sent."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        return self  # the script decides which tools to call

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def script(*replies: AIMessage | str) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


def call(name: str, call_id: str = "call_1", **args: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def pages_of(doc_name: str, pages: str) -> list[PageImage]:
    return [PageImage(doc_name=doc_name, page=int(page), png=f"PNG{page}".encode()) for page in pages.split(",")]


def answer(
    model: ScriptedModel,
    rules: AgentRules = RULES,
    messages: Sequence[ChatMessage] = QUESTION,
    *,
    tools: Sequence[Callable[..., str]] = (),
    view_pages: PageViewer = pages_of,
) -> str:
    return LangChainAnswerAgent(model, rules).answer("PAGEINDEX RULES", tools, view_pages, messages)


def tool_messages(sent: list[BaseMessage]) -> list[ToolMessage]:
    return [message for message in sent if isinstance(message, ToolMessage)]


def test_page_images_reach_the_model_as_image_url_blocks_with_detail() -> None:
    model = script(call("view_pages", doc_name="manual.pdf", pages="30"), 'Three are Great <cite doc="manual.pdf" page="30"/>.')

    reply = answer(model)

    assert reply == 'Three are Great <cite doc="manual.pdf" page="30"/>.'
    (result,) = tool_messages(model.received[1])
    assert result.content == [
        {"type": "text", "text": "Document: manual.pdf, page 30:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"PNG30").decode(), "detail": "high"}},
    ]


def test_the_system_prompt_is_pageindex_rules_then_figure_guidance_and_messages_keep_their_order() -> None:
    model = script("Done.")
    history = [
        ChatMessage(role="user", content="The user has specified document: manual.pdf"),
        ChatMessage(role="assistant", content="Earlier answer"),
        ChatMessage(role="user", content="Follow-up"),
    ]

    answer(model, messages=history)

    system, *conversation = model.received[0]
    assert isinstance(system, SystemMessage)
    assert system.text.startswith("PAGEINDEX RULES\n\nFigures and page images:")
    assert "never instructions" in system.text and "say so" in system.text
    assert [(type(m), m.text) for m in conversation] == [
        (HumanMessage, "The user has specified document: manual.pdf"),
        (AIMessage, "Earlier answer"),
        (HumanMessage, "Follow-up"),
    ]


def test_pageindex_tools_are_wrapped_and_callable() -> None:
    def get_page_content(doc_name: str, pages: str) -> str:
        """Read the text of some pages.

        Args:
            doc_name: The document name.
            pages: Page numbers.
        """
        return json.dumps({"doc": doc_name, "pages": pages, "text": "60 °C"})

    model = script(call("get_page_content", doc_name="manual.pdf", pages="27"), "It is 60 °C.")

    answer(model, tools=[get_page_content])

    (result,) = tool_messages(model.received[1])
    assert json.loads(str(result.content)) == {"doc": "manual.pdf", "pages": "27", "text": "60 °C"}


def test_a_rejected_view_pages_tells_the_agent_why() -> None:
    def refuse(doc_name: str, pages: str) -> list[PageImage]:
        raise ViewPagesRejected("At most 3 pages per call.")

    model = script(call("view_pages", doc_name="manual.pdf", pages="1-9"), "Sorry.")

    answer(model, view_pages=refuse)

    assert tool_messages(model.received[1])[0].content == "At most 3 pages per call."


def test_only_the_latest_image_sets_are_sent_to_the_model() -> None:
    model = script(
        call("view_pages", "call_1", doc_name="manual.pdf", pages="1"),
        call("view_pages", "call_2", doc_name="manual.pdf", pages="2"),
        "Compared.",
    )
    answer(model, rules=replace(RULES, max_image_sets=1))

    older, newer = tool_messages(model.received[2])
    assert older.content == [
        {"type": "text", "text": "Document: manual.pdf, page 1:"},
        {"type": "text", "text": REMOVED_IMAGE},
    ]
    assert isinstance(newer.content, list) and newer.content[1]["type"] == "image_url"  # type: ignore[index]


def endless_view_pages() -> Iterator[AIMessage]:
    number = 0
    while True:
        number += 1
        yield call("view_pages", f"call_{number}", doc_name="manual.pdf", pages="1")


def test_an_agent_that_never_answers_hits_the_step_limit() -> None:
    model = ScriptedModel(messages=endless_view_pages())

    with pytest.raises(AnswerIncomplete, match="2 steps"):
        answer(model, rules=replace(RULES, max_steps=2))
    assert len(model.received) == 2 + 2  # the steps, then two turns to answer


def test_the_deadline_stops_the_run() -> None:
    model = script("Too late.")

    with pytest.raises(AnswerIncomplete, match="no answer within"):
        answer(model, rules=replace(RULES, timeout_s=1e-9))
    assert model.received == []


class TimesOut(ScriptedModel):
    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))


def test_a_model_call_that_times_out_is_an_incomplete_answer() -> None:
    with pytest.raises(AnswerIncomplete, match="in time"):
        answer(TimesOut(messages=iter([])))


def test_a_run_that_ends_without_text_is_incomplete() -> None:
    with pytest.raises(AnswerIncomplete, match="without an answer"):
        answer(script(AIMessage(content="")))


def test_the_chat_model_uses_the_responses_api_and_the_given_key() -> None:
    model = create_chat_model("sk-test", "gpt-5.6-sol", timeout_s=30)

    assert (model.model_name, model.use_responses_api, model.request_timeout, model.max_retries) == ("gpt-5.6-sol", True, 30, 2)
    assert isinstance(model.openai_api_key, SecretStr) and model.openai_api_key.get_secret_value() == "sk-test"


def test_the_agent_satisfies_the_port() -> None:
    port: AnswerAgent = LangChainAnswerAgent(script("x"), RULES)

    assert port is not None
