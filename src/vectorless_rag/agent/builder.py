"""The answering agent (implements ports.AnswerAgent; spec 5.6).

A LangChain agent, built per question around a model shared by all questions: the user's read-only
PageIndex tools plus view_pages, PageIndex's instructions plus our figure guidance, the limits, and a
recorder that returns the run (agent-runs spec 2.2).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from openai import APITimeoutError
from pydantic import SecretStr

from vectorless_rag.agent.middleware import keep_latest_page_images, stop_at_deadline
from vectorless_rag.agent.prompts import FIGURE_GUIDANCE
from vectorless_rag.agent.recording import RunRecorder
from vectorless_rag.agent.view_pages import ImageDetail, view_pages_tool
from vectorless_rag.models import AgentRun, ChatMessage, RunLabels
from vectorless_rag.operations import AnswerIncomplete, PageViewer

# After the tool-call limit blocks further calls, a couple of model turns to write the answer.
EXTRA_MODEL_CALLS = 2
# Retries of one model call on 429/5xx. The SDK also retries a call that timed out, so a call that
# keeps hanging can hold a question for (1 + MODEL_RETRIES) x AGENT_TIMEOUT_S before it fails.
MODEL_RETRIES = 2

ReasoningSummary = Literal["off", "auto", "detailed"]  # AGENT_REASONING_SUMMARY


@dataclass(frozen=True)
class AgentRules:
    max_steps: int  # AGENT_MAX_STEPS: tool calls per question
    view_pages_max_calls: int
    max_image_sets: int  # view_pages results that keep their images in the context
    image_detail: ImageDetail
    timeout_s: float


def create_chat_model(api_key: str, model: str, timeout_s: float, *, reasoning_summary: ReasoningSummary) -> ChatOpenAI:
    """The answering model, on the Responses API: Chat Completions silently drops images in tool
    results, and rejects tools with reasoning for gpt-5.6-sol (Spike B). Unless `reasoning_summary`
    is "off", the model also returns summaries of its reasoning; the effort stays the model's default."""
    reasoning = None if reasoning_summary == "off" else {"summary": reasoning_summary}
    return ChatOpenAI(
        model=model, api_key=SecretStr(api_key), use_responses_api=True, timeout=timeout_s, max_retries=MODEL_RETRIES,
        reasoning=reasoning,
    )


class LangChainAnswerAgent:
    def __init__(self, model: BaseChatModel, rules: AgentRules, callbacks: Sequence[BaseCallbackHandler] = ()) -> None:
        """`callbacks` go with every run, e.g. a tracing handler (agent-runs spec Part B); none for now."""
        self._model = model
        self._rules = rules
        self._callbacks = list(callbacks)

    def answer(
        self,
        instructions: str,
        tools: Sequence[Callable[..., str]],
        view_pages: PageViewer,
        messages: Sequence[ChatMessage],
        labels: RunLabels,
    ) -> AgentRun:
        recorder = RunRecorder()
        agent = create_agent(
            self._model,
            tools=[*(StructuredTool.from_function(tool) for tool in tools), view_pages_tool(view_pages, self._rules.image_detail)],
            system_prompt=f"{instructions}\n\n{FIGURE_GUIDANCE}",
            middleware=[recorder, *self._middleware()],
        )
        config: RunnableConfig = {
            "callbacks": self._callbacks,
            "metadata": {"trace_id": labels.trace_id, "user_key": labels.user_key},
        }
        try:
            result = agent.invoke({"messages": [{"role": message.role, "content": message.content} for message in messages]}, config)
            return recorder.run(final_answer(result["messages"]))
        except ModelCallLimitExceededError as error:
            raise recorder.incomplete(f"no answer within {self._rules.max_steps} steps") from error
        except APITimeoutError as error:
            raise recorder.incomplete("the model did not answer in time") from error
        except AnswerIncomplete as error:  # the deadline, or a run that ended without an answer
            raise recorder.incomplete(str(error)) from error

    def _middleware(self) -> list[AgentMiddleware[Any, Any, Any]]:  # each has its own state type
        rules = self._rules
        return [
            stop_at_deadline(rules.timeout_s),
            ModelCallLimitMiddleware(run_limit=rules.max_steps + EXTRA_MODEL_CALLS, exit_behavior="error"),
            # "continue": a blocked call gets an error result and the model answers with what it has.
            ToolCallLimitMiddleware(run_limit=rules.max_steps),
            ToolCallLimitMiddleware(tool_name="view_pages", run_limit=rules.view_pages_max_calls),
            keep_latest_page_images(rules.max_image_sets),
        ]


def final_answer(messages: Sequence[AnyMessage]) -> str:
    """The text of the agent's last message, which must be an answer, not a tool call."""
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and not last.tool_calls and last.text.strip():
        return str(last.text)
    raise AnswerIncomplete("the agent stopped without an answer")
