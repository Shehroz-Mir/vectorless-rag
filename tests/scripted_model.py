"""A chat model that replies from a script, for testing the agent without the network."""
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatResult
from pydantic import Field


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


def usage(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> UsageMetadata:
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
        "output_token_details": {"reasoning": reasoning_tokens},
    }
