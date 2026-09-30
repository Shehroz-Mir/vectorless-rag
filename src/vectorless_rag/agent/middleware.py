"""Agent middleware: keep page images from filling the context (spec 5.7) and stop at the deadline.

Both only look at or rewrite what one model call is sent; the agent's state is left as it is.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, before_model, wrap_model_call
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.runtime import Runtime

from vectorless_rag.operations import AnswerIncomplete

IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})
REMOVED_IMAGE = "[page image removed to save context; call view_pages again to see it]"


def keep_latest_page_images(max_sets: int) -> AgentMiddleware:
    """Only the latest `max_sets` tool results keep their images in what the model is sent. A page
    image costs ~3k input tokens on every model call it stays in (Spike B)."""

    @wrap_model_call
    def trim_old_page_images(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(messages=without_old_images(request.messages, max_sets)))

    return trim_old_page_images


def stop_at_deadline(seconds: float) -> AgentMiddleware:
    """Before each model call, end the run once `seconds` have passed since this middleware was made
    (one per question). Each call also has its own timeout, so a question can overrun by at most one call."""
    deadline = time.monotonic() + seconds

    @before_model
    def check_deadline(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if time.monotonic() >= deadline:
            raise AnswerIncomplete(f"no answer within {seconds:g} s")
        return None

    return check_deadline


def without_old_images(messages: Sequence[AnyMessage], max_sets: int) -> list[AnyMessage]:
    """Tool results with images, except the latest `max_sets`, lose their images; their labels stay."""
    with_images = [i for i, message in enumerate(messages) if isinstance(message, ToolMessage) and _has_image(message)]
    stale = set(with_images[: max(0, len(with_images) - max_sets)])
    return [_without_images(message) if i in stale else message for i, message in enumerate(messages)]


def _is_image(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") in IMAGE_BLOCK_TYPES


def _has_image(message: AnyMessage) -> bool:
    return isinstance(message.content, list) and any(_is_image(block) for block in message.content)


def _without_images(message: AnyMessage) -> AnyMessage:
    if not isinstance(message.content, list):
        return message  # plain text has no images
    content = [{"type": "text", "text": REMOVED_IMAGE} if _is_image(block) else block for block in message.content]
    return message.model_copy(update={"content": content})
