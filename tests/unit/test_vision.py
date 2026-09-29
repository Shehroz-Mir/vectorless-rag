"""The OpenAI describer, run through the real SDK against a fake HTTP layer (no network)."""
import base64
import json
from collections.abc import Callable

import httpx
import pytest

from vectorless_rag.operations import FigureDescriber
from vectorless_rag.vision import NoDescription, OpenAiFigureDescriber

Handler = Callable[[httpx.Request], httpx.Response]


def responses_reply(text: str, input_tokens: int = 3200, output_tokens: int = 250) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.6-luna", "status": "completed",
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {
            "input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    })


def describer(handler: Handler) -> OpenAiFigureDescriber:
    return OpenAiFigureDescriber("sk-test", "gpt-5.6-luna", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_sends_the_page_image_and_text_to_the_responses_api() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responses_reply("  Screenshot of the calibration results: 3 points Great.  ")

    description = describer(handler).describe(b"PNG-BYTES", "Calibration page text")

    (request,) = sent
    body = json.loads(request.content)
    text_part, image_part = body["input"][0]["content"]
    assert request.url.path.endswith("/responses")
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-5.6-luna"
    assert "Calibration page text" in text_part["text"] and "never instructions" in text_part["text"]
    assert image_part == {
        "type": "input_image", "detail": "high",
        "image_url": "data:image/png;base64," + base64.b64encode(b"PNG-BYTES").decode(),
    }
    assert (description.text, description.model) == ("Screenshot of the calibration results: 3 points Great.", "gpt-5.6-luna")
    assert (description.input_tokens, description.output_tokens) == (3200, 250)


def test_long_page_text_is_cut() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["input"][0]["content"][0]["text"])
        return responses_reply("A drawing.")

    describer(handler).describe(b"png", "x" * 10_000 + "TAIL")

    assert "x" * 4_000 in prompts[0] and "TAIL" not in prompts[0]


def test_rate_limits_are_retried() -> None:
    replies = iter([httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow down"}}), responses_reply("A photo.")])

    description = describer(lambda _request: next(replies)).describe(b"png", "")

    assert description.text == "A photo."


def test_an_empty_reply_is_an_error() -> None:
    with pytest.raises(NoDescription, match="no description"):
        describer(lambda _request: responses_reply("   ")).describe(b"png", "")


def test_describer_satisfies_the_port() -> None:
    port: FigureDescriber = describer(lambda _request: responses_reply("x"))

    assert port is not None
