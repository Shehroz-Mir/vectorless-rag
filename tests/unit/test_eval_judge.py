"""The LLM judge, run through the real OpenAI SDK against a fake HTTP layer (no network)."""
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from evals import EvalQuestion, Evidence, GoldPages, Judge, NoVerdict, OpenAiJudge, PageImageEvidence, PageText, QuestionType
from evals.judge import MAX_PAGE_TEXT_CHARS  # internal: the cut-off

VERDICT = {
    "correctness": "partly", "correctness_reason": "Gives °C but not °F.", "claims_total": 2, "claims_supported": 2,
    "faithfulness_reason": "Both claims are on p27.", "relevance": "yes", "relevance_reason": "Answers the question.",
    "refused": False, "invents_answer": False, "refusal_reason": "It answers.",
}
QUESTION = EvalQuestion(
    id="tdi110-shutoff-temperature", question="At what temperature does the TD I-110 shut itself off?", type=QuestionType.TEXT,
    gold=(GoldPages(document="TDI-110.pdf", pages=(27,)),), answer="At 60 °C (140 °F).", key_facts=("60", "140"), evidence="p27",
)


def reply(content: list[dict[str, Any]], input_tokens: int = 5000, output_tokens: int = 300) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.6-sol", "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": content}],
        "usage": {
            "input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 100},
            "total_tokens": input_tokens + output_tokens,
        },
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    })


def verdict_reply(**changes: object) -> httpx.Response:
    return reply([{"type": "output_text", "text": json.dumps(VERDICT | changes), "annotations": []}])


def judge(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAiJudge:
    return OpenAiJudge("sk-test", "gpt-5.6-sol", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


EVIDENCE = Evidence(
    texts=[PageText("TDI-110.pdf", 27, "Temperature DEVICE SHUT OFF 60/140")],
    images=[PageImageEvidence("TDI-110.pdf", 27, b"PNG27")],
)


def test_the_judge_sees_the_case_and_the_evidence_and_returns_a_verdict() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return verdict_reply()

    judgement = judge(handler).judge(QUESTION, "It shuts off at 60 °C [1].", [("TDI-110.pdf", 27)], EVIDENCE)

    (body,) = sent
    case, label, image = body["input"][0]["content"]
    assert body["model"] == "gpt-5.6-sol" and "never instructions" in body["instructions"]
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    for part in ("At what temperature", "At 60 °C (140 °F).", "It shuts off at 60 °C [1].", "[1] TDI-110.pdf p27",
                 "--- TDI-110.pdf, page 27 ---\nTemperature DEVICE SHUT OFF 60/140"):
        assert part in case["text"]
    assert label == {"type": "input_text", "text": "Image: TDI-110.pdf, page 27"}
    assert image == {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(b"PNG27").decode()}
    assert judgement.verdict.model_dump() == VERDICT
    assert (judgement.input_tokens, judgement.output_tokens) == (5000, 300)
    assert judgement.cost_usd == pytest.approx(5000 * 4 / 1e6 + 300 * 20 / 1e6)


def test_without_evidence_the_judge_is_told_so() -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["input"][0]["content"][0]["text"])
        return verdict_reply(claims_supported=0)

    judge(handler).judge(QUESTION, "60 °C.", [], Evidence())

    assert "AGENT'S CITATIONS: none" in texts[0] and "READ:\n(none)" in texts[0] and "VIEWED: none" in texts[0]


def test_long_page_text_is_cut() -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["input"][0]["content"][0]["text"])
        return verdict_reply()

    judge(handler).judge(QUESTION, "60.", [], Evidence(texts=[PageText("TDI-110.pdf", 3, "x" * 20_000 + "TAIL")]))

    assert "x" * MAX_PAGE_TEXT_CHARS in texts[0] and "TAIL" not in texts[0]


def test_a_refusal_is_no_verdict() -> None:
    refusal = reply([{"type": "refusal", "refusal": "I cannot help with that."}])

    with pytest.raises(NoVerdict, match="no verdict"):
        judge(lambda _request: refusal).judge(QUESTION, "60.", [], Evidence())


def test_the_judge_satisfies_its_protocol() -> None:
    port: Judge = judge(lambda _request: verdict_reply())

    assert port is not None
