"""Eval settings (agent-runs spec 4.3, 4.4): the judge model and the price table.

The service's own settings (its models, limits and OPENAI_API_KEY) come from vectorless_rag.config, as in
the service; only the eval's composition root (evals/run.py) loads either.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Per 1M tokens (input, output), from the OpenAI pricing page on 2026-09-29 (docs/spike-findings.md).
# Cached input is cheaper; every input token is priced in full here, so costs are upper bounds.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore", frozen=True)

    eval_judge_model: Annotated[str, Field(min_length=1)] = "gpt-5.6-sol"  # EVAL_JUDGE_MODEL


def load_eval_settings(env_file: Path | None = Path(".env")) -> EvalSettings:
    """Settings from the environment, plus `env_file` when given (environment wins)."""
    return EvalSettings(_env_file=env_file)  # pyright: ignore[reportCallIssue]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """What the tokens cost on `model`; None for a model missing from the price table.
    Reasoning tokens are billed as output and are already part of `output_tokens`."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
