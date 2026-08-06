"""OpenRouter/Gemini client wrapped with Instructor for structured output.

Uses JSON mode because most free models don't support tool/function calling.
Adds resilience: on rate-limit (429) it backs off and, for daily-cap exhaustion,
falls back to a secondary model that has its own separate quota.
"""

from __future__ import annotations

import time
from typing import Type, TypeVar

import instructor
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from . import telemetry
from .config import settings

T = TypeVar("T", bound=BaseModel)

_client: instructor.Instructor | None = None


def get_client() -> instructor.Instructor:
    global _client
    if _client is None:
        base = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout,
        )
        _client = instructor.from_openai(base, mode=instructor.Mode.JSON)
    return _client


def _model_chain(primary: str) -> list[str]:
    """Primary model followed by the configured fallback (if distinct)."""
    chain = [primary]
    fb = settings.fallback_model
    if fb and fb != primary:
        chain.append(fb)
    return chain


def structured(
    *,
    model: str,
    response_model: Type[T],
    system: str,
    user: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    node: str = "",
) -> T:
    """Structured call with backoff + model fallback.

    - Per-minute rate limits: exponential backoff, retry same model.
    - Per-day quota exhaustion: skip straight to the fallback model (separate quota).
    """
    if settings.llm_provider == "anthropic":
        from . import llm_anthropic

        return llm_anthropic.structured(
            model=model, response_model=response_model, system=system, user=user, node=node
        )

    client = get_client()
    last_err: Exception | None = None

    for model_name in _model_chain(model):
        for attempt in range(max_retries):
            started = time.monotonic()
            try:
                result, completion = client.chat.completions.create_with_completion(
                    model=model_name,
                    response_model=response_model,
                    max_retries=1,  # instructor's own validation retry
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                usage = getattr(completion, "usage", None)
                telemetry.record(
                    telemetry.Call(
                        model=model_name,  # the model that actually served it, post-fallback
                        latency_s=time.monotonic() - started,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        node=node,
                    )
                )
                return result
            except RateLimitError as e:
                last_err = e
                if "PerDay" in str(e) or "per day" in str(e).lower():
                    break  # daily cap — backoff won't help, try the fallback model
                time.sleep(min(30.0, 4.0 * (2**attempt)))  # per-minute limit — wait it out
            except Exception as e:  # noqa: BLE001 — surface after exhausting the chain
                last_err = e
                break

    raise last_err if last_err else RuntimeError("structured() failed with no error captured")
