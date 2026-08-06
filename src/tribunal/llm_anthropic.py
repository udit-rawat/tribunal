"""Anthropic provider using the official SDK.

Kept separate from the OpenAI-compatible path because Claude Fable 5 has a different request
contract, and mixing the two in one function made both harder to read:

* **Thinking is always on** — the `thinking` parameter must be omitted entirely. An explicit
  `{"type": "disabled"}` is rejected with a 400.
* **No sampling parameters** — `temperature` / `top_p` / `top_k` all return 400.
* **Depth is controlled by `output_config.effort`**, not a token budget.
* **Requests can be declined** by safety classifiers, returning HTTP 200 with
  `stop_reason == "refusal"`, so `content` must never be read before checking it.
"""

from __future__ import annotations

import time
from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

from . import telemetry
from .config import settings

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None

# Structured outputs reject numeric/length constraints; Pydantic emits them from Field(ge=, le=).
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "format",
}


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Zero-arg construction also picks up an `ant auth login` profile, so only pass the key
        # when one is configured explicitly.
        _client = (
            anthropic.Anthropic(api_key=settings.llm_api_key, timeout=settings.request_timeout)
            if settings.llm_api_key
            else anthropic.Anthropic(timeout=settings.request_timeout)
        )
    return _client


def strict_schema(node: Any) -> Any:
    """Rewrite a Pydantic JSON schema into the strict subset structured outputs accepts.

    Every object must set `additionalProperties: false` and list *all* of its properties as
    required — Pydantic omits fields that have defaults, which the API rejects.
    """
    if isinstance(node, list):
        return [strict_schema(n) for n in node]
    if not isinstance(node, dict):
        return node

    out = {k: strict_schema(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") == "object" or "properties" in out:
        props = out.get("properties", {})
        out["additionalProperties"] = False
        out["required"] = list(props.keys())
    return out


def structured(
    *,
    model: str,
    response_model: Type[T],
    system: str,
    user: str,
    node: str = "",
    max_retries: int = 2,
) -> T:
    """One structured call against Claude, returning a validated Pydantic object."""
    client = get_client()
    schema = strict_schema(response_model.model_json_schema())

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": settings.anthropic_max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": settings.anthropic_effort,
            "format": {"type": "json_schema", "schema": schema},
        },
    }
    # Server-side fallback: if safety classifiers decline the request, the API transparently
    # re-runs it on the fallback model inside the same call rather than returning nothing.
    if settings.anthropic_fallback_model:
        kwargs["betas"] = ["server-side-fallback-2026-06-01"]
        kwargs["fallbacks"] = [{"model": settings.anthropic_fallback_model}]

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            resp = client.beta.messages.create(**kwargs)

            if resp.stop_reason == "refusal":
                detail = getattr(resp, "stop_details", None)
                category = getattr(detail, "category", None) if detail else None
                raise RuntimeError(f"request declined by safety classifiers (category={category})")

            usage = resp.usage
            telemetry.record(
                telemetry.Call(
                    model=resp.model,  # the model that actually served it, post-fallback
                    latency_s=time.monotonic() - started,
                    prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                    node=node,
                )
            )

            text = next((b.text for b in resp.content if b.type == "text"), "")
            return response_model.model_validate_json(text)

        except anthropic.RateLimitError as e:
            last_err = e
            time.sleep(min(30.0, 4.0 * (2**attempt)))
        except Exception as e:  # noqa: BLE001 — surfaced after retries are exhausted
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(1.5)

    raise last_err if last_err else RuntimeError("anthropic structured() failed with no error")
