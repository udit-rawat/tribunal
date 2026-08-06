"""Per-call telemetry: which model ran, how long it took, what it cost.

`structured()` appends a record for every completion; the web layer drains the buffer after each
graph node so the UI can attribute model, latency and spend to the stage that incurred them.

Thread-safe because LangGraph runs the two advocate nodes concurrently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# USD per 1M tokens, (prompt, completion). Anything absent is treated as free — which is correct
# for the free tiers this project targets, and keeps the cost line honest rather than invented.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class Call:
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Which graph node made the call. Required for correct attribution: the two advocates run
    # concurrently, so draining by arrival order credits both calls to whichever finishes first.
    node: str = ""

    @property
    def cost_usd(self) -> float:
        base = self.model.split(":")[0]  # strip OpenRouter ':free' suffix
        if self.model.endswith(":free") or base not in PRICES:
            return 0.0
        pin, pout = PRICES[base]
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "latency_s": round(self.latency_s, 2),
            "tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


_buffer: list[Call] = []
_lock = threading.Lock()


def record(call: Call) -> None:
    with _lock:
        _buffer.append(call)


def drain() -> list[Call]:
    """Take everything recorded since the last drain."""
    with _lock:
        out, _buffer[:] = _buffer[:], []
    return out


def summarise(calls: list[Call]) -> dict | None:
    """Collapse a node's calls into one compact readout for the UI."""
    if not calls:
        return None
    return {
        "model": calls[0].model,
        "calls": len(calls),
        "latency_s": round(sum(c.latency_s for c in calls), 2),
        "tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
        "cost_usd": round(sum(c.cost_usd for c in calls), 6),
    }


@dataclass
class ToolCall:
    """A non-LLM call made on the request path: a search API, or the embedding/index step."""

    tool: str
    query: str
    results: int
    latency_s: float
    ok: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "query": self.query,
            "results": self.results,
            "latency_s": round(self.latency_s, 2),
            "ok": self.ok,
            "note": self.note,
        }


_tools: list[ToolCall] = []


def record_tool(call: ToolCall) -> None:
    with _lock:
        _tools.append(call)


def drain_tools() -> list[ToolCall]:
    with _lock:
        out, _tools[:] = _tools[:], []
    return out


def bucket(calls: list[Call]) -> dict[str, list[Call]]:
    """Group calls by the node that made them."""
    out: dict[str, list[Call]] = {}
    for c in calls:
        out.setdefault(c.node, []).append(c)
    return out
