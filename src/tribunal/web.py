"""FastAPI app: serves the single-page client and streams verification progress over SSE.

Run: `uvicorn tribunal.web:app --reload` then open http://127.0.0.1:8000

The graph takes tens of seconds, most of it in retrieval and the two advocate calls. Streaming each
node as it completes turns that wait into visible progress instead of a blank page.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from . import cache, telemetry
from .graph import build_graph
from .schemas import Brief

app = FastAPI(title="Tribunal")

_STATIC = pathlib.Path(__file__).parent / "static"

# Human-readable stage labels, in pipeline order, for the client's progress rail.
STAGES = [
    ("decomposer", "Framing the charges"),
    ("retriever", "Gathering evidence"),
    ("prosecutor", "Prosecution brief"),
    ("defender", "Defence brief"),
    ("citation_verifier", "Verifying citations"),
    ("judge", "Ruling"),
]


def _brief_payload(brief: Brief | None) -> dict:
    brief = brief or Brief()
    return {
        "summary": brief.summary,
        "arguments": [
            {"point": a.point, "quote": a.quote, "source_url": a.source_url}
            for a in brief.arguments
        ],
    }


def _node_payload(node: str, update: dict) -> dict | None:
    """Translate a raw LangGraph node update into something the client can render."""
    if node == "decomposer":
        return {
            "sub_claims": [
                {"text": sc.text, "queries": sc.search_queries}
                for sc in update.get("sub_claims", [])
            ]
        }
    if node == "retriever":
        evidence = update.get("evidence", [])
        sources: dict[str, str] = {}
        for e in evidence:
            if e.source_url and e.source_url not in sources:
                sources[e.source_url] = e.source_title or e.source_url
        return {
            "chunks": len(evidence),
            "sources": [{"url": u, "title": t} for u, t in list(sources.items())[:12]],
            # every search / embedding call the RAG node actually made
            "trace": [t.as_dict() for t in telemetry.drain_tools()],
        }
    if node == "prosecutor":
        return _brief_payload(update.get("prosecution"))
    if node == "defender":
        return _brief_payload(update.get("defense"))
    if node == "citation_verifier":
        return {
            "verified": update.get("verified_citations", 0),
            "dropped": update.get("dropped_citations", 0),
            "prosecution": _brief_payload(update.get("verified_prosecution")),
            "defense": _brief_payload(update.get("verified_defense")),
            "stricken": [
                {"point": a.point, "quote": a.quote} for a in update.get("stricken", [])
            ],
        }
    if node == "judge":
        result = update.get("result")
        return result.model_dump(mode="json") if result else None
    return None


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _friendly_error(exc: Exception) -> str:
    """Turn provider failures into something a human can act on."""
    text = str(exc)
    if "429" in text or "rate limit" in text.lower():
        if "per-day" in text.lower() or "free-models-per-day" in text or "PerDay" in text:
            return (
                "Daily free-tier quota is exhausted. Wait for the provider's reset, switch "
                "LLM_API_KEY to another provider, or add credits."
            )
        return "Rate limited by the provider. Wait a moment and try again."
    if "401" in text or "invalid api key" in text.lower():
        return "The provider rejected the API key. Check LLM_API_KEY in your .env."
    if "TimeoutError" in type(exc).__name__ or "timeout" in text.lower():
        return "The request timed out before a verdict was reached. Try again."
    return f"{type(exc).__name__}: {text[:300]}"


def _event_stream(claim: str) -> Iterator[str]:
    import uuid

    yield _sse("start", {"claim": claim, "stages": [{"id": i, "label": l} for i, l in STAGES]})

    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    telemetry.drain()  # discard anything left over from a previous run
    run_started = time.monotonic()
    totals = {"tokens": 0, "cost_usd": 0.0, "calls": 0}

    # Replay a cached run verbatim: same stages, same trace, zero tokens spent.
    replay = cache.get("stream", claim)
    if replay:
        for ev in replay:
            yield _sse("stage", ev)
        yield _sse("done", {**(replay[-1].get("_totals") or {}), "cached": True})
        return

    captured: list[dict] = []
    pending: dict[str, list[telemetry.Call]] = {}
    try:
        mark = time.monotonic()
        for chunk in graph.stream({"claim": claim}, config=config, stream_mode="updates"):
            # Bucket by originating node: the advocates run concurrently, so arrival order
            # cannot be used to decide which call belongs to which branch.
            for n, calls in telemetry.bucket(telemetry.drain()).items():
                pending.setdefault(n, []).extend(calls)

            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                payload = _node_payload(node, update)
                if payload is None:
                    continue
                meta = telemetry.summarise(pending.pop(node, [])) or {}
                # Wall clock covers retrieval and embedding too, not just the model call.
                meta["elapsed_s"] = round(time.monotonic() - mark, 2)
                mark = time.monotonic()
                totals["tokens"] += meta.get("tokens", 0)
                totals["cost_usd"] += meta.get("cost_usd", 0.0)
                totals["calls"] += meta.get("calls", 0)
                event = {"node": node, "data": payload, "meta": meta}
                captured.append(event)
                yield _sse("stage", event)

        summary = {
            "elapsed_s": round(time.monotonic() - run_started, 2),
            "tokens": totals["tokens"],
            "cost_usd": round(totals["cost_usd"], 6),
            "calls": totals["calls"],
        }
        # Only cache complete runs — a partial sequence would replay as a broken verdict.
        if captured and any(e["node"] == "judge" for e in captured):
            captured[-1]["_totals"] = summary
            cache.put("stream", claim, captured)
        yield _sse("done", summary)
    except Exception as exc:  # noqa: BLE001 — surfaced to the client, never a stack trace
        yield _sse("error", {"message": _friendly_error(exc)})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/stream")
def stream(claim: str) -> StreamingResponse:
    """SSE progress stream. GET because EventSource only speaks GET."""
    return StreamingResponse(
        _event_stream(claim),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
