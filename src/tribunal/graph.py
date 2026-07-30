"""LangGraph wiring + the top-level `verify_claim` entrypoint used by all three interfaces.

v2 topology (Prosecutor and Defender fan out in parallel, then fan back in), with an optional
human-in-the-loop review gate before the Judge:

    decomposer -> retriever -> prosecutor ┐
                            └-> defender  ┴-> citation_verifier -> review_gate -> judge
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    citation_verifier,
    decomposer,
    defender,
    judge,
    prosecutor,
    retriever,
    review_gate,
)
from .observability import get_callbacks
from .state import TribunalState


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(TribunalState)
    g.add_node("decomposer", decomposer)
    g.add_node("retriever", retriever)
    g.add_node("prosecutor", prosecutor)
    g.add_node("defender", defender)
    g.add_node("citation_verifier", citation_verifier)
    g.add_node("review_gate", review_gate)
    g.add_node("judge", judge)

    g.add_edge(START, "decomposer")
    g.add_edge("decomposer", "retriever")
    # fan-out: retriever kicks off both advocates in the same superstep
    g.add_edge("retriever", "prosecutor")
    g.add_edge("retriever", "defender")
    # fan-in: citation_verifier waits for BOTH advocates before running
    g.add_edge("prosecutor", "citation_verifier")
    g.add_edge("defender", "citation_verifier")
    g.add_edge("citation_verifier", "review_gate")  # HITL pause happens here (if enabled)
    g.add_edge("review_gate", "judge")
    g.add_edge("judge", END)
    # Checkpointer = durable state; required for the review_gate interrupt/resume to work.
    return g.compile(checkpointer=MemorySaver())


def _run(claim: str) -> dict:
    """Invoke the graph once and return the raw final state."""
    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "callbacks": get_callbacks()}
    return graph.invoke({"claim": claim}, config=config)


def verify_claim(claim: str) -> dict:
    """Run the full pipeline and return the verdict as a plain dict.

    HITL is off by default here, so this never pauses. Interactive review lives in the CLI.
    """
    final = _run(claim)
    # mode="json" serializes the Verdict enum to its string value ("False", not "Verdict.FALSE")
    result = final["result"].model_dump(mode="json")
    result["dropped_citations"] = final.get("dropped_citations", 0)
    result["verified_citations"] = final.get("verified_citations", 0)
    return result


def verify_claim_detailed(claim: str) -> dict:
    """Same as `verify_claim` but also returns the retrieved contexts.

    Used by the RAGAS eval, which needs the evidence that grounded the verdict in order to score
    faithfulness (is the answer supported by the retrieved context?).
    """
    final = _run(claim)
    result = final["result"]
    return {
        "verdict": result.model_dump(mode="json"),
        "contexts": [e.text for e in final.get("evidence", [])],
        "answer": f"{result.verdict.value}. {result.summary} {result.reasoning}".strip(),
        "dropped_citations": final.get("dropped_citations", 0),
    }
