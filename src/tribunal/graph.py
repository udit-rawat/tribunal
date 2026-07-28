"""LangGraph wiring + the top-level `verify_claim` entrypoint used by all three interfaces.

v2 topology (Prosecutor and Defender fan out in parallel, then fan back in):

    decomposer -> retriever -> prosecutor ┐
                            └-> defender  ┴-> citation_verifier -> judge
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .agents import (
    citation_verifier,
    decomposer,
    defender,
    judge,
    prosecutor,
    retriever,
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
    g.add_node("judge", judge)

    g.add_edge(START, "decomposer")
    g.add_edge("decomposer", "retriever")
    # fan-out: retriever kicks off both advocates in the same superstep
    g.add_edge("retriever", "prosecutor")
    g.add_edge("retriever", "defender")
    # fan-in: citation_verifier waits for BOTH advocates before running
    g.add_edge("prosecutor", "citation_verifier")
    g.add_edge("defender", "citation_verifier")
    g.add_edge("citation_verifier", "judge")
    g.add_edge("judge", END)
    return g.compile()


def verify_claim(claim: str) -> dict:
    """Run the full pipeline and return the verdict as a plain dict."""
    graph = build_graph()
    final = graph.invoke({"claim": claim}, config={"callbacks": get_callbacks()})
    # mode="json" serializes the Verdict enum to its string value ("False", not "Verdict.FALSE")
    result = final["result"].model_dump(mode="json")
    result["dropped_citations"] = final.get("dropped_citations", 0)
    return result
