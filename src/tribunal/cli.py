"""CLI: `tribunal "the earth is flat"` prints a verdict card."""

from __future__ import annotations

import argparse
import json

from .graph import verify_claim

_ICON = {"True": "✅", "False": "❌", "Misleading": "⚠️", "Unverifiable": "❓"}


def _run_with_review(claim: str) -> dict:
    """HITL run: pause at the review gate on low-confidence cases and take human guidance."""
    import uuid

    from langgraph.types import Command

    from .config import settings
    from .graph import build_graph

    settings.hitl_enabled = True
    graph = build_graph()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    state = graph.invoke({"claim": claim}, config=cfg)
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print("\n  ⏸  HUMAN REVIEW REQUESTED")
        print(f"     {payload.get('reason')}")
        print(
            f"     evidence chunks: {payload.get('evidence_count')} · "
            f"prosecution pts: {payload.get('prosecution_points')} · "
            f"defense pts: {payload.get('defense_points')}"
        )
        note = input("     Your guidance (blank = approve as-is): ").strip()
        state = graph.invoke(Command(resume={"note": note}), config=cfg)

    result = state["result"].model_dump(mode="json")
    result["dropped_citations"] = state.get("dropped_citations", 0)
    return result


def _print_card(r: dict, claim: str) -> None:
    verdict = r["verdict"]
    print()
    print(f"  CLAIM: {claim}")
    print(f"  {_ICON.get(verdict, '•')} VERDICT: {verdict}  (confidence {r['confidence']:.0%})")
    print(f"  {r['summary']}")
    print()
    print(f"  Reasoning: {r['reasoning']}")
    if r.get("dropped_citations"):
        print(f"\n  🛡️  Citation Verifier dropped {r['dropped_citations']} unverified/hallucinated quote(s).")
    if r.get("citations"):
        print("\n  Citations:")
        for c in r["citations"]:
            mark = "+" if c["supports"] else "-"
            print(f"   [{mark}] \"{c['quote'][:160]}\"")
            if c.get("source_url"):
                print(f"       {c['source_url']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(prog="tribunal", description="Fact-check a factual claim.")
    p.add_argument("claim", nargs="+", help="the claim to verify")
    p.add_argument("--json", action="store_true", help="print raw JSON instead of a card")
    p.add_argument(
        "--review",
        action="store_true",
        help="human-in-the-loop: pause for your input on low-confidence cases",
    )
    args = p.parse_args()

    claim = " ".join(args.claim)
    result = _run_with_review(claim) if args.review else verify_claim(claim)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_card(result, claim)


if __name__ == "__main__":
    main()
