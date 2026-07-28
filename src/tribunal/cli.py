"""CLI: `tribunal "the earth is flat"` prints a verdict card."""

from __future__ import annotations

import argparse
import json

from .graph import verify_claim

_ICON = {"True": "✅", "False": "❌", "Misleading": "⚠️", "Unverifiable": "❓"}


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
    args = p.parse_args()

    claim = " ".join(args.claim)
    result = verify_claim(claim)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_card(result, claim)


if __name__ == "__main__":
    main()
