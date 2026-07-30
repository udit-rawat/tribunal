"""Verdict-accuracy eval against the golden set.

    python evals/run_evals.py                  # full set
    python evals/run_evals.py --limit 8        # first 8 claims (quota-friendly)
    python evals/run_evals.py --sleep 4        # pace requests to dodge rate limits
    python evals/run_evals.py --json out.json  # also write raw results

Reports two accuracy figures:

* **strict**   — exact label match.
* **polarity** — collapses the labels into {supported, contested, unknown}. `False` and
  `Misleading` both mean "do not trust this claim as stated", and the boundary between them is
  genuinely subjective, so polarity measures whether the system got the *direction* right
  independent of that judgment call.

API errors (rate limits, timeouts) are counted and reported separately rather than scored as wrong
answers — a quota failure is not a reasoning failure, and folding the two together flatters or
punishes the model for the wrong reason.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tribunal.graph import verify_claim  # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("golden.jsonl")
LABELS = ["True", "False", "Misleading", "Unverifiable"]

# Collapse the 4-way taxonomy into directional buckets.
POLARITY = {
    "True": "supported",
    "False": "contested",
    "Misleading": "contested",
    "Unverifiable": "unknown",
}


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _bar(label: str, hit: int, total: int, width: int = 18) -> str:
    filled = round(width * hit / total) if total else 0
    pct = f"{hit / total:.0%}" if total else "n/a"
    return f"  {label:<14} {'#' * filled}{'.' * (width - filled)}  {hit}/{total} ({pct})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the golden-set verdict eval.")
    ap.add_argument("--limit", type=int, help="only run the first N claims")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between claims")
    ap.add_argument("--json", dest="json_out", help="write raw results to this path")
    args = ap.parse_args()

    rows = load_golden()
    if args.limit:
        rows = rows[: args.limit]

    results: list[dict] = []
    strict_hits = 0
    polarity_hits = 0
    errors = 0
    kept_quotes = 0
    dropped_quotes = 0
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # label -> [hits, total]
    confusion: Counter[tuple[str, str]] = Counter()

    # flush=True throughout: stdout is block-buffered when piped, which otherwise hides all
    # progress until the run ends — unusable for a job that takes minutes per claim.
    print(f"Running {len(rows)} claims through Tribunal...\n", flush=True)
    for i, row in enumerate(rows, 1):
        expected = row["expected"]
        got: str | None = None
        err: str | None = None
        try:
            out = verify_claim(row["claim"])
            got = out["verdict"]
            kept_quotes += out.get("verified_citations", 0)
            dropped_quotes += out.get("dropped_citations", 0)
        except Exception as e:  # keep the harness alive; record and move on
            err = f"{type(e).__name__}: {e}"
            errors += 1

        if err:
            mark, shown = "!", "ERROR"
        else:
            strict_ok = got == expected
            polar_ok = POLARITY.get(got) == POLARITY[expected]
            strict_hits += strict_ok
            polarity_hits += polar_ok
            per_label[expected][0] += strict_ok
            per_label[expected][1] += 1
            confusion[(expected, got)] += 1
            mark = "PASS" if strict_ok else ("~" if polar_ok else "FAIL")
            shown = got or "?"

        print(
            f" [{i:>2}] {mark:<4} expected={expected:<13} got={shown:<13} | {row['claim'][:52]}",
            flush=True,
        )
        results.append({"claim": row["claim"], "expected": expected, "got": got, "error": err})

        if args.sleep and i < len(rows):
            time.sleep(args.sleep)

    scored = len(rows) - errors
    print(f"\n{'=' * 66}")
    if scored:
        print(f"Strict accuracy   : {strict_hits}/{scored} = {strict_hits / scored:.0%}")
        print(f"Polarity accuracy : {polarity_hits}/{scored} = {polarity_hits / scored:.0%}")
    else:
        print("Nothing scored — every claim errored.")
    if errors:
        print(f"API errors        : {errors}/{len(rows)} (excluded from accuracy)")

    total_quotes = kept_quotes + dropped_quotes
    if total_quotes:
        # Deterministic, not LLM-judged: share of advocate quotes found verbatim in the evidence.
        print(
            f"Quote grounding   : {kept_quotes}/{total_quotes} = "
            f"{kept_quotes / total_quotes:.0%} verified verbatim "
            f"({dropped_quotes} dropped as unsupported)"
        )

    if scored:
        print("\nPer-label (strict):")
        for label in LABELS:
            hits, total = per_label[label]
            if total:
                print(_bar(label, hits, total))

        misses = {k: v for k, v in confusion.items() if k[0] != k[1]}
        if misses:
            print("\nConfusion (expected -> got):")
            for (exp, got_), n in sorted(misses.items(), key=lambda kv: -kv[1]):
                print(f"  {exp:<13} -> {got_:<13} x{n}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nRaw results written to {args.json_out}")


if __name__ == "__main__":
    main()
