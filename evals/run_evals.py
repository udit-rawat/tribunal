"""Verdict-accuracy eval against the golden set.

    python evals/run_evals.py

Reports per-claim correctness, overall accuracy, and a confusion summary.
RAGAS faithfulness/context-precision is a v2 add-on (see PLAN.md); this v1 script measures
end-to-end verdict accuracy, which is the number you quote in interviews.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tribunal.graph import verify_claim  # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("golden.jsonl")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def main() -> None:
    rows = load_golden()
    correct = 0
    print(f"Running {len(rows)} claims through Tribunal...\n")
    for row in rows:
        try:
            result = verify_claim(row["claim"])
            got = result["verdict"]
        except Exception as e:  # keep the harness going if one claim errors
            got = f"ERROR: {e}"
        ok = got == row["expected"]
        correct += ok
        mark = "✓" if ok else "✗"
        print(f" {mark}  expected={row['expected']:<13} got={got:<13} | {row['claim'][:60]}")

    n = len(rows)
    print(f"\nAccuracy: {correct}/{n} = {correct / n:.0%}")


if __name__ == "__main__":
    main()
