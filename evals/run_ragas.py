"""RAGAS eval — scores retrieval quality and answer grounding, not just the final label.

    pip install -e '.[evals]'
    python evals/run_ragas.py --limit 5

`run_evals.py` asks "was the verdict right?". This asks "was the verdict *earned*?" — a system can
reach the correct label from irrelevant evidence, which looks fine on accuracy and fails in
production.

Metrics:

* **faithfulness**     — is every statement in the answer supported by the retrieved context?
  This is the direct hallucination measure.
* **answer_relevancy** — does the answer actually address the claim?

Both use an LLM judge, so each sample costs several extra calls. Default limit is deliberately
small; raise it only if your quota can take it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tribunal.config import settings  # noqa: E402
from tribunal.embeddings import embed, embed_one  # noqa: E402
from tribunal.graph import verify_claim_detailed  # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("golden.jsonl")


class LocalEmbeddings:
    """LangChain-compatible adapter over Tribunal's local ONNX embeddings.

    Keeps the eval free and offline instead of billing an embeddings API.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_one(text)


def build_judge():
    """LLM + embeddings wrappers RAGAS uses to grade samples."""
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=settings.strong_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
        timeout=settings.request_timeout,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(LocalEmbeddings())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAGAS faithfulness / relevancy scoring.")
    ap.add_argument("--limit", type=int, default=5, help="number of claims to score (default 5)")
    ap.add_argument("--json", dest="json_out", help="write per-sample scores to this path")
    args = ap.parse_args()

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy

    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    rows = rows[: args.limit]

    samples = []
    print(f"Collecting {len(rows)} samples through the pipeline...\n")
    for i, row in enumerate(rows, 1):
        try:
            out = verify_claim_detailed(row["claim"])
        except Exception as e:
            print(f" [{i:>2}] SKIP  {type(e).__name__} | {row['claim'][:50]}")
            continue
        if not out["contexts"]:
            print(f" [{i:>2}] SKIP  no evidence retrieved | {row['claim'][:50]}")
            continue
        print(f" [{i:>2}] ok    {out['verdict']['verdict']:<13} | {row['claim'][:50]}")
        samples.append(
            SingleTurnSample(
                user_input=row["claim"],
                response=out["answer"],
                retrieved_contexts=out["contexts"],
            )
        )

    if not samples:
        print("\nNo scorable samples collected — check quota and retrieval.")
        return

    llm, embeddings = build_judge()
    print(f"\nScoring {len(samples)} samples with RAGAS (LLM judge)...")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings)],
    )

    print(f"\n{'=' * 60}")
    print(result)
    if args.json_out:
        result.to_pandas().to_json(args.json_out, orient="records", indent=2)
        print(f"\nPer-sample scores written to {args.json_out}")


if __name__ == "__main__":
    main()
