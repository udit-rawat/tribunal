"""The v2 agents. Each is a LangGraph node: reads state, returns a partial state update.

Pipeline: Decomposer -> Retriever -> [Prosecutor || Defender] -> Citation Verifier -> Judge
"""

from __future__ import annotations

import re

from .config import AGENT_MODELS, settings
from .llm import structured
from .schemas import Brief, Decomposition, EvidenceChunk, VerdictResult
from .retrieval.sources import gather_sources
from .retrieval.store import build_index, retrieve
from .state import TribunalState


def _format_evidence(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "NO EVIDENCE FOUND."
    return "\n\n".join(
        f"[{i}] ({e.source_title} — {e.source_url})\n{e.text}"
        for i, e in enumerate(evidence, 1)
    )


# --------------------------------------------------------------------------- #
# 1. Decomposer
# --------------------------------------------------------------------------- #
def decomposer(state: TribunalState) -> dict:
    """Claim -> atomic sub-claims + search queries."""
    result = structured(
        model=AGENT_MODELS["decomposer"],
        response_model=Decomposition,
        system=(
            "You break a factual claim into atomic, independently checkable sub-claims. "
            "For each, write 1-3 neutral search queries that would surface evidence either way. "
            "Do not take a side. Keep sub-claims minimal and non-overlapping."
        ),
        user=f"Claim: {state['claim']}",
    )
    return {"sub_claims": result.sub_claims}


# --------------------------------------------------------------------------- #
# 2. Retriever (RAG node)
# --------------------------------------------------------------------------- #
def retriever(state: TribunalState) -> dict:
    """Gather hybrid sources, build an ephemeral index, retrieve per sub-claim."""
    sub_claims = state["sub_claims"]
    queries: list[str] = []
    for sc in sub_claims:
        queries.extend(sc.search_queries or [sc.text])

    docs = gather_sources(queries)
    table = build_index(docs)

    evidence: list[EvidenceChunk] = []
    seen: set[str] = set()
    for sc in sub_claims:
        for chunk in retrieve(table, sc.text, settings.top_k):
            if chunk.text not in seen:
                seen.add(chunk.text)
                evidence.append(chunk)
    return {"evidence": evidence}


# --------------------------------------------------------------------------- #
# 3a/3b. Prosecutor + Defender (parallel adversarial branches)
# --------------------------------------------------------------------------- #
def _make_brief(state: TribunalState, agent_key: str, stance: str) -> Brief:
    evidence = state.get("evidence", [])
    return structured(
        model=AGENT_MODELS[agent_key],
        response_model=Brief,
        system=(
            f"You are the {stance}. Build the strongest possible case that the claim is "
            f"{'FALSE or unsupported' if stance == 'Prosecutor' else 'TRUE and well-supported'}. "
            "Every argument MUST include a `quote` copied VERBATIM from the provided evidence — "
            "do not paraphrase quotes, do not invent them. If the evidence gives you nothing for "
            "your side, return an empty arguments list and say so in the summary. Never fabricate."
        ),
        user=f"Claim: {state['claim']}\n\nEvidence:\n{_format_evidence(evidence)}",
    )


def prosecutor(state: TribunalState) -> dict:
    return {"prosecution": _make_brief(state, "prosecutor", "Prosecutor")}


def defender(state: TribunalState) -> dict:
    return {"defense": _make_brief(state, "defender", "Defender")}


# --------------------------------------------------------------------------- #
# 4. Citation Verifier (deterministic guardrail — no LLM call)
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _quote_supported(quote: str, evidence_norm: list[str]) -> bool:
    """A quote is verified if it (or a long contiguous span of it) appears in the evidence."""
    q = _normalize(quote)
    if len(q) < 12:  # too short to be a meaningful, checkable citation
        return False
    if any(q in t for t in evidence_norm):
        return True
    span = q[:60]  # tolerate trailing truncation/ellipsis differences
    return any(span in t for t in evidence_norm)


def citation_verifier(state: TribunalState) -> dict:
    """Drop any argument whose quote is NOT found verbatim in the retrieved evidence."""
    evidence_norm = [_normalize(e.text) for e in state.get("evidence", [])]

    def verify(brief: Brief) -> tuple[Brief, int]:
        kept = [p for p in brief.arguments if _quote_supported(p.quote, evidence_norm)]
        dropped = len(brief.arguments) - len(kept)
        return Brief(arguments=kept, summary=brief.summary), dropped

    vp, dp = verify(state.get("prosecution", Brief()))
    vd, dd = verify(state.get("defense", Brief()))
    return {"verified_prosecution": vp, "verified_defense": vd, "dropped_citations": dp + dd}


# --------------------------------------------------------------------------- #
# 4b. Review Gate (human-in-the-loop, durable state)
# --------------------------------------------------------------------------- #
def _is_uncertain(state: TribunalState) -> bool:
    """Uncertain = thin evidence or neither side has any verified support."""
    if len(state.get("evidence", [])) < settings.hitl_min_evidence:
        return True
    verified = state.get("verified_prosecution", Brief()).arguments + state.get(
        "verified_defense", Brief()
    ).arguments
    return len(verified) == 0


def review_gate(state: TribunalState) -> dict:
    """Pause for a human before judging low-confidence cases (only when HITL is enabled).

    Uses LangGraph's dynamic `interrupt()`, so the run halts on a checkpoint and resumes
    with whatever the reviewer passes back via `Command(resume=...)`.
    """
    if not settings.hitl_enabled or not _is_uncertain(state):
        return {}

    from langgraph.types import interrupt  # local import keeps core path dependency-light

    note = interrupt(
        {
            "reason": "low-confidence: thin or unverified evidence — human review requested",
            "claim": state["claim"],
            "evidence_count": len(state.get("evidence", [])),
            "prosecution_points": len(state.get("verified_prosecution", Brief()).arguments),
            "defense_points": len(state.get("verified_defense", Brief()).arguments),
        }
    )
    text = note.get("note", "") if isinstance(note, dict) else str(note or "")
    return {"human_note": text}


# --------------------------------------------------------------------------- #
# 5. Judge
# --------------------------------------------------------------------------- #
def _format_brief(brief: Brief) -> str:
    if not brief.arguments:
        return f"(no verified arguments)\nSummary: {brief.summary}"
    lines = [f"- {p.point}\n    quote: \"{p.quote}\" [{p.source_url}]" for p in brief.arguments]
    return "\n".join(lines) + f"\nSummary: {brief.summary}"


def judge(state: TribunalState) -> dict:
    """Weigh the two verified briefs and rule. Citations must come from verified evidence."""
    prosecution = state.get("verified_prosecution", Brief())
    defense = state.get("verified_defense", Brief())
    human_note = state.get("human_note", "")
    human_block = (
        f"\n\nA human reviewer flagged this case and added guidance: {human_note}\n"
        "Weigh it, but stay grounded in the verified evidence."
        if human_note
        else ""
    )
    result = structured(
        model=AGENT_MODELS["judge"],
        response_model=VerdictResult,
        system=(
            "You are the Judge. You are given a claim and two briefs whose quotes have already been "
            "verified against the source evidence. Rule with one verdict: True, False, Misleading, "
            "or Unverifiable.\n"
            "- Unverifiable: neither side has real supporting evidence.\n"
            "- Misleading: literally true but creates a false impression.\n"
            "Weigh the strength of the VERIFIED evidence on each side, not rhetoric. Every citation "
            "you output must come from the briefs below. Set confidence honestly in [0,1]."
        ),
        user=(
            f"Claim: {state['claim']}\n\n"
            f"PROSECUTION (claim is false/unsupported):\n{_format_brief(prosecution)}\n\n"
            f"DEFENSE (claim is true/supported):\n{_format_brief(defense)}"
            f"{human_block}"
        ),
    )
    return {"result": result}
