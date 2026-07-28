"""LangGraph shared state.

Prosecutor and Defender run as parallel branches, but they write to *distinct* keys
(`prosecution` / `defense`), so LangGraph merges their updates without a custom reducer.
"""

from __future__ import annotations

from typing import TypedDict

from .schemas import Brief, EvidenceChunk, SubClaim, VerdictResult


class TribunalState(TypedDict, total=False):
    claim: str
    sub_claims: list[SubClaim]
    evidence: list[EvidenceChunk]
    # adversarial briefs (parallel branches)
    prosecution: Brief
    defense: Brief
    # after the Citation Verifier drops any quote not found in the evidence
    verified_prosecution: Brief
    verified_defense: Brief
    dropped_citations: int
    human_note: str  # optional guidance injected by a human reviewer (HITL)
    result: VerdictResult
