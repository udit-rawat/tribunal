"""Pydantic schemas. These double as Instructor `response_model`s for structured output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    MISLEADING = "Misleading"
    UNVERIFIABLE = "Unverifiable"


class SubClaim(BaseModel):
    text: str = Field(description="One atomic, independently checkable statement.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="1-3 web/Wikipedia search queries that would surface evidence.",
    )


class Decomposition(BaseModel):
    sub_claims: list[SubClaim]


class EvidenceChunk(BaseModel):
    text: str
    source_url: str = ""
    source_title: str = ""
    score: float = 0.0  # lower distance = closer match


class SubClaimAnalysis(BaseModel):
    sub_claim: str
    supporting: list[str] = Field(
        default_factory=list, description="Points from the evidence that SUPPORT the sub-claim."
    )
    refuting: list[str] = Field(
        default_factory=list, description="Points from the evidence that REFUTE the sub-claim."
    )
    assessment: str = Field(description="Neutral one-line read on where the evidence leans.")


class Analysis(BaseModel):
    per_sub_claim: list[SubClaimAnalysis]
    overall_reasoning: str


class ArgumentPoint(BaseModel):
    point: str = Field(description="One argument for this side.")
    quote: str = Field(description="A VERBATIM snippet from the evidence that backs this point.")
    source_url: str = ""


class Brief(BaseModel):
    arguments: list[ArgumentPoint] = Field(default_factory=list)
    summary: str = ""


class Citation(BaseModel):
    quote: str = Field(description="Verbatim snippet from the evidence.")
    source_url: str = ""
    supports: bool = Field(description="True if this quote supports the claim, False if it refutes.")


class VerdictResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(description="One-line ruling.")
    reasoning: str
    citations: list[Citation] = Field(default_factory=list)
