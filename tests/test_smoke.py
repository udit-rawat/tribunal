"""Offline smoke tests — no network, no API key. Verify wiring, not model quality."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def test_chunking_overlap():
    from tribunal.retrieval.chunking import chunk_text

    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)


def test_schemas_roundtrip():
    from tribunal.schemas import Verdict, VerdictResult

    r = VerdictResult(verdict=Verdict.TRUE, confidence=0.9, summary="s", reasoning="r")
    assert r.model_dump()["verdict"] == "True"


def test_graph_builds():
    from tribunal.graph import build_graph

    graph = build_graph()
    assert graph is not None


def test_citation_verifier_drops_hallucinations():
    """Deterministic guardrail: keep quotes found in evidence, drop the invented one."""
    from tribunal.agents import citation_verifier
    from tribunal.schemas import ArgumentPoint, Brief, EvidenceChunk

    evidence = [EvidenceChunk(text="Honey has very low moisture and high acidity, preventing spoilage.")]
    real = ArgumentPoint(point="Honey resists spoilage", quote="low moisture and high acidity", source_url="x")
    fake = ArgumentPoint(point="Honey is radioactive", quote="honey emits gamma radiation constantly", source_url="y")

    state = {
        "evidence": evidence,
        "prosecution": Brief(arguments=[fake], summary="p"),
        "defense": Brief(arguments=[real], summary="d"),
    }
    out = citation_verifier(state)
    assert len(out["verified_defense"].arguments) == 1  # real quote kept
    assert len(out["verified_prosecution"].arguments) == 0  # fabricated quote dropped
    assert out["dropped_citations"] == 1
