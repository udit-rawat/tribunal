"""Ephemeral LanceDB index, one table per claim. No server, file-based, low RAM."""

from __future__ import annotations

import time
import uuid

import lancedb

from .. import telemetry
from ..config import settings
from ..embeddings import embed, embed_one
from ..schemas import EvidenceChunk
from .chunking import chunk_text


def build_index(docs: list[dict]):
    """Chunk + embed all docs into a fresh LanceDB table. Returns the table (or None)."""
    rows: list[dict] = []
    for doc in docs:
        for ch in chunk_text(doc["text"]):
            rows.append({"text": ch, "url": doc.get("url", ""), "title": doc.get("title", "")})
    if not rows:
        return None

    started = time.monotonic()
    vectors = embed([r["text"] for r in rows])
    for r, v in zip(rows, vectors):
        r["vector"] = v

    db = lancedb.connect(settings.lancedb_path)
    name = f"claim_{uuid.uuid4().hex[:8]}"
    table = db.create_table(name, data=rows, mode="overwrite")
    telemetry.record_tool(
        telemetry.ToolCall(
            tool="embed+index",
            query=f"{len(docs)} docs → {len(rows)} chunks",
            results=len(rows),
            latency_s=time.monotonic() - started,
            note=f"{settings.embed_model.split('/')[-1]} · {len(vectors[0])}d · LanceDB",
        )
    )
    return table


def retrieve(table, query: str, k: int) -> list[EvidenceChunk]:
    """Top-k nearest chunks for a query."""
    if table is None:
        return []
    qv = embed_one(query)
    rows = table.search(qv).limit(k).to_list()
    return [
        EvidenceChunk(
            text=row["text"],
            source_url=row.get("url", ""),
            source_title=row.get("title", ""),
            score=float(row.get("_distance", 0.0)),
        )
        for row in rows
    ]
