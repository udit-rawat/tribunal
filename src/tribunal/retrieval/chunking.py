"""Word-window chunking with overlap. Simple and dependency-free for v1."""

from __future__ import annotations


def chunk_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks
