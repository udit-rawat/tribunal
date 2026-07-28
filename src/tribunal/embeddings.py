"""Local MiniLM embeddings via fastembed (ONNX). Free, offline, ~50MB — no torch.

384-dim, normalized by fastembed for sentence-transformers models. Ideal for M1/8GB.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    return [v.tolist() for v in model.embed(list(texts))]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
