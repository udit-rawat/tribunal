"""Central configuration. Reads .env once and exposes typed settings + per-agent models."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider (OpenAI-compatible endpoint: OpenRouter, Google AI Studio, etc.)
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"

    # Per-agent models (see .env.example)
    strong_model: str = "google/gemini-2.0-flash-exp:free"
    fast_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    # Fallback used when the primary model hits a daily quota cap (separate quota pool).
    fallback_model: str = ""

    # Embeddings (local)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    lancedb_path: str = ".lancedb"
    top_k: int = 5
    max_sources_per_query: int = 4

    # Langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    request_timeout: int = 60
    # Separate, shorter budget for evidence search — these run on the request path and a hung
    # socket would otherwise stall the whole pipeline with no upper bound.
    search_timeout: int = 15

    # Cache: searches for speed, verdicts for quota. TTL of 0 disables expiry.
    cache_enabled: bool = True
    cache_path: str = ".tribunal-cache.db"
    cache_ttl_hours: int = 168  # a week — long enough to demo, short enough to catch stale facts

    # How many evidence searches to run at once. They are independent network calls, so running
    # them sequentially made retrieval the slowest part of the pipeline.
    search_concurrency: int = 6

    # Human-in-the-loop: pause before the Judge on low-confidence cases for a human to weigh in.
    hitl_enabled: bool = False
    hitl_min_evidence: int = 2  # fewer retrieved chunks than this = "uncertain", trigger review


settings = Settings()

# Which model each agent uses. Decomposer + Judge do the reasoning-heavy work.
AGENT_MODELS: dict[str, str] = {
    "decomposer": settings.strong_model,
    "retriever": settings.fast_model,
    "prosecutor": settings.fast_model,
    "defender": settings.fast_model,
    "judge": settings.strong_model,
}
