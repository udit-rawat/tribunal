"""Optional Langfuse tracing. If keys are absent, returns no callbacks (a no-op)."""

from __future__ import annotations

from .config import settings


def get_callbacks() -> list:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse.callback import CallbackHandler

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        ]
    except Exception:
        # Langfuse version mismatch or import failure should never break verification.
        return []
