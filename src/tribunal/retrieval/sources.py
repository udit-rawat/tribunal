"""Hybrid evidence sourcing: Wikipedia first, DuckDuckGo web fallback.

Every network call is wrapped so a single failing source never crashes a verification.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

import httpx

from ..config import settings

# Wikimedia's robot policy (https://w.wiki/4wJS) rejects generic agents with 403 — the UA must
# identify the tool and give a contact route, or every request silently returns no results.
_UA = {"User-Agent": "Tribunal/0.1 (https://github.com/udit-rawat/tribunal)"}
_WIKI_API = "https://en.wikipedia.org/w/api.php"

log = logging.getLogger("tribunal.retrieval")


def _with_deadline(fn: Callable[[], Any], timeout: float) -> Any:
    """Run `fn` on a daemon thread and give up after `timeout` seconds.

    Search clients expose no reliable timeout knob (`ddgs` takes opaque **kwargs), and a throttled
    or blackholed endpoint otherwise blocks the request path forever. The worker is a daemon so an
    abandoned socket can never hold up interpreter exit either.
    """
    box: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            box.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box.put(("err", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        kind, value = box.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"search exceeded {timeout}s budget") from None
    if kind == "err":
        raise value
    return value


def wikipedia_search(query: str, limit: int = 2) -> list[dict]:
    """Search Wikipedia, return plain-text extracts for the top hits."""
    docs: list[dict] = []
    try:
        with httpx.Client(timeout=settings.request_timeout, headers=_UA) as c:
            resp = c.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                },
            )
            # Surface hard failures (e.g. 403 from the robot policy) instead of degrading in
            # silence — a dead primary source otherwise looks identical to "no results found".
            resp.raise_for_status()
            hits = (
                resp.json()
                .get("query", {})
                .get("search", [])
            )
            for hit in hits:
                title = hit["title"]
                pages = (
                    c.get(
                        _WIKI_API,
                        params={
                            "action": "query",
                            "prop": "extracts",
                            "explaintext": 1,
                            "titles": title,
                            "exlimit": 1,
                            "format": "json",
                        },
                    )
                    .json()
                    .get("query", {})
                    .get("pages", {})
                )
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract:
                        docs.append(
                            {
                                "title": title,
                                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                "text": extract,
                            }
                        )
    except Exception as exc:
        log.warning("wikipedia search failed for %r: %s", query, exc)
    return docs


def ddg_search(query: str, limit: int = 4) -> list[dict]:
    """Keyless DuckDuckGo web search. Returns snippet-level bodies.

    The explicit timeout matters: without it a throttled or blackholed connection blocks the whole
    pipeline indefinitely, since this runs on the request path.
    """
    docs: list[dict] = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS

        def _fetch() -> list[dict]:
            with DDGS() as d:
                return list(d.text(query, max_results=limit))

        for r in _with_deadline(_fetch, settings.search_timeout):
            body = r.get("body", "")
            if body:
                docs.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", "") or r.get("url", ""),
                        "text": body,
                    }
                )
    except Exception as exc:
        log.warning("duckduckgo search failed for %r: %s", query, exc)
    return docs


def gather_sources(queries: list[str]) -> list[dict]:
    """Wikipedia-first hybrid gather across all queries, de-duplicated by URL."""
    docs: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        results = wikipedia_search(q, limit=2)
        if results:
            results += ddg_search(q, limit=2)  # augment with a couple web hits
        else:
            results = ddg_search(q, limit=settings.max_sources_per_query)  # fall back fully
        for d in results:
            url = d.get("url", "")
            if url and url not in seen and d.get("text"):
                seen.add(url)
                docs.append(d)
    return docs
