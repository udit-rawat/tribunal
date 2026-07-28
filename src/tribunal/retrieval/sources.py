"""Hybrid evidence sourcing: Wikipedia first, DuckDuckGo web fallback.

Every network call is wrapped so a single failing source never crashes a verification.
"""

from __future__ import annotations

import httpx

from ..config import settings

_UA = {"User-Agent": "Tribunal/0.1 (fact-check portfolio project)"}
_WIKI_API = "https://en.wikipedia.org/w/api.php"


def wikipedia_search(query: str, limit: int = 2) -> list[dict]:
    """Search Wikipedia, return plain-text extracts for the top hits."""
    docs: list[dict] = []
    try:
        with httpx.Client(timeout=settings.request_timeout, headers=_UA) as c:
            hits = (
                c.get(
                    _WIKI_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": limit,
                        "format": "json",
                    },
                )
                .json()
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
    except Exception:
        pass
    return docs


def ddg_search(query: str, limit: int = 4) -> list[dict]:
    """Keyless DuckDuckGo web search. Returns snippet-level bodies."""
    docs: list[dict] = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS

        with DDGS() as d:
            for r in d.text(query, max_results=limit):
                body = r.get("body", "")
                if body:
                    docs.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", "") or r.get("url", ""),
                            "text": body,
                        }
                    )
    except Exception:
        pass
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
