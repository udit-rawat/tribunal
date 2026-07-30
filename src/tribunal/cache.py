"""Small SQLite-backed cache with TTL.

Two things are worth caching, for different reasons:

* **searches** — retrieval dominates wall-clock time, and repeated queries across claims are common.
* **verdicts** — the LLM calls are what burn the daily free-tier quota, which has been the binding
  constraint on this project. A cached verdict costs nothing and returns instantly.

stdlib only: sqlite3 handles concurrent readers and gives durability without another dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from typing import Any

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(settings.cache_path, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            " ns TEXT NOT NULL, k TEXT NOT NULL, v TEXT NOT NULL, ts REAL NOT NULL,"
            " PRIMARY KEY (ns, k))"
        )
        _conn.commit()
    return _conn


def key_for(text: str) -> str:
    """Normalised hash so 'The Sun is a star.' and ' the sun is a STAR ' share an entry."""
    norm = re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", text).strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:32]


def get(ns: str, text: str) -> Any | None:
    if not settings.cache_enabled:
        return None
    ttl = settings.cache_ttl_hours * 3600
    with _lock:
        row = _connect().execute(
            "SELECT v, ts FROM entries WHERE ns=? AND k=?", (ns, key_for(text))
        ).fetchone()
    if not row:
        return None
    value, ts = row
    if ttl > 0 and time.time() - ts > ttl:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def put(ns: str, text: str, value: Any) -> None:
    if not settings.cache_enabled:
        return
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO entries (ns, k, v, ts) VALUES (?,?,?,?)",
            (ns, key_for(text), json.dumps(value), time.time()),
        )
        conn.commit()


def clear(ns: str | None = None) -> int:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM entries" + (" WHERE ns=?" if ns else ""), (ns,) if ns else ())
        conn.commit()
        return cur.rowcount
