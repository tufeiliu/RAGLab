"""Persistent SQLite caches for embeddings and chat-model responses.

Embedding vectors are deterministic for a given model and text, so caching them
is always safe: rebuilding an index with different chunking only pays for chunks
it has never seen. Response caching returns the stored answer for an identical
prompt and model; it makes reruns free and repeatable but hides run-to-run
variation, so it is optional.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any
import warnings

from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumps, loads

log = logging.getLogger("raglab.cache")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    # WAL lets background jobs and the app read and write concurrently.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class CachedEmbeddings(Embeddings):
    """Wrap any LangChain embedder with a persistent per-model vector cache."""

    def __init__(self, embedder: Embeddings, model: str, path: str | Path):
        self.embedder, self.model, self.path = embedder, model, Path(path)
        self.hits = self.misses = 0
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")

    def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        found = {}
        with closing(_connect(self.path)) as conn:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                marks = ",".join("?" * len(chunk))
                for key, vector in conn.execute(f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", chunk):
                    found[key] = json.loads(vector)
        return found

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [_key(self.model, t) for t in texts]
        found = self._lookup(keys)
        missing = list(dict.fromkeys(k for k in keys if k not in found))
        if missing:
            by_key = dict(zip(keys, texts))
            vectors = self.embedder.embed_documents([by_key[k] for k in missing])
            with closing(_connect(self.path)) as conn, conn:
                conn.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                 [(k, json.dumps(v)) for k, v in zip(missing, vectors)])
            found.update(zip(missing, vectors))
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        log.debug("Embedded %d texts with %s: %d cached, %d new", len(texts), self.model,
                  len(texts) - len(missing), len(missing))
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SQLiteResponseCache(BaseCache):
    """LangChain chat-model cache keyed by prompt and model parameters."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def lookup(self, prompt: str, llm_string: str) -> Any:
        with closing(_connect(self.path)) as conn:
            row = conn.execute("SELECT value FROM responses WHERE key = ?", (_key(llm_string, prompt),)).fetchone()
        with self._lock:
            if row:
                self.hits += 1
            else:
                self.misses += 1
        if not row:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # loads() is flagged beta; the payload is our own.
            return loads(row[0], allowed_objects="core")

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)",
                         (_key(llm_string, prompt), dumps(return_val)))

    def clear(self, **kwargs: Any) -> None:
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("DELETE FROM responses")


def cache_stats(cache_dir: str | Path) -> dict[str, int]:
    """Row counts and on-disk size, for display."""
    cache_dir = Path(cache_dir)
    stats = {}
    for name, table in (("embeddings", "embeddings"), ("responses", "responses")):
        path = cache_dir / f"{name}.sqlite"
        count = 0
        if path.is_file():
            with closing(_connect(path)) as conn:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.OperationalError:
                    count = 0
        size = sum(p.stat().st_size for p in cache_dir.glob(f"{name}.sqlite*") if p.is_file())
        stats[name] = count
        stats[f"{name}_bytes"] = size
    return stats
