"""Content-addressed vector indexes.

An index is identified by the exact files it contains (name + content hash) and
the settings that change its chunks or vectors: embedding model, chunk size and
overlap, and LLM preprocessing/summaries. A finished index is never modified.
Asking for the same files and settings again reuses it; changing anything builds
a separate index, so chunks from different settings never mix and re-uploading
a file never duplicates it.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from .core import RAGConfig, RAGLab

log = logging.getLogger("raglab.indexes")
MARKER = "index_ready.json"
# Bump when loading or chunking changes, so indexes built the old way are rebuilt.
# 2: running PDF headers/footers are removed before chunking.
FORMAT_VERSION = 2
COLLECTION = "raglab"


class IndexNotBuilt(LookupError):
    pass


def file_digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def index_settings(cfg: RAGConfig) -> dict[str, Any]:
    settings = {"embedding_model": cfg.embedding_model, "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap, "preprocess": cfg.preprocess,
                "hierarchical": cfg.hierarchical}
    if cfg.preprocess or cfg.hierarchical:
        # The LLM writes index text in these modes, so it is part of the index.
        settings["chat_model"] = cfg.chat_model
    return settings


def index_key(cfg: RAGConfig, files: list[tuple[str, str]]) -> str:
    """files: (name, sha256) pairs. Order does not matter."""
    payload = json.dumps({"settings": index_settings(cfg), "files": sorted(files), "format": FORMAT_VERSION},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe_files(paths: list[str | Path]) -> list[tuple[str, str]]:
    return [(Path(p).name, file_digest(p)) for p in paths]


def read_index(root: str | Path, key: str) -> dict | None:
    marker = Path(root) / key / MARKER
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_indexes(root: str | Path) -> list[dict]:
    rows = [read_index(root, p.parent.name) for p in Path(root).glob(f"*/{MARKER}")]
    return sorted((r for r in rows if r), key=lambda r: r.get("built_at", ""), reverse=True)


def open_index(cfg: RAGConfig, root: str | Path, files: list[tuple[str, str]], **clients) -> RAGLab:
    """Open a finished index for querying. Raises IndexNotBuilt if it does not exist yet."""
    key = index_key(cfg, files)
    if read_index(root, key) is None:
        raise IndexNotBuilt(key)
    return RAGLab(replace(cfg, db_dir=str(Path(root) / key), collection_name=COLLECTION), **clients)


def build_index(cfg: RAGConfig, paths: list[str | Path], root: str | Path, *,
                on_progress: Callable[[str, int, int], None] | None = None, **clients) -> dict:
    """Build (or reuse) the index for these files and settings and return its record."""
    from .loaders import load_file

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = describe_files(paths)
    key = index_key(cfg, files)
    existing = read_index(root, key)
    if existing:
        log.info("Reusing index %s (%d chunks, built %s)", key, existing["chunks"], existing["built_at"])
        return existing
    with (root / f"{key}.lock").open("a") as lock:
        # Two jobs asking for the same index: the second waits, then reuses it.
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_index(root, key)
        if existing:
            log.info("Reusing index %s built by another job", key)
            return existing
        directory = root / key
        if directory.exists():
            log.warning("Removing incomplete index %s left by an interrupted build", key)
            shutil.rmtree(directory)
        started = time.perf_counter()
        documents = []
        for path in paths:
            parsed = load_file(path)
            for doc in parsed:
                doc.metadata["source"] = Path(path).name
            log.info("Read %s: %d sections", Path(path).name, len(parsed))
            documents.extend(parsed)
        lab = RAGLab(replace(cfg, db_dir=str(directory), collection_name=COLLECTION), **clients)
        chunks = lab.ingest_documents(documents, on_progress=on_progress)
        record = {"key": key, "settings": index_settings(cfg),
                  "files": [{"name": n, "sha256": d} for n, d in files],
                  "chunks": chunks, "seconds": round(time.perf_counter() - started, 1),
                  "built_at": datetime.now(timezone.utc).isoformat()}
        (directory / MARKER).write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.info("Built index %s: %d chunks from %d files in %.1fs", key, chunks, len(files), record["seconds"])
        return record


def delete_index(root: str | Path, key: str) -> None:
    if not key.isalnum():
        raise ValueError("Invalid index key")
    shutil.rmtree(Path(root) / key, ignore_errors=True)
    (Path(root) / f"{key}.lock").unlink(missing_ok=True)
