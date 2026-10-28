"""Named document collections stored on disk.

raglab_data/collections/<slug>/
    collection.json   name, files (with sha256) and test sets
    files/            uploaded documents, stored once per distinct content
    tests/            test-question sets (JSONL)
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

from raglab_app import COLLECTIONS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = "".join(ch for ch in base if ch.isalnum() or ch in "._- ").strip() or "upload.txt"
    return base.lstrip(".") or "upload.txt"


def _dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,60}", slug or ""):
        raise ValueError("Invalid collection id")
    return COLLECTIONS / slug


def load(slug: str) -> dict:
    meta = json.loads((_dir(slug) / "collection.json").read_text(encoding="utf-8"))
    meta["slug"] = slug
    return meta


def list_collections() -> list[dict]:
    rows = []
    for path in sorted(COLLECTIONS.glob("*/collection.json")):
        try:
            rows.append(load(path.parent.name))
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda c: c["name"].lower())


def create(name: str, *, benchmark: str | None = None) -> str:
    name = name.strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("Collection name must have 1–80 characters")
    if any(c["name"].lower() == name.lower() for c in list_collections()):
        raise ValueError(f"A collection called {name!r} already exists")
    stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "collection"
    slug = stem
    while (COLLECTIONS / slug).exists():
        slug = f"{stem}-{uuid.uuid4().hex[:4]}"
    directory = _dir(slug)
    (directory / "files").mkdir(parents=True)
    (directory / "tests").mkdir()
    meta = {"name": name, "created_at": _now(), "files": {}, "tests": {}}
    if benchmark:
        meta["benchmark"] = benchmark  # marks a collection loaded from raglab_app.benchmarks
    _write(directory / "collection.json", meta)
    return slug


def add_files(slug: str, uploads: list[tuple[str, bytes]]) -> tuple[list[str], list[str]]:
    """Store uploads; returns (added, skipped). Identical content is stored only once."""
    directory = _dir(slug)
    meta = load(slug)
    known = {info["sha256"]: name for name, info in meta["files"].items()}
    added, skipped = [], []
    for original, data in uploads:
        digest = hashlib.sha256(data).hexdigest()
        if digest in known:
            skipped.append(f"{original} (same content as {known[digest]})")
            continue
        name = safe_filename(original)
        stem, suffix = Path(name).stem, Path(name).suffix
        n = 2
        while name in meta["files"]:
            name = f"{stem} ({n}){suffix}"
            n += 1
        (directory / "files" / name).write_bytes(data)
        meta["files"][name] = {"sha256": digest, "bytes": len(data), "added_at": _now()}
        known[digest] = name
        added.append(name)
    meta.pop("slug")
    _write(directory / "collection.json", meta)
    return added, skipped


def remove_file(slug: str, name: str) -> None:
    directory = _dir(slug)
    meta = load(slug)
    if name not in meta["files"]:
        raise ValueError(f"No file named {name!r}")
    (directory / "files" / name).unlink(missing_ok=True)
    del meta["files"][name]
    meta.pop("slug")
    _write(directory / "collection.json", meta)


def file_paths(slug: str) -> list[Path]:
    directory = _dir(slug)
    return [directory / "files" / name for name in sorted(load(slug)["files"])]


def file_identity(slug: str) -> list[tuple[str, str]]:
    """(name, sha256) pairs: what raglab.indexes uses to identify an index."""
    return sorted((name, info["sha256"]) for name, info in load(slug)["files"].items())


def test_set_name(name: str) -> str:
    return Path(safe_filename(name)).with_suffix(".jsonl").name


def add_test_set(slug: str, original: str, data: bytes) -> str:
    """Validate and store an uploaded JSONL test set; returns its stored name."""
    from raglab.evaluation import load_tests
    directory = _dir(slug)
    name = test_set_name(original)
    target = directory / "tests" / name
    tmp = target.with_suffix(".upload")
    tmp.write_bytes(data)
    try:
        cases = load_tests(tmp)
    except (ValueError, UnicodeDecodeError) as exc:
        tmp.unlink()
        raise ValueError(f"{original}: {exc}") from exc
    if not cases:
        tmp.unlink()
        raise ValueError(f"{original} contains no questions")
    tmp.replace(target)
    _register_test_set(slug, name, [vars(c) for c in cases])
    return name


EVIDENCE_SEPARATOR = "\n---\n"  # between several evidence passages in one table cell


def read_test_set(slug: str, name: str) -> list[dict]:
    """Questions as dicts: question, reference_answer, keywords (list), category, evidence (list of dicts)."""
    from raglab.evaluation import load_tests
    return [vars(c) for c in load_tests(test_path(slug, name))]


def save_test_set(slug: str, name: str, cases: list[dict]) -> str:
    """Create or overwrite a test set from typed-in questions; blank questions are dropped."""
    from raglab.evaluation import TestCase
    from raglab.evidence import normalise_evidence
    cleaned = []
    for case in cases:
        question = str(case.get("question") or "").strip()
        if not question:
            continue
        keywords = case.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        evidence = case.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [part.strip() for part in evidence.split(EVIDENCE_SEPARATOR.strip()) if part.strip()]
        cleaned.append(vars(TestCase(question=question,
                                     reference_answer=str(case.get("reference_answer") or "").strip(),
                                     keywords=list(keywords),
                                     category=str(case.get("category") or "").strip() or "unspecified",
                                     evidence=normalise_evidence(evidence))))
    if not cleaned:
        raise ValueError("Add at least one question")
    name = test_set_name(name)
    target = _dir(slug) / "tests" / name
    tmp = target.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cleaned), encoding="utf-8")
    tmp.replace(target)
    _register_test_set(slug, name, cleaned)
    return name


def _register_test_set(slug: str, name: str, cases: list[dict]) -> None:
    meta = load(slug)
    meta["tests"][name] = {"questions": len(cases),
                           "with_reference": sum(bool(str(c["reference_answer"]).strip()) for c in cases),
                           "with_keywords": sum(bool(c["keywords"]) for c in cases),
                           "with_evidence": sum(bool(c.get("evidence")) for c in cases), "added_at": _now()}
    meta.pop("slug")
    _write(_dir(slug) / "collection.json", meta)


def remove_test_set(slug: str, name: str) -> None:
    directory = _dir(slug)
    meta = load(slug)
    meta["tests"].pop(name, None)
    (directory / "tests" / Path(name).name).unlink(missing_ok=True)
    meta.pop("slug")
    _write(directory / "collection.json", meta)


def test_path(slug: str, name: str) -> Path:
    if name not in load(slug)["tests"]:
        raise ValueError(f"No test set named {name!r}")
    return _dir(slug) / "tests" / name


def delete(slug: str) -> None:
    shutil.rmtree(_dir(slug))

