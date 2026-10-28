"""Retrieval failure diagnostics; works on previously produced evaluation rows."""
from __future__ import annotations

from typing import Any


def analyze_failure(row: dict[str, Any], *, accuracy_threshold: float = 3,
                    coverage_threshold: float = 0.8) -> dict[str, Any]:
    keywords = row.get("expected_keywords", [])
    docs = row.get("retrieved_documents", [])
    text = "\n".join(d.get("content", "") for d in docs).casefold()
    missing = [key for key in keywords if key.casefold() not in text]
    judge = row.get("judge") or {}
    coverage = row.get("keyword_coverage", 0.0)
    accuracy = judge.get("accuracy")
    faithful = judge.get("faithfulness")
    flags = []
    if keywords and (missing or coverage < coverage_threshold):
        flags.append("retrieval_keyword_gap")
    if accuracy is not None and accuracy < accuracy_threshold:
        flags.append("answer_accuracy_failure")
    if faithful is not None and faithful < accuracy_threshold:
        flags.append("grounding_failure")
    if docs and accuracy is not None and accuracy < accuracy_threshold and not missing:
        flags.append("possible_generation_failure")
    if not docs:
        flags.append("no_retrieved_documents")
    return {"question": row.get("question", ""), "flags": flags,
            "missing_keywords": missing, "retrieved_count": len(docs),
            "sources": [d.get("metadata", {}).get("source", "unknown") for d in docs],
            "retrieved_ids": [d.get("id") for d in docs if d.get("id")],
            "coverage": coverage, "accuracy": accuracy, "faithfulness": faithful,
            "generated_answer": row.get("generated_answer", ""),
            "reference_answer": row.get("reference_answer", "")}


def failure_report(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return [analyze_failure(row) for row in evaluation.get("results", [])]


def selected_ids_from_texts(collection, retrieved_documents: list[dict]) -> list[str]:
    """Match exact content and source when RAG output has no stored Chroma IDs.

    Duplicate chunks can resolve to multiple IDs; all matches are highlighted.
    This is an inspection convenience, not ground-truth source attribution.
    """
    targets = {(d.get("content", ""), d.get("metadata", {}).get("source"))
               for d in retrieved_documents}
    ids = []
    offset = 0
    while offset < collection.count():
        fetched = collection.get(limit=500, offset=offset, include=["documents", "metadatas"])
        for doc_id, txt, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            if (txt, (meta or {}).get("source")) in targets:
                ids.append(doc_id)
        offset += len(fetched["ids"])
        if not fetched["ids"]:
            break
    return ids
