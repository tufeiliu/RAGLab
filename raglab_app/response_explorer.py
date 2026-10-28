"""UI-only helpers for inspecting saved RAG benchmark answers and retrievals.

No model calls, filesystem writes, or application state here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalise_passages(retrieved: Any) -> list[dict[str, Any]]:
    """Preserve the saved retrieval order; show scores only when recorded."""
    if not isinstance(retrieved, list):
        return []
    result = []
    for rank, raw in enumerate(retrieved, start=1):
        if isinstance(raw, str):
            raw = {"content": raw}
        if not isinstance(raw, dict):
            raw = {"content": _text(raw)}
        meta = raw.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        source = raw.get("source") or meta.get("source") or meta.get("file_name") or "Unknown source"
        source = _text(source)
        # Avoid showing the user's machine-specific directory path in the UI.
        filename = Path(source).name
        # Prior ingestion encoded a suffix (e.g. filing.txt-0) in the source.
        score = next((raw[k] for k in ("similarity_score", "similarity", "score", "distance")
                      if raw.get(k) is not None), None)
        result.append({
            "rank": rank,
            "source": filename,
            "source_full": source,
            "page": meta.get("page", meta.get("page_number")),
            "chunk": meta.get("chunk", meta.get("chunk_index")),
            "score": score,
            "score_label": next((k for k in ("similarity_score", "similarity", "score", "distance")
                                 if raw.get(k) is not None), None),
            "content": _text(raw.get("content", raw.get("page_content", raw.get("text", "")))),
        })
    return result


def pair_question_rows(report: dict, left: str, right: str) -> list[dict]:
    """Match the SAME questions across configurations, not list positions.

    Handles repeated identical question texts by their occurrence index.
    """
    experiments = {e.get("name"): e for e in report.get("experiments", [])}
    if left not in experiments or right not in experiments:
        return []

    def indexed_rows(exp: dict) -> dict[tuple[str, int], dict]:
        seen: dict[str, int] = {}
        indexed = {}
        for row in exp.get("results", []):
            question = _text(row.get("question")).strip()
            occurrence = seen.get(question, 0)
            seen[question] = occurrence + 1
            indexed[(question, occurrence)] = row
        return indexed

    lhs, rhs = indexed_rows(experiments[left]), indexed_rows(experiments[right])
    return [{"question": question, "left": lhs[key], "right": rhs[key]}
            for key in lhs if key in rhs
            for question in [key[0]]]


def row_view(row: dict) -> dict:
    """Read the current JSON result schema with safe fallbacks."""
    judge = row.get("judge") or {}
    finance = row.get("finance") or {}
    return {
        "question": _text(row.get("question")),
        "answer": _text(row.get("generated_answer", row.get("answer", ""))),
        "reference": _text(row.get("reference_answer", row.get("reference", ""))),
        "feedback": _text(judge.get("feedback", "")) if isinstance(judge, dict) else "",
        "accuracy": judge.get("accuracy") if isinstance(judge, dict) else None,
        "source_hit": finance.get("source_hit") if isinstance(finance, dict) else None,
        "precision_at_k": row.get("precision_at_k"),
        "recall_at_k": row.get("recall_at_k"),
        "retrieved": normalise_passages(row.get("retrieved_documents", row.get("retrieved", []))),
    }
