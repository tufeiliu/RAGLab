"""Retrieval scores against gold evidence passages.

A test question can name the passage(s) that contain its answer ("evidence"),
optionally with the file they come from. A retrieved chunk counts as relevant
when the words it shares with an evidence passage make up a large enough share of
the shorter of the two texts. Normalising by the shorter text means a small chunk
lying entirely inside a long evidence table counts, and so does a large chunk that
contains a short evidence sentence.

How large a share depends on how long the shorter text is: a long table sharing
30% of its words with a chunk is the same region of the document, but a short
sentence sharing 30% may just share a few common topic words. So the required
share falls from 60% for texts of 15 words or fewer to `threshold` (30%) at 60
words and more. On the FinanceBench example this agrees with the benchmark's own
evidence labels on 19 of 20 questions; on a single-topic paper it keeps each
evidence sentence to the one or few chunks that actually contain it.

This is a lexical approximation of relevance, far closer to the truth than
keyword presence, but not a human relevance judgement.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Iterable

STOP = set('the and for from what which who how that this with into have has were was are not its their '
           'than then also been being but can may will would should could our your they them these those'.split())
MIN_TERMS = 8  # chunks or evidence shorter than this are too small to match reliably
SHORT, LONG, SHORT_SHARE = 15, 60, 0.6  # required share: SHORT_SHARE up to SHORT words, `threshold` from LONG


def terms(text: str) -> Counter:
    """Content words and normalised numbers ('1,577.0' and '1577' are the same term)."""
    out = []
    for word in re.findall(r'[a-z]+|\d[\d,]*(?:\.\d+)?', str(text).lower()):
        if word in STOP or len(word) == 1:
            continue
        if word[0].isdigit():
            try:
                word = str(float(word.replace(',', ''))).removesuffix('.0')
            except ValueError:
                pass
        out.append(word)
    return Counter(out)


def normalise_evidence(evidence: Any) -> list[dict]:
    """Accept a string, a list of strings, or dicts with text/source (FinanceBench keys too)."""
    if not evidence:
        return []
    items = [evidence] if isinstance(evidence, (str, dict)) else list(evidence)
    out = []
    for item in items:
        if isinstance(item, str):
            item = {'text': item}
        if not isinstance(item, dict):
            continue
        text = str(item.get('text') or item.get('evidence_text') or '').strip()
        source = str(item.get('source') or item.get('doc_name') or item.get('evidence_doc_name') or '').strip()
        if text:
            out.append({'text': text, 'source': source or None})
    return out


def _source_id(value: str) -> str:
    """'docs/3M_2018_10K.txt' and '3M_2018_10K' identify the same file."""
    name = Path(str(value).replace('\\', '/')).name
    return re.sub(r'\.[A-Za-z0-9]{1,5}(?:-\d+)?$', '', name).casefold()


def _source_of(doc: dict) -> str:
    meta = doc.get('metadata') or {}
    return str(meta.get('source_file') or meta.get('source') or '')


def required_share(words: int, threshold: float) -> float:
    if words <= SHORT:
        return SHORT_SHARE
    if words >= LONG:
        return threshold
    return SHORT_SHARE - (SHORT_SHARE - threshold) * (words - SHORT) / (LONG - SHORT)


def is_relevant(doc: dict, item: dict, threshold: float) -> bool:
    if item['source'] and _source_id(_source_of(doc)) != _source_id(item['source']):
        return False
    a, b = terms(doc.get('content') or doc.get('page_content') or ''), terms(item['text'])
    smaller = min(sum(a.values()), sum(b.values()))
    if smaller < MIN_TERMS:
        return False
    return sum((a & b).values()) / smaller >= required_share(smaller, threshold)


def evidence_metrics(docs: Iterable[Any], evidence: Any, *, k: int = 5, threshold: float = 0.3) -> dict:
    """Precision, recall, hit rate, MRR and nDCG of the top-k chunks against gold evidence.

    docs may be LangChain Documents or {'content', 'metadata'} dicts. Returns all
    metrics as None when the question has no usable evidence.
    """
    gold = [e for e in normalise_evidence(evidence) if sum(terms(e['text']).values()) >= MIN_TERMS]
    keys = ('evidence_precision', 'evidence_recall', 'evidence_hit', 'evidence_mrr', 'evidence_ndcg')
    if not gold:
        return dict.fromkeys(keys)
    top = [d if isinstance(d, dict) else {'content': d.page_content, 'metadata': dict(d.metadata)}
           for d in list(docs)[:k]]
    relevant = [any(is_relevant(d, e, threshold) for e in gold) for d in top]
    found = [any(is_relevant(d, e, threshold) for d in top) for e in gold]
    first = next((i + 1 for i, rel in enumerate(relevant) if rel), None)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevant))
    # Ideal: as many relevant chunks as there are evidence passages, ranked first.
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {'evidence_precision': sum(relevant) / max(len(top), 1), 'evidence_recall': sum(found) / len(gold),
            'evidence_hit': float(any(relevant)), 'evidence_mrr': 1 / first if first else 0.0,
            'evidence_ndcg': min(1.0, dcg / ideal) if ideal else None}
