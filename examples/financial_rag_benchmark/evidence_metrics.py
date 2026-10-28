"""Post-retrieval FinanceBench evidence relevance proxies (never used for retrieval).

The resulting Precision@K/MRR/nDCG are NOT human-judged passage metrics.
One evidence snippet may match multiple retrieved windows; Recall@K is
fraction of annotated evidence snippets covered, not fraction of all relevant
chunks in the original corpus.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from statistics import mean

STOP = set('the and for from what which who how that this with into have has were was are not its their year fiscal statement reported consolidated financial results company'.split())


def terms(text: str) -> Counter:
    words = re.findall(r'[a-z]+|\d[\d,]*(?:\.\d+)?', str(text).lower())
    normalized = []
    for word in words:
        if word in STOP or len(word) == 1:
            continue
        if word[0].isdigit():
            try:
                word = str(float(word.replace(',', ''))).removesuffix('.0')
            except ValueError:
                pass
        normalized.append(word)
    return Counter(normalized)


def matching_score(doc: dict, evidence: dict) -> float:
    """Fraction of gold tokens found in one retrieved chunk, source restricted."""
    gold_source = str(evidence.get('doc_name') or evidence.get('evidence_doc_name') or '').strip()
    doc_meta = doc.get('metadata') or {}
    source = str(doc_meta.get('source_file') or doc_meta.get('source') or '')
    if not gold_source or not source:
        return 0.0
    # Compare filing identifiers, not arbitrary substrings of the absolute path.
    # Chunks may carry a stem (source_file) or a source path with a '-0' suffix.
    source_id = str(doc_meta.get('source_file') or '').strip()
    if not source_id:
        source_id = re.sub(r'\.txt(?:-\d+)?$', '', source.rsplit('/', 1)[-1], flags=re.I)
    if source_id.casefold() != gold_source.casefold():
        return 0.0
    gold = terms(evidence.get('evidence_text') or '')
    found = terms(doc.get('content') or '')
    return sum((gold & found).values()) / sum(gold.values()) if gold else 0.0


def score_retrieval(docs: list[dict], evidence: list[dict], *, k: int = 5,
                    threshold: float = 0.30) -> dict:
    """Evidence-based *proxy* labels; do not present as independently verified judgments."""
    if k < 1 or not 0 < threshold <= 1:
        raise ValueError('k >= 1 and 0 < threshold <= 1 required')
    gold = [e for e in (evidence or []) if isinstance(e, dict) and
            (e.get('doc_name') or e.get('evidence_doc_name')) and terms(e.get('evidence_text') or '')]
    if not gold:
        return dict(precision_at_k=None, recall_at_k=None, hit_at_k=None,
                    mrr=None, ndcg_at_k=None, relevant_retrieved=None,
                    evidence_count=0, metric_status='unavailable_no_usable_gold_evidence')
    top = docs[:k]
    relevance = [int(any(matching_score(doc, ev) >= threshold for ev in gold)) for doc in top]
    gold_hit = [any(matching_score(doc, ev) >= threshold for doc in top) for ev in gold]
    first = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
    # Multiple annotations may be covered by the same chunk. This ideal DCG uses
    # one relevant passage per annotated snippet as an approximation, NOT corpus qrels.
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    ideal_count = min(len(gold), k)
    ideal = sum(1 / math.log2(i + 2) for i in range(ideal_count))
    return dict(precision_at_k=sum(relevance) / k, recall_at_k=sum(gold_hit) / len(gold),
                hit_at_k=int(any(relevance)), mrr=1 / first if first else 0.0,
                ndcg_at_k=min(1.0, dcg / ideal) if ideal else None,
                relevant_retrieved=sum(relevance), evidence_count=len(gold),
                metric_status='heuristic_evidence_overlap_not_human_qrels')


def enrich_retrieval(report: dict, gold_rows: list[dict], threshold: float = 0.30) -> dict:
    lookup = {r['question']: r for r in gold_rows}
    keys = ('precision_at_k', 'recall_at_k', 'hit_at_k', 'mrr', 'ndcg_at_k')
    for exp in report['experiments']:
        k = int((exp.get('config') or {}).get('top_k', 5))
        for row in exp.get('results', []):
            source = lookup.get(row['question'], {})
            metrics = score_retrieval(row.get('retrieved_documents', []), source.get('evidence') or [],
                                      k=k, threshold=threshold)
            row['evidence_retrieval'] = metrics
            for key in keys:
                row[key] = metrics[key]
        for key in keys:
            vals = [r[key] for r in exp.get('results', []) if r[key] is not None]
            exp['summary'][key] = mean(vals) if vals else None
        exp['summary']['retrieval_scored_questions'] = sum(r['mrr'] is not None for r in exp.get('results', []))
        # Correct earlier dummy null MRR by replacing with explicit evidence proxy metric.
        exp['summary']['ndcg'] = exp['summary']['ndcg_at_k']
    report['leaderboard'] = [dict(name=e['name'], models=e['models'], **e['summary']) for e in report['experiments']]
    report.setdefault('research_metadata', {})['retrieval_labels'] = (
        f'Heuristic source-restricted lexical overlap, threshold={threshold}; '
        'NOT human relevance labels; precision@k denominator is k; '
        'recall@k denominator is annotated evidence snippets; '
        'nDCG ideal ranking approximated using one relevant window per evidence snippet.')
    return report
