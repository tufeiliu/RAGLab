"""Finance-specific, transparent post-hoc diagnostics; does not call an LLM."""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
from statistics import mean

STOP = set('the a an and or of to for in on at by is are was were as with from this that it its per year fiscal what how did does company financial'.split())


def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+', str(text).lower()) if len(t) > 2 and t not in STOP]


def evidence_recall(retrieved: list[dict], evidence: list[dict], *, threshold: float = 0.50):
    """Fraction of gold evidence snippets with >= threshold token recall in a matching source.

    Weak heuristic: lexical evidence overlap, NOT a gold-relevance-label benchmark.
    """
    if not evidence:
        return None
    covered = 0
    for ev in evidence:
        gold = Counter(tokens(ev.get('evidence_text', '')))
        if not gold:
            continue
        source_name = str(ev.get('evidence_doc_name', ''))
        # Ensure the retrieved snippet came from the annotated source filing.
        candidates = [d.get('content', '') for d in retrieved
                      if source_name and source_name in str(d.get('metadata', {}).get('source', ''))]
        best = max((sum((gold & Counter(tokens(c))).values()) / sum(gold.values()) for c in candidates), default=0)
        covered += best >= threshold
    return covered / len(evidence)


def source_hit(retrieved, doc_name):
    return int(any(str(doc_name) in str(d.get('metadata', {}).get('source', '')) for d in retrieved))


def _numeric_values(text):
    """Conservative literal-number tokens. Excludes years and avoids claiming unit equivalence."""
    values = []
    for match in re.finditer(r'(?<!\w)(?:\$\s*)?[-+]?\d[\d,]*(?:\.\d+)?\s*%?', str(text)):
        s = match.group().strip().replace(',', '').replace('$', '').replace(' ', '')
        suffix = '%' if s.endswith('%') else ''
        try:
            val = float(s.removesuffix('%'))
        except ValueError:
            continue
        if not suffix and 1900 <= val <= 2100 and val.is_integer():
            continue
        values.append((val, suffix))
    return values


def numeric_overlap(answer, reference):
    """Diagnostic only; not numeric answer accuracy (units, ranges and derivations matter)."""
    gold = _numeric_values(reference)
    if not gold:
        return None
    predicted = _numeric_values(answer)
    return sum(any(u == v and math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-6)
                   for y, v in predicted) for x, u in gold) / len(gold)


def enrich(report, original_rows):
    lookup = {r['question']: r for r in original_rows}
    for exp in report['experiments']:
        for result in exp['results']:
            gold = lookup[result['question']]
            docs = result.get('retrieved_documents', [])
            result['finance'] = {
                'financebench_id': gold['financebench_id'],
                'document': gold['doc_name'],
                'question_reasoning': gold.get('question_reasoning'),
                'source_hit': source_hit(docs, gold['doc_name']),
                'evidence_lexical_recall': evidence_recall(docs, gold.get('evidence') or []),
                'numeric_overlap_diagnostic': numeric_overlap(result['generated_answer'], gold['answer']),
            }
        rows = exp['results']
        for key in ('source_hit', 'evidence_lexical_recall', 'numeric_overlap_diagnostic'):
            values = [r['finance'][key] for r in rows if r['finance'][key] is not None]
            exp['summary'][key] = mean(values) if values else None
    report['leaderboard'] = [{**row, **{k: next(e['summary'][k] for e in report['experiments'] if e['name'] == row['name'])
        for k in ('source_hit','evidence_lexical_recall','numeric_overlap_diagnostic')}} for row in report['leaderboard']]
    return report


def paired_bootstrap(report, metric='evidence_lexical_recall', baseline='baseline', n=2000, seed=42):
    """Paired question-level bootstrap CI for improvement over a named baseline."""
    experiments = {e['name']: e for e in report['experiments']}
    if baseline not in experiments:
        raise ValueError(f'Missing baseline run {baseline}')
    base = {r['question']: r['finance'][metric] for r in experiments[baseline]['results']}
    rng = random.Random(seed)
    out = []
    for name, exp in experiments.items():
        if name == baseline:
            continue
        diffs = [r['finance'][metric] - base[r['question']]
                 for r in exp['results'] if r['question'] in base
                 and r['finance'][metric] is not None and base[r['question']] is not None]
        if not diffs:
            continue
        samples = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
        out.append({'experiment': name, 'baseline': baseline, 'metric': metric, 'paired_n': len(diffs),
                    'mean_delta': mean(diffs), 'ci95_low': samples[int(n*.025)],
                    'ci95_high': samples[min(n - 1, int(n*.975))]})
    return out
