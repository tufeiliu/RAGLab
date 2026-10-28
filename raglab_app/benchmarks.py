"""Ready-made benchmarks: public document sets with test questions and answer keys.

Loading a benchmark creates (or refreshes) an ordinary collection, so it goes
through the same Ask -> Evaluate -> Results workflow as your own documents and
its scores are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from raglab_app import ROOT, collections

FINANCE = ROOT / 'examples' / 'financial_rag_benchmark'
FINANCE_DATA = FINANCE / 'data'


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    summary: str
    details: str
    scores: str


FINANCEBENCH = Benchmark(
    'financebench', 'FinanceBench',
    'Real company filings (10-K/10-Q) and analyst questions. Hard and realistic: long documents, many tables.',
    'A public benchmark by Patronus AI (open sample: 150 questions over 84 filings). Questions need number '
    'look-ups, calculations or judgement, e.g. "What is the FY2018 capital expenditure for 3M?". Each comes '
    'with the correct answer and the exact passage of the filing that supports it. Licence: CC BY-NC 4.0.',
    'Answers against reference answers (LLM judge); search against the gold evidence passages, which is '
    'stricter than keywords.')
ORTHOFINDER = Benchmark(
    'orthofinder', 'OrthoFinder paper (example)',
    'One research paper and 11 questions about it. Small, quick and cheap: a good first test.',
    'The Nature Methods paper "OrthoFinder: improved phylogenetic orthology inference with enhanced accuracy and '
    'scalability" (Emms, Liu, Belcher, Holmes & Kelly, 2026). Questions about the paper itself (authors, '
    'dates), its results and its methods. Questions about the paper itself are a known weak spot of plain '
    'RAG: try "Include document opening".',
    'Answers against reference answers (LLM judge); search against evidence passages from the paper, and keywords.')
BENCHMARKS = {b.key: b for b in (ORTHOFINDER, FINANCEBENCH)}
ORTHOFINDER_DIR = ROOT / 'examples' / 'orthofinder'


def _find(key: str) -> dict | None:
    return next((c for c in collections.list_collections() if c.get('benchmark') == key), None)


def loaded(key: str) -> dict | None:
    return _find(key)


def financebench_status() -> dict:
    """What FinanceBench data is on disk: prepared questions/filings and the full open sample size."""
    questions = FINANCE_DATA / 'questions.jsonl'
    rows = [json.loads(line) for line in questions.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if questions.is_file() else []
    docs = sorted({r['doc_name'] for r in rows})
    present = [d for d in docs if (FINANCE_DATA / 'documents' / f'{d}.txt').is_file()]
    raw = FINANCE_DATA / 'raw' / 'financebench.jsonl'
    full = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if raw.is_file() else []
    return {'questions': len(rows), 'documents': len(present), 'ready': bool(rows) and len(present) == len(docs),
            'full_questions': len(full) or 150, 'full_documents': len({r['doc_name'] for r in full}) or 84}


def load(key: str) -> str:
    """Create or refresh the benchmark's collection and return its id."""
    if key == 'financebench':
        return _load_financebench()
    if key == 'orthofinder':
        return _load_orthofinder()
    raise ValueError(f'Unknown benchmark {key!r}')


def _load_orthofinder() -> str:
    existing = _find('orthofinder')
    if existing:
        return existing['slug']
    slug = collections.create(ORTHOFINDER.name, benchmark='orthofinder')
    collections.add_files(slug, [(p.name, p.read_bytes()) for p in sorted(ORTHOFINDER_DIR.glob('*.pdf'))])
    collections.add_test_set(slug, 'orthofinder questions.jsonl', (ORTHOFINDER_DIR / 'tests.jsonl').read_bytes())
    return slug


def _load_financebench() -> str:
    status = financebench_status()
    if not status['ready']:
        raise FileNotFoundError('FinanceBench data is not prepared yet. Download it first.')
    rows = [json.loads(line) for line in (FINANCE_DATA / 'questions.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    existing = _find('financebench')
    slug = existing['slug'] if existing else collections.create(FINANCEBENCH.name, benchmark='financebench')
    wanted = {f"{d}.txt" for d in {r['doc_name'] for r in rows}}
    have = set(collections.load(slug)['files'])
    for name in sorted(have - wanted):
        collections.remove_file(slug, name)
    collections.add_files(slug, [(name, (FINANCE_DATA / 'documents' / name).read_bytes()) for name in sorted(wanted - have)])
    collections.save_test_set(slug, 'financebench', [{
        'question': r['question'], 'reference_answer': r['answer'],
        'category': (r.get('question_reasoning') or r.get('question_type') or 'unspecified').split(' OR ')[0],
        'evidence': [{'text': e['evidence_text'], 'source': e.get('doc_name') or r['doc_name']}
                     for e in r.get('evidence', [])],
    } for r in rows])
    return slug
