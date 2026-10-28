"""UI controller for the FinanceBench example. No RAG implementation lives here.

The Streamlit app builds experiment plans here and runs run.py on them; benchmark data
and indexes remain owned by the example.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CATALOG = PROJECT / 'models.yaml'
DEFAULT_PLAN = HERE / 'experiments.yaml'
RESULTS = HERE / 'results'
RUNNER = HERE / 'run.py'

TECHNIQUES = {
    'query_rewrite': 'Query rewriting',
    'query_fusion': 'Original + rewritten query fusion',
    'rerank': 'LLM reranking',
    'improved_prompt': 'Enhanced generation prompt',
}
RETRIEVAL_MODES = {
    'vector': 'Vector (unfiltered)',
    'filtered_vector': 'Vector (filing-filtered)',
    'hybrid': 'Hybrid BM25 + vector (filing-filtered)',
}


def model_choices(kind: str) -> list[tuple[str, str]]:
    if kind not in ('chat', 'embedding'):
        raise ValueError('Unknown model kind')
    data = yaml.safe_load(CATALOG.read_text(encoding='utf-8')) or {}
    section = (data.get('models') or {}).get(kind) or {}
    if not isinstance(section, dict):
        raise ValueError(f'models.{kind} in models.yaml must be a mapping')
    return [(f"{row.get('label', alias)} ({alias})", alias)
            for alias, row in section.items() if isinstance(row, dict)]


def defaults() -> dict:
    plan = yaml.safe_load(DEFAULT_PLAN.read_text(encoding='utf-8')) or {}
    return plan.get('settings', {})


def load_report(report_path: str | Path) -> dict:
    path = Path(report_path).resolve()
    allowed = (RESULTS / 'ui_runs').resolve()
    from raglab_app.jobs import JOBS
    from raglab_app.library import SAVED, SHIPPED
    trusted = (allowed, JOBS.resolve(), SAVED.resolve(), SHIPPED.resolve())
    if not any(path.is_relative_to(base) for base in trusted) and path != (RESULTS / 'report.json').resolve():
        raise ValueError('Report must be a local FinanceBench result')
    report = json.loads(path.read_text(encoding='utf-8'))
    # Historical reports may have been saved before evidence enrichment was
    # reflected in the leaderboard. Recalculate locally from stored passages.
    experiments = report.get('experiments', [])
    if experiments and any(
        e.get('summary', {}).get('precision_at_k') is None
        for e in experiments
    ):
        gold_file = HERE / 'data' / 'questions.jsonl'
        if gold_file.is_file():
            try:
                from evidence_metrics import enrich_retrieval
                gold = [json.loads(s) for s in gold_file.read_text(encoding='utf-8').splitlines() if s.strip()]
                original_q = {r.get('question') for r in gold}
                report_q = {r.get('question') for e in experiments for r in e.get('results', [])}
                if report_q <= original_q:
                    enrich_retrieval(report, gold)
            except (KeyError, TypeError, ValueError) as exc:
                report.setdefault('research_metadata', {})['retrieval_repair_warning'] = str(exc)
    # Experiment summaries, not an earlier copied leaderboard, are authoritative.
    report['leaderboard'] = [
        {'name': e['name'], 'models': e.get('models', {}), **e.get('summary', {})}
        for e in experiments
    ]
    return report


# Grid-search extension. Values are validated rather than evaluated as code.
from itertools import product

MAX_GRID_RUNS = 32

def parse_integer_values(value, *, label: str, minimum: int, maximum: int) -> list[int]:
    """Accept '300, 500, 700', '[300, 500]', or a single integer."""
    if isinstance(value, int):
        raw = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        content = str(value).strip().strip('[]')
        raw = [v.strip() for v in content.split(',')]
    if not raw or any(str(v).strip() == '' for v in raw):
        raise ValueError(f'{label}: provide at least one integer')
    values = []
    for item in raw:
        try:
            if isinstance(item, bool) or str(item).strip() != str(int(str(item).strip())):
                raise ValueError
            v = int(item)
        except (ValueError, TypeError):
            raise ValueError(f'{label}: {item!r} is not an integer') from None
        if not minimum <= v <= maximum:
            raise ValueError(f'{label}: {v} is outside [{minimum}, {maximum}]')
        if v not in values:
            values.append(v)
    return values


def parse_model_aliases(value, kind: str) -> list[str]:
    """Models must be defined in models.yaml; arbitrary raw model IDs are unsafe."""
    available = {alias for _, alias in model_choices(kind)}
    if isinstance(value, str):
        items = [v.strip() for v in value.split(',') if v.strip()]
    else:
        items = [str(v).strip() for v in (value or [])]
    if not items:
        raise ValueError(f'Select at least one {kind} model')
    unknown = set(items) - available
    if unknown:
        raise ValueError(f'Unknown {kind} alias(es) {sorted(unknown)}. Add them to models.yaml first.')
    return list(dict.fromkeys(items))


def make_grid_plan(*, chat: str, embeddings, judge: str, judge_enabled: bool,
                   top_ks, window_lines, window_overlaps,
                   retrievals, techniques: list[str], compare_baseline: bool,
                   mode: str = 'grid') -> dict:
    """Materialize a validated plan of independent experiments.

    Only the current supported FinanceBench switches are exposed. Each run
    carries its own settings, so index-time parameters are respected.
    """
    if mode not in {'single', 'grid'}:
        raise ValueError('mode must be single or grid')
    if chat not in {x for _, x in model_choices('chat')} or judge not in {x for _, x in model_choices('chat')}:
        raise ValueError('Chat and judge models must exist in models.yaml')
    es = parse_model_aliases(embeddings, 'embedding')
    ks = parse_integer_values(top_ks, label='Top-K', minimum=1, maximum=50)
    ws = parse_integer_values(window_lines, label='Window lines', minimum=8, maximum=300)
    overlaps = parse_integer_values(window_overlaps, label='Line overlap', minimum=0, maximum=299)
    if isinstance(retrievals, str):
        rs = [v.strip() for v in retrievals.split(',') if v.strip()]
    else:
        rs = list(retrievals or [])
    if not rs or set(rs) - set(RETRIEVAL_MODES):
        raise ValueError('Choose one or more supported retrieval modes')
    if set(techniques or []) - set(TECHNIQUES):
        raise ValueError('Unknown technique')
    if mode == 'single' and any(len(x) != 1 for x in (es, ks, ws, overlaps, rs)):
        raise ValueError('Single mode requires one value per field; switch to Grid Search for multiple values')
    combinations = [(e, k, w, o, r) for e,k,w,o,r in product(es,ks,ws,overlaps,rs) if o < w]
    if not combinations:
        raise ValueError('All combinations are invalid: overlap must be smaller than window lines')
    skipped = len(es)*len(ks)*len(ws)*len(overlaps)*len(rs)-len(combinations)
    if len(combinations) + int(bool(compare_baseline)) > MAX_GRID_RUNS:
        raise ValueError(f'{len(combinations)} valid configurations plus baseline exceed the {MAX_GRID_RUNS}-run cost safety cap')
    common = {'chat': chat, 'judge': judge, 'judge_enabled': bool(judge_enabled)}
    runs = []
    if compare_baseline:
        e,k,w,o,r = combinations[0]
        runs.append({'name':'baseline', 'retrieval':'vector', 'techniques':{},
                     'settings':{'embedding':e, 'top_k':k, 'window_lines':w, 'window_overlap':o}})
    t = {key:True for key in (techniques or [])}
    for n,(e,k,w,o,r) in enumerate(combinations,1):
        name = f'grid_{n:03d}_{r}_k{k}_w{w}_o{o}_{e}'
        runs.append({'name':name, 'retrieval':r, 'techniques':t.copy(),
                     'settings':{'embedding':e,'top_k':k,'window_lines':w,'window_overlap':o}})
    return {'settings':common, 'runs':runs,
            'research_ui':{'grid_mode':mode,'invalid_combinations_skipped':skipped}}
