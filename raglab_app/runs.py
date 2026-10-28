"""One row per run, for the run lists on the Workspace and Results pages."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from raglab_app.options import TECHNIQUES

STATUS = {'queued': 'Waiting', 'running': 'Running', 'completed': 'Finished', 'failed': 'Failed'}
HEADLINE = {'accuracy': 'Accuracy', 'evidence_recall': 'Evidence recall', 'context_recall': 'Context recall',
            'keyword_coverage': 'Keyword coverage'}


def _plan(job: dict) -> dict:
    try:
        return yaml.safe_load(Path(job['plan']).read_text(encoding='utf-8')) or {}
    except (KeyError, OSError, yaml.YAMLError):
        return {}


def _values(runs: list[dict], key: str) -> str:
    values = sorted({r.get('settings', {}).get(key) for r in runs} - {None}, key=str)
    return ', '.join(map(str, values))


def _techniques(runs: list[dict]) -> str:
    names = set()
    for run in runs:
        flags = run.get('techniques', {})
        on = [k for k, (_, _, overrides) in TECHNIQUES.items() if all(flags.get(f) == v for f, v in overrides.items())]
        if 'query_fusion' in on:
            on.remove('query_rewrite')  # fusion includes the rewrite; show it once
        names.update(TECHNIQUES[k][0] for k in on)
    return ', '.join(sorted(names)) or 'none'


def duration(job: dict) -> str | None:
    if not job.get('started_at'):
        return None
    seconds = (job.get('finished_at') or datetime.now().timestamp()) - job['started_at']
    return f'{seconds / 60:.0f} min' if seconds >= 90 else f'{seconds:.0f} s'


def overview(job: dict, summaries: list[dict] | None = None) -> dict:
    """Name, status, settings (from the run's plan, so also for unfinished runs) and headline scores.

    summaries: the run's per-setup summaries when finished; the best setup's scores are shown.
    """
    plan = _plan(job)
    runs = plan.get('runs', [])
    row = {'ID': f"#{job['number']}" if job.get('number') else None,
           'Run': (job.get('display_name') or 'Run').removeprefix('Ask: '), 'Status': STATUS.get(job.get('status'), job.get('status')),
           'Questions': f"Ask: {job.get('question', '')[:60]}" if job.get('kind') == 'ask'
                        else (job.get('test_set') or '').removesuffix('.jsonl') or None,
           'Setups': len(runs) or job.get('configurations'),
           'Chunk size': _values(runs, 'chunk_size') or _values(runs, 'window_lines') or None,
           'Top-K': _values(runs, 'top_k') or None,
           'Techniques': ('finance-tuned, ' if job.get('kind') == 'financial' else '') + _techniques(runs)}
    if summaries:
        best = max(summaries, key=lambda s: s.get('accuracy') or 0)
        row.update({label: None if best.get(k) is None else round(float(best[k]), 2) for k, label in HEADLINE.items()})
    row['Started'] = job['created_at'][:16].replace('T', ' ')
    row['Duration'] = duration(job)
    return row
