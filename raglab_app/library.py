"""A library of evaluation results to browse: shipped examples plus results you saved.

Each entry is a folder with meta.json (title, description, what to look at) and
report.json (the evaluation report, in the same format as a job's report).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from raglab_app import DATA, ROOT

SHIPPED = ROOT / 'examples' / 'library'
SAVED = DATA / 'library'


def entries() -> list[dict]:
    rows = []
    for base, origin in ((SHIPPED, 'example'), (SAVED, 'saved')):
        for meta_file in sorted(base.glob('*/meta.json')):
            try:
                meta = json.loads(meta_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            rows.append({**meta, 'id': f'{origin}-{meta_file.parent.name}', 'origin': origin,
                         'report': str(meta_file.parent / 'report.json')})
    return rows


def as_job(entry: dict) -> dict:
    """The fields the results view needs, as if the entry were a finished job."""
    return {'id': entry['id'], 'kind': entry.get('kind', 'experiment'), 'report': entry['report'],
            'display_name': entry['title'], 'status': 'completed'}


def save(job: dict, title: str, description: str, what_to_look_at: str = '') -> str:
    """Copy a finished evaluation into the library; returns the entry folder name."""
    title = title.strip()
    if not title:
        raise ValueError('Give the entry a title')
    stem = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:50] or 'entry'
    folder = SAVED / stem
    n = 2
    while folder.exists():
        folder = SAVED / f'{stem}-{n}'
        n += 1
    folder.mkdir(parents=True)
    shutil.copyfile(job['report'], folder / 'report.json')
    (folder / 'meta.json').write_text(json.dumps({
        'title': title, 'description': description.strip(), 'what_to_look_at': what_to_look_at.strip(),
        'kind': job['kind'], 'source': job.get('collection_name'),
        'added_at': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
    return folder.name


def remove(entry: dict) -> None:
    if entry['origin'] != 'saved':
        raise ValueError('Shipped examples cannot be removed from the app')
    shutil.rmtree(Path(entry['report']).parent)
