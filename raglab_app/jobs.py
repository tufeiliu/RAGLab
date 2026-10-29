"""Persistent local background jobs for the RAGLab Streamlit app.

Each job is an OS process with its own folder under raglab_data/jobs/<id>/:
status.json (state and metadata), job.log (timestamped log) and, for experiments,
report.json. Jobs survive page changes and browser refreshes. This is a local
workstation runner, not a distributed scheduler.

Kinds:
    ask         one question asked on the Workspace page (recorded when answered, no process)
    prepare     download more of the FinanceBench sample and refresh its collection
    experiment  compare RAG configurations on a collection's test questions
    financial   FinanceBench with its finance-tuned retrieval pipeline
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from raglab_app import JOBS, ROOT

PROGRESS_PREFIX = 'RAGLAB_PROGRESS '


def _write(path: Path, payload: dict) -> None:
    tmp = path.parent / (path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


RUN_KINDS = ('experiment', 'financial', 'ask')  # what users see as "runs"; these get a run number


def _numbers() -> list[int]:
    """Run numbers in use, plus the highest ever given (kept in a file, so deleting runs never frees one)."""
    try:
        numbers = [int((JOBS / 'last_run_number').read_text(encoding='utf-8'))]
    except (OSError, ValueError):
        numbers = []
    for path in JOBS.glob('*/status.json'):
        try:
            numbers.append(int(json.loads(path.read_text(encoding='utf-8')).get('number') or 0))
        except (OSError, ValueError):
            continue
    return numbers


def _new_job(kind: str, **meta) -> tuple[str, Path]:
    run_id = uuid.uuid4().hex[:12]
    directory = JOBS / run_id
    directory.mkdir(parents=True)
    if kind in RUN_KINDS:
        recent_jobs(1)  # number any older runs first, so numbers follow creation order
        # Short permanent ID shown as "#7"; never reused, also after deletions.
        meta['number'] = max(_numbers(), default=0) + 1
        (JOBS / 'last_run_number').write_text(str(meta['number']), encoding='utf-8')
    _write(directory / 'status.json', {'id': run_id, 'kind': kind, 'status': 'queued',
                                       'created_at': _now(), 'log': str(directory / 'job.log'), **meta})
    return run_id, directory


def start_experiment(slug: str, collection_name: str, test_set: str, plan: dict,
                     display_name: str, cache_responses: bool) -> str:
    import yaml
    run_id, directory = _new_job('experiment', collection=slug, collection_name=collection_name,
                                 test_set=test_set, display_name=display_name.strip()[:100],
                                 configurations=len(plan.get('runs', [])),
                                 cache_responses=cache_responses)
    (directory / 'experiments.yaml').write_text(yaml.safe_dump(plan, sort_keys=False), encoding='utf-8')
    _patch(run_id, report=str(directory / 'report.json'), plan=str(directory / 'experiments.yaml'))
    _launch(directory, 'experiment')
    return run_id


def record_answer(slug: str, collection_name: str, question: str, report: dict, plan: dict,
                  display_name: str) -> str:
    """Save an answered Ask question as a finished run, so it can be listed and compared like any run."""
    import yaml
    run_id, directory = _new_job('ask', collection=slug, collection_name=collection_name, question=question,
                                 display_name=display_name[:100], configurations=1)
    (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    (directory / 'experiments.yaml').write_text(yaml.safe_dump(plan, sort_keys=False), encoding='utf-8')
    _patch(run_id, status='completed', report=str(directory / 'report.json'),
           plan=str(directory / 'experiments.yaml'), finished_at=datetime.now(timezone.utc).timestamp())
    return run_id


def start_prepare(max_documents: int, max_questions: int) -> str:
    run_id, directory = _new_job('prepare', display_name='Download FinanceBench',
                                 max_documents=int(max_documents), max_questions=int(max_questions))
    _launch(directory, 'prepare')
    return run_id


def start_financial(plan: dict, display_name: str | None = None, *, collection: dict | None = None,
                    test_set: str | None = None) -> str:
    """collection/test_set link the run to the FinanceBench collection, so Results can compare it
    with generic evaluations on the same questions."""
    import yaml
    run_id, directory = _new_job('financial',
                                 display_name=(display_name or 'Financial RAG comparison').strip()[:100],
                                 configurations=len(plan.get('runs', [])),
                                 collection=collection['slug'] if collection else None,
                                 collection_name=collection['name'] if collection else 'FinanceBench',
                                 test_set=test_set)
    (directory / 'experiments.yaml').write_text(yaml.safe_dump(plan, sort_keys=False), encoding='utf-8')
    _patch(run_id, report=str(directory / 'report.json'), plan=str(directory / 'experiments.yaml'))
    _launch(directory, 'financial')
    return run_id


def _patch(run_id: str, **changes) -> None:
    file = JOBS / run_id / 'status.json'
    meta = json.loads(file.read_text(encoding='utf-8'))
    meta.update(changes)
    _write(file, meta)


def _launch(directory: Path, kind: str) -> None:
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), env.get('PYTHONPATH', '')]))
    with (directory / 'job.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'raglab_app.job_worker', str(directory), kind],
                                   cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    _patch(directory.name, pid=process.pid)


def progress_event(**details) -> None:
    """Emit a structured progress line; the UI parses these from job.log."""
    print(PROGRESS_PREFIX + json.dumps(details, separators=(',', ':')), flush=True)


def progress_from_log(log_text: str, kind: str, state: str) -> dict:
    """Parse the latest progress event. Never infer counts from a successful exit.

    Two shapes are understood: experiment events (total_runs/completed_runs and
    total_questions/question_done) and simple events (done/total/unit).
    """
    events = []
    for line in log_text.splitlines():
        if line.startswith(PROGRESS_PREFIX):
            try:
                value = json.loads(line[len(PROGRESS_PREFIX):])
                if isinstance(value, dict):
                    events.append(value)
            except (ValueError, TypeError):
                continue
    if not events:
        return {'stage': state, 'fraction': 1.0 if state == 'completed' else 0.0,
                'completed_runs': None, 'total_runs': None, 'question_done': None,
                'total_questions': None}
    item = events[-1]
    if 'total_runs' in item:
        runs = max(1, int(item.get('total_runs') or 1))
        completed = min(runs, max(0, int(item.get('completed_runs') or 0)))
        questions = max(0, int(item.get('total_questions') or 0))
        done = min(questions, max(0, int(item.get('question_done') or 0)))
        fraction = (completed + (done / questions if questions else 0)) / runs
        result = {'completed_runs': completed, 'total_runs': runs,
                  'question_done': done, 'total_questions': questions,
                  'current_run': item.get('current_run'),
                  'successful': item.get('successful'), 'failed': item.get('failed')}
    else:
        total = max(0, int(item.get('total') or 0))
        done = min(total, max(0, int(item.get('done') or 0)))
        fraction = done / total if total else 0.0
        result = {'done': done, 'total': total, 'unit': item.get('unit', 'items')}
    fraction = 1.0 if state == 'completed' else min(fraction, 0.99)
    return {'stage': item.get('stage', state), 'fraction': max(0.0, min(1.0, fraction)), **result}


def _check_id(run_id: str) -> None:
    # Never take arbitrary paths from a UI input.
    if not run_id or not all(c in '0123456789abcdef' for c in run_id):
        raise ValueError('Invalid experiment ID')


def rename_job(run_id: str, display_name: str) -> None:
    _check_id(run_id)
    label = display_name.strip()
    if not 1 <= len(label) <= 100:
        raise ValueError('Experiment name must have 1–100 characters')
    _patch(run_id, display_name=label)


def delete_job(run_id: str) -> None:
    """Remove a finished or failed run and everything it saved. Running jobs cannot be deleted."""
    import shutil
    _check_id(run_id)
    if status(run_id).get('status') in ('queued', 'running'):
        raise ValueError('This run is still in progress; wait for it to finish before deleting it')
    shutil.rmtree(JOBS / run_id)


def read_log(run_id: str) -> str:
    """Full log without the machine-readable progress lines."""
    _check_id(run_id)
    path = JOBS / run_id / 'job.log'
    text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
    return '\n'.join(line for line in text.splitlines() if not line.startswith(PROGRESS_PREFIX))


def status(run_id: str) -> dict:
    _check_id(run_id)
    directory = JOBS / run_id
    meta = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    log_text = (directory / 'job.log').read_text(encoding='utf-8', errors='replace') if (directory / 'job.log').exists() else ''
    meta['log_tail'] = '\n'.join([line for line in log_text.splitlines() if not line.startswith(PROGRESS_PREFIX)][-25:])
    meta['progress'] = progress_from_log(log_text, meta.get('kind', ''), meta['status'])
    if meta['status'] in ('queued', 'running') and not _alive(meta.get('pid')):
        # The worker died without recording an outcome (killed, machine restarted).
        meta.update(status='failed', error='The background process stopped unexpectedly. See the log.')
        _write(directory / 'status.json', {k: v for k, v in meta.items() if k not in ('log_tail', 'progress')})
    return meta


def _alive(pid) -> bool:
    if not pid:
        return True  # Not launched yet.
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # A finished child stays a zombie until reaped; treat it as gone.
        return Path(f'/proc/{int(pid)}/stat').read_text().split()[2] != 'Z'
    except (OSError, IndexError):
        return True


def _created_at(directory: Path) -> str:
    """Jobs from older versions lack created_at; their oldest file marks the start."""
    times = [p.stat().st_mtime for p in directory.rglob('*') if p.is_file()]
    stamp = min(times) if times else directory.stat().st_mtime
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def _plan_runs(directory: Path) -> int | None:
    import yaml
    try:
        plan = yaml.safe_load((directory / 'experiments.yaml').read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        return None
    return len(plan.get('runs') or []) or None


def recent_jobs(limit: int = 20, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Newest first. Fills in fields missing from jobs created by older versions."""
    rows = []
    for path in JOBS.glob('*/status.json'):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        meta.setdefault('created_at', _created_at(path.parent))
        if meta.get('kind') == 'financial' and 'configurations' not in meta:
            meta['configurations'] = _plan_runs(path.parent)
        rows.append(meta)
    _number_older_runs(rows)
    rows = [m for m in rows if not kinds or m.get('kind') in kinds]
    rows.sort(key=lambda m: m['created_at'], reverse=True)
    return rows[:limit]


def _number_older_runs(rows: list[dict]) -> None:
    """Runs made before run numbers existed get one, oldest first, saved so it never changes."""
    missing = sorted((m for m in rows if m.get('kind') in RUN_KINDS and not m.get('number')),
                     key=lambda m: m['created_at'])
    next_number = max([int(m.get('number') or 0) for m in rows] + _numbers(), default=0) + 1
    for meta in missing:
        meta['number'] = next_number
        try:
            _patch(meta['id'], number=next_number)
        except (OSError, ValueError):
            pass
        next_number += 1
