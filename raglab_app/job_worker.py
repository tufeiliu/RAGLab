"""Subprocess worker for background jobs; started by raglab_app.jobs. Don't run directly."""
from __future__ import annotations

import fcntl
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from raglab_app import CACHE, CATALOG, INDEXES
from raglab_app.jobs import _write, progress_event
from raglab_app.logs import setup_job_logging

log = logging.getLogger('raglab.job')


def run_experiment(directory: Path, meta: dict) -> None:
    from raglab.evaluation import load_tests
    from raglab.experiments import run_experiments
    from raglab_app import collections
    paths = collections.file_paths(meta['collection'])
    tests = load_tests(collections.test_path(meta['collection'], meta['test_set']))
    log.info('Experiment on collection %r (%d files) with test set %r (%d questions)',
             meta['collection_name'], len(paths), meta['test_set'], len(tests))
    report = run_experiments(directory / 'experiments.yaml', CATALOG, tests, [str(p) for p in paths],
                             index_root=INDEXES, cache_dir=CACHE,
                             cache_responses=meta.get('cache_responses', False), progress=progress_event)
    report['collection'] = {'name': meta['collection_name'], 'files': [p.name for p in paths],
                            'test_set': meta['test_set']}
    (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


def run_financial(directory: Path, meta: dict) -> None:
    from examples.financial_rag_benchmark.ui_backend import RUNNER, PROJECT
    command = [sys.executable, '-u', str(RUNNER), '--plan', str(directory / 'experiments.yaml'),
               '--catalog', str(CATALOG), '--output', str(directory / 'report.json'),
               '--cache-dir', str(CACHE)]
    # Serialize benchmark work to protect the example's shared Chroma caches.
    with (directory.parent / 'financial_index.lock').open('a') as lock:
        log.info('Waiting for other FinanceBench runs to finish')
        fcntl.flock(lock, fcntl.LOCK_EX)
        log.info('Running FinanceBench benchmark')
        result = subprocess.run(command, cwd=PROJECT, check=False)
    if result.returncode:
        raise RuntimeError(f'Benchmark exited with code {result.returncode}')
    if not (directory / 'report.json').exists():
        raise FileNotFoundError('Runner did not save a report')
    if meta.get('collection'):
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        report['collection'] = {'name': meta['collection_name'], 'test_set': meta['test_set'],
                                'files': sorted(collections_files(meta['collection']))}
        report['pipeline'] = 'finance-tuned'
        (directory / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


def collections_files(slug: str) -> list[str]:
    from raglab_app import collections
    return list(collections.load(slug)['files'])


def run_prepare(directory: Path, meta: dict) -> None:
    from raglab_app import benchmarks
    command = [sys.executable, '-u', str(benchmarks.FINANCE / 'prepare.py'),
               '--max-documents', str(meta['max_documents']), '--max-questions', str(meta['max_questions'])]
    log.info('Downloading FinanceBench: up to %d filings and %d questions', meta['max_documents'], meta['max_questions'])
    result = subprocess.run(command, cwd=benchmarks.FINANCE, check=False)
    if result.returncode:
        raise RuntimeError(f'prepare.py exited with code {result.returncode}')
    status = benchmarks.financebench_status()
    log.info('FinanceBench ready: %d questions over %d filings', status['questions'], status['documents'])
    if benchmarks.loaded('financebench'):
        benchmarks.load('financebench')
        log.info('Refreshed the FinanceBench collection')


RUNNERS = {'experiment': run_experiment, 'financial': run_financial, 'prepare': run_prepare}


def main(directory: Path, kind: str) -> None:
    setup_job_logging()
    state_file = directory / 'status.json'
    meta = json.loads(state_file.read_text(encoding='utf-8'))
    meta['status'] = 'running'
    meta['started_at'] = time.time()
    _write(state_file, meta)
    try:
        if kind not in RUNNERS:
            raise ValueError(f'Unknown job kind: {kind}')
        RUNNERS[kind](directory, meta)
        meta['status'] = 'completed'
        log.info('Job finished in %.0fs', time.time() - meta['started_at'])
    except Exception as exc:
        meta['status'] = 'failed'
        meta['error'] = str(exc)
        log.exception('Job failed')
    finally:
        meta['finished_at'] = time.time()
        _write(state_file, meta)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
