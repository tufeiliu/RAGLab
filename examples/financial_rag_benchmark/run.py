"""FinanceBench experiments: common RAGLab models/evaluator, finance-only retrieval research.

Gold answers/evidence and gold document IDs are used only AFTER generation for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'experiments.yaml')
    parser.add_argument('--catalog', type=Path, default=ROOT.parents[1] / 'models.yaml')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'report.json')
    parser.add_argument('--rebuild-index', action='store_true')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='Reuse embeddings and model responses cached in this folder')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT.parents[1] / '.env')
    from raglab.catalog import ModelCatalog, chat_client, embedding_client
    from raglab.core import RAGConfig
    from raglab.evaluation import LLMJudge, load_tests, evaluate_dataset
    from raglab.failure import failure_report
    from raglab.cache import CachedEmbeddings, SQLiteResponseCache
    from raglab.experiments import expand_grid
    from financial_plan import load_financial_plan
    from financial_retrieval import FinancialRAGLab, windows_from_file
    from scoring import enrich, paired_bootstrap
    from evidence_metrics import enrich_retrieval

    gold = [json.loads(s) for s in (args.data_dir / 'questions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    tests = load_tests(args.data_dir / 'tests.jsonl')
    expected = {r['doc_name'] for r in gold}
    paths = sorted(p for p in (args.data_dir / 'documents').glob('*.txt') if p.stem in expected)
    if not paths or len(gold) != len(tests) or {p.stem for p in paths} != expected:
        parser.error('Missing or inconsistent source filings. Run prepare.py first.')
    plan = load_financial_plan(args.plan)
    defaults = plan.get('settings', {})
    catalog = ModelCatalog(args.catalog)
    specs = expand_grid(plan)
    # Build table-aware chunks for each unique ingestion setting, not once globally.
    # Otherwise a grid search over window sizes would silently use one index.
    chunk_cache = {}
    responses = SQLiteResponseCache(args.cache_dir / 'responses.sqlite') if args.cache_dir else None

    def embedder_for(spec):
        client = embedding_client(spec)
        return CachedEmbeddings(client, spec.model, args.cache_dir / 'embeddings.sqlite') if args.cache_dir else client
    print(f'Running {len(tests)} questions from {len(paths)} source filings across {len(specs)} runs', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    def progress_event(**details):
        print('RAGLAB_PROGRESS ' + json.dumps(details, separators=(',', ':')), flush=True)
    progress_event(stage='preparing', total_runs=len(specs), total_questions=len(tests),
                   completed_runs=0, question_done=0)
    for i, spec in enumerate(specs, 1):
        progress_event(stage='indexing', total_runs=len(specs), completed_runs=i-1,
                       current_run=spec['name'], total_questions=len(tests), question_done=0)

        settings = {**defaults, **spec.get('settings', {})}
        techniques = spec.get('techniques', {})
        mode = spec.get('retrieval', 'vector')
        if mode not in {'vector', 'filtered_vector', 'hybrid'}:
            raise ValueError(f"Invalid retrieval mode in {spec['name']}: {mode}")
        chat_alias = settings['chat']
        embed_alias = settings['embedding']
        judge_alias = settings.get('judge', chat_alias)
        chat = catalog.get('chat', chat_alias)
        embed = catalog.get('embedding', embed_alias)
        judge_spec = catalog.get('chat', judge_alias) if settings.get('judge_enabled', True) else None
        config = RAGConfig(chat_model=chat.model, embedding_model=embed.model,
                           top_k=int(settings.get('top_k', 5)),
                           query_rewrite=bool(techniques.get('query_rewrite', False)),
                           query_fusion=bool(techniques.get('query_fusion', False)),
                           query_expansion=int(techniques.get('query_expansion', 1)),
                           rerank=bool(techniques.get('rerank', False)),
                           improved_prompt=bool(techniques.get('improved_prompt', False)))
        window_lines = int(settings.get('window_lines', 32))
        window_overlap = int(settings.get('window_overlap', 10))
        if not (8 <= window_lines <= 300 and 0 <= window_overlap < window_lines):
            raise ValueError(f"Invalid chunk parameters in {spec['name']}: lines={window_lines}, overlap={window_overlap}")
        chunk_key = (window_lines, window_overlap)
        if chunk_key not in chunk_cache:
            chunk_cache[chunk_key] = [chunk for path in paths for chunk in windows_from_file(
                path, lines_per_chunk=window_lines, overlap=window_overlap)]
        chunks = chunk_cache[chunk_key]
        if not chunks:
            raise ValueError(f"Empty chunk collection for {spec['name']}")
        print(f"Index settings for {spec['name']}: lines={window_lines}, overlap={window_overlap}, chunks={len(chunks)}", flush=True)
        hasher = hashlib.sha256()
        hasher.update(f'{embed.model}|table_windows|{window_lines}|{window_overlap}'.encode())
        for path in paths:
            hasher.update(str(path.resolve()).encode())
            hasher.update(path.read_bytes())
        index_key = hasher.hexdigest()[:16]
        config.db_dir = str(args.data_dir / 'indexes' / index_key)
        config.collection_name = 'financial'
        index_marker = Path(config.db_dir) / 'index_ready.json'
        lab = FinancialRAGLab(config, llm=chat_client(chat, cache=responses), embedder=embedder_for(embed))
        # Validate the actual collection, not just a stale marker file.
        # RAGLab suffixes collection_name with the embedding model.
        actual_count = lab.store._collection.count()
        if args.rebuild_index or not index_marker.exists() or actual_count != len(chunks):
            # Recreate only the cache index for this exact corpus/configuration.
            lab.store.delete_collection()
            from langchain_chroma import Chroma
            lab.store = Chroma(collection_name=lab.collection, persist_directory=config.db_dir,
                               embedding_function=lab.embedder)
            lab.store.add_documents(chunks)
            index_marker.parent.mkdir(parents=True, exist_ok=True)
            index_marker.write_text(json.dumps({'key': index_key, 'documents': len(chunks)}))
        actual_count = lab.store._collection.count()
        if actual_count != len(chunks):
            raise RuntimeError(f'Index incomplete: {lab.collection} has {actual_count} vectors; expected {len(chunks)}')
        lab.configure_retrieval(chunks, mode)
        judge = LLMJudge(llm=chat_client(judge_spec, cache=responses)) if judge_spec else None
        progress_event(stage='evaluating', total_runs=len(specs), completed_runs=i-1,
                       current_run=spec['name'], total_questions=len(tests), question_done=0)
        result = evaluate_dataset(lab, tests, judge=judge,
            on_progress=lambda done, total, ok, failed: progress_event(
                stage='evaluating', total_runs=len(specs), completed_runs=i-1,
                current_run=spec['name'], total_questions=total or len(tests),
                question_done=done, successful=ok, failed=failed))
        result.update(name=spec['name'], models={'chat': chat_alias, 'embedding': embed_alias,
                       'judge': judge_alias if judge else None}, index_key=index_key, index_dir=config.db_dir, index_collection=lab.collection,
                      config={'retrieval': mode, **settings, **techniques})
        # FinanceBench has no expected-keyword labels. Zero would falsely imply failed retrieval.
        for row in result['results']:
            for key in ('mrr', 'ndcg', 'keyword_coverage', 'keyword_coverage_pct'):
                row[key] = None
        for key in ('mrr', 'ndcg', 'keyword_coverage'):
            result['summary'][key] = None
        result['failures'] = failure_report(result)
        reports.append(result)
        progress_event(stage='run_complete', total_runs=len(specs), completed_runs=i,
                       current_run=spec['name'], total_questions=len(tests), question_done=len(tests))
        print(f"Completed {i}/{len(specs)}: {spec['name']} ({result['summary']['successful']} successful, {result['summary']['failed']} failed)", flush=True)
        if result['errors']:
            print('  First error:', result['errors'][0]['error'], file=sys.stderr, flush=True)
    reports.sort(key=lambda e: e['summary'].get('accuracy') or 0, reverse=True)
    report = {'experiments': reports,
              'leaderboard': [{'name': e['name'], 'models': e['models'], **e['summary']} for e in reports],
              'ranking_rule': 'mean judge accuracy (descriptive; inspect retrieval evidence and uncertainty)'}
    enrich(report, gold)
    report['research_metadata'] = {
        'benchmark': 'FinanceBench open-source sample', 'questions': len(gold),
        'document_names': sorted(expected), 'gold_evidence_in_index': False,
        'keyword_based_metrics': 'N/A: tests have no keywords; evidence-based proxy metrics added separately',
        'evidence_lexical_recall': 'heuristic, not judged-document relevance',
        'retrieval_modes': {'vector': 'unfiltered vector baseline',
                            'filtered_vector': 'question-derived metadata filter + vector',
                            'hybrid': 'question-derived metadata filter + BM25/vector reciprocal-rank fusion'}}
    enrich_retrieval(report, gold)
    # The same evidence scores RAGLab's general pipeline reports, so finance-tuned
    # and generic runs on these questions can be compared like for like.
    from raglab.evidence import evidence_metrics
    from statistics import mean
    evidence_by_question = {g['question']: g.get('evidence') or [] for g in gold}
    general_keys = ('evidence_precision', 'evidence_recall', 'evidence_hit', 'evidence_mrr', 'evidence_ndcg')
    for experiment in report['experiments']:
        k = int(experiment.get('config', {}).get('top_k', 5))
        for row in experiment['results']:
            row.update(evidence_metrics(row.get('retrieved_documents', []), evidence_by_question.get(row['question']), k=k))
        for key in general_keys:
            values = [r[key] for r in experiment['results'] if r.get(key) is not None]
            experiment['summary'][key] = mean(values) if values else None
    # Scoring may mutate summaries; always rebuild the leaderboard at the end.
    report['leaderboard'] = [
        {'name': e['name'], 'models': e['models'], **e['summary']}
        for e in report['experiments']
    ]
    report['paired_bootstrap'] = paired_bootstrap(report)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', args.output)
    print(json.dumps(report['leaderboard'], indent=2))


if __name__ == '__main__':
    main()
