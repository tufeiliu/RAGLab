import importlib.util
from pathlib import Path

file = Path(__file__).resolve().parents[1] / 'evidence_metrics.py'
spec = importlib.util.spec_from_file_location('evidence_metrics_under_test', file)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_real_financebench_doc_name():
    docs = [{'metadata': {'source_file': '3M_2018_10K'}, 'content': '2018 Purchases of property plant and equipment 1,577'}]
    gold = [{'doc_name': '3M_2018_10K', 'evidence_text': 'Purchases of property plant and equipment 1,577'}]
    r = e.score_retrieval(docs, gold, k=1)
    assert r['metric_status'] == 'heuristic_evidence_overlap_not_human_qrels'
    assert r['mrr'] == 1.0 and r['recall_at_k'] == 1.0


def test_legacy_field_and_wrong_document():
    gold = [{'evidence_doc_name': '3M_2018_10K', 'evidence_text': 'capital expenditure 1577'}]
    docs = [{'metadata': {'source_file': '3M_2022_10K'}, 'content': 'capital expenditure 1577'}]
    assert e.score_retrieval(docs, gold)['mrr'] == 0.0


def test_absent_evidence_remains_unavailable():
    assert e.score_retrieval([], [])['mrr'] is None
