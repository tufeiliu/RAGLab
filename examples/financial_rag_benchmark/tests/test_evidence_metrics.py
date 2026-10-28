import importlib.util
from pathlib import Path

module = Path(__file__).resolve().parents[1] / 'evidence_metrics.py'
spec = importlib.util.spec_from_file_location('finance_evidence_metrics', module)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_correct_filing_and_table_evidence():
    evidence = [{'evidence_doc_name': '3M_2018_10K', 'evidence_text': 'Purchases of property plant and equipment (1,577) 2018'}]
    docs = [
        {'metadata': {'source_file': '3M_2023Q2_10Q'}, 'content': 'Purchases property plant equipment 1,577 2018'},
        {'metadata': {'source_file': '3M_2018_10K'}, 'content': '2018 Purchases of property, plant and equipment (1,577)'}]
    scores = e.score_retrieval(docs, evidence, k=2)
    assert scores['mrr'] == 0.5
    assert scores['recall_at_k'] == 1
    assert scores['precision_at_k'] == 0.5
    assert 0 < scores['ndcg_at_k'] <= 1


def test_missing_gold_is_unavailable():
    scores = e.score_retrieval([], [], k=5)
    assert scores['mrr'] is None
    assert scores['precision_at_k'] is None


def test_missing_evidence_scores_zero():
    scores = e.score_retrieval([{'metadata': {'source_file': 'x'}, 'content': 'unrelated'}],
                               [{'evidence_doc_name': '3M_2018_10K', 'evidence_text': 'cash flow purchases plant 1577'}], k=5)
    assert scores['mrr'] == 0
    assert scores['precision_at_k'] == 0
    assert scores['recall_at_k'] == 0
