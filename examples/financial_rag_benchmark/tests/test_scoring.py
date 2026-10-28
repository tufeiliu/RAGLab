from pathlib import Path
import importlib.util

PATH = Path(__file__).resolve().parents[1] / 'scoring.py'
spec = importlib.util.spec_from_file_location('finance_scoring', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_source_hit():
    docs = [{'content': 'Revenue 42', 'metadata': {'source': 'folder/AAA_2022_10K.txt'}}]
    assert mod.source_hit(docs, 'AAA_2022_10K') == 1
    assert mod.source_hit(docs, 'BBB_2022_10K') == 0


def test_evidence_source_condition():
    docs = [{'content': 'company recorded cash flow from operations total 42 million dollars',
             'metadata': {'source': 'BBB_2022_10K.txt'}}]
    evidence = [{'evidence_doc_name': 'AAA_2022_10K',
                 'evidence_text': 'company recorded cash flow from operations total 42 million dollars'}]
    assert mod.evidence_recall(docs, evidence) == 0
    docs[0]['metadata']['source'] = 'AAA_2022_10K.txt'
    assert mod.evidence_recall(docs, evidence) == 1


def test_numeric_diagnostic_not_accuracy():
    assert mod.numeric_overlap('100 dollars', '100 dollars') == 1
    assert mod.numeric_overlap('200 dollars', '100 dollars') == 0
    assert mod.numeric_overlap('n/a', 'no values') is None


def test_paired_bootstrap():
    def exp(name, vals):
        return {'name': name, 'results': [{'question': str(i), 'finance': {'evidence_lexical_recall': v}}
                                          for i, v in enumerate(vals)]}
    report = {'experiments': [exp('baseline',[0, 0, 0]), exp('better',[1,1,1])]}
    r = mod.paired_bootstrap(report, n=100)
    assert r[0]['mean_delta'] == 1
    assert r[0]['ci95_low'] == 1
