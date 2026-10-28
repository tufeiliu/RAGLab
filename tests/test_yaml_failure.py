"""Unit tests that do not require paid API calls."""
import pytest
from raglab.catalog import ModelCatalog
from raglab.failure import analyze_failure, failure_report
from raglab.experiments import load_plan, expand_grid


def test_catalog_selects_key_env(tmp_path, monkeypatch):
    file = tmp_path / 'models.yaml'
    file.write_text('models:\n  chat:\n    mini:\n      label: test\n      provider: openai\n      model: gpt-4.1-nano\n      api_key_env: TEST_KEY\n')
    registry = ModelCatalog(file)
    assert not registry.choices('chat')
    with pytest.raises(EnvironmentError):
        registry.get('chat', 'mini')
    monkeypatch.setenv('TEST_KEY','abc')
    assert registry.get('chat', 'mini').model == 'gpt-4.1-nano'


def test_failure_labels_and_sources():
    row = {'question':'x', 'expected_keywords':['found','missing'],
           'retrieved_documents':[{'content':'found', 'metadata': {'source':'abc'}}],
           'keyword_coverage':0.5, 'judge':{'accuracy':2, 'faithfulness':1}}
    diag = analyze_failure(row)
    assert 'retrieval_keyword_gap' in diag['flags']
    assert 'grounding_failure' in diag['flags']
    assert diag['sources'] == ['abc']
    assert diag['missing_keywords'] == ['missing']
    assert len(failure_report({'results':[row]})) == 1


def test_plan_grid(tmp_path):
    file = tmp_path / 'experiments.yaml'
    file.write_text('runs:\n  - name: baseline\n    techniques: {}\ngrid:\n  rerank: [true, false]\n  query_expansion: [1, 3]\n')
    assert len(expand_grid(load_plan(file))) == 4
