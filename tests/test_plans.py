import pytest

from raglab.experiments import expand_grid
from raglab_app.plans import build_plan, parse_ints


def test_parse_ints():
    assert parse_ints('300, 500,500', label='x', low=1, high=999) == [300, 500]
    with pytest.raises(ValueError):
        parse_ints('3.5', label='x', low=1, high=9)
    with pytest.raises(ValueError):
        parse_ints('', label='x', low=1, high=9)


def test_each_technique_separately_with_grid():
    plan = build_plan(chat='c', judge=None, embeddings=['e'], chunk_sizes=[300, 600], chunk_overlap=50,
                      top_ks=[5], techniques=['rerank', 'query_expansion'], combine=False, baseline=True)
    names = [r['name'] for r in plan['runs']]
    assert names == ['baseline · chunk300', 'rerank · chunk300', 'query_expansion · chunk300',
                     'baseline · chunk600', 'rerank · chunk600', 'query_expansion · chunk600']
    assert plan['runs'][2]['techniques'] == {'query_expansion': 3}
    assert plan['settings']['judge_enabled'] is False
    assert expand_grid(plan) == plan['runs']  # valid for raglab.experiments


def test_combined_and_limits():
    plan = build_plan(chat='c', judge='j', embeddings=['e'], chunk_sizes=[500], chunk_overlap=50, top_ks=[5],
                      techniques=['query_fusion', 'rerank'], combine=True, baseline=False)
    assert [r['techniques'] for r in plan['runs']] == [{'query_rewrite': True, 'query_fusion': True, 'rerank': True}]
    with pytest.raises(ValueError, match='smaller'):
        build_plan(chat='c', judge=None, embeddings=['e'], chunk_sizes=[100], chunk_overlap=100, top_ks=[5],
                   techniques=[], combine=False, baseline=True)
    with pytest.raises(ValueError, match='limit'):
        build_plan(chat='c', judge=None, embeddings=['e'], chunk_sizes=list(range(100, 1000, 100)),
                   chunk_overlap=10, top_ks=[3, 5, 8, 10], techniques=[], combine=False, baseline=True)
