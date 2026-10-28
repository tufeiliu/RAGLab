from raglab_app import results_charts as mod


def test_scores_preserved_and_missing_not_zero():
    summaries = [{'name': 'hybrid', 'mrr': .783, 'ndcg': None},
                 {'name': 'baseline', 'mrr': .328, 'ndcg': .4}]
    data = mod.chart_data(summaries, mod.available(summaries, mod.RETRIEVAL_METRICS))
    assert len(data) == 3
    assert data.loc[(data.Run == 'hybrid') & (data.Metric == 'Keyword MRR'), 'Score'].iloc[0] == .783


def test_only_measured_metrics_are_shown():
    summaries = [{'name': 'a', 'accuracy': 3.0, 'keyword_coverage': None}]
    assert list(mod.available(summaries, mod.ANSWER_METRICS)) == ['accuracy']
    assert mod.available(summaries, mod.RETRIEVAL_METRICS) == {}
    assert mod.score_chart(summaries, {}, max_score=1) is None


def test_answer_scale():
    chart = mod.score_chart([{'name': 'a', 'accuracy': 2.5}], mod.ANSWER_METRICS, max_score=5).to_dict()
    assert chart['encoding']['y']['scale']['domain'] == [0, 5]
