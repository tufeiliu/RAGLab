"""Metric definitions, ratings and charts for experiment results.

Presentation only: scores are read from saved reports. No API calls or
re-evaluation are performed.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

ANSWER_METRICS = {
    'accuracy': ('Accuracy', 'Is the answer correct compared with the reference answer? (LLM judge, 1–5)'),
    'completeness': ('Completeness', 'Does the answer cover everything in the reference answer? (LLM judge, 1–5)'),
    'relevance': ('Relevance', 'Does the answer address the question directly? (LLM judge, 1–5)'),
    'faithfulness': ('Faithfulness', 'Is every claim supported by the retrieved passages, i.e. not made up? (LLM judge, 1–5)'),
}
RETRIEVAL_METRICS = {
    # Against gold evidence passages (when test questions have them): the precise measure.
    'evidence_recall': ('Evidence recall', 'Share of the gold evidence passages that the search found '
                        '(a retrieved chunk overlaps the passage). 1 = every needed passage was retrieved.'),
    'evidence_precision': ('Evidence precision', 'Share of the retrieved chunks that overlap a gold evidence passage. '
                           'Even a perfect search returns some extra chunks when K is larger than the number of '
                           'passages needed, so 0.6 is already good.'),
    'evidence_hit': ('Evidence hit rate', 'Share of questions where at least one retrieved chunk overlaps the gold evidence.'),
    'evidence_mrr': ('Evidence MRR', 'How near the top the first evidence chunk is: 1 = always first, 0.5 = typically second.'),
    'evidence_ndcg': ('Evidence nDCG', 'Rank-weighted evidence hits: 1 = all evidence chunks ranked first.'),
    # Judged by the LLM judge from the reference answer, when no evidence passages are given.
    'context_recall': ('Context recall', "How much of the correct answer's information the retrieved passages "
                       'contain, as judged by the LLM judge. 1 = all of it.'),
    'context_precision': ('Context precision', 'Share of the retrieved passages that the judge found to contain '
                          'information from the correct answer. Naturally lower when K is large; 0.6 is already good.'),
    'context_hit': ('Context hit rate', 'Share of questions where at least one retrieved passage contains the answer (judged).'),
    'context_mrr': ('Context MRR', 'How near the top the first passage containing the answer is (judged). 1 = first.'),
    'context_ndcg': ('Context nDCG', 'Rank-weighted judged relevance: 1 = all relevant passages ranked first.'),
    # Against expected keywords: a cheap proxy.
    'mrr': ('Keyword MRR', 'How near the top the first passage containing each expected keyword is. '
                           '1 = always the first passage, 0.5 = typically second, 0 = never retrieved.'),
    'ndcg': ('Keyword nDCG', 'Like keyword MRR, but credits every passage containing a keyword, weighted by rank.'),
    'keyword_coverage': ('Keyword coverage', 'Share of the expected keywords found anywhere in the retrieved passages.'),
    # Older FinanceBench reports, scored by the example's own evidence heuristic.
    'precision_at_k': ('Precision@K', 'Share of the top-K passages that overlap the gold evidence.'),
    'recall_at_k': ('Recall@K', 'Share of gold evidence spans found in the top-K passages.'),
    'hit_at_k': ('Hit@K', 'Share of questions with at least one evidence passage in the top-K.'),
    'ndcg_at_k': ('nDCG@K', 'Rank-weighted evidence overlap.'),
    'source_hit': ('Source hit', 'Share of questions where a passage came from the correct source document.'),
}
EVIDENCE_KEYS = ('evidence_recall', 'evidence_precision', 'evidence_hit', 'evidence_mrr', 'evidence_ndcg')
JUDGED_KEYS = ('context_recall', 'context_precision', 'context_hit', 'context_mrr', 'context_ndcg')
LEGACY_EVIDENCE_KEYS = ('precision_at_k', 'recall_at_k', 'hit_at_k', 'ndcg_at_k')
# (good, fair) thresholds, as in the course's evaluation dashboard; precision is naturally lower.
THRESHOLDS = {'answer': (4.5, 4.0), 'retrieval': (0.9, 0.75), 'precision': (0.6, 0.4)}


def rating(key: str, value: float | None) -> tuple[str, str] | None:
    """('Good' | 'Fair' | 'Poor', badge colour) for a summary score."""
    if value is None:
        return None
    kind = 'answer' if key in ANSWER_METRICS else 'precision' if 'precision' in key else 'retrieval'
    good, fair = THRESHOLDS[kind]
    if value >= good:
        return 'Good', 'green'
    if value >= fair:
        return 'Fair', 'orange'
    return 'Poor', 'red'


def display(key: str, value: float) -> str:
    if key in ANSWER_METRICS:
        return f'{value:.2f}/5'
    if key in ('keyword_coverage', 'evidence_recall', 'evidence_hit', 'context_recall', 'context_hit'):
        return f'{value:.0%}'
    return f'{value:.2f}'


def available(summaries: list[dict], metrics: dict) -> dict:
    """Only metrics that at least one run actually measured."""
    return {key: value for key, value in metrics.items()
            if any(s.get(key) is not None for s in summaries)}


def chart_data(summaries: list[dict], metrics: dict) -> pd.DataFrame:
    """Long-form data. Missing values stay missing (never become zero)."""
    rows = [{'Run': s['name'], 'Metric': label, 'Score': float(s[key])}
            for s in summaries for key, (label, _) in metrics.items() if s.get(key) is not None]
    return pd.DataFrame(rows, columns=['Run', 'Metric', 'Score'])


def score_chart(summaries: list[dict], metrics: dict, *, max_score: float) -> alt.Chart | None:
    """Grouped bars: one group per metric, one bar per run. None if nothing was measured."""
    data = chart_data(summaries, metrics)
    if data.empty:
        return None
    order = [label for label, _ in metrics.values()]
    return alt.Chart(data).mark_bar().encode(
        x=alt.X('Metric:N', sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset('Run:N'),
        y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, max_score]), title=f'Score (0–{max_score:g})'),
        color=alt.Color('Run:N', title='Setup', legend=alt.Legend(orient='bottom', columns=1, labelLimit=0)),
        tooltip=['Run', 'Metric', alt.Tooltip('Score:Q', format='.3f')],
    ).properties(height=320)


def by_category(results: list[dict], metric: str) -> pd.DataFrame:
    """Mean of one per-question metric for each question category."""
    def value(row):
        return (row.get('judge') or {}).get(metric) if metric in ANSWER_METRICS else row.get(metric)
    rows = [{'Category': r.get('category') or 'unspecified', 'value': v}
            for r in results if (v := value(r)) is not None]
    if not rows:
        return pd.DataFrame(columns=['Category', 'Score', 'Questions'])
    frame = pd.DataFrame(rows)
    return (frame.groupby('Category')['value'].agg(['mean', 'count']).reset_index()
            .rename(columns={'mean': 'Score', 'count': 'Questions'}))


def category_chart(frame: pd.DataFrame, *, title: str, max_score: float) -> alt.Chart:
    return alt.Chart(frame).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, max_score]), title=title),
        y=alt.Y('Category:N', sort='-x', title=None),
        tooltip=['Category', alt.Tooltip('Score:Q', format='.2f'), 'Questions'],
    ).properties(height=max(120, 28 * len(frame)))
