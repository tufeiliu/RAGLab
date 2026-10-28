from langchain_core.documents import Document

from raglab.evaluation import TestCase, evaluate_result, load_tests
from raglab.evidence import evidence_metrics, normalise_evidence

TABLE = ('Consolidated Statement of Cash Flows (Millions) 2018 2017 Purchases of property, plant and equipment '
         '(1,577) (1,373) Proceeds from sale of businesses 846 1,065 Net cash used in investing activities '
         '(222) (3,086) Depreciation and amortization 1,488 1,544')


def doc(text, source='3M_2018_10K.txt'):
    return Document(page_content=text, metadata={'source': source})


def test_small_chunk_inside_long_evidence_is_relevant():
    chunk = 'Purchases of property, plant and equipment (1577) (1373) Proceeds from sale of businesses 846 1065'
    noise = 'The board of directors met four times during the year to discuss governance and strategy matters.'
    m = evidence_metrics([doc(noise), doc(chunk)], [{'text': TABLE, 'source': '3M_2018_10K'}], k=2)
    assert m['evidence_hit'] == 1 and m['evidence_mrr'] == 0.5 and m['evidence_precision'] == 0.5
    assert m['evidence_recall'] == 1


def test_wrong_source_is_not_relevant():
    m = evidence_metrics([doc(TABLE, source='3M_2022_10K.txt')], [{'text': TABLE, 'source': '3M_2018_10K'}], k=5)
    assert m['evidence_hit'] == 0 and m['evidence_mrr'] == 0


def test_plain_text_evidence_without_source():
    m = evidence_metrics([{'content': TABLE, 'metadata': {}}], TABLE, k=5)
    assert m['evidence_hit'] == 1


def test_no_evidence_means_not_measured():
    assert set(evidence_metrics([doc(TABLE)], [], k=5).values()) == {None}


def test_financebench_rows_and_unknown_fields_load(tmp_path):
    file = tmp_path / 't.jsonl'
    file.write_text('{"question": "Capex?", "reference_answer": "$1577", "category": null, "company": "3M", '
                    '"evidence": [{"evidence_text": "' + TABLE + '", "doc_name": "3M_2018_10K"}]}\n'
                    '{"question": "Q2", "keywords": "a, b"}\n')
    cases = load_tests(file)
    assert cases[0].evidence == [{'text': TABLE, 'source': '3M_2018_10K'}]
    assert cases[0].category == 'unspecified' and cases[1].keywords == ['a', 'b']
    assert normalise_evidence('x') == [{'text': 'x', 'source': None}]


def test_evaluate_result_includes_evidence_scores():
    class Result:
        answer, documents = 'A', [doc(TABLE)]
    row = evaluate_result(Result(), TestCase(question='q', evidence=[TABLE]), k=5)
    assert row['evidence_hit'] == 1 and row['mrr'] is None


def test_judged_search_scores_from_supporting_passages():
    from raglab.evaluation import JudgeScores, judged_retrieval_metrics
    scores = JudgeScores(feedback='', accuracy=1, completeness=1, relevance=5, supporting_passages=[3, 9],
                         context_recall=3)
    m = judged_retrieval_metrics(scores, retrieved=5, k=5)
    assert m['context_hit'] == 1 and m['context_mrr'] == 1 / 3 and m['context_precision'] == 0.2
    assert m['context_recall'] == 0.5 and 0 < m['context_ndcg'] < 1
    none = judged_retrieval_metrics(JudgeScores(feedback='', accuracy=1, completeness=1, relevance=1), 5, 5)
    assert none['context_hit'] == 0 and none['context_recall'] is None


def test_evidence_wins_over_judged_scores_in_the_display():
    from raglab_app.report_view import retrieval_metrics_for
    both = retrieval_metrics_for([{'evidence_recall': 1.0, 'context_recall': 0.5}])
    assert 'evidence_recall' in both and 'context_recall' not in both
    assert 'context_recall' in retrieval_metrics_for([{'context_recall': 0.5}])


def test_judge_scores_a_declined_answer_as_wrong():
    from raglab.evaluation import JudgeScores, LLMJudge

    class Structured:
        def invoke(self, messages):
            return JudgeScores(declined=True, feedback='Correctly says the context lacks it.',
                               accuracy=5, completeness=3, relevance=5)

    class FakeLLM:
        def with_structured_output(self, schema):
            return Structured()

    scores = LLMJudge(llm=FakeLLM()).evaluate('Who?', 'The context does not say.', 'Yi Liu', [])
    assert scores.accuracy == 1 and scores.completeness == 1 and scores.relevance == 5
    assert scores.feedback.startswith('No answer was given')
