"""find_best_rag end to end with stand-in models (no API calls)."""
import hashlib

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from raglab import experiments, find_best_rag, plan_search, search


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    for module in (experiments, search):
        monkeypatch.setattr(module, 'chat_client', lambda spec, cache=None: FakeListChatModel(responses=['42.'] * 100))
        monkeypatch.setattr(module, 'embedding_client', lambda spec: FakeEmbeddings())


def test_plan_search_crosses_every_option():
    plan = plan_search({'chunk_size': [200, 400], 'techniques': [[], ['rerank', 'opening']]}, judge=False)
    assert [r['name'] for r in plan['runs']] == ['chunk 200 · baseline', 'chunk 200 · rerank+opening',
                                                'chunk 400 · baseline', 'chunk 400 · rerank+opening']
    assert plan['runs'][1]['techniques'] == {'rerank': True, 'opening': True}
    assert plan['runs'][2]['settings']['chunk_overlap'] == 80  # 20% of the chunk size by default
    assert plan['settings']['judge_enabled'] is False


@pytest.mark.parametrize('bad, message', [({'temperature': [0]}, 'Cannot search'),
                                          ({'techniques': [['magic']]}, 'Unknown technique'),
                                          ({'chunk_size': [100], 'chunk_overlap': [100]}, 'smaller'),
                                          ({'chunk_size': list(range(100, 1000, 10))}, 'exceed')])
def test_plan_search_rejects_bad_options(bad, message):
    with pytest.raises(ValueError, match=message):
        plan_search(bad)


def test_find_best_rag_ranks_setups_and_opens_the_winner(tmp_path, fake_models):
    doc = tmp_path / 'notes.txt'
    answer = 'The deep thought computer worked out that the answer to life, the universe and everything is 42. '
    bananas = 'Ripe bananas turn yellow because chlorophyll breaks down and reveals carotenoid pigments underneath. '
    doc.write_text(answer * 20 + bananas * 20)
    tests = [{'question': 'What is the answer?', 'evidence': answer, 'keywords': '42'},
             {'question': 'What colour are bananas?', 'evidence': [bananas]}]
    result = find_best_rag([doc], tests, {'chunk_size': [100, 300], 'top_k': [1, 3]}, judge=False,
                           workdir=tmp_path / 'work')
    assert result.metric == 'evidence_mrr'
    board = result.leaderboard
    assert len(board) == 4 and list(board.index) == [1, 2, 3, 4]
    assert board['evidence_mrr'].is_monotonic_decreasing
    assert result.best['name'] == board.loc[1, 'setup']
    rag = result.best_rag()
    assert rag.config.chunk_size == result.best['settings']['chunk_size']
    assert rag.ask('What is the answer?').answer == '42.'
    assert (tmp_path / 'r.json').exists() is False and result.save(tmp_path / 'r.json').is_file()
    # Second search over the same files reuses the indexes: no new ones appear.
    built = sorted(p.name for p in (tmp_path / 'work' / 'indexes').iterdir())
    find_best_rag(tmp_path / 'notes.txt', tests, {'chunk_size': [100]}, judge=False, workdir=tmp_path / 'work')
    assert sorted(p.name for p in (tmp_path / 'work' / 'indexes').iterdir()) == built


def test_failed_runs_rank_last():
    summaries = [{'accuracy': 5, 'failed': 1}, {'accuracy': 3, 'failed': 0}, {'accuracy': 4, 'failed': 0}]
    assert experiments.rank(summaries, 'accuracy') == [2, 1, 0]


class JudgingModel(FakeListChatModel):
    """Answers '42.'; as a judge, rates relevance 4 and says passage 1 helps."""
    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda messages: schema(feedback='ok', accuracy=5, completeness=5, relevance=4,
                                                      faithfulness=5, context_recall=5, supporting_passages=[1]))


@pytest.fixture
def judging_models(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    for module in (experiments, search):
        monkeypatch.setattr(module, 'chat_client', lambda spec, cache=None: JudgingModel(responses=['42.'] * 200))
        monkeypatch.setattr(module, 'embedding_client', lambda spec: FakeEmbeddings())


def test_tests_can_be_one_question_a_list_or_a_file(tmp_path):
    assert [c.question for c in search._test_cases('Who wrote it?')] == ['Who wrote it?']
    assert [c.question for c in search._test_cases(['A?', {'question': 'B?', 'keywords': 'x, y'}])] == ['A?', 'B?']
    (tmp_path / 't.jsonl').write_text('{"question": "C?"}\n')
    assert [c.question for c in search._test_cases(str(tmp_path / 't.jsonl'))] == ['C?']
    with pytest.raises(ValueError, match='needs a question'):
        search._test_cases(['  '])


def test_questions_without_answers_are_judged_without_a_reference(tmp_path, judging_models):
    (tmp_path / 'notes.txt').write_text('The answer to life, the universe and everything is 42. ' * 30)
    result = find_best_rag(tmp_path, 'What is the answer?', {'top_k': [1, 3]}, workdir=tmp_path / 'work')
    assert result.metric == 'relevance'
    row = result.report['experiments'][0]['results'][0]
    assert row['judge']['accuracy'] is None and row['judge']['relevance'] == 4  # no correctness without a reference
    assert row['context_precision'] is not None
    assert set(result.leaderboard.columns) >= {'relevance', 'faithfulness', 'answered', 'context_precision'}
    # A list of questions, one of them with a correct answer: that one is also scored for accuracy.
    mixed = find_best_rag(tmp_path, ['What is the answer?', {'question': 'Is it 42?', 'reference_answer': 'Yes'}],
                          workdir=tmp_path / 'work')
    rows = mixed.report['experiments'][0]['results']
    assert [r['judge']['accuracy'] for r in rows] == [None, 5]


def test_without_judge_every_question_needs_an_answer_key(tmp_path, fake_models):
    (tmp_path / 'a.txt').write_text('text ' * 50)
    with pytest.raises(ValueError, match='no answer key'):
        find_best_rag(tmp_path, ['Q?'], judge=False, workdir=tmp_path / 'work')


def test_reference_free_judge_marks_a_declined_answer_irrelevant():
    from raglab.evaluation import JudgeScores, LLMJudge

    class Declining(FakeListChatModel):
        def with_structured_output(self, schema, **kwargs):
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(lambda m: JudgeScores(feedback='honest', declined=True, relevance=5,
                                                        faithfulness=5, accuracy=4))
    scores = LLMJudge(llm=Declining(responses=['x']), without_reference=True).evaluate('Q?', 'The text does not say.', '')
    assert scores.relevance == 1 and scores.accuracy is None
