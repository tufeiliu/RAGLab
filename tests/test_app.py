"""Render every page headlessly against temporary data folders (no API calls)."""
import hashlib
import json

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

import raglab_app
from raglab_app import collections, jobs, models, ui


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(collections, 'COLLECTIONS', tmp_path / 'collections')
    monkeypatch.setattr(raglab_app, 'INDEXES', tmp_path / 'indexes')
    monkeypatch.setattr(ui, 'INDEXES', tmp_path / 'indexes')
    clients = {'llm': FakeListChatModel(responses=['Forty-two.'] * 20), 'embedder': FakeEmbeddings()}
    monkeypatch.setattr(models, 'make_clients', lambda *a, **k: clients)
    monkeypatch.setattr(ui, 'make_clients', lambda *a, **k: clients)
    (tmp_path / 'jobs').mkdir()
    return tmp_path


def add_job(root, job_id, **meta):
    (root / 'jobs' / job_id).mkdir()
    (root / 'jobs' / job_id / 'status.json').write_text(json.dumps({'id': job_id, **meta}))


TABS = {'ask': ':material/chat: Ask a question', 'test': ':material/checklist: Test questions',
        'compare': ':material/compare_arrows: Compare runs'}


def app(page=None, tab=None):
    at = AppTest.from_file('../app.py', default_timeout=60)
    if tab:
        at.session_state['workspace_tab'] = TABS[tab]
    at.run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_workspace_without_workspaces(data):
    at = app()
    assert 'New workspace' in at.info[0].value


def test_guide_and_library_render(data):
    at = app('app_pages/guide.py')
    assert at.markdown[0].value.startswith('# RAGLab guide')
    assert [t.label for t in at.tabs] == ['Overview', 'Using the app', 'Test questions', 'Settings', 'Techniques',
                                          'Scores', 'Improving results', 'Python API', 'Benchmarks',
                                          'Storage and limits']
    # Code blocks stay inside their section (the Python comments are not headings).
    assert '# the app' in next(t for t in at.tabs if t.label == 'Python API').markdown[0].value
    at = app('app_pages/library.py')
    assert any('FinanceBench: plain vector search' in m.value for m in at.markdown)
    next(b for b in at.button if b.label == 'Open').click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert 'Which setup did best?' in [s.value for s in at.subheader]
    assert len(at.get('vega_lite_chart')) >= 3  # heatmap plus answer and retrieval bar charts


def test_results_empty_and_listing(data):
    assert any('No runs yet' in i.value for i in app('app_pages/results.py').info)
    add_job(data, 'aaaaaaaaaaaa', kind='experiment', status='running', display_name='Chunk sweep',
            collection_name='Papers', created_at='2026-10-27T10:00:00+00:00', pid=None)
    add_job(data, 'bbbbbbbbbbbb', kind='financial', status='failed', error='boom')  # old job: no name/date
    add_job(data, 'cccccccccccc', kind='index', status='completed')  # retired kind: hidden
    at = app('app_pages/results.py')
    table = at.dataframe[0].value
    assert list(table['Run']) == ['Run', 'Chunk sweep']
    assert list(table['Status']) == ['Failed', 'Running']
    assert 'boom' in at.error[0].value


def test_finished_evaluation_shows_scorecard_and_categories(data):
    rows = [{'question': f'Q{i}?', 'category': cat, 'generated_answer': 'A', 'keyword_coverage': 0.5, 'mrr': mrr,
             'judge': {'accuracy': acc}, 'retrieved_documents': []}
            for i, (cat, mrr, acc) in enumerate([('facts', 1.0, 5), ('facts', 0.5, 4), ('dates', 0.2, 2)])]
    report = {'collection': {'name': 'Papers', 'files': ['a.txt'], 'test_set': 't.jsonl'},
              'experiments': [{'name': 'my setup', 'summary': {'total': 3, 'successful': 3, 'accuracy': 3.7,
                                                               'mrr': 0.57, 'keyword_coverage': 0.5, 'latency_s': 1.2},
                               'config': {'chunk_size': 500, 'top_k': 5}, 'models': {'chat': 'm'},
                               'results': rows, 'failures': [], 'errors': []}]}
    add_job(data, 'dddddddddddd', kind='experiment', status='completed', display_name='Done',
            report=str(data / 'jobs' / 'dddddddddddd' / 'report.json'))
    (data / 'jobs' / 'dddddddddddd' / 'report.json').write_text(json.dumps(report))
    at = app('app_pages/results.py')
    assert [m.label for m in at.metric] == ['Keyword MRR', 'Keyword coverage', 'Accuracy']
    assert len(at.get('vega_lite_chart')) >= 2  # answer and retrieval charts, also for a single setup
    assert [m.value for m in at.metric] == ['0.57', '50%', '3.70/5']
    assert 'Question by question' in [s.value for s in at.subheader]


def test_workspace_evaluate_section(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'hello world'), ('copy.txt', b'hello world')])
    collections.add_test_set(slug, 'qs.jsonl', b'{"question": "Hi?", "keywords": ["hello"]}\n')
    at = app(tab='test')
    assert any('`a.txt`' in m.value for m in at.markdown) and not any('copy.txt' in m.value for m in at.markdown)
    assert any('Will score: search, roughly (1/1 questions)' in c.value for c in at.caption)
    assert any(b.label == 'Run' for b in at.button)
    # The batch shortcut (in the Test questions tab) previews its setups
    assert next(d.value for d in at.dataframe if 'Setup' in d.value)['Setup'].tolist() == ['baseline', 'rerank', 'opening']
    assert any('No runs yet' in c.value for c in at.caption)
    at = app('app_pages/results.py')
    at.session_state['results_tab'] = ':material/checklist: Test questions'
    at.run()
    assert next(d.value for d in at.dataframe if 'Test set' in d.value)['Questions'].tolist() == [1]


def test_typed_test_set_round_trip(data):
    slug = collections.create('Papers')
    name = collections.save_test_set(slug, 'mine', [
        {'question': ' Who? ', 'reference_answer': 'Ann', 'keywords': 'Ann, author', 'category': ''},
        {'question': '', 'reference_answer': 'dropped'}])
    assert collections.read_test_set(slug, name) == [
        {'question': 'Who?', 'keywords': ['Ann', 'author'], 'reference_answer': 'Ann', 'category': 'unspecified',
         'evidence': []}]
    collections.add_files(slug, [('a.txt', b'hello')])
    at = app(tab='test')
    at.selectbox(key=f'edit_set_{slug}').set_value('mine.jsonl').run()
    editor = next(d.value for d in at.dataframe if 'reference_answer' in d.value)
    assert editor['question'].tolist() == ['Who?']


def test_ask_builds_index_answers_and_scores(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'The answer to everything is 42. ' * 30)])
    at = app()
    at.text_area(key='ask_question').input('What is the answer?')
    at.text_input(key='ask_keywords').input('42, everything')
    at.button(key='FormSubmitter:ask-Ask and evaluate').click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('Forty-two.' in m.value for m in at.markdown)
    assert 'Evaluation of this answer' in [s.value for s in at.subheader]
    assert [m.label for m in at.metric] == ['Keyword MRR', 'Keyword nDCG', 'Keyword coverage']
    assert at.metric[2].value == '100%'
    assert at.get('vega_lite_chart')  # full evaluation, with its chart
    at.text_input(key='ask_keywords').input('')
    at.text_area(key='ask_evidence').input('The answer to everything is 42. The answer to everything is 42. The answer to everything is 42.')
    at.button(key='FormSubmitter:ask-Ask and evaluate').click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.label for m in at.metric][:2] == ['Evidence recall', 'Evidence precision']
    assert at.metric[0].value == '100%'


def test_rename_job(data):
    add_job(data, 'aaaaaaaaaaaa', kind='experiment', status='queued')
    jobs.rename_job('aaaaaaaaaaaa', '  Top-k sweep ')
    assert jobs.recent_jobs()[0]['display_name'] == 'Top-k sweep'


def test_financebench_loads_with_evidence_and_offers_both_pipelines(data):
    from raglab_app import benchmarks
    status = benchmarks.financebench_status()
    if not status['ready']:
        pytest.skip('FinanceBench example data not prepared')
    slug = benchmarks.load('financebench')
    assert benchmarks.load('financebench') == slug  # refreshing reuses the collection
    collection = collections.load(slug)
    info = collection['tests']['financebench.jsonl']
    assert info['questions'] == status['questions'] and info['with_evidence'] == status['questions']
    first = collections.read_test_set(slug, 'financebench.jsonl')[0]
    assert first['evidence'][0]['source'] + '.txt' in collection['files']
    at = app(tab='test')
    total = status['questions']
    assert any(f'search, precisely ({total}/{total} questions)' in c.value for c in at.caption)
    at.segmented_control(key='eval_pipeline').set_value('finance').run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(b.label == 'Run evaluation' for b in at.button)


def test_old_finance_report_gets_evidence_scores_and_comparison(data):
    from raglab_app import benchmarks
    from raglab_app.report_view import add_evidence_scores, comparable_key
    if not benchmarks.financebench_status()['ready']:
        pytest.skip('FinanceBench example data not prepared')
    gold = [json.loads(line) for line in (benchmarks.FINANCE_DATA / 'questions.jsonl').read_text().splitlines()]
    evidence = gold[0]['evidence'][0]
    report = {'experiments': [{'name': 'hybrid', 'summary': {}, 'config': {'top_k': 5}, 'results': [{
        'question': gold[0]['question'],
        'retrieved_documents': [{'content': evidence['evidence_text'], 'metadata': {'source_file': evidence['doc_name']}}]}]}]}
    scored = add_evidence_scores(report)['experiments'][0]
    assert scored['summary']['evidence_recall'] == 1.0 and scored['results'][0]['evidence_mrr'] == 1.0
    slug = benchmarks.load('financebench')
    old = {'id': 'a', 'kind': 'financial'}
    new_generic = {'id': 'b', 'kind': 'experiment', 'collection': slug, 'test_set': 'financebench.jsonl'}
    assert comparable_key(old) == comparable_key(new_generic)


def test_ask_can_start_from_a_saved_test_question(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'The answer to everything is 42. ' * 30)])
    collections.save_test_set(slug, 'mine', [{'question': 'What is the answer?', 'reference_answer': '42',
                                              'keywords': '42', 'evidence': [{'text': 'x', 'source': 'a.txt'}]}])
    at = app()
    at.selectbox(key='ask_pick').set_value(0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.text_area(key='ask_question').value == 'What is the answer?'
    assert at.text_area(key='ask_reference').value == '42'
    assert at.text_input(key='ask_keywords').value == '42'


def test_answer_map_draws_question_answer_and_ranked_passages(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', ''.join(f'Paragraph {i} talks about topic {i}. ' * 6 + '\n\n'
                                                     for i in range(30)).encode())])
    at = app()
    at.text_area(key='ask_question').input('What does paragraph 3 say?')
    at.button(key='FormSubmitter:ask-Ask and evaluate').click().run()
    at.button(key='answer_map_draw').click().run()
    assert not at.exception, [e.value for e in at.exception]
    table = next(d.value for d in at.dataframe if 'Rank' in d.value)
    assert table['Rank'].tolist()[:2] == ['#1', '#2']
    assert {'Similarity to question', 'Similarity to answer'} <= set(table.columns)


def make_runs(data, slug):
    collections.save_test_set(slug, 'qs', [{'question': 'Who?', 'reference_answer': 'Ann'},
                                           {'question': 'When?', 'reference_answer': '2015'}])
    for job_id, chunk, accuracy in (('aaaaaaaaaaaa', 200, 2), ('bbbbbbbbbbbb', 100, 4)):
        rows = [{'question': q, 'generated_answer': 'x', 'judge': {'accuracy': accuracy}, 'retrieved_documents': []}
                for q in ('Who?', 'When?')]
        report = {'experiments': [{'name': 'current settings', 'results': rows, 'failures': [], 'errors': [],
                                   'summary': {'total': 2, 'successful': 2, 'accuracy': accuracy},
                                   'config': {'chunk_size': chunk, 'chunk_overlap': 50, 'top_k': 5},
                                   'models': {'chat': 'm', 'embedding': 'e'}}]}
        directory = data / 'jobs' / job_id
        add_job(data, job_id, kind='experiment', status='completed', collection=slug, collection_name='Papers',
                test_set='qs.jsonl', display_name=f'chunks {chunk}', created_at=f'2026-10-28T1{chunk // 100}:00:00+00:00',
                report=str(directory / 'report.json'), plan=str(directory / 'experiments.yaml'), configurations=1)
        (directory / 'report.json').write_text(json.dumps(report))
        (directory / 'experiments.yaml').write_text(
            f'runs:\n- name: current settings\n  techniques: {{rerank: true}}\n  settings: {{chunk_size: {chunk}, top_k: 5}}\n')


def test_workspace_runs_list_compare_and_delete(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'hello world')])
    make_runs(data, slug)
    at = app()
    runs = next(d.value for d in at.dataframe if 'Run' in d.value and 'Chunk size' in d.value)
    assert runs['Chunk size'].tolist() == ['200', '100'] and runs['Techniques'].tolist() == ['LLM re-ranking'] * 2
    assert runs['Accuracy'].tolist() == [2, 4]
    # Tick both runs and press Compare: the Results page opens with them selected
    ids = [j['id'] for j in jobs.recent_jobs(500, kinds=jobs.RUN_KINDS) if j.get('collection') == slug]
    at.session_state[ui.selection_key(f'runs_{slug}', ids)] = {'selection': {'rows': [0, 1], 'columns': [], 'cells': []}}
    at.run()
    next(b for b in at.button if b.label == 'Compare 2 runs').click().run()
    assert not at.exception, [e.value for e in at.exception]
    scores = next(d.value for d in at.dataframe if 'Accuracy' in getattr(d.value, 'columns', []) and 'Run' not in d.value)
    assert list(scores.index) == ['#2 · chunks 200', '#1 · chunks 100']  # permanent run numbers, oldest first
    jobs.delete_job('aaaaaaaaaaaa')
    assert [j['id'] for j in jobs.recent_jobs()] == ['bbbbbbbbbbbb']


def test_run_numbers_are_never_reused(data):
    first = jobs.record_answer('papers', 'Papers', 'Q?', {'experiments': []}, {'runs': []}, 'one')
    second = jobs.record_answer('papers', 'Papers', 'Q?', {'experiments': []}, {'runs': []}, 'two')
    assert jobs.status(second)['number'] == 2
    jobs.delete_job(second)
    third = jobs.record_answer('papers', 'Papers', 'Q?', {'experiments': []}, {'runs': []}, 'three')
    assert jobs.status(first)['number'] == 1 and jobs.status(third)['number'] == 3


def test_worker_keeps_a_rename_made_while_running(data, monkeypatch):
    from raglab_app import job_worker
    add_job(data, 'dddddddddddd', kind='experiment', status='queued', display_name='old')

    def runner(directory, meta):
        jobs.rename_job('dddddddddddd', 'renamed meanwhile')
    monkeypatch.setitem(job_worker.RUNNERS, 'experiment', runner)
    monkeypatch.setattr(job_worker, 'setup_job_logging', lambda: None)
    job_worker.main(data / 'jobs' / 'dddddddddddd', 'experiment')
    meta = jobs.status('dddddddddddd')
    assert meta['status'] == 'completed' and meta['display_name'] == 'renamed meanwhile'


def test_running_job_cannot_be_deleted(data):
    add_job(data, 'cccccccccccc', kind='experiment', status='running', pid=None)
    with pytest.raises(ValueError, match='in progress'):
        jobs.delete_job('cccccccccccc')


def test_asking_records_comparable_runs(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'The answer to everything is 42. ' * 30)])
    at = app()
    assert any(b.label == 'Compare runs' and b.disabled for b in at.button)  # actions shown even without runs
    for _ in range(2):
        at.text_area(key='ask_question').input('What is the answer?')
        at.text_input(key='ask_keywords').input('42')
        at.button(key='FormSubmitter:ask-Ask and evaluate').click().run()
        assert not at.exception, [e.value for e in at.exception]
    recorded = jobs.recent_jobs(kinds=('ask',))
    assert len(recorded) == 2 and recorded[0]['question'] == 'What is the answer?'
    runs = next(d.value for d in at.dataframe if 'Run' in d.value and 'Questions' in d.value)
    assert runs['Questions'].tolist() == ['Ask: What is the answer?'] * 2
    ids = [j['id'] for j in jobs.recent_jobs(500, kinds=jobs.RUN_KINDS) if j.get('collection') == slug]
    at.session_state[ui.selection_key(f'runs_{slug}', ids)] = {'selection': {'rows': [0, 1], 'columns': [], 'cells': []}}
    at.run()
    next(b for b in at.button if b.label == 'Compare 2 runs').click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('Keyword coverage' in getattr(d.value, 'columns', []) and 'Run' not in d.value for d in at.dataframe)


def test_compare_asked_questions_side_by_side(data):
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'The answer to everything is 42. ' * 30)])
    at = app()
    for question in ('What is the answer?', 'What is everything?'):
        at.text_area(key='ask_question').input(question)
        at.text_input(key='ask_keywords').input('42')
        at.button(key='FormSubmitter:ask-Ask and evaluate').click().run()
    ids = [j['id'] for j in jobs.recent_jobs(500, kinds=jobs.RUN_KINDS) if j.get('collection') == slug]
    at.session_state[ui.selection_key(f'runs_{slug}', ids)] = {'selection': {'rows': [0, 1], 'columns': [], 'cells': []}}
    at.run()
    next(b for b in at.button if b.label == 'Compare 2 runs').click().run()
    assert not at.exception, [e.value for e in at.exception]
    table = next(d.value for d in at.dataframe if getattr(d.value, 'index', None) is not None
                 and d.value.index.name == 'Question')
    assert list(table.index) == ['Q1. What is everything?', 'Q2. What is the answer?']
    assert any('answered different questions' in c.value for c in at.caption)
    legend = next(d.value for d in at.dataframe if list(getattr(d.value, 'columns', [])) == ['ID', 'Question', 'Runs'])
    assert legend.to_dict('records') == [{'ID': 'Q1', 'Question': 'What is everything?', 'Runs': '#2'},
                                         {'ID': 'Q2', 'Question': 'What is the answer?', 'Runs': '#1'}]
    settings = next(d.value for d in at.dataframe if 'Run' in getattr(d.value, 'columns', []) and 'Chunk size' in d.value)
    assert [run.split(' · ')[:2] for run in settings['Run']] == [['#2', 'Q1'], ['#1', 'Q2']]


def test_sidebar_starts_from_settings_yaml(data, tmp_path, monkeypatch):
    from raglab_app import settings
    path = tmp_path / 'settings.yaml'
    path.write_text('defaults:\n  chunk_size: 1200\n  chunk_overlap: 700\n  top_k: 8\n  techniques: [rerank, nope]\n')
    monkeypatch.setattr(settings, 'SETTINGS_FILE', path)
    slug = collections.create('Papers')
    collections.add_files(slug, [('a.txt', b'hello')])
    at = app()
    values = {n.label: n.value for n in at.sidebar.number_input}
    assert values['Chunk size (characters)'] == 1200 and values['Chunk overlap (characters)'] == 700
    assert values['Passages given to the model (top-K)'] == 8
    assert any('unknown techniques ignored: nope' in w.value for w in at.sidebar.warning)
    assert any('More than half of each chunk' in c.value for c in at.sidebar.caption)  # 700 of 1200


def test_orthofinder_example_loads_with_answer_keys(data):
    from raglab_app import benchmarks
    at = app('app_pages/library.py')
    next(b for b in at.button if b.key == 'load_orthofinder').click().run()
    assert not at.exception, [e.value for e in at.exception]
    workspace = benchmarks.loaded('orthofinder')
    assert list(workspace['files']) == ['orthofinder_nature_methods_2026.pdf']
    info = workspace['tests']['orthofinder questions.jsonl']
    assert (info['questions'], info['with_reference'], info['with_evidence']) == (11, 11, 10)
