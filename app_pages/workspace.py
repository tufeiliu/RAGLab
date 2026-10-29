"""The workspace: files, asking, test questions and evaluation, all on one page."""
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from raglab.evaluation import LLMJudge, TestCase, evaluate_result
from raglab.indexes import open_index
from raglab.loaders import SUPPORTED_EXTENSIONS
from raglab_app import INDEXES, collections, finance_form, runs, test_editor
from raglab_app.jobs import delete_job, recent_jobs, record_answer, start_experiment
from raglab_app.models import choices, make_clients, make_config
from raglab_app.options import HELP, INDEX_TIME, QUERY_TIME, TECHNIQUES
from raglab_app.plans import build_plan, parse_ints
from raglab_app.settings import load as load_settings
from raglab_app.report_view import comparable_key, render, render_report, run_entries, single_answer_report
from raglab_app.results_charts import RETRIEVAL_METRICS
from raglab_app.ui import ensure_index, job_progress, pick_workspace, selection_key, when

st.title('Workspace')
st.caption('Upload files, ask questions about them, and measure how well the answers are. '
           'New here? Read the Guide page first.')

workspace = pick_workspace()
if workspace is None:
    st.info('Create a workspace with **New workspace** above to start.', icon=':material/arrow_upward:')
    st.stop()
slug = workspace['slug']


# --- Settings (sidebar): used by both Ask and Evaluate -------------------------------------
chat_models, embedding_models = choices('chat'), choices('embedding')
defaults = load_settings(chat_models, embedding_models)
with st.sidebar:
    st.header('RAG settings')
    st.caption('Used for asking **and** for evaluating, so an evaluation measures exactly what you tried. '
               'They start from settings.yaml in the project folder.')
    for problem in defaults.problems:
        st.warning(f'settings.yaml: {problem}', icon=':material/warning:')
    chat = st.selectbox('Answer model', list(chat_models), index=list(chat_models).index(defaults.answer_model),
                        format_func=chat_models.get, help=HELP['chat'])
    top_k = st.number_input('Passages given to the model (top-K)', *defaults.limits['top_k'], defaults.top_k,
                            help=HELP['top_k'])
    with st.expander('How documents are split'):
        st.caption('Changing these prepares the documents again (automatic; text embedded before is free).')
        embedding = st.selectbox('Embedding model', list(embedding_models),
                                 index=list(embedding_models).index(defaults.embedding_model),
                                 format_func=embedding_models.get, help=HELP['embedding'])
        chunk_size = st.number_input('Chunk size (characters)', *defaults.limits['chunk_size'], defaults.chunk_size,
                                     step=50, help=HELP['chunk_size'])
        chunk_overlap = st.number_input('Chunk overlap (characters)', 0, chunk_size - 1,
                                        min(defaults.chunk_overlap, chunk_size - 1), step=25,
                                        help=HELP['chunk_overlap'] + ' Must be smaller than the chunk size.')
        if chunk_overlap > chunk_size // 2:
            st.caption(':material/warning: More than half of each chunk repeats its neighbour, so the top passages '
                       'will largely overlap. Usually 10-25% of the chunk size is enough.')
        index_extras = st.pills('LLM-written index content', sorted(INDEX_TIME), selection_mode='multi',
                                default=[t for t in defaults.techniques if t in INDEX_TIME],
                                format_func=lambda k: TECHNIQUES[k][0],
                                help='\n\n'.join(f'**{TECHNIQUES[k][0]}**: {TECHNIQUES[k][1]}' for k in sorted(INDEX_TIME)))
    with st.expander('Techniques'):
        st.caption('Optional extra steps that can improve answers. Each adds model calls. Hover for details.')
        techniques = st.pills('Use', QUERY_TIME, selection_mode='multi', label_visibility='collapsed',
                              default=[t for t in defaults.techniques if t in QUERY_TIME],
                              format_func=lambda k: TECHNIQUES[k][0],
                              help='\n\n'.join(f'**{TECHNIQUES[k][0]}**: {TECHNIQUES[k][1]}' for k in QUERY_TIME))
    cache_responses = st.toggle('Reuse cached answers', value=defaults.reuse_cached_answers, help=HELP['cache_responses'])

index_settings = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **{k: True for k in index_extras})
query_settings = {'top_k': top_k}
for key in techniques:
    query_settings.update(TECHNIQUES[key][2])


# --- 1. Files -------------------------------------------------------------------------------------
def add_uploads(key: str) -> None:
    """Store files as soon as they are picked, then clear the uploader for the next batch."""
    uploads = st.session_state.get(key) or []
    if uploads:
        st.session_state['upload_notice'] = collections.add_files(slug, [(f.name, f.getvalue()) for f in uploads])
    st.session_state['uploader_version'] = st.session_state.get('uploader_version', 0) + 1


files = workspace['files']
st.header('1. Files', divider='gray')
if files:
    names = sorted(files)
    st.markdown(f'Answers are searched for in **{len(names)} file(s)**: ' + ', '.join(f'`{n}`' for n in names[:10])
                + (f' and {len(names) - 10} more' if len(names) > 10 else ''))
with st.expander('Add or remove files', icon=':material/upload_file:', expanded=not files):
    upload_key = f"uploader_{slug}_{st.session_state.get('uploader_version', 0)}"
    st.file_uploader('Add files (saved as soon as you pick them)', type=[x.lstrip('.') for x in SUPPORTED_EXTENSIONS],
                     accept_multiple_files=True, key=upload_key, on_change=add_uploads, args=(upload_key,),
                     help='A file already in this workspace is skipped, even under another name. Scanned PDFs need OCR first.')
    if files:
        file_table = pd.DataFrame([{'File': n, 'Size (KB)': round(i['bytes'] / 1024, 1), 'Added': when(i['added_at'])}
                                   for n, i in sorted(files.items())])
        picked = st.dataframe(file_table, hide_index=True, on_select='rerun', selection_mode='multi-row',
                              key=selection_key(f'files_{slug}', file_table['File']))
        chosen = [file_table.iloc[i]['File'] for i in picked.selection.rows]
        if chosen and st.button(f'Remove {len(chosen)} selected file(s)', icon=':material/delete:'):
            for name in chosen:
                collections.remove_file(slug, name)
            st.rerun()
if 'upload_notice' in st.session_state:
    added, skipped = st.session_state.pop('upload_notice')
    if added:
        st.success(f"Added {', '.join(added)}.", icon=':material/check:')
    for item in skipped:
        st.info(f'Already in this workspace, skipped: {item}', icon=':material/content_copy:')
if not files:
    st.stop()


@st.cache_resource(max_entries=6, show_spinner=False)
def get_lab(identity: tuple, chat: str, embedding: str, settings: tuple, cache_responses: bool):
    """One open index per settings combination; reused across reruns and sessions."""
    cfg = make_config(chat, embedding, **dict(settings))
    return open_index(cfg, INDEXES, list(identity), **make_clients(chat, embedding, cache_responses=cache_responses))


def current_lab():
    ensure_index(workspace, chat, embedding, **index_settings)
    return get_lab(tuple(collections.file_identity(slug)), chat, embedding,
                   tuple(sorted({**index_settings, **query_settings}.items())), cache_responses)


def answer_map(saved: dict, method: str) -> tuple:
    """Question, answer and correct answer projected with all chunks; returns (figure, similarity rows)."""
    import numpy as np
    from langchain_core.documents import Document
    from raglab.core import RAGLab
    from raglab.visualization import map_chroma
    lab = current_lab()
    docs = saved['result']['documents']
    ids = [RAGLab.chunk_id(Document(page_content=d['content'], metadata=d.get('metadata') or {})) for d in docs]
    texts = {'Question': saved['result']['question'], 'Answer': saved['result']['answer'],
             'Correct answer': saved['test'].get('reference_answer', '')}
    texts = {label: text for label, text in texts.items() if text and text.strip()}
    with st.spinner('Embedding the question and answers, and projecting all chunks…'):
        projected = map_chroma(lab.store, method=method, selected_ids=ids, extra_points=texts,
                               embedding_function=lab.embedder, max_points=1500)
    figure = projected.figure(color_by='source', title=f'{method.upper()} projection of all chunks')
    stored = lab.store._collection.get(ids=ids, include=['embeddings'])
    by_id = dict(zip(stored['ids'], stored['embeddings']))
    vectors = {label: np.asarray(lab.embedder.embed_query(text)) for label, text in texts.items()}

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rows = []
    for rank, (doc_id, doc) in enumerate(zip(ids, docs), 1):
        if doc_id not in by_id:
            continue
        chunk = np.asarray(by_id[doc_id])
        meta = doc.get('metadata') or {}
        rows.append({'Rank': f'#{rank}', 'Passage': ' '.join(doc['content'].split())[:90],
                     'Page': meta.get('page'),
                     **{f'Similarity to {label.lower()}': round(cosine(vectors[label], chunk), 3) for label in texts}})
    return figure, rows


def record_ask_run(test, result, scored: dict) -> None:
    """Keep the answer as a run, with its settings, so it shows in the Runs list and can be compared."""
    used = [*index_extras, *techniques]
    flags = {}
    for key in used:
        flags.update(TECHNIQUES[key][2])
    report = single_answer_report(
        scored, config={**index_settings, **query_settings},
        models={'chat': chat, 'embedding': embedding, 'judge': chat if scored.get('judge') else None})
    report['experiments'][0]['results'][0]['category'] = 'asked'
    report['collection'] = {'name': workspace['name'], 'files': sorted(files), 'question': test.question}
    plan = {'runs': [{'name': 'this answer', 'techniques': flags,
                      'settings': {'embedding': embedding, 'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap,
                                   'top_k': top_k}}]}
    record_answer(slug, workspace['name'], test.question, report, plan,
                  f"Ask: chunks {chunk_size}, top-K {top_k}, "
                  f"{', '.join(TECHNIQUES[k][0] for k in used) or 'no technique'}")


# --- 2. Ask or test -------------------------------------------------------------------------------------
st.header('2. Ask or test', divider='gray')
ask_tab, test_tab = st.tabs(
    [':material/chat: Ask a question', ':material/checklist: Test questions'], key='workspace_tab', on_change='rerun')
tests = workspace['tests']
evaluations = recent_jobs(500, kinds=('experiment', 'financial', 'ask'))
mine = [j for j in evaluations if j.get('collection') == slug]


def pick_test_set(key: str):
    """Test-set selector plus a note on what its answer key lets RAGLab score."""
    if not tests:
        st.info('No test questions yet. Save them from the Ask tab, or write them in the Test questions tab.')
        return None, None
    name = st.selectbox('Test questions', sorted(tests), key=key,
                        format_func=lambda t: f"{t.removesuffix('.jsonl')} ({tests[t]['questions']} questions)")
    info = tests[name]
    total = info['questions']
    scored = [(what, count) for what, count in (('answers and search (judged)', info['with_reference']),
              ('search, precisely', info.get('with_evidence', 0)), ('search, roughly', info['with_keywords'])) if count]
    if scored:
        st.caption('Will score: ' + ' · '.join(f'{what} ({count}/{total} questions)' for what, count in scored))
    else:
        st.warning('These questions have no answer key (correct answer, passage or keywords), so nothing can be scored.')
    return name, info


def start_run(test_set: str, info: dict, plan: dict, default_name: str, key: str) -> None:
    runs = len(plan['runs'])
    answers = runs * info['questions']
    judged = plan['settings'].get('judge_enabled')
    st.caption(f"{runs} setup(s) · {answers:,} answers{f' + {answers:,} judge scores' if judged else ''} "
               '(techniques add calls per answer). Runs in the background; you can keep working.')
    c1, c2 = st.columns([3, 1], vertical_alignment='bottom')
    name = c1.text_input('Name for this run', value=default_name, max_chars=100, key=f'{key}_name')
    if c2.button('Run', type='primary', icon=':material/play_arrow:', width='stretch', key=f'{key}_run'):
        start_experiment(slug, workspace['name'], test_set, plan, name, cache_responses)
        st.rerun()


def show_latest(jobs: list[dict], title: str) -> None:
    if not jobs:
        return
    job = jobs[0]
    st.subheader(f"{title}: {job.get('display_name') or 'Evaluation'}")
    st.caption(f"Started {when(job['created_at'])}. Earlier runs are on the Results page.")
    if job.get('status') in ('queued', 'running'):
        @st.fragment(run_every='3s')
        def progress(job_id):
            if job_progress(job_id) not in ('queued', 'running'):
                st.rerun(scope='app')
        progress(job['id'])
    elif job.get('status') == 'failed':
        st.error(job.get('error') or 'Failed; see its log on the Results page.')
    elif job.get('report') and Path(job['report']).is_file():
        render(job, evaluations)


def judge_toggle(info: dict, key: str) -> bool:
    return st.toggle('Score answers with the LLM judge', value=defaults.judge and bool(info['with_reference']),
                     disabled=not info['with_reference'], help=HELP['judge'], key=key)


# --- Ask a question ---
if ask_tab.open:
    with ask_tab:
        st.caption('Ask one question. You get the answer and the chunks it was based on, then, if you say what the '
                   'correct answer is, its full evaluation.')
        saved_questions = [(name, i, case) for name in sorted(workspace['tests'])
                           for i, case in enumerate(collections.read_test_set(slug, name))]


        def use_test_question():
            """Fill the form from a saved test question (runs before the form is drawn)."""
            choice = st.session_state.get('ask_pick')
            if choice is None:
                return
            case = saved_questions[choice][2]
            st.session_state.update(
                ask_question=case['question'], ask_reference=case['reference_answer'],
                ask_keywords=', '.join(case['keywords']),
                ask_evidence=collections.EVIDENCE_SEPARATOR.join(e['text'] for e in case['evidence']),
                ask_evidence_original=case['evidence'])


        if saved_questions:
            st.selectbox('Start from a saved test question (optional)', range(len(saved_questions)), index=None,
                         key='ask_pick', on_change=use_test_question,
                         placeholder=f'{len(saved_questions)} saved: pick one, or type a new question below',
                         format_func=lambda i: saved_questions[i][2]['question'][:120])

        with st.form('ask'):
            question = st.text_area('Your question', height=80, key='ask_question',
                                    placeholder='e.g. Who are the authors of this paper?')
            with st.expander('Score the answer (optional): say what the correct answer is', icon=':material/rule:',
                             expanded=bool(st.session_state.get('ask_reference') or st.session_state.get('ask_evidence'))):
                left, middle, right = st.columns(3)
                reference = left.text_area('Correct answer', height=100, key='ask_reference',
                                           help='Scores the answer with the LLM judge (one extra model call).')
                evidence = middle.text_area('Passage that contains the answer', height=100, key='ask_evidence',
                                            placeholder='Copied from the document',
                                            help='Scores the search precisely: was this passage retrieved, and how high?')
                keywords = right.text_input('Keywords the right passage contains', key='ask_keywords',
                                            placeholder='e.g. Emms, Liu', help='Comma separated. A quick, rough search check.')
            asked = st.form_submit_button('Ask and evaluate', type='primary', icon=':material/send:',
                                          help='Returns the answer and the chunks it used, then scores it if you gave an answer key.')


        def evidence_for(text: str) -> list:
            """The saved passages (with their source files) if the text is unchanged, else what was typed."""
            original = st.session_state.get('ask_evidence_original') or []
            if original and collections.EVIDENCE_SEPARATOR.join(e['text'] for e in original) == text:
                return original
            return [text.strip()] if text.strip() else []


        if asked:
            if not question.strip():
                st.warning('Type a question first.')
            else:
                try:
                    lab = current_lab()
                    test = TestCase(question=question.strip(), reference_answer=reference.strip(),
                                    keywords=[k.strip() for k in keywords.split(',') if k.strip()],
                                    evidence=evidence_for(evidence))
                    with st.spinner('Searching the files and writing the answer…'):
                        started = time.perf_counter()
                        result = lab.ask(test.question)
                        seconds = time.perf_counter() - started
                    with st.spinner('Scoring the answer…'):
                        scored = evaluate_result(result, test, LLMJudge(llm=lab.llm) if test.reference_answer else None, top_k)
                    scored['latency_s'] = round(seconds, 3)
                    record_ask_run(test, result, scored)
                    st.session_state['last_answer'] = {'result': result.to_dict(), 'scores': scored, 'seconds': seconds,
                                                       'techniques': list(techniques), 'test': vars(test), 'workspace': slug,
                                                       'asked_at': time.time()}
                except Exception as exc:
                    st.error(f'The question failed: {exc}')

        saved = st.session_state.get('last_answer')
        if saved and saved.get('workspace') == slug:
            result, scores = saved['result'], saved['scores']
            answer_col, sources_col = st.columns([3, 2], gap='large')
            with answer_col:
                st.markdown('**Answer**')
                with st.container(border=True):
                    st.markdown(result['answer'])
                st.caption(f"{saved['seconds']:.1f}s · {len(result['documents'])} passages · techniques: "
                           f"{', '.join(TECHNIQUES[t][0] for t in saved['techniques']) or 'none'}")
                with st.container(horizontal=True):
                    with st.popover('Save as a test question', icon=':material/playlist_add:',
                                    help='Keep this question and its answer key, to run it again in the Test questions tab.'):
                        sets = sorted(workspace['tests'])
                        NEW = '+ New test set'
                        target = st.selectbox('Test set', [*sets, NEW])
                        name = st.text_input('Name', value='my questions') if target == NEW else target
                        if st.button('Save question', type='primary'):
                            try:
                                if target == NEW and collections.test_set_name(name) in sets:
                                    raise ValueError(f'{collections.test_set_name(name)} already exists; choose it in the list instead.')
                                existing = collections.read_test_set(slug, target) if target != NEW else []
                                stored = collections.save_test_set(slug, name, existing + [saved['test']])
                                st.toast(f'Saved to {stored}', icon=':material/check:')
                            except ValueError as exc:
                                st.error(str(exc))
                    st.download_button('Download (JSON)', json.dumps(saved, indent=2, default=str), 'rag_answer.json',
                                       icon=':material/download:')
            with sources_col:
                st.markdown('**Sources**: the passages the model was given, best match first')
                for i, doc in enumerate(result['documents'], 1):
                    meta = doc.get('metadata') or {}
                    where = ' · '.join(str(x) for x in (meta.get('source', 'unknown'),
                                       f"page {meta['page']}" if meta.get('page') is not None else None) if x)
                    with st.expander(f'#{i} · {where}', expanded=i == 1):
                        st.text(doc.get('content', ''))
                st.caption('Wrong answer and the right information is not among these? Then the search is the problem: '
                           'try bigger chunks, more passages, or the query techniques in the sidebar.')
                with st.expander('How the answer was produced', icon=':material/route:'):
                    st.markdown('\n'.join(f'- Search query: {q}' for q in result['queries']))
                    st.markdown('\n'.join(f'- {step}' for step in result['trace']))

            st.subheader('Evaluation of this answer')
            if not scores.get('judge') and all(scores.get(k) is None for k in RETRIEVAL_METRICS):
                st.info('Not evaluated: open "Score the answer" above, give the correct answer (and, if you have them, the '
                        'passage or keywords), and ask again.', icon=':material/info:')
            else:
                render_report(single_answer_report(
                    scores, config={**index_settings, **query_settings},
                    models={'chat': chat, 'embedding': embedding, 'judge': chat if scores.get('judge') else None}),
                    f"ask_{saved.get('asked_at')}", show_questions=False, note_sample=False)

            with st.expander('Show the question, answer and passages on a map (UMAP / t-SNE)', icon=':material/scatter_plot:'):
                st.caption('Every chunk of your files is a grey point, placed so that chunks with similar meaning sit close '
                           'together. Red diamonds are the passages the model was given, numbered by rank. The star is your '
                           'question, the square the answer and the triangle your correct answer. If the correct answer sits '
                           'far from the numbered passages, the search looked in the wrong place. Positions are approximate; '
                           'the table gives the exact similarities. Uses the current settings in the sidebar.')
                c1, c2 = st.columns([1, 3], vertical_alignment='bottom')
                method = c1.segmented_control('Projection', ['umap', 'tsne'], default='umap', key='answer_map_method',
                                              format_func={'umap': 'UMAP', 'tsne': 't-SNE'}.get) or 'umap'
                if c2.button('Draw map', icon=':material/scatter_plot:', key='answer_map_draw'):
                    try:
                        st.session_state['answer_map'] = (saved.get('asked_at'), method, *answer_map(saved, method))
                    except Exception as exc:
                        st.error(f'Could not draw the map: {exc}')
                drawn = st.session_state.get('answer_map')
                if drawn and drawn[0] == saved.get('asked_at'):  # only for the answer it was drawn for
                    st.plotly_chart(drawn[2], key='answer_map_figure')
                    st.markdown('**Exact similarity of each passage** (cosine, in the original embedding space; '
                                'higher = closer in meaning)')
                    st.dataframe(pd.DataFrame(drawn[3]), hide_index=True)
                    st.caption('The search orders passages by similarity to the question. A passage much more similar to '
                               'the correct answer than the others is a good one; if none is, the information the correct '
                               'answer needs was not retrieved.')


# --- Test questions ---
if test_tab.open:
    with test_tab:
        st.caption('Run a whole set of questions with known answers through RAG with the **current settings** '
                   '(sidebar). Each run is kept: change a setting (say the chunk size, or a technique) and run again, '
                   'then compare the runs in the Compare tab. Each question can have a **correct answer** (scores '
                   'the answer, and judges the search), the **passage** that contains it (scores the search '
                   'precisely) and/or **keywords** (a quick search check). Twenty or more give a fair picture.')
        with st.expander('Write, edit or upload test questions', icon=':material/edit_note:', expanded=not tests):
            test_editor.render(workspace)
        test_set, info = pick_test_set('test_set_single')
        if test_set:
            finance = workspace.get('benchmark') == 'financebench' and st.segmented_control(
                'Retrieval pipeline', ['generic', 'finance'], default='generic', key='eval_pipeline',
                format_func={'generic': 'Generic RAG (sidebar settings)', 'finance': 'Finance-tuned'}.get,
                help="Finance-tuned: FinanceBench's own retrieval, which keeps tables intact, searches only the "
                     'filing the question names, and mixes keyword (BM25) and vector search.') == 'finance'
            if finance:
                finance_form.render(workspace, test_set)
            else:
                judge_on = judge_toggle(info, 'judge_single')
                current = [*index_extras, *techniques]
                st.caption(f"Settings of this run: answer model {chat_models[chat]}, top-K {top_k}, chunks of "
                           f"{chunk_size} characters (overlap {chunk_overlap}), techniques: "
                           f"{', '.join(TECHNIQUES[k][0] for k in current) or 'none'}. Change them in the sidebar.")
                plan = build_plan(chat=chat, judge=chat if judge_on else None, embeddings=[embedding],
                                  chunk_sizes=[chunk_size], chunk_overlap=chunk_overlap, top_ks=[top_k],
                                  techniques=current, combine=True, baseline=False)
                plan['runs'][0]['name'] = 'current settings'
                start_run(test_set, info, plan, f"chunks {chunk_size}, top-K {top_k}, "
                          f"{', '.join(TECHNIQUES[k][0] for k in current) or 'no technique'}", 'single')
        with st.expander('Shortcut: run several setups in one go', icon=':material/bolt:'):
            st.caption('Instead of changing the sidebar and running again each time, run a batch: each technique on '
                       'its own (or all together) against a baseline, optionally with several chunk sizes, top-K '
                       'values or embedding models. Each setup appears in the Runs list when it finishes.')
            if test_set:
                judge_on = judge_toggle(info, 'judge_compare')
                run_techniques = st.pills('Techniques', list(TECHNIQUES), selection_mode='multi',
                                          default=['rerank', 'opening'], format_func=lambda k: TECHNIQUES[k][0],
                                          help='\n\n'.join(f'**{label}**: {text}' for label, text, _ in TECHNIQUES.values()))
                c1, c2 = st.columns(2)
                combine = c1.segmented_control('Test the techniques', [False, True], default=False,
                                               format_func={False: 'Each on its own', True: 'All together'}.get) or False
                baseline = c2.checkbox('Also run without any technique (baseline)', value=True)
                c1, c2, c3 = st.columns(3)
                chunk_text = c1.text_input('Chunk sizes', str(chunk_size), help=HELP['chunk_size'] + ' Comma separated.')
                overlap = c2.number_input('Chunk overlap', 0, defaults.limits['chunk_size'][1] - 1, chunk_overlap, 25,
                                          help=HELP['chunk_overlap'])
                top_k_text = c3.text_input('Top-K values', str(top_k), help=HELP['top_k'] + ' Comma separated.')
                try:
                    plan = build_plan(chat=chat, judge=chat if judge_on else None, embeddings=[embedding],
                                      chunk_sizes=parse_ints(chunk_text, label='Chunk sizes',
                                                             low=defaults.limits['chunk_size'][0],
                                                             high=defaults.limits['chunk_size'][1]),
                                      chunk_overlap=int(overlap),
                                      top_ks=parse_ints(top_k_text, label='Top-K values', low=defaults.limits['top_k'][0],
                                                        high=defaults.limits['top_k'][1]),
                                      techniques=run_techniques, combine=bool(combine), baseline=baseline)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.dataframe(pd.DataFrame([{'Setup': r['name'], 'Chunk size': r['settings']['chunk_size'],
                                                'Top-K': r['settings']['top_k']} for r in plan['runs']]), hide_index=True)
                    start_run(test_set, info, plan, f"batch of {len(plan['runs'])} setups", 'compare')
        show_latest([j for j in mine if j['kind'] != 'ask'], 'Latest run')

# --- 3. Runs -----------------------------------------------------------------------------------------------
st.header('3. Runs', divider='gray')
st.caption('Everything run in this workspace. Tick runs to compare them, open one, or delete the ones you no '
           'longer need. Full results and comparisons are on the Results page.')
# Read again: a question asked above in this same rerun has just been recorded.
mine = [j for j in recent_jobs(500, kinds=('experiment', 'financial', 'ask')) if j.get('collection') == slug]
if not mine:
    st.caption('No runs yet. Every question you ask, and every run of test questions, is listed here.')
    chosen = []
else:
    run_rows = []
    for job in mine:
        summaries = None
        if job.get('status') == 'completed' and job.get('report') and Path(job['report']).is_file():
            summaries = [e['experiment'].get('summary', {}) for e in run_entries(job)]
        run_rows.append(runs.overview(job, summaries))
    picked = st.dataframe(pd.DataFrame(run_rows).dropna(axis=1, how='all'), hide_index=True,
                          on_select='rerun', selection_mode='multi-row',
                          key=selection_key(f'runs_{slug}', [j['id'] for j in mine]))
    chosen = [mine[i] for i in picked.selection.rows]
finished = [j for j in chosen if j.get('status') == 'completed']
same_questions = len({comparable_key(j) for j in finished}) == 1
with st.container(horizontal=True):
    if st.button(f'Compare {len(finished)} runs' if finished else 'Compare runs', icon=':material/compare_arrows:',
                 disabled=len(finished) < 2 or not same_questions,
                 help='Tick two or more finished runs: questions you asked, or runs of the same test set. Opens '
                      'the comparison on the Results page with them selected.'):
        st.session_state['compare_runs'] = [j['id'] for j in finished]
        st.session_state['results_goto'] = 'compare'
        st.switch_page('app_pages/results.py')
    if st.button('Open', icon=':material/open_in_new:', disabled=len(chosen) != 1,
                 help='Shows the ticked run in full on the Results page.'):
        st.session_state['selected_experiment'] = chosen[0]['id']
        st.session_state['results_goto'] = 'runs'
        st.switch_page('app_pages/results.py')
    with st.popover(f'Delete {len(chosen)} runs' if chosen else 'Delete', icon=':material/delete:', disabled=not chosen):
        st.write('Deletes the ticked runs, their results and logs. This cannot be undone.')
        if st.button('Delete', type='primary', key='confirm_delete_runs'):
            problems = []
            for job in chosen:
                try:
                    delete_job(job['id'])
                except (OSError, ValueError) as exc:
                    problems.append(f"{job.get('display_name') or job['id']}: {exc}")
            for problem in problems:
                st.error(problem)
            if not problems:
                st.rerun()
if len(finished) >= 2 and not same_questions:
    st.warning('Questions you asked and runs of a test set (or of two different test sets) cannot be compared '
               'with each other. Tick only asked questions, or only runs of one test set.')
elif mine and not chosen:
    st.caption('Tick runs to compare, open or delete them. Asking the same question again with other settings '
               'gives runs you can compare.')

