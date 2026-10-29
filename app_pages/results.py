"""Results: every run in full, comparisons between runs, and the saved test questions."""
from pathlib import Path

import pandas as pd
import streamlit as st

from raglab_app import collections, library, runs
from raglab_app.jobs import delete_job, read_log, recent_jobs, rename_job
from raglab_app.report_view import comparable_key, compare_runs, render, run_entries
from raglab_app.ui import job_progress, page_intro, selection_key

KINDS = ('experiment', 'financial', 'ask')
TABS = {'runs': ':material/list: Runs', 'compare': ':material/compare_arrows: Compare runs',
        'tests': ':material/checklist: Test questions'}

page_intro('Results', 'Everything you have run. **Runs** shows one run in full; **Compare runs** puts several side '
           'by side to see which settings work best; **Test questions** lists the questions you have saved.')

if 'results_goto' in st.session_state:  # set by the Workspace's Compare / Open buttons
    st.session_state['results_tab'] = TABS[st.session_state.pop('results_goto')]
    # A run asked for by name wins over a row clicked earlier: start the table without a selection.
    st.session_state['run_table_version'] = st.session_state.get('run_table_version', 0) + 1
runs_tab, compare_tab, tests_tab = st.tabs(list(TABS.values()), key='results_tab', on_change='rerun')
jobs = recent_jobs(500, kinds=KINDS)


def name(job: dict) -> str:
    return job.get('display_name') or ('FinanceBench run' if job['kind'] == 'financial' else 'Run')


def finished(job: dict) -> bool:
    return job.get('status') == 'completed' and bool(job.get('report')) and Path(job['report']).is_file()


# --- Runs ---------------------------------------------------------------------------------------------------
if runs_tab.open:
    with runs_tab:
        if not jobs:
            st.info('No runs yet. Run your test questions on the Workspace page.')
            st.page_link('app_pages/workspace.py', label='Workspace', icon=':material/work:')
        else:
            table = pd.DataFrame([{'Workspace': j.get('collection_name') or 'FinanceBench',
                                   **runs.overview(j, [e['experiment'].get('summary', {}) for e in run_entries(j)]
                                                   if finished(j) else None)} for j in jobs])
            picked = st.dataframe(table.dropna(axis=1, how='all'), hide_index=True, on_select='rerun',
                                  selection_mode='single-row',
                                  key=selection_key(f"experiment_table_{st.session_state.get('run_table_version', 0)}",
                                                    [j['id'] for j in jobs]))
            if picked.selection.rows:
                job = jobs[picked.selection.rows[0]]
            else:
                preferred = st.session_state.get('selected_experiment')
                job = next((j for j in jobs if j['id'] == preferred), jobs[0])
                st.caption('Showing the latest run. Select a row to open another.')
            job_id = job['id']

            header, actions = st.columns([3, 2], vertical_alignment='bottom')
            header.subheader(name(job))
            header.caption(' · '.join(filter(None, [
                f"started {job['created_at'][:16].replace('T', ' ')}",
                f"took {runs.duration(job)}" if job.get('finished_at') else None,
                'cached answers reused' if job.get('cache_responses') else None])))
            with actions.container(horizontal=True):
                with st.popover('Rename', icon=':material/edit:'):
                    new_name = st.text_input('Name', value=name(job), max_chars=100, key=f'rename_{job_id}')
                    if st.button('Save name', key=f'save_name_{job_id}', type='primary'):
                        try:
                            rename_job(job_id, new_name)
                            st.rerun()
                        except (OSError, ValueError) as exc:
                            st.error(str(exc))
                with st.popover('Delete', icon=':material/delete:'):
                    st.write('Deletes this run, its results and log. This cannot be undone.')
                    if st.button('Delete run', type='primary', key=f'delete_{job_id}'):
                        try:
                            delete_job(job_id)
                            st.session_state.pop('selected_experiment', None)
                            st.rerun()
                        except (OSError, ValueError) as exc:
                            st.error(str(exc))

            @st.fragment(run_every='3s')
            def progress_panel(job_id: str) -> None:
                if job_progress(job_id) not in ('queued', 'running'):
                    st.rerun(scope='app')

            report = Path(job['report']) if job.get('report') else None
            if job.get('status') in ('queued', 'running'):
                st.caption('The first run on new files also prepares them for search. This updates by itself.')
                progress_panel(job_id)
            elif job.get('status') == 'failed':
                st.error(job.get('error') or 'See the log below.', title='This run failed')
            elif not finished(job):
                st.warning('This run finished but its results file is missing.')
            else:
                render(job, jobs)
                with st.popover('Save to the library', icon=':material/bookmark_add:'):
                    st.caption('Keep this run in the Library, with a note on what it shows.')
                    title = st.text_input('Title', value=name(job), key=f'lib_title_{job_id}')
                    description = st.text_area('What was tested', key=f'lib_desc_{job_id}')
                    lesson = st.text_area('What to look at', key=f'lib_look_{job_id}')
                    if st.button('Save', type='primary', key=f'lib_save_{job_id}'):
                        try:
                            library.save(job, title, description, lesson)
                            st.toast('Saved to the library', icon=':material/check:')
                        except (OSError, ValueError) as exc:
                            st.error(str(exc))

            with st.expander('Log', icon=':material/receipt_long:'):
                st.caption('Timestamped record of the run: documents prepared or reused, cache hits, per-setup '
                           'results, errors.')
                log_text = read_log(job_id)
                st.code(log_text[-20000:] or 'No log output yet', language='text', height=300)
                with st.container(horizontal=True):
                    st.download_button('Download log', log_text, f'{job_id}.log', icon=':material/download:',
                                       key=f'log_{job_id}')
                    if report and report.is_file():
                        st.download_button('Download results (JSON)', report.read_bytes(), f'{job_id}_report.json',
                                           mime='application/json', icon=':material/download:', key=f'raw_{job_id}')

# --- Compare runs -------------------------------------------------------------------------------------------
if compare_tab.open:
    with compare_tab:
        st.caption('Pick runs on the same documents and test questions, e.g. run 1 with chunk size 200, run 2 '
                   'with 100, run 3 with 200 plus re-ranking, and see side by side which works best.')
        groups: dict[tuple, list[dict]] = {}
        for job in jobs:
            if finished(job) and comparable_key(job):
                groups.setdefault(comparable_key(job), []).append(job)
        groups = {k: v for k, v in groups.items() if sum(len(run_entries(j)) for j in v) >= 2}
        if not groups:
            st.info('Nothing to compare yet: you need two or more finished runs on the same test questions.')
        else:
            def group_label(key):
                job = groups[key][0]
                questions = ('questions asked' if job['kind'] == 'ask'
                             else (job.get('test_set') or 'financebench.jsonl').removesuffix('.jsonl'))
                return f"{job.get('collection_name') or 'FinanceBench'} · {questions} ({len(groups[key])} runs)"
            preselect = st.session_state.pop('compare_runs', None)
            keys = list(groups)
            start = next((i for i, k in enumerate(keys) if preselect and any(j['id'] in preselect for j in groups[k])), 0)
            key = st.selectbox('Documents and test questions', keys, index=start, format_func=group_label)
            entries = [e for job in groups[key] for e in run_entries(job)]
            ids = [f"{e['job']['id']}:{e['experiment']['name']}" for e in entries]
            def picker_label(e):
                question = f" · \u201c{e['job']['question'][:50]}\u201d" if e['job']['kind'] == 'ask' else ''
                return f"{e['label']}{question} ({e['job']['created_at'][:16].replace('T', ' ')})"
            labels = dict(zip(ids, map(picker_label, entries)))
            select_key = f'compare_pick_{"_".join(map(str, key))}'
            if preselect:
                st.session_state[select_key] = [i for i in ids if i.split(':')[0] in preselect]
            chosen = st.multiselect('Runs to compare', ids, format_func=labels.get, key=select_key,
                                    placeholder='Choose two or more runs')
            if len(chosen) < 2:
                st.caption('Choose two or more runs. Tip: tick runs in the Workspace\'s Runs list and press Compare.')
            else:
                compare_runs([entries[ids.index(i)] for i in chosen], f'results_{select_key}')

# --- Test questions -----------------------------------------------------------------------------------------
if tests_tab.open:
    with tests_tab:
        sets = [(w, set_name, info) for w in collections.list_collections() for set_name, info in sorted(w['tests'].items())]
        if not sets:
            st.caption('None yet. Save questions from Ask on the Workspace page, or write them in its Test questions tab.')
        else:
            test_table = pd.DataFrame([{'Workspace': w['name'], 'Test set': set_name.removesuffix('.jsonl'),
                                        'Questions': info['questions'], 'With correct answer': info['with_reference'],
                                        'With passage': info.get('with_evidence', 0), 'With keywords': info['with_keywords']}
                                       for w, set_name, info in sets])
            picked_set = st.dataframe(test_table, hide_index=True, on_select='rerun', selection_mode='single-row',
                                      key=selection_key('test_sets', [f"{w['slug']}/{n}" for w, n, _ in sets]))
            if picked_set.selection.rows:
                w, set_name, _ = sets[picked_set.selection.rows[0]]
                st.dataframe(pd.DataFrame([{'Question': c['question'], 'Correct answer': c['reference_answer'],
                                            'Keywords': ', '.join(c['keywords']), 'Passages': len(c['evidence'])}
                                           for c in collections.read_test_set(w['slug'], set_name)]), hide_index=True)
                st.caption('Edit them on the Workspace page, in the Test questions tab.')
