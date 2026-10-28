"""Library: example evaluation results to learn from, plus results you saved, and benchmarks to run."""
import streamlit as st

from raglab_app import benchmarks, library
from raglab_app.jobs import recent_jobs, start_prepare
from raglab_app.report_view import load, render
from raglab_app.results_charts import ANSWER_METRICS, RETRIEVAL_METRICS, display
from raglab_app.ui import job_progress, page_intro

page_intro('Library', 'Finished evaluations to learn from: what good and poor results look like, and how settings '
           'and techniques change them. Open one to explore it exactly like your own results. Save your own '
           'evaluations here from the Results page.')

entries = library.entries()
if not entries:
    st.info('The library is empty. Save an evaluation from the Results page.')
else:
    for entry in entries:
        with st.container(border=True):
            st.markdown(f"#### {entry['title']}")
            st.caption(('Example' if entry['origin'] == 'example' else 'Saved by you') +
                       (f" · {entry['source']}" if entry.get('source') else ''))
            st.markdown(entry.get('description', ''))
            job = library.as_job(entry)
            report = load(job['id'], job['report'], job['kind'], 0.0)
            with st.container(horizontal=True):
                for experiment in report['experiments'][:4]:
                    s = experiment['summary']
                    search = next((k for k in ('evidence_recall', 'keyword_coverage') if s.get(k) is not None), None)
                    shown = [(RETRIEVAL_METRICS[k][0], display(k, s[k])) for k in [search] if k]
                    if s.get('accuracy') is not None:
                        shown.append((ANSWER_METRICS['accuracy'][0], display('accuracy', s['accuracy'])))
                    with st.container(border=True):
                        st.markdown(f"**{experiment['name']}**")
                        st.caption(' · '.join(f'{label} {value}' for label, value in shown) or 'no scores')
            if entry.get('what_to_look_at'):
                st.info(entry['what_to_look_at'], title='What to look at', icon=':material/visibility:')
            with st.container(horizontal=True):
                if st.button('Open', key=f"open_{entry['id']}", icon=':material/open_in_full:'):
                    st.session_state['library_open'] = entry['id']
                if entry['origin'] == 'saved' and st.button('Remove from library', key=f"rm_{entry['id']}"):
                    library.remove(entry)
                    st.rerun()

    opened = next((e for e in entries if e['id'] == st.session_state.get('library_open')), None)
    if opened:
        st.header(opened['title'], divider='gray')
        render(library.as_job(opened))

st.header('Try it yourself', divider='gray')
st.caption('Documents with test questions whose answers are already known, so you can try RAGLab straight away and '
           'compare results. Loading one creates a workspace with its files and questions.')
for benchmark in benchmarks.BENCHMARKS.values():
    finance = benchmark.key == 'financebench'
    status = benchmarks.financebench_status() if finance else None
    with st.container(border=True):
        st.markdown(f'**{benchmark.name}**: {benchmark.summary}')
        st.caption(benchmark.details)
        st.caption(f':material/rule: **Scored on:** {benchmark.scores}')
        if finance:
            st.caption(f":material/database: **On this computer:** {status['questions']} questions over "
                       f"{status['documents']} filings (open sample: {status['full_questions']} questions, "
                       f"{status['full_documents']} filings). Its workspace also offers the **finance-tuned** "
                       'retrieval pipeline, to compare with generic RAG.')
            if not status['ready']:
                st.warning('The filings are not on this computer yet. Use **Download more** below to fetch them '
                           'from the public FinanceBench repository, then load the workspace.',
                           icon=':material/cloud_download:')
        with st.container(horizontal=True):
            existing = benchmarks.loaded(benchmark.key)
            if st.button(f"{'Open' if existing and not finance else 'Refresh' if existing else 'Load'} as a workspace",
                         key=f'load_{benchmark.key}', disabled=bool(status) and not status['ready'],
                         type='primary', icon=':material/download:'):
                with st.spinner(f'Loading {benchmark.name}…'):
                    st.session_state['active_collection'] = benchmarks.load(benchmark.key)
                st.switch_page('app_pages/workspace.py')
            if finance:
                with st.popover('Download more', icon=':material/cloud_download:'):
                    st.markdown('Downloads questions and their filings (PDFs) from the public FinanceBench repository '
                                'and extracts the text. Takes a few minutes; filings that fail to download are skipped.')
                    n_docs = st.number_input('Filings', 1, status['full_documents'], min(10, status['full_documents']))
                    n_questions = st.number_input('Questions at most', 1, status['full_questions'], 30)
                    if st.button('Download', type='primary'):
                        start_prepare(n_docs, n_questions)
                        st.rerun()

running = [j for j in recent_jobs(5, kinds=('prepare',)) if j.get('status') in ('queued', 'running')]
if running:
    @st.fragment(run_every='3s')
    def prepare_progress(job_id):
        st.markdown('**Downloading FinanceBench…** You can keep working; this updates by itself.')
        if job_progress(job_id) not in ('queued', 'running'):
            st.rerun(scope='app')
    prepare_progress(running[0]['id'])
last = recent_jobs(1, kinds=('prepare',))
if last and last[0].get('status') == 'failed':
    st.error(last[0].get('error', 'Download failed'), title='The last FinanceBench download failed')
