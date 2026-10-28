"""Streamlit widgets shared by several pages."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from raglab.indexes import build_index, index_key, read_index
from raglab_app import INDEXES, collections
from raglab_app.jobs import status as job_status
from raglab_app.models import make_clients, make_config

def when(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime('%Y-%m-%d %H:%M')


def page_intro(title: str, purpose: str) -> None:
    st.title(title)
    st.markdown(purpose)


def pick_workspace() -> dict | None:
    """Workspace switcher plus "New workspace"; the choice is remembered across pages."""
    available = collections.list_collections()
    by_slug = {c['slug']: c for c in available}
    current = st.session_state.get('active_collection')
    with st.container(horizontal=True, vertical_alignment='bottom'):
        slug = None
        if available:
            slugs = list(by_slug)
            slug = st.selectbox('Workspace', slugs, index=slugs.index(current) if current in slugs else 0,
                                format_func=lambda s: by_slug[s]['name'],
                                help='A workspace holds a set of files, the test questions about them and their '
                                     'evaluations. Use one workspace per topic so answers never mix unrelated files.')
            st.session_state['active_collection'] = slug
        with st.popover('New workspace', icon=':material/create_new_folder:'):
            with st.form('new_workspace', clear_on_submit=True, border=False):
                name = st.text_input('Name', placeholder='e.g. Research papers')
                if st.form_submit_button('Create', type='primary'):
                    try:
                        st.session_state['active_collection'] = collections.create(name)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    return by_slug.get(slug)


def index_record(collection: dict, chat: str, embedding: str, **index_settings) -> tuple[str, dict | None]:
    """(key, record) of the index for this collection and settings; record is None if not built."""
    cfg = make_config(chat, embedding, **index_settings)
    key = index_key(cfg, collections.file_identity(collection['slug']))
    return key, read_index(INDEXES, key)


def ensure_index(collection: dict, chat: str, embedding: str, **index_settings) -> dict:
    """Return the index record, building it here with a progress bar if it does not exist yet."""
    _, record = index_record(collection, chat, embedding, **index_settings)
    if record:
        return record
    with st.status('Preparing the documents for search (first time with these settings)…', expanded=True) as box:
        st.write('Splitting the files into chunks and embedding each chunk. Chunks embedded before come from the '
                 'cache, so only new text costs anything.')
        bar = st.progress(0.0)

        def show(stage, done, total):
            what = {'embedding': 'Embedding chunks', 'summarising': 'Writing chunk summaries (one model call each)',
                    'preparing': 'Cleaning sections with the LLM'}.get(stage, stage)
            bar.progress(done / total if total else 0.0, text=f'{what}: {done}/{total}')

        record = build_index(make_config(chat, embedding, **index_settings),
                             collections.file_paths(collection['slug']), INDEXES, on_progress=show,
                             **make_clients(chat, embedding, cache_responses=True))
        box.update(label=f"Documents ready: {record['chunks']:,} chunks indexed", state='complete', expanded=False)
    return record


def job_progress(job_id: str) -> str:
    """Draw progress for a queued or running job and return its status."""
    try:
        info = job_status(job_id)
    except (OSError, ValueError) as exc:
        st.error(f'Unable to read progress: {exc}')
        return 'failed'
    state = info.get('status', 'queued')
    p = info.get('progress') or {}
    if state == 'queued':
        st.info('Waiting to start…', icon=':material/hourglass:')
    elif state == 'running':
        stage = str(p.get('stage') or 'running').replace('_', ' ')
        if p.get('total_runs') and p.get('total_questions'):
            st.progress(float(p.get('fraction') or 0.0),
                        text=f"{round(100 * float(p.get('fraction') or 0))}% · {stage}")
            with st.container(horizontal=True):
                st.metric('Setups finished', f"{p.get('completed_runs') or 0}/{p['total_runs']}", border=True)
                st.metric('Questions answered in this setup', f"{p.get('question_done') or 0}/{p['total_questions']}", border=True)
            if p.get('current_run'):
                st.caption(f"Now running: {p['current_run']}")
        else:
            st.progress(0.0, text='Starting…')
        with st.expander('Latest log lines'):
            st.code(info.get('log_tail') or 'No log output yet', language='text')
    return state
