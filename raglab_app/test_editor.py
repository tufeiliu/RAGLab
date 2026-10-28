"""Editor for a document set's test questions (typed into a table, or uploaded as JSONL)."""
import pandas as pd
import streamlit as st

from raglab_app import collections
from raglab_app.options import TEST_EXAMPLE, TEST_FORMAT


def render(collection: dict) -> None:
    slug = collection['slug']
    tests = collections.load(slug)['tests']
    mode = st.segmented_control('How', ['type', 'upload'], default='type', key='test_mode',
                                format_func={'type': 'Type questions', 'upload': 'Upload a file'}.get,
                                label_visibility='collapsed') or 'type'
    if mode == 'type':
        NEW = '+ New test set'
        if 'saved_test_set' in st.session_state:  # select the set just saved (must happen before the widget)
            st.session_state[f'edit_set_{slug}'] = st.session_state.pop('saved_test_set')
        target = st.selectbox('Test set to edit', [NEW, *sorted(tests)], key=f'edit_set_{slug}',
                              format_func=lambda t: t if t == NEW else f"{t} ({tests[t]['questions']} questions)")
        if target == NEW:
            set_name = st.text_input('Name of the new test set', value='my questions', max_chars=60)
            rows = [{'question': '', 'reference_answer': '', 'keywords': '', 'evidence': '', 'category': ''}]
        else:
            set_name = target
            rows = [{**c, 'keywords': ', '.join(c['keywords']),
                     'evidence': collections.EVIDENCE_SEPARATOR.join(e['text'] for e in c['evidence']),
                     'category': '' if c['category'] == 'unspecified' else c['category']}
                    for c in collections.read_test_set(slug, target)]
            sources = {c['question']: c['evidence'] for c in collections.read_test_set(slug, target)}
        version = st.session_state.get('test_editor_version', 0)  # bumped after saving to reset the editor
        edited = st.data_editor(
            pd.DataFrame(rows, columns=['question', 'reference_answer', 'keywords', 'evidence', 'category']),
            num_rows='dynamic', hide_index=True, key=f'test_editor_{slug}_{target}_{version}',
            column_config={
                'question': st.column_config.TextColumn('Question', required=True, width='large'),
                'reference_answer': st.column_config.TextColumn('Reference answer', width='large',
                    help='What a correct answer says. Needed for answer scores.'),
                'keywords': st.column_config.TextColumn('Keywords', width='medium',
                    help='Comma separated words a relevant passage must contain, e.g. 2015, Lisbon. Needed for retrieval scores.'),
                'evidence': st.column_config.TextColumn('Evidence passage', width='large',
                    help='The text in the document that contains the answer, copied from the document. Needed for '
                         'precise search scores. Several passages: separate them with a line containing only ---.'),
                'category': st.column_config.TextColumn('Category', help='Optional label, e.g. "dates" or "people"; results are broken down by it.'),
            })
        st.caption('Click a cell to type. Add a row with the + at the bottom; remove one by selecting it and pressing Delete.')
        with st.container(horizontal=True):
            if st.button('Save test set', type='primary', icon=':material/save:'):
                try:
                    if target == NEW and collections.test_set_name(set_name) in tests:
                        raise ValueError(f'A test set called {collections.test_set_name(set_name)} already exists; pick it above to edit it.')
                    records = edited.fillna('').to_dict('records')
                    if target != NEW:
                        # Keep each passage's source file when the passage text was not edited.
                        for record in records:
                            original = sources.get(record['question'], [])
                            if collections.EVIDENCE_SEPARATOR.join(e['text'] for e in original) == record['evidence']:
                                record['evidence'] = original
                    saved = collections.save_test_set(slug, set_name, records)
                    st.session_state['test_editor_version'] = version + 1
                    st.session_state['saved_test_set'] = saved
                    st.toast(f'Saved {saved}', icon=':material/check:')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if target != NEW and st.button('Delete this test set', icon=':material/delete:'):
                collections.remove_test_set(slug, target)
                st.rerun()
    else:
        st.markdown(TEST_FORMAT)
        st.code(TEST_EXAMPLE, language='json')
        with st.form('upload_tests', clear_on_submit=True):
            test_upload = st.file_uploader('Test set (.jsonl)', type=['jsonl'])
            if st.form_submit_button('Add test set', icon=':material/upload:') and test_upload:
                try:
                    stored = collections.add_test_set(slug, test_upload.name, test_upload.getvalue())
                    st.toast(f'Added {stored}', icon=':material/check:')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
