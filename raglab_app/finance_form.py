"""FinanceBench's finance-tuned retrieval pipeline: settings form and start button."""
import streamlit as st

from raglab_app.jobs import start_financial
from raglab_app.models import choices
from raglab_app.options import HELP


def render(collection: dict, test_set: str) -> None:
    chat_models, embedding_models = choices('chat'), choices('embedding')
    st.caption('Finance-tuned retrieval runs on the prepared FinanceBench filings with the same questions and '
               'the same evidence scoring, so it can be compared directly with a generic run.')
    from ui_backend import defaults, model_choices, make_grid_plan, RETRIEVAL_MODES, TECHNIQUES as FIN_TECHNIQUES
    D = defaults()
    fin_chat_aliases = [a for _, a in model_choices('chat')]
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        fin_chat = c1.selectbox('Answer model', fin_chat_aliases, format_func=chat_models.get, help=HELP['chat'],
                                index=fin_chat_aliases.index(D['chat']) if D.get('chat') in fin_chat_aliases else 0)
        fin_judge = c2.selectbox('Judge model', fin_chat_aliases, format_func=chat_models.get,
                                 index=fin_chat_aliases.index(D['judge']) if D.get('judge') in fin_chat_aliases else 0)
        fin_judge_on = c3.toggle('Score answers with the judge', value=D.get('judge_enabled', True))
        fin_embeddings = st.multiselect('Embedding models', [a for _, a in model_choices('embedding')],
                                        format_func=embedding_models.get,
                                        default=[D.get('embedding')] if D.get('embedding') else None)
        c1, c2, c3 = st.columns(3)
        fin_top_k = c1.text_input('Top-K values', value=str(D.get('top_k', 5)), help=HELP['top_k'])
        fin_lines = c2.text_input('Window size (lines)', value=str(D.get('window_lines', 32)),
                                  help='Filings are cut into windows of this many lines so tables stay whole.')
        fin_overlap = c3.text_input('Window overlap (lines)', value=str(D.get('window_overlap', 10)))
        fin_modes = st.multiselect('Search methods', list(RETRIEVAL_MODES), default=['hybrid'],
                                   format_func=RETRIEVAL_MODES.get)
        fin_techniques = st.pills('Techniques (applied together)', list(FIN_TECHNIQUES), selection_mode='multi',
                                  default=['rerank'], format_func=FIN_TECHNIQUES.get)
        fin_baseline = st.checkbox('Also run a plain vector-search baseline', value=True)
    name = st.text_input('Name for this evaluation', value='FinanceBench · finance-tuned', max_chars=100)
    if st.button('Run evaluation', type='primary', icon=':material/play_arrow:'):
        try:
            fin_plan = make_grid_plan(chat=fin_chat, embeddings=fin_embeddings, judge=fin_judge,
                                      judge_enabled=fin_judge_on, top_ks=fin_top_k, window_lines=fin_lines,
                                      window_overlaps=fin_overlap, retrievals=fin_modes, techniques=fin_techniques,
                                      compare_baseline=fin_baseline, mode='grid')
            st.session_state['selected_experiment'] = start_financial(
                fin_plan, display_name=name, collection=collection, test_set=test_set)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.rerun()
