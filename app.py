"""RAGLab Streamlit app. Launch: streamlit run app.py"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
# The FinanceBench backend lives with its example and uses flat imports.
FINANCE = str(ROOT / 'examples' / 'financial_rag_benchmark')
if FINANCE not in sys.path:
    sys.path.insert(0, FINANCE)

from raglab_app.logs import setup_app_logging

setup_app_logging()
st.set_page_config(page_title='RAGLab', page_icon=':material/science:', layout='wide')

page = st.navigation([
    st.Page('app_pages/workspace.py', title='Workspace', icon=':material/work:', default=True),
    st.Page('app_pages/results.py', title='Results', icon=':material/bar_chart:'),
    st.Page('app_pages/library.py', title='Library', icon=':material/local_library:'),
    st.Page('app_pages/guide.py', title='Guide', icon=':material/menu_book:'),
], position='top')
page.run()
