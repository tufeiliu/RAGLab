"""Guide: what RAGLab does and how to use it, rendered from docs/guide.md."""
import streamlit as st

from raglab_app import ROOT

st.markdown((ROOT / 'docs' / 'guide.md').read_text(encoding='utf-8'))
