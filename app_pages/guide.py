"""Guide: what RAGLab does and how to use it, rendered from docs/guide.md, one tab per section."""
import streamlit as st

from raglab_app import ROOT


def sections(text):
    """Split markdown at '## ' headings (outside code blocks) into the intro and (title, body) pairs."""
    intro, parts, fenced = [], [], False
    for line in text.splitlines():
        if line.startswith('```'):
            fenced = not fenced
        if line.startswith('## ') and not fenced:
            parts.append((line[3:].strip(), []))
        else:
            (parts[-1][1] if parts else intro).append(line)
    return '\n'.join(intro), [(title, '\n'.join(body)) for title, body in parts]


intro, parts = sections((ROOT / 'docs' / 'guide.md').read_text(encoding='utf-8'))
st.markdown(intro)
for tab, (title, body) in zip(st.tabs([title for title, _ in parts]), parts):
    with tab:
        st.markdown(body)
