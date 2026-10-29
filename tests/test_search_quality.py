"""Regression tests for running PDF headers and duplicate passages crowding out the context."""
from langchain_core.documents import Document

from raglab.core import RAGConfig, RAGLab
from raglab.loaders import _strip_running_lines


def test_running_headers_and_page_numbers_are_removed():
    topics = ['methods', 'results', 'genes', 'trees', 'species', 'discussion']
    pages = [(n, f'Article   https://doi.org/10.1/x\nThis page discusses {topics[n - 1]}.\nPage {n} of 9')
             for n in range(1, 7)]
    cleaned = _strip_running_lines(pages)
    assert all('doi.org' not in text and 'of 9' not in text for _, text in cleaned)
    assert cleaned[2][1] == 'This page discusses genes.'
    assert _strip_running_lines(pages[:2]) == pages[:2]  # too few pages to tell


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k):
        return self.docs[:k]


def test_duplicate_passages_do_not_shrink_the_context():
    header = [Document(page_content='Article https://doi.org/x', metadata={'source': 'a.pdf', 'page': p, 'chunk': 0})
              for p in range(4)]
    body = [Document(page_content=f'Real passage {i}', metadata={'source': 'a.pdf', 'page': 9, 'chunk': i})
            for i in range(6)]
    lab = RAGLab.__new__(RAGLab)
    lab.config = RAGConfig(top_k=5)
    lab.store = FakeStore(header + body)
    lab._ask = lambda instruction, text: 'answer'
    result = lab.ask('q')
    assert len(result.documents) == 5
    assert [d.page_content for d in result.documents][:2] == ['Article https://doi.org/x', 'Real passage 0']


class OpeningStore(FakeStore):
    def get(self, where, include):
        source = where['$and'][0]['source']
        hits = [d for d in self.docs if d.metadata['source'] == source and d.metadata['chunk'] == 0]
        return {'documents': [d.page_content for d in hits], 'metadatas': [d.metadata for d in hits]}


def test_document_opening_is_added_for_files_found():
    title = Document(page_content='OrthoFinder. David M. Emms, Yi Liu', metadata={'source': 'p.pdf', 'page': 1, 'chunk': 0})
    later = Document(page_content='Methods start here', metadata={'source': 'p.pdf', 'page': 4, 'chunk': 0})
    hit = Document(page_content='Author contributions: D.M.E. wrote it', metadata={'source': 'p.pdf', 'page': 9, 'chunk': 3})
    lab = RAGLab.__new__(RAGLab)
    lab.config = RAGConfig(top_k=1, opening=True)
    lab.store = OpeningStore([hit, later, title])
    lab._ask = lambda instruction, text: 'answer'
    result = lab.ask('who is the second author?')
    assert [d.page_content for d in result.documents] == ['Author contributions: D.M.E. wrote it',
                                                          'OrthoFinder. David M. Emms, Yi Liu']
    assert 'Added the opening passage of 1 file(s)' in result.trace


def test_judge_is_told_that_refusing_is_wrong():
    import inspect
    from raglab.evaluation import LLMJudge
    assert 'cannot answer' in inspect.getsource(LLMJudge.evaluate)


class QueryStore(FakeStore):
    """'q' finds alpha0..alpha9; any other query finds beta0..beta3, then alpha5."""
    def similarity_search(self, query, k):
        if query == 'q':
            found = [f'alpha{i}' for i in range(10)]
        else:
            found = [f'beta{i}' for i in range(4)] + ['alpha5']
        return [Document(page_content=text, metadata={}) for text in found[:k]]

    def get(self, include):
        return {'documents': ['alpha0', 'unrelated words', 'alpha0 alpha1 neighbour'], 'metadatas': [{}, {}, {}]}


def query_lab(**settings):
    lab = RAGLab.__new__(RAGLab)
    lab.config = RAGConfig(top_k=3, **settings)
    lab.store = QueryStore([])
    lab._ask = lambda instruction, text: 'other wording' if 'queries' in instruction or 'Rewrite' in instruction else 'answer'
    return [d.page_content for d in lab.ask('q').documents]


def test_extra_queries_change_what_is_retrieved():
    # Before: the first query's results filled the top-K, so expansion and fusion changed nothing.
    assert query_lab() == ['alpha0', 'alpha1', 'alpha2']
    # Reciprocal rank fusion: alpha5, found by both queries, comes first; then the two queries interleave.
    assert query_lab(query_expansion=2) == ['alpha5', 'alpha0', 'beta0']
    assert query_lab(query_rewrite=True, query_fusion=True) == ['alpha5', 'alpha0', 'beta0']


def test_neighbour_expansion_adds_chunks_after_the_top_k():
    assert query_lab(graph=True) == ['alpha0', 'alpha1', 'alpha2', 'alpha0 alpha1 neighbour']
