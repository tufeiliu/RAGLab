"""Index identity, reuse and caching, with fake models (no API calls)."""
import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from raglab.cache import CachedEmbeddings, SQLiteResponseCache, cache_stats
from raglab.core import RAGConfig
from raglab.indexes import build_index, open_index, describe_files, list_indexes, index_key


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def corpus(tmp_path):
    doc = tmp_path / 'notes.txt'
    doc.write_text('\n\n'.join(f'Paragraph {i}: the warehouse in city {i} stores item {i * 7}.' for i in range(40)))
    return [doc]


def test_same_files_and_settings_reuse_one_index(tmp_path):
    files, embedder = corpus(tmp_path), CountingEmbeddings()
    cfg = RAGConfig(chunk_size=120, chunk_overlap=20, embedding_model='fake')
    first = build_index(cfg, files, tmp_path / 'idx', embedder=embedder, llm=object())
    calls = embedder.calls
    again = build_index(cfg, files, tmp_path / 'idx', embedder=embedder, llm=object())
    assert again['key'] == first['key'] and embedder.calls == calls
    lab = open_index(cfg, tmp_path / 'idx', describe_files(files), embedder=embedder, llm=object())
    assert lab.store._collection.count() == first['chunks']


def test_changed_setting_builds_separate_index(tmp_path):
    files = corpus(tmp_path)
    a = RAGConfig(chunk_size=120, chunk_overlap=20, embedding_model='fake')
    b = RAGConfig(chunk_size=300, chunk_overlap=20, embedding_model='fake')
    assert index_key(a, describe_files(files)) != index_key(b, describe_files(files))
    # Query-time options do not change the index.
    c = RAGConfig(chunk_size=120, chunk_overlap=20, embedding_model='fake', top_k=9, rerank=True)
    assert index_key(a, describe_files(files)) == index_key(c, describe_files(files))
    for cfg in (a, b):
        build_index(cfg, files, tmp_path / 'idx', embedder=CountingEmbeddings(), llm=object())
    assert len(list_indexes(tmp_path / 'idx')) == 2


def test_reingesting_does_not_duplicate_chunks(tmp_path):
    files = corpus(tmp_path)
    cfg = RAGConfig(chunk_size=120, chunk_overlap=20, embedding_model='fake')
    record = build_index(cfg, files, tmp_path / 'idx', embedder=CountingEmbeddings(), llm=object())
    lab = open_index(cfg, tmp_path / 'idx', describe_files(files), embedder=CountingEmbeddings(), llm=object())
    from raglab.loaders import load_file
    sections = load_file(files[0])
    for section in sections:
        section.metadata['source'] = files[0].name  # as build_index records it
    lab.ingest_documents(sections)
    assert lab.store._collection.count() == record['chunks']


def test_embedding_cache_skips_known_texts(tmp_path):
    inner = CountingEmbeddings()
    cached = CachedEmbeddings(inner, 'fake', tmp_path / 'embeddings.sqlite')
    first = cached.embed_documents(['a', 'b'])
    assert cached.embed_documents(['b', 'c', 'a']) == [first[1], inner.embed_query('c'), first[0]]
    assert inner.calls == 3 + 1  # a, b, c, plus the direct embed_query above
    assert cached.hits == 2 and cached.misses == 3
    assert CachedEmbeddings(CountingEmbeddings(), 'other-model', tmp_path / 'embeddings.sqlite').embed_query('a') != []
    assert cache_stats(tmp_path)['embeddings'] == 4


def test_response_cache_returns_stored_answer(tmp_path):
    cache = SQLiteResponseCache(tmp_path / 'responses.sqlite')
    llm = FakeListChatModel(responses=['first', 'second'], cache=cache)
    assert llm.invoke('hello').content == 'first'
    assert llm.invoke('hello').content == 'first'
    assert llm.invoke('different').content == 'second'
    assert cache.hits == 1


def test_chunk_summaries_report_progress_per_chunk(tmp_path):
    files = corpus(tmp_path)
    seen = []
    llm = FakeListChatModel(responses=['a summary'] * 500)
    cfg = RAGConfig(chunk_size=120, chunk_overlap=20, embedding_model='fake', hierarchical=True)
    record = build_index(cfg, files, tmp_path / 'idx', embedder=CountingEmbeddings(), llm=llm,
                         on_progress=lambda stage, done, total: seen.append((stage, done, total)))
    summaries = [s for s in seen if s[0] == 'summarising']
    assert summaries and summaries[-1][1] == summaries[-1][2] == record['chunks'] // 2
