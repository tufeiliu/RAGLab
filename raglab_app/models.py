"""Model aliases from models.yaml, turned into RAG configs and cached clients.

The app and background jobs both go through here, so an index built by a job
has exactly the identity the playground later looks up.
"""
from __future__ import annotations

from raglab.cache import CachedEmbeddings, SQLiteResponseCache
from raglab.catalog import ModelCatalog, chat_client, embedding_client
from raglab.core import RAGConfig
from raglab_app import CACHE, CATALOG


def catalog() -> ModelCatalog:
    return ModelCatalog(CATALOG)


def choices(kind: str) -> dict[str, str]:
    """alias -> label, for every model in models.yaml (with or without an API key)."""
    return {alias: spec.label for alias, spec in catalog().choices(kind, available_only=False).items()}


def make_config(chat: str, embedding: str, **settings) -> RAGConfig:
    models = catalog()
    return RAGConfig(**settings, chat_model=models.get('chat', chat, require_key=False).model,
                     embedding_model=models.get('embedding', embedding, require_key=False).model)


def make_clients(chat: str, embedding: str, *, cache_responses: bool) -> dict:
    """llm and embedder keyword arguments for RAGLab / raglab.indexes, with shared caches."""
    models = catalog()
    chat_spec, embed_spec = models.get('chat', chat), models.get('embedding', embedding)
    responses = SQLiteResponseCache(CACHE / 'responses.sqlite') if cache_responses else None
    return {'llm': chat_client(chat_spec, cache=responses),
            'embedder': CachedEmbeddings(embedding_client(embed_spec), embed_spec.model,
                                         CACHE / 'embeddings.sqlite')}
