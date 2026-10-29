"""A transparent educational implementation of ten RAG techniques.

All LLM-dependent methods require an OpenAI API key and incur API costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
log = logging.getLogger("raglab")


@dataclass
class RAGConfig:
    # Technique 1: chunking R&D
    chunk_size: int = 500
    chunk_overlap: int = 100
    # Technique 2: embedding model R&D
    embedding_model: str = field(default_factory=lambda: os.getenv("RAGLAB_EMBED_MODEL", "text-embedding-3-small"))
    chat_model: str = field(default_factory=lambda: os.getenv("RAGLAB_CHAT_MODEL", "gpt-5-mini"))
    top_k: int = 5
    # Technique switches
    improved_prompt: bool = False       # 3
    preprocess: bool = False            # 4 (index time)
    query_rewrite: bool = False         # 5
    query_fusion: bool = False          # search both original and rewritten queries
    query_expansion: int = 1            # 6: total queries, 1 disables
    rerank: bool = False                # 7
    hierarchical: bool = False         # 8: summarize chunks at index time
    graph: bool = False                # 9: lexical overlapping-neighbor expansion
    agentic: bool = False              # 10: LLM selects vector retrieval or safe read-only SQLite
    opening: bool = False              # add each found file's opening passage (title, authors, date)
    sqlite_path: str | None = None
    db_dir: str = "raglab_db"
    collection_name: str = "raglab"

    def validate(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        if self.top_k < 1 or self.query_expansion < 1:
            raise ValueError("top_k and query_expansion must be >= 1")


@dataclass
class RAGResult:
    question: str
    answer: str
    documents: list[Document]
    queries: list[str]
    trace: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer,
                "queries": self.queries, "trace": self.trace,
                "documents": [{"content": d.page_content, "metadata": d.metadata} for d in self.documents]}


class RAGLab:
    def __init__(self, config: RAGConfig | None = None, *, llm=None, embedder=None):
        self.config = config or RAGConfig()
        self.config.validate()
        self.llm = llm if llm is not None else ChatOpenAI(model=self.config.chat_model, temperature=0)
        self.embedder = embedder if embedder is not None else OpenAIEmbeddings(model=self.config.embedding_model)
        # Separate collection for each embedding model; collection names cannot clash.
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "-", self.config.embedding_model)
        self.collection = (self.config.collection_name + "-" + safe_model)[:60]
        self.store = Chroma(collection_name=self.collection,
                            persist_directory=self.config.db_dir,
                            embedding_function=self.embedder)

    def _ask(self, instruction: str, text: str) -> str:
        response = self.llm.invoke([SystemMessage(content=instruction), HumanMessage(content=text)])
        return str(response.content).strip()

    @staticmethod
    def chunk_id(doc: Document) -> str:
        """Stable ID from provenance and text: re-adding a chunk overwrites it instead of duplicating it."""
        m = doc.metadata
        raw = "|".join(str(m.get(k)) for k in ("source", "page", "sheet", "row_start", "chunk", "text_type"))
        return hashlib.sha256((raw + "|" + doc.page_content).encode("utf-8")).hexdigest()[:32]

    def ingest_documents(self, documents: list[Document], *, batch_size: int = 64,
                         on_progress: Callable[[str, int, int], None] | None = None) -> int:
        """Index parsed documents, keeping source/page/sheet/row metadata.

        on_progress(stage, done, total) is called while LLM preprocessing runs
        ("preparing", per document) and while chunks are embedded ("embedding").
        """
        started = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = []
        for number, document in enumerate(documents, 1):
            original = document.page_content
            if original.strip():
                text = (self._ask("Clean text for search. Preserve all facts, numbers and row labels.", original)
                        if self.config.preprocess else original)
                for index, chunk in enumerate(splitter.split_text(text)):
                    chunks.append(Document(page_content=chunk,
                                           metadata={**document.metadata, "chunk": index, "text_type": "original"}))
            if self.config.preprocess and on_progress:
                on_progress("preparing", number, len(documents))
        if self.config.hierarchical:
            # One model call per chunk: report progress per chunk, it is the slow part.
            originals = list(chunks)
            log.info("Writing summaries for %d chunks (one model call each)", len(originals))
            for number, chunk in enumerate(originals, 1):
                summary = self._ask("Summarize this passage for retrieval, retaining important facts and numbers.",
                                    chunk.page_content)
                chunks.append(Document(page_content=summary, metadata={**chunk.metadata, "text_type": "summary"}))
                if on_progress:
                    on_progress("summarising", number, len(originals))
        unique = {self.chunk_id(c): c for c in chunks}
        if len(unique) < len(chunks):
            log.info("Skipped %d duplicate chunks", len(chunks) - len(unique))
        ids, chunks = list(unique), list(unique.values())
        log.info("Split %d document sections into %d chunks (size %d, overlap %d)",
                 len(documents), len(chunks), self.config.chunk_size, self.config.chunk_overlap)
        for start in range(0, len(chunks), batch_size):
            self.store.add_documents(chunks[start:start + batch_size], ids=ids[start:start + batch_size])
            if on_progress:
                on_progress("embedding", min(start + batch_size, len(chunks)), len(chunks))
        log.info("Indexed %d chunks with %s in %.1fs", len(chunks), self.config.embedding_model,
                 time.perf_counter() - started)
        return len(chunks)

    def ingest_texts(self, texts: list[str], source_prefix: str = "text") -> int:
        docs = [Document(page_content=text, metadata={"source": f"{source_prefix}-{i}"})
                for i, text in enumerate(texts)]
        return self.ingest_documents(docs)

    def ingest_files(self, paths: list[str]) -> int:
        """Parse supported file types and ingest chunks with provenance metadata."""
        from .loaders import load_file
        documents = [doc for path in paths for doc in load_file(path)]
        return self.ingest_documents(documents)

    @staticmethod
    def _unique(docs: list[Document]) -> list[Document]:
        seen = set()
        output = []
        for doc in docs:
            # Same text is the same information, even if it was repeated on several pages.
            key = (doc.metadata.get("text_type"), " ".join(doc.page_content.split()))
            if key not in seen:
                seen.add(key)
                output.append(doc)
        return output

    @classmethod
    def _fuse(cls, rankings: list[list[Document]], k: int = 60) -> list[Document]:
        """Reciprocal rank fusion: passages ranked high by several queries come first.

        With one query this is that query's ranking, minus duplicates.
        """
        scores, docs = {}, {}
        for ranking in rankings:
            for rank, doc in enumerate(cls._unique(ranking), 1):
                key = (doc.metadata.get("text_type"), " ".join(doc.page_content.split()))
                scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
                docs.setdefault(key, doc)
        return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]

    def _openings(self, docs: list[Document], max_files: int = 3) -> list[Document]:
        """First chunk of each file among docs: where titles, authors and dates usually are."""
        out = []
        for source in list(dict.fromkeys(d.metadata.get("source") for d in docs if d.metadata.get("source")))[:max_files]:
            found = self.store.get(where={"$and": [{"source": source}, {"chunk": 0}]},
                                   include=["documents", "metadatas"])
            pairs = [(m or {}, text) for text, m in zip(found.get("documents", []), found.get("metadatas", []))
                     if (m or {}).get("text_type", "original") == "original"]
            if pairs:
                meta, text = min(pairs, key=lambda p: p[0].get("page") or 0)
                out.append(Document(page_content=text, metadata=meta))
        return out

    def _expand_graph(self, docs: list[Document], max_extra: int = 3) -> list[Document]:
        """Lightweight graph approximation: connect chunks sharing lexical entities/terms.

        This is not knowledge-graph extraction or GraphRAG community summarization.
        """
        all_data = self.store.get(include=["documents", "metadatas"])
        words = lambda s: set(re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,}\b", s.lower()))
        seeds = set().union(*(words(d.page_content) for d in docs)) if docs else set()
        have = {d.page_content for d in docs}
        scored = []
        for content, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
            if content and content not in have:  # the retrieved chunks themselves overlap most
                overlap = len(seeds & words(content))
                if overlap:
                    scored.append((overlap, Document(page_content=content, metadata=meta or {})))
        scored.sort(key=lambda x: x[0], reverse=True)
        return self._unique(docs + [d for _, d in scored[:max_extra]])

    def _rerank(self, question: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        indexed = "\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(docs))
        response = self._ask("Rank passage IDs by relevance to the question. Return ONLY a JSON array of integer IDs, best first. Do not invent IDs.",
                             f"Question: {question}\nPassages:\n{indexed}")
        try:
            indices = json.loads(response)
            ordered = [docs[i] for i in indices if type(i) is int and 0 <= i < len(docs)]
            return self._unique(ordered + docs)
        except (ValueError, TypeError):
            return docs

    def _sql_readonly(self, question: str) -> str:
        if not self.config.sqlite_path:
            return "SQL tool unavailable"
        uri = f"file:{Path(self.config.sqlite_path).resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
            schema = "\n".join(str(row) for row in tables)
            sql = self._ask("Translate the question to a single SQLite SELECT statement. Return ONLY SQL. No code fences or comments. Use the provided schema.",
                            f"Schema:\n{schema}\nQuestion: {question}")
            sql = sql.strip().rstrip(";")
            if not re.match(r"(?is)^select\b", sql) or ";" in sql:
                return "SQL rejected: only single SELECT queries are allowed"
            # Defense in depth: query-only SQLite authorizer; no writes, pragmas or attaches.
            def auth(action, arg1, arg2, dbname, source):
                return sqlite3.SQLITE_OK if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION} else sqlite3.SQLITE_DENY
            conn.set_authorizer(auth)
            conn.set_progress_handler(lambda: 1, 100000)
            try:
                rows = conn.execute(f"SELECT * FROM ({sql}) LIMIT 20").fetchall()
                return f"SQL: {sql}\nRows: {rows}"
            except sqlite3.Error as exc:
                return f"SQL query failed: {exc}"

    def ask(self, question: str, history: str = "") -> RAGResult:
        cfg = self.config
        started = time.perf_counter()
        trace = []
        queries = [question]
        if cfg.query_rewrite:
            rewritten = self._ask("Rewrite the user question as one standalone search query. Preserve intent and constraints; output only query.",
                                  f"Conversation: {history}\nQuestion: {question}")
            queries = [question, rewritten] if cfg.query_fusion else [rewritten]
            trace.append(f"Rewritten query: {rewritten}")
        if cfg.query_expansion > 1:
            response = self._ask(f"Generate {cfg.query_expansion - 1} different search queries expanding the question. Return one query per line, no numbering.",
                                 f"Question: {queries[0]}")
            new_queries = [re.sub(r"^\s*\d+[.)]\s*", "", x).strip() for x in response.splitlines() if x.strip()]
            queries += new_queries[: cfg.query_expansion - 1]
            trace.append(f"Expanded to {len(queries)} queries")
        tool_result = ""
        if cfg.agentic:
            route = self._ask("Choose a tool: return exactly VECTOR or SQL. Use SQL only for calculations or structured database questions when relevant.", question).upper()
            trace.append(f"Agent route: {route}")
            if route.startswith("SQL") and cfg.sqlite_path:
                tool_result = self._sql_readonly(question)
        docs = []
        if not tool_result:
            # Extra candidates, so that dropping duplicates still leaves top_k passages.
            docs = self._fuse([self.store.similarity_search(q, k=cfg.top_k * 2) for q in queries])
            candidates = len(docs)
            if cfg.rerank:
                docs = self._rerank(question, docs)
                trace.append("LLM re-ranking applied")
            docs = docs[: cfg.top_k]
            trace.append(f"Search returned {candidates} candidate passages; kept the best {len(docs)}")
            if cfg.graph:
                kept = len(docs)
                docs = self._expand_graph(docs)
                trace.append(f"Graph-neighbor expansion: added {len(docs) - kept} passages")
            if cfg.opening:
                openings = [d for d in self._openings(docs) if d not in docs]
                docs = docs + openings
                trace.append(f"Added the opening passage of {len(openings)} file(s)")
        context = tool_result or "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))
        instruction = "Answer the question using the provided context. If the context is insufficient, say so. Do not invent facts."
        if cfg.improved_prompt:
            instruction += f"\nToday's date: {date.today().isoformat()}. Separate confirmed facts from uncertainty. Cite passage numbers like [1] when appropriate."
        retrieved_at = time.perf_counter()
        answer = self._ask(instruction, f"Conversation:\n{history}\nContext:\n{context}\nQuestion: {question}")
        done = time.perf_counter()
        trace.append(f"Retrieval {retrieved_at - started:.1f}s, generation {done - retrieved_at:.1f}s")
        log.info("Answered %r in %.1fs (%d passages, %d queries)", question[:80], done - started, len(docs), len(queries))
        return RAGResult(question, answer, docs, queries, trace)

    def evaluate(self, tests, judge=None, k: int | None = None):
        """Evaluate a sequence of TestCase items against this RAG configuration."""
        from .evaluation import evaluate_dataset
        return evaluate_dataset(self, tests, judge=judge, k=k)

    def compare(self, question: str, variants: dict[str, dict], reference: str | None = None) -> list[dict]:
        """Run variants; includes simple retrieval-independent exact match baseline, not judge scoring."""
        results = []
        for name, overrides in variants.items():
            cfg = replace(self.config, **overrides)
            run = RAGLab(cfg)
            res = run.ask(question)
            record = {"name": name, "answer": res.answer, "trace": res.trace,
                      "retrieved": len(res.documents)}
            if reference is not None:
                record["exact_match"] = int(res.answer.strip().lower() == reference.strip().lower())
            results.append(record)
        return results
