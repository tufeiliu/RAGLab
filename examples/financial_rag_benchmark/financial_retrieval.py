"""FinanceBench example-only retrieval and table-preserving ingestion.

No reference answers, document IDs from the gold annotations, or evidence texts
are used in query routing or retrieval. All routing uses question text and the
names/metadata of indexed source filings.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import re

from langchain_core.documents import Document
from raglab.core import RAGLab, RAGResult

STOP = set("the a an for of from what which how was were in on at to is are did do by with and or as amount give response question details shown relying fiscal year usd millions million company statement".split())
PAGE = re.compile(r"\[PDF PAGE\s+(\d+)\]")
DOC = re.compile(r"^(?P<company>.+)_(?P<year>20\d{2})(?P<period>Q[1-4])?_(?P<type>10K|10Q)$", re.I)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOP]


def windows_from_file(path: Path, *, lines_per_chunk: int = 32, overlap: int = 10) -> list[Document]:
    """Keep complete financial-statement rows and nearby year/unit headers together."""
    if not (lines_per_chunk > overlap >= 0):
        raise ValueError("Require lines_per_chunk > overlap >= 0")
    text = path.read_text(encoding="utf-8")
    sections = PAGE.split(text)
    pages = [(0, sections[0])]
    pages += [(int(sections[i]), sections[i + 1]) for i in range(1, len(sections) - 1, 2)]
    match = DOC.match(path.stem)
    filing = match.groupdict() if match else {}
    docs = []
    for page_number, page_text in pages:
        lines = [line.rstrip() for line in page_text.splitlines() if line.strip()]
        for start in range(0, len(lines), lines_per_chunk - overlap):
            snippet = "\n".join(lines[start:start + lines_per_chunk])
            if not snippet.strip():
                continue
            md = {"source": f"{path}-0", "source_file": path.stem, "page": page_number,
                  "line_start": start, "chunk": len(docs), "text_type": "original",
                  "company": filing.get("company") or "", "fiscal_year": int(filing["year"]) if filing.get("year") else 0,
                  "filing_type": filing.get("type") or ""}
            docs.append(Document(page_content=snippet, metadata=md))
            if start + lines_per_chunk >= len(lines):
                break
    return docs


def identify_filing(question: str, available: list[str]) -> str | None:
    """Conservatively infer filing from a question, never from the gold answer."""
    q = question.lower()
    year_match = re.search(r"\b(?:fy|fiscal\s*(?:year)?\s*|20)(20\d{2}|\d{2})\b", q, re.I)
    # FY2018, fiscal 2018, and bare 2018 are all supported.
    if not year_match:
        year_match = re.search(r"\b(20\d{2})\b", q)
    year = None
    if year_match:
        yy = year_match.group(1)
        year = f"20{yy}" if len(yy) == 2 else yy
    candidates = []
    for name in available:
        match = DOC.match(name)
        if not match:
            continue
        group = match.groupdict()
        company = group["company"].lower().replace("_", " ")
        if year is not None and group["year"] != year:
            continue
        if not re.search(r"(?<![a-z0-9])" + re.escape(company) + r"(?![a-z0-9])", q):
            continue
        if "fy" in q or "annual" in q or "10-k" in q:
            if group["type"].lower() != "10k":
                continue
        candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def bm25_search(query: str, docs: list[Document], k: int) -> list[Document]:
    """Small, dependency-free BM25 for an educational corpus; no train/eval leakage."""
    if not docs:
        return []
    terms = set(words(query))
    corpus = [Counter(words(d.page_content)) for d in docs]
    lengths = [sum(c.values()) for c in corpus]
    avglen = sum(lengths) / len(lengths) or 1
    dfs = Counter(t for counts in corpus for t in counts)
    scored = []
    for doc, counts, length in zip(docs, corpus, lengths):
        score = 0.0
        for term in terms:
            tf = counts[term]
            if tf:
                idf = math.log(1 + (len(docs) - dfs[term] + 0.5) / (dfs[term] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / avglen))
        if score:
            scored.append((score, doc))
    return [doc for score, doc in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]


def financial_search_terms(question: str) -> str:
    """Rule-based synonym expansion, intentionally independent of gold evidence."""
    extras = []
    if re.search(r"\bcapex\b|capital expenditure", question, re.I):
        extras.append("purchases of property plant and equipment PP&E investing activities")
    if re.search(r"free cash flow", question, re.I):
        extras.append("operating cash flow purchases property plant equipment")
    return question + " " + " ".join(extras)


@dataclass
class FinancialRetriever:
    lab: RAGLab
    documents: list[Document]
    candidate_k: int = 30

    def search(self, question: str, top_k: int, mode: str = "hybrid") -> tuple[list[Document], str | None]:
        names = sorted({d.metadata["source_file"] for d in self.documents})
        matched = identify_filing(question, names) if mode in {"filtered_vector", "hybrid"} else None
        eligible = [d for d in self.documents if matched is None or d.metadata["source_file"] == matched]
        filt = {"source_file": matched} if matched is not None else None
        q = financial_search_terms(question) if mode == "hybrid" else question
        search_kwargs = {"k": max(self.candidate_k, top_k)}
        if filt is not None:
            search_kwargs["filter"] = filt
        vectors = self.lab.store.similarity_search(q, **search_kwargs)
        if mode != "hybrid":
            return vectors[:top_k], matched
        lexical = bm25_search(q, eligible, max(self.candidate_k, top_k))
        # Reciprocal-rank fusion. Stable identifiers use file, page and line range.
        def key(d):
            m = d.metadata
            return m["source_file"], m["page"], m["line_start"]
        scores = Counter()
        lookup = {}
        for ranked in (vectors, lexical):
            for pos, doc in enumerate(ranked, 1):
                ident = key(doc)
                lookup[ident] = doc
                scores[ident] += 1 / (60 + pos)
        ids = sorted(scores, key=lambda ident: scores[ident], reverse=True)
        return [lookup[ident] for ident in ids[:top_k]], matched


class FinancialRAGLab(RAGLab):
    """Example-specific RAG pipeline; reuses RAGLab models, storage and methods."""

    def configure_retrieval(self, docs: list[Document], mode: str):
        if mode not in {"vector", "filtered_vector", "hybrid"}:
            raise ValueError(f"Unknown finance retrieval mode: {mode}")
        self.finance_mode = mode
        self.finance_retriever = FinancialRetriever(self, docs)

    def ask(self, question: str, history: str = "") -> RAGResult:
        if not hasattr(self, "finance_retriever"):
            return super().ask(question, history)
        cfg = self.config
        queries = [question]
        trace = []
        if cfg.query_rewrite:
            rewritten = self._ask("Rewrite as a standalone search query, retaining company, fiscal year, and requested metric. Return query only.",
                                  f"Conversation: {history}\nQuestion: {question}")
            queries = [question, rewritten] if cfg.query_fusion else [rewritten]
            trace.append(f"Rewritten: {rewritten}")
        if cfg.query_expansion > 1:
            response = self._ask(f"Return {cfg.query_expansion - 1} related search queries, one per line, retaining company and fiscal year.", question)
            queries += [x.strip() for x in response.splitlines() if x.strip()][:cfg.query_expansion - 1]
        all_docs = []
        # Always route using ORIGINAL user question to prevent a rewrite dropping constraints.
        for q in queries:
            retrieved, matched = self.finance_retriever.search(question if self.finance_mode != "vector" else q,
                                                               cfg.top_k * 4, self.finance_mode)
            if self.finance_mode != "vector" and q != question:
                # Search the rewritten/expanded query only inside the same original-question document scope.
                if matched:
                    subset = self.finance_retriever.search(q, cfg.top_k * 4, self.finance_mode)[0]
                    subset = [d for d in subset if d.metadata.get("source_file") == matched]
                    retrieved = self._unique(retrieved + subset)
            all_docs.extend(retrieved)
            trace.append(f"Retrieval {self.finance_mode}: matched filing={matched or 'none'}")
        docs = self._unique(all_docs)
        if cfg.rerank:
            docs = self._rerank(question, docs)
            trace.append("LLM reranking")
        docs = docs[:cfg.top_k]
        # The user question is preserved; retrieved passage numbers are supplied explicitly.
        context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
        prompt = ("Answer using only supplied filing evidence. For financial tables read the year headings and units carefully. "
                  "For cash outflows reported in parentheses, report a positive magnitude if asked for amount spent. "
                  "If the evidence is insufficient, say so; never invent figures. Cite passage numbers when possible.")
        if cfg.improved_prompt:
            prompt += " Distinguish observed reported values from estimates or guidance."
        answer = self._ask(prompt, f"Context:\n{context}\n\nQuestion: {question}")
        return RAGResult(question, answer, docs, queries, trace)
