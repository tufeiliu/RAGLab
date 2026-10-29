"""Find the best RAG setup for your documents in one call.

    from raglab import find_best_rag

    search = find_best_rag(
        documents=["paper.pdf"],                     # files, or a folder
        tests="tests.jsonl",                         # questions with known answers
        search={"chunk_size": [400, 800], "top_k": [3, 5],
                "techniques": [[], ["rerank"], ["rerank", "opening"]]},
    )
    print(search.leaderboard)                        # every setup, best first
    print(search.best["settings"])                   # the winner's settings
    rag = search.best_rag()                          # the winning pipeline, ready to use
    print(rag.ask("Who wrote the paper?").answer)

Every combination of the values in `search` is run on the same questions and scored the same way as
in the app; the best is the one with the highest ranking metric (see raglab.experiments.RANKING_METRICS).
Indexes and embeddings are cached under `workdir`, so a second search over the same files reuses them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from .catalog import ModelCatalog, chat_client, embedding_client
from .core import RAGConfig, RAGLab
from .evaluation import TestCase, load_tests
from .experiments import SETTINGS, TECHNIQUE_FLAGS, rank, ranking_metric, run_experiments

log = logging.getLogger("raglab.search")

PROJECT_CATALOG = Path(__file__).resolve().parents[1] / "models.yaml"
SEARCHABLE = {"chunk_size", "chunk_overlap", "top_k", "techniques", "chat", "embedding"}
MAX_SETUPS = 64


@dataclass
class SearchResult:
    """The outcome of find_best_rag: every setup's scores, ranked, and the winning pipeline."""
    report: dict                       # full report, same format as the app's (raglab.experiments)
    metric: str | None                 # what the setups were ranked by
    order: list[int]                   # indexes into report["experiments"], best first
    documents: list[str]
    catalog: str
    workdir: str
    setups: list[dict] = field(default_factory=list)  # what was searched: settings + techniques per setup

    @property
    def best(self) -> dict:
        """The winning setup: name, settings, techniques, models and summary scores."""
        return self._describe(self.order[0])

    @property
    def ranking(self) -> list[dict]:
        return [self._describe(i) for i in self.order]

    @property
    def leaderboard(self):
        """A pandas DataFrame: one row per setup, best first, with its settings and main scores."""
        import pandas as pd
        rows = []
        for place, entry in enumerate(self.ranking, 1):
            s = entry["summary"]
            rows.append({"rank": place, "setup": entry["name"], **entry["settings"],
                         "techniques": ", ".join(entry["techniques"]) or "none",
                         **{k: s.get(k) for k in ("accuracy", "relevance", "faithfulness", "answered",
                                                  "evidence_recall", "evidence_mrr", "context_recall",
                                                  "context_precision", "mrr", "keyword_coverage", "latency_s",
                                                  "failed")}})
        return pd.DataFrame(rows).dropna(axis=1, how="all").set_index("rank")

    def best_rag(self, *, cache_responses: bool = False) -> RAGLab:
        """The winning pipeline, opened on its (already built) index and ready to `.ask()`."""
        return self.rag(0, cache_responses=cache_responses)

    def rag(self, which: int | str, *, cache_responses: bool = False) -> RAGLab:
        """Any setup's pipeline, by rank position (0 = best) or by name."""
        from .cache import CachedEmbeddings, SQLiteResponseCache
        from .indexes import describe_files, open_index
        experiments = self.report["experiments"]
        index = self.order[which] if isinstance(which, int) else \
            next(i for i, e in enumerate(experiments) if e["name"] == which)
        experiment = experiments[index]
        catalog = ModelCatalog(self.catalog)
        chat = catalog.get("chat", experiment["models"]["chat"])
        embed = catalog.get("embedding", experiment["models"]["embedding"])
        cache = Path(self.workdir) / "cache"
        settings = {k: v for k, v in experiment["config"].items() if k in SETTINGS}
        cfg = RAGConfig(**settings, chat_model=chat.model, embedding_model=embed.model)
        responses = SQLiteResponseCache(cache / "responses.sqlite") if cache_responses else None
        return open_index(cfg, Path(self.workdir) / "indexes", describe_files(self.documents),
                          llm=chat_client(chat, cache=responses),
                          embedder=CachedEmbeddings(embedding_client(embed), embed.model, cache / "embeddings.sqlite"))

    def save(self, path: str | Path) -> Path:
        """Write the full report as JSON (the same format the app shows)."""
        path = Path(path)
        path.write_text(json.dumps({**self.report, "ranking_metric": self.metric}, indent=2, default=str),
                        encoding="utf-8")
        return path

    def _describe(self, index: int) -> dict:
        experiment = self.report["experiments"][index]
        setup = self.setups[index] if index < len(self.setups) else {}
        config = experiment["config"]
        return {"name": experiment["name"], "models": experiment["models"],
                "settings": {k: config.get(k) for k in ("chunk_size", "chunk_overlap", "top_k")},
                "techniques": setup.get("technique_names", []),
                "summary": experiment["summary"]}

    def __repr__(self) -> str:
        best = self.best
        return (f"<SearchResult: {len(self.order)} setups ranked by {self.metric}; best {best['name']!r} "
                f"({self.metric}={best['summary'].get(self.metric)})>")


def plan_search(search: dict | None = None, *, chat: str = "openai_gpt5_mini", embedding: str = "openai_small",
                judge: bool | str = True) -> dict:
    """Turn a search space into an experiment plan (raglab.experiments format) without running it.

    search keys (each a list of values to try; all combinations are run):
      chunk_size, chunk_overlap, top_k, chat, embedding  e.g. [400, 800]
      techniques   lists of technique names, e.g. [[], ["rerank"], ["rerank", "opening"]]; see TECHNIQUE_FLAGS
    chunk_overlap defaults to 20% of each chunk size.
    """
    search = dict(search or {})
    unknown = set(search) - SEARCHABLE
    if unknown:
        raise ValueError(f"Cannot search over {sorted(unknown)}; choose from {sorted(SEARCHABLE)}")
    space = {key: (list(values) if isinstance(values, (list, tuple)) else [values]) for key, values in search.items()}
    if any(not values for values in space.values()):
        raise ValueError("Every search option needs at least one value")
    technique_sets = [list(t) if isinstance(t, (list, tuple)) else [t] for t in space.pop("techniques", [[]])]
    for names in technique_sets:
        bad = set(names) - set(TECHNIQUE_FLAGS)
        if bad:
            raise ValueError(f"Unknown technique(s) {sorted(bad)}; choose from {sorted(TECHNIQUE_FLAGS)}")
    defaults = RAGConfig()
    fields = ["chat", "embedding", "chunk_size", "chunk_overlap", "top_k"]
    space.setdefault("chat", [chat])
    space.setdefault("embedding", [embedding])
    space.setdefault("chunk_size", [defaults.chunk_size])
    space.setdefault("top_k", [defaults.top_k])
    overlaps = space.pop("chunk_overlap", [None])
    combos = list(product(*(space[f] for f in fields if f != "chunk_overlap"), overlaps, technique_sets))
    if len(combos) > MAX_SETUPS:
        raise ValueError(f"{len(combos)} setups exceed the limit of {MAX_SETUPS}; every setup answers every "
                         "question, so narrow the search")
    varied = {f for f in fields if len(space.get(f, overlaps)) > 1} | ({"techniques"} if len(technique_sets) > 1 else set())
    runs = []
    for chat_alias, embed_alias, size, top_k, overlap, names in combos:
        overlap = size // 5 if overlap is None else overlap
        if not 0 <= overlap < size:
            raise ValueError(f"chunk_overlap {overlap} must be smaller than chunk_size {size}")
        flags = {}
        for name in names:
            flags.update(TECHNIQUE_FLAGS[name])
        label = {"chat": chat_alias, "embedding": embed_alias, "chunk_size": f"chunk {size}",
                 "chunk_overlap": f"overlap {overlap}", "top_k": f"top-k {top_k}",
                 "techniques": "+".join(names) or "baseline"}
        name = " · ".join(label[f] for f in [*fields, "techniques"] if f in varied) or "baseline"
        runs.append({"name": name, "techniques": flags, "technique_names": list(names),
                     "settings": {"chat": chat_alias, "embedding": embed_alias, "chunk_size": size,
                                  "chunk_overlap": overlap, "top_k": top_k}})
    judge_alias = judge if isinstance(judge, str) else chat
    return {"settings": {"chat": chat, "embedding": embedding, "judge": judge_alias, "judge_enabled": bool(judge)},
            "runs": runs}


def find_best_rag(documents: str | Path | Iterable[str | Path], tests: str | Path | Iterable[str | TestCase | dict],
                  search: dict | None = None, *, chat: str = "openai_gpt5_mini", embedding: str = "openai_small",
                  judge: bool | str = True, metric: str | None = None, workdir: str | Path = "raglab_data",
                  catalog: str | Path | None = None, cache_responses: bool = False,
                  progress: Callable[..., None] | None = None) -> SearchResult:
    """Run every combination in `search` on the test questions and rank the setups.

    documents  file paths, or a folder (every supported file in it)
    tests      a JSONL test file; one question as a string; or a list of questions (strings), dicts with
               question and optionally reference_answer, keywords, evidence and category, or TestCase objects.
               Questions without an answer key are judged without one: is the answer relevant, grounded
               in the passages, and did the search find passages that help (no correctness score).
    search     values to try; see plan_search. None runs the defaults once.
    chat, embedding  model aliases from models.yaml, used unless `search` varies them
    judge      True: score answers with the answer model as judge; an alias: use that model; False: no judge
               (then setups are ranked on search quality, which needs evidence passages or keywords)
    metric     rank by this summary metric instead of the automatic choice (e.g. "faithfulness")
    workdir    where indexes and caches are kept; the app uses "raglab_data", so sharing it shares the cache
    cache_responses  reuse stored model answers for identical prompts (cheaper reruns, same answers)
    progress   optional callback receiving progress events (stage, completed_runs, total_runs, ...)
    """
    from .loaders import SUPPORTED_EXTENSIONS
    paths = [documents] if isinstance(documents, (str, Path)) else list(documents)
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No documents found")
    cases = _test_cases(tests)
    keyed = [c for c in cases if c.reference_answer.strip() or c.keywords or c.evidence]
    if not judge and len(keyed) < len(cases):
        raise ValueError(f"{len(cases) - len(keyed)} question(s) have no answer key (reference_answer, evidence "
                         "or keywords), so without the judge nothing can score them. Use judge=True, which judges "
                         "answers without a reference, or add answer keys.")
    if len(cases) < 5:
        log.warning("Only %d question(s): the ranking reflects these questions and may not generalise; "
                    "ten or more give a fairer picture", len(cases))
    catalog = Path(catalog) if catalog else (Path("models.yaml") if Path("models.yaml").is_file() else PROJECT_CATALOG)
    plan = plan_search(search, chat=chat, embedding=embedding, judge=judge)
    plan["settings"]["judge_without_reference"] = bool(judge)
    workdir = Path(workdir)
    log.info("Searching %d setups over %d files with %d questions", len(plan["runs"]), len(files), len(cases))
    report = run_experiments(plan, catalog, cases, [str(f) for f in files], index_root=workdir / "indexes",
                             cache_dir=workdir / "cache", cache_responses=cache_responses, progress=progress)
    summaries = [e["summary"] for e in report["experiments"]]
    chosen = ranking_metric(summaries, metric)
    if chosen is None:
        log.warning("No score is available for every setup (no reference answers, evidence or keywords); "
                    "the ranking only reflects failures and speed")
    report["ranking_metric"] = chosen
    return SearchResult(report=report, metric=chosen, order=rank(summaries, chosen),
                        documents=[str(f) for f in files], catalog=str(catalog), workdir=str(workdir),
                        setups=plan["runs"])


def _test_cases(tests) -> list[TestCase]:
    """A JSONL file, one question string, or a list of strings / dicts / TestCase objects."""
    if isinstance(tests, Path) or (isinstance(tests, str) and tests.endswith(".jsonl")):
        return load_tests(tests)
    if isinstance(tests, (str, TestCase, dict)):
        tests = [tests]
    cases = [_test_case(t) for t in tests]
    if not cases:
        raise ValueError("No test questions")
    return cases


def _test_case(item: str | TestCase | dict) -> TestCase:
    if isinstance(item, TestCase):
        return item
    if isinstance(item, str):
        item = {"question": item}
    if not str(item.get("question") or "").strip():
        raise ValueError(f"Each test needs a question: {item!r}")
    from .evidence import normalise_evidence
    keywords = item.get("keywords") or []
    return TestCase(question=item["question"].strip(), reference_answer=str(item.get("reference_answer") or ""),
                    keywords=[k.strip() for k in keywords.split(",")] if isinstance(keywords, str) else list(keywords),
                    category=item.get("category") or "unspecified",
                    evidence=normalise_evidence(item.get("evidence") or []))
