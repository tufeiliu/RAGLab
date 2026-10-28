"""Declarative, comparable RAG experiments from experiments.yaml."""
from __future__ import annotations
from itertools import product
from pathlib import Path
import logging
from typing import Any
import yaml

from .catalog import ModelCatalog, chat_client, embedding_client
from .core import RAGConfig
from .evaluation import LLMJudge, evaluate_dataset
from .failure import failure_report

log = logging.getLogger("raglab.experiments")

TECHNIQUES = {"improved_prompt", "preprocess", "query_rewrite", "query_fusion",
              "query_expansion", "rerank", "hierarchical", "graph", "agentic", "opening"}
SETTINGS = set(RAGConfig.__dataclass_fields__) - {"chat_model", "embedding_model", "db_dir", "collection_name"}


def load_plan(path: str | Path) -> dict:
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not plan.get("runs"):
        raise ValueError("Experiment YAML must contain a nonempty runs list")
    for run in plan["runs"]:
        invalid = set(run.get("techniques", {})) - TECHNIQUES
        invalid |= set(run.get("settings", {})) - SETTINGS - {"chat", "embedding", "judge"}
        if invalid:
            raise ValueError(f"Unsupported experiment overrides: {sorted(invalid)}")
        if not run.get("name"):
            raise ValueError("Each run needs a name")
    return plan


def expand_grid(plan: dict) -> list[dict]:
    """Optional grid expands parameter values to combinations, with a hard safety cap."""
    items = list(plan.get("runs", []))
    grid = plan.get("grid", {})
    if not grid:
        return items
    if not all(isinstance(v, list) and v for v in grid.values()):
        raise ValueError("grid options must be nonempty lists")
    fields = list(grid)
    expanded = []
    for base in items:
        for values in product(*(grid[key] for key in fields)):
            candidate = {**base, "techniques": dict(base.get("techniques", {})),
                         "settings": dict(base.get("settings", {}))}
            suffix = []
            for key, val in zip(fields, values):
                target = "techniques" if key in TECHNIQUES else "settings"
                if target == "settings" and key not in SETTINGS | {"chat", "embedding", "judge"}:
                    raise ValueError(f"Invalid grid setting: {key}")
                candidate[target][key] = val
                suffix.append(f"{key}={val}")
            candidate["name"] = base["name"] + "__" + "__".join(suffix)
            expanded.append(candidate)
            if len(expanded) > 64:
                raise ValueError("Grid would create >64 runs; narrow the search to limit API cost")
    return expanded


def run_experiments(plan_path: str | Path, catalog_path: str | Path, tests,
                    documents: list[str], *, index_root: str | Path | None = None,
                    cache_dir: str | Path | None = None, cache_responses: bool = False,
                    progress=None) -> dict[str, Any]:
    """Run named models/technique combinations on exactly the same test cases.

    Indexes are shared and cached by file content and index-time settings (see
    raglab.indexes), so runs differing only in query-time techniques reuse one
    index. With cache_dir, embeddings are always cached there; model responses
    are cached too when cache_responses is True.

    progress(**event) receives stage, completed_runs, total_runs, current_run,
    question_done and total_questions.
    """
    from .cache import CachedEmbeddings, SQLiteResponseCache
    from .indexes import build_index, open_index, describe_files

    plan = load_plan(plan_path)
    catalog = ModelCatalog(catalog_path)
    defaults = plan.get("settings", {})
    runs = expand_grid(plan)
    tests = list(tests)
    if not documents:
        raise ValueError("Provide at least one document")
    if not tests:
        raise ValueError("Provide at least one test question")
    index_root = Path(index_root or defaults.get("db_dir", "raglab_experiment_db"))
    response_cache = SQLiteResponseCache(Path(cache_dir) / "responses.sqlite") if cache_dir and cache_responses else None
    files = describe_files(documents)
    emit = progress or (lambda **event: None)
    # Resolve all models before building indexes or spending money on questions.
    selected_models = set()
    for spec in runs:
        config = spec.get("settings", {})
        selected_models.add(("chat", config.get("chat", defaults.get("chat", "openai_gpt5_mini"))))
        selected_models.add(("embedding", config.get("embedding", defaults.get("embedding", "openai_small"))))
        if config.get("judge_enabled", defaults.get("judge_enabled", False)):
            selected_models.add(("chat", config.get("judge", defaults.get("judge", config.get("chat", defaults.get("chat", "openai_gpt5_mini"))))))
    for kind, alias in sorted(selected_models):
        catalog.get(kind, alias)  # Validate aliases and credentials early.
    log.info("Running %d configurations on %d questions over %d files", len(runs), len(tests), len(files))
    reports = []
    for number, spec in enumerate(runs, 1):
        state = dict(total_runs=len(runs), completed_runs=number - 1, current_run=spec["name"],
                     total_questions=len(tests), question_done=0)
        alias_chat = spec.get("settings", {}).get("chat", defaults.get("chat", "openai_gpt5_mini"))
        alias_embed = spec.get("settings", {}).get("embedding", defaults.get("embedding", "openai_small"))
        alias_judge = spec.get("settings", {}).get("judge", defaults.get("judge", alias_chat))
        chat = catalog.get("chat", alias_chat)
        embed = catalog.get("embedding", alias_embed)
        judge_spec = catalog.get("chat", alias_judge) if defaults.get("judge_enabled", False) else None
        merged = {k: v for k, v in defaults.items() if k in SETTINGS}
        merged.update({k: v for k, v in spec.get("settings", {}).items() if k in SETTINGS})
        merged.update(spec.get("techniques", {}))
        cfg = RAGConfig(**merged, chat_model=chat.model, embedding_model=embed.model)
        embedder = embedding_client(embed)
        if cache_dir:
            embedder = CachedEmbeddings(embedder, embed.model, Path(cache_dir) / "embeddings.sqlite")
        clients = dict(llm=chat_client(chat, cache=response_cache), embedder=embedder)
        log.info("[%d/%d] %s: chunk %s/%s, top-k %s, techniques %s", number, len(runs), spec["name"],
                 cfg.chunk_size, cfg.chunk_overlap, cfg.top_k, spec.get("techniques") or "none")
        emit(stage="indexing", **state)
        record = build_index(cfg, documents, index_root,
                             on_progress=lambda stage, done, total: emit(stage=f"indexing ({stage} {done}/{total})", **state),
                             **clients)
        lab = open_index(cfg, index_root, files, **clients)
        judge = LLMJudge(llm=chat_client(judge_spec, cache=response_cache)) if judge_spec else None
        emit(stage="evaluating", **state)
        report = evaluate_dataset(lab, tests, judge=judge, on_progress=lambda done, total, ok, failed: emit(
            stage="evaluating", **{**state, "question_done": done}, successful=ok, failed=failed))
        report["name"] = spec["name"]
        report["models"] = {"chat": alias_chat, "embedding": alias_embed, "judge": alias_judge if judge else None}
        report["config"] = {**merged, "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "top_k": cfg.top_k}
        report["index_key"] = record["key"]
        report["index_dir"] = lab.config.db_dir
        report["index_collection"] = lab.collection
        report["failures"] = failure_report(report)
        reports.append(report)
        summary = report["summary"]
        log.info("[%d/%d] %s done: %d ok, %d failed", number, len(runs), spec["name"],
                 summary["successful"], summary["failed"])
        if report["errors"]:
            log.warning("First error in %s: %s", spec["name"], report["errors"][0]["error"])
        emit(stage="run_complete", **{**state, "completed_runs": number, "question_done": len(tests)})
    if response_cache:
        log.info("Response cache: %d hits, %d new calls", response_cache.hits, response_cache.misses)
    def score(report):
        summary = report["summary"]
        # Rank using accuracy when judged; otherwise retrieval score.
        return (summary.get("accuracy") if summary.get("accuracy") is not None else summary.get("ndcg")) or 0.0
    ranking = sorted(reports, key=score, reverse=True)
    return {"leaderboard": [{"name": r["name"], "models": r["models"],
                              **r["summary"]} for r in ranking], "experiments": reports,
            "ranking_rule": "accuracy if judge enabled, else keyword-based nDCG; inspect all metrics"}
