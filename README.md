# RAGLab

Composable RAG technique playground with a single Streamlit app for running and comparing experiments.

## Run

```bash
pip install -e '.[viz,documents]'
# put OPENAI_API_KEY in .env
streamlit run app.py
```

The app has four pages; the **Guide** page (also at [docs/guide.md](docs/guide.md)) explains everything in detail.

- **Workspace**: everything on one page: your files → ask a question (answer, sources, scores when you give the correct answer) → test questions → evaluate them with the current settings, or compare settings. The sidebar settings apply to both asking and evaluating.
- **Results**: your saved test questions and every evaluation: scores, breakdown by category, question-by-question diagnosis, comparison between evaluations. Evaluations can be saved to the Library.
- **Library**: finished evaluations to learn from (a FinanceBench example ships with the project), and benchmarks to run yourself. FinanceBench loads as a workspace with gold-evidence answer keys and offers a finance-tuned retrieval pipeline next to generic RAG.
- **Guide**: how RAG works, the workflow, what every score means, what to change when something fails, and how to use the package from Python.

## How indexing and caching work

- **Indexes are content-addressed** (`raglab/indexes.py`). An index is identified by the exact files (name + SHA-256) and the settings that change chunks or vectors: embedding model, chunk size, overlap, and LLM preprocessing/summaries. A finished index is never modified. Asking again with the same files and settings reuses it; changing anything builds a separate index. Re-uploading a file never duplicates chunks.
- **Embeddings are always cached** (`raglab/cache.py`) per model and text, so rebuilding with different chunking only pays for chunks never seen before.
- **Model responses** can optionally be cached ("Reuse cached answers"): an identical prompt returns the stored answer, making reruns free and repeatable.

Long-running work (index builds, experiments) runs in background processes, so it survives page changes and browser refreshes. This is a local-workstation runner: do not expose it publicly without authentication, upload limits and per-user isolation.

## Data and logs

```
raglab_data/
  collections/<name>/   files/, tests/, collection.json
  indexes/<key>/        Chroma index + index_ready.json (files, settings, chunk count)
  cache/                embeddings.sqlite, responses.sqlite
  jobs/<id>/            status.json, job.log (timestamped), report.json
  logs/app.log          interactive questions from the Ask page
```

## Code layout

```
app.py                 Streamlit entry point (navigation)
app_pages/             Workspace, Results, Library, Guide
raglab/                Core library: RAG pipeline, loaders, indexes, caches, evaluation, experiments, visualization
raglab_app/            App support: collections, background jobs, logging, plans, result views
examples/              quickstart.py, the OrthoFinder paper example, the FinanceBench research example, and library/ (shipped example results)
docs/guide.md          The user guide shown on the Guide page
models.yaml            Model aliases (OpenAI, Anthropic) used everywhere in the app
settings.yaml          Default RAG settings for the app (models, chunk size/overlap, top-K, techniques, judge)
tests/                 pytest suite (no API calls)
```

## Notes

- Evidence scores count a retrieved chunk as relevant when it shares enough words with the evidence passage, measured against the shorter of the two: 60% for short sentences, falling to 30% for long passages, restricted to the evidence's file when one is given. On FinanceBench this agrees with the benchmark's own evidence labels on 19 of 20 questions, but it is a lexical approximation, not a human relevance judgement; nor are keyword scores or LLM-judge scores. See `examples/financial_rag_benchmark/README.md` for the full list of limitations.
- Scanned PDFs are not OCR'd. Legacy `.doc` files need the `antiword` system tool.
- Tests: `python -m pytest -q tests examples/financial_rag_benchmark/tests`

## Data and licences

- **FinanceBench** ([patronus-ai/financebench](https://github.com/patronus-ai/financebench)) by Patronus AI is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This repository includes 10 of its questions with their gold answers and evidence (`examples/financial_rag_benchmark/data/`) and one example result built from them (`examples/library/financebench-hybrid-vs-vector/`). Those files are shared under the same licence: attribution required, non-commercial use only. The filings themselves are not included; the app downloads them from the FinanceBench repository.
- **OrthoFinder paper** (`examples/orthofinder/`): Emms et al., *Nature Methods* (2026), https://doi.org/10.1038/s41592-026-03126-6. Redistributed under the paper's open-access licence.

