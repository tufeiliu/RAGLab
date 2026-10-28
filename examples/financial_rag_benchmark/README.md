# FinanceBench: a Stage 1 financial RAG research case study

This is a **self-contained research example built ON TOP OF** the public `raglab` Python API. No finance-specific code is placed inside `raglab/` or `src/raglab/`. The original library is unchanged.

## Research question

Which RAG pipeline (baseline, rewriting, expansion, reranking, fusion + reranking) best retrieves evidence and answers numerical questions from annual reports, under a fixed benchmark sample?

FinanceBench open sample: 150 annotated Q&A examples with gold evidence. **Source filings** form the searchable corpus. Questions, gold answers, justifications, and gold evidence are held out of that corpus and used only during evaluation. The original FinanceBench sample is CC BY-NC 4.0; consult its license before distribution/commercial use. Dataset: https://github.com/patronus-ai/financebench

## Layout

```
examples/financial_rag_benchmark/
  README.md
  experiments.yaml        # default settings and runs for run.py
  prepare.py              # download annotations and PDFs; extract filing text
  run.py                  # run the finance-tuned pipeline over a plan
  financial_plan.py       # validate experiment plans
  financial_retrieval.py  # table-aware windows, filing filter, BM25 + vector hybrid search
  scoring.py              # source hit, numeric diagnostics, paired bootstrap
  evidence_metrics.py     # gold-evidence Precision/Recall/MRR/nDCG proxies
  ui_backend.py           # helpers the Streamlit app uses to build plans and read reports
  tests/
  data/questions.jsonl    # the 10 shipped FinanceBench questions with gold answers and evidence
  data/tests.jsonl        # the same questions as RAGLab test cases
  data/raw/ pdfs/ documents/ indexes/   # downloaded or generated, gitignored
  results/                # generated reports, gitignored
```

## Run

The easiest route is the app: **Library → FinanceBench → Download more**, then **Load as a workspace**. Results then appear on the Results page.

From the command line (project root):

```bash
pip install -e '.[viz,documents]'
# Set OPENAI_API_KEY in .env; never commit .env
python examples/financial_rag_benchmark/prepare.py --max-documents 4 --max-questions 20
python examples/financial_rag_benchmark/run.py
pytest -q examples/financial_rag_benchmark/tests
```

A full 150-question benchmark is possible with `--max-documents 200 --max-questions 150`, provided the PDFs can be obtained. Some provider PDF links block automated downloads; if so, download the associated FinanceBench PDFs manually into `data/pdfs/<doc_name>.pdf`, then run preparation again. PDF text extraction is imperfect on tables and scanned documents; visually audit representative pages. **No gold evidence is inserted into the index.**

For the default benchmark, questions are selected deterministically by filing and identifier (not a random/representative sample). Accordingly, results are illustrative and must not be presented as representative of all FinanceBench questions. Change command-line settings in `experiments.yaml`; model aliases are in the project's `models.yaml`; secrets remain in `.env`.

## Important limitations

1. `raglab`'s MRR and nDCG are **keyword-presence proxies**, not true gold-document relevance metrics. Here keywords are intentionally empty, so these values are **not meaningful**. Use `source_hit` and `evidence_lexical_recall` instead. Evidence recall is a lexical heuristic (the max token overlap between a retrieved chunk and each gold evidence span, conditioned on matching source filing), not a rigorous graded relevance label. For a published paper, implement passage-level gold evidence annotation and standard Recall@k/nDCG.
2. `numeric_overlap_diagnostic` checks literal numbers; it is **not** a measure of financial numerical correctness, equivalent units, arithmetic, or numerical reasoning. Have a human audit a stratified sample of judge results and derived calculations.
3. Judge scores are subjective. The current RAGLab runner does not record token cost or exact per-question latency; do **not** make cost/latency claims from this report.
4. A judge-enabled run makes multiple billable model calls. Test on 3-5 questions first. The runner caches indexes for matching chunking/embedding settings, but changes in source text trigger new indexes.
5. Compare configurations over **identical questions and a shared source corpus**; paired bootstrap uses question-level resampling, which is not a guarantee of generalization. Shared companies/filings induce dependence; for publication, use company/document-blocked resampling and stronger holdout designs.
6. UMAP/t-SNE are exploratory projections, not relevance metrics. Query overlay can cost one embedding API call.
7. **Known core-runner limitation:** RAGLab's `run_experiments` can swallow per-question exceptions into `errors`. Check `successful` and `failed` counts and do not rank runs with materially different completion rates; rerun or fix those failures first.

No live run is asserted here; you must supply accessible filings and valid provider keys.
