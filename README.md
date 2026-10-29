# RAGLab

**Find out how well AI answers questions about your documents, and which setup answers them best.**

RAGLab answers questions about your own files (papers, reports, policies, manuals) using retrieval-augmented generation (RAG): it searches your documents for the relevant passages, then has a language model answer from those passages. Every answer shows the passages it came from.

What makes it a lab: you can **measure** how good the answers are, see **why** a question failed (the search missed the passage, or the model misread it), and **compare** settings and techniques side by side until you find what works for your documents.

## What you can do with it

- **Ask questions** about a set of documents and check the sources behind each answer.
- **Test a setup** on questions whose answers you know, and get scores for both the search and the answers.
- **Compare** chunk sizes, models and techniques such as re-ranking or query expansion, and see which one helps.
- **Search automatically** from Python: try many combinations in one call and get the best pipeline back, ready to use.
- **Learn how RAG works** with ready-made examples: a research paper and the FinanceBench financial benchmark.

## Get started

You need Python 3.10 or newer and an [OpenAI API key](https://platform.openai.com/api-keys). An Anthropic key is optional, for Claude models.

```bash
cd RAGLab
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[viz,documents]'                     # add ,anthropic for Claude models
echo "OPENAI_API_KEY=sk-..." > .env                   # your key; .env is never committed
streamlit run app.py
```

The app opens in your browser. A good first five minutes:

1. Go to **Library → Try it yourself → OrthoFinder paper → Load as a workspace**. This gives you one research paper and 11 questions with known answers.
2. In the **Workspace**, ask *"Who is the second author of this paper?"* and look at the passages the answer came from.
3. Open the **Test questions** tab and press **Run**. You get scores for every question, and a diagnosis for those that failed.
4. In the sidebar, switch on **Include document opening** and run again. Then tick both runs under **Runs** and press **Compare**.

To use your own documents, create a new workspace and upload them. PDF, Word, Excel, CSV, text, Markdown, JSON and HTML files are supported.

## The app

| Page | What it is for |
|---|---|
| **Workspace** | Your files, asking questions, test questions and runs, all on one page. The sidebar holds the settings and techniques. |
| **Results** | Every run in full, runs compared side by side, and your saved test questions. |
| **Library** | Finished example evaluations to learn from, and benchmarks you can run. |
| **Guide** | How it all works: settings, every technique, every score and how to read it, what to change when something fails, and the Python API. |

The Guide is also in this repository: [docs/guide.md](docs/guide.md).

## Use it from Python

`find_best_rag` tries every combination of the settings you list on your test questions, scores each one, and hands back the winner:

```python
from raglab import find_best_rag

search = find_best_rag(
    documents=["paper.pdf"],                       # files, or a folder
    tests="tests.jsonl",                           # a test file, one question, or a list of questions
    search={"chunk_size": [400, 800],
            "top_k": [3, 6],
            "techniques": [[], ["rerank"], ["rerank", "opening"]]},
)
print(search.leaderboard)                          # every setup, best first
rag = search.best_rag()                            # the winning pipeline
print(rag.ask("Who are the authors?").answer)
```

To try it on the example paper:

```bash
python examples/find_best_rag.py --dry-run     # list the setups it would try (free)
python examples/find_best_rag.py --quick       # 2 setups, about 45 model calls
python examples/find_best_rag.py --help        # all options
```

See the Guide's **Python API** section for the arguments, test question formats, how the best setup is chosen, and the lower-level building blocks.

## Good to know

- **Costs.** Every answer is a call to your model provider, and so is every judge score and every model-based technique. Testing 10 questions on 4 setups is about 80 calls. Run the examples with `--dry-run` or `--quick` first.
- **Nothing is computed twice.** Search indexes and embeddings are cached in `raglab_data/`, so trying new settings on the same files only pays for what is new.
- **Scores are estimates.** Answer scores come from a language model acting as judge, and search scores from word overlap with the passages you marked. Both are useful for comparing setups; check important results by hand.
- **Your data stays local.** Files, indexes and results are stored in `raglab_data/` on your machine; only the text sent to the model provider leaves it.
- **Single user, local only.** Don't expose the app on a public network: it has no login or upload limits.
- **Scanned PDFs** need OCR before upload. Legacy `.doc` files need the `antiword` tool.

## Settings

- **`settings.yaml`**: the app's starting settings (models, chunk size, top-K, techniques, judge) and the allowed ranges.
- **`models.yaml`**: the model names you can choose, and which API key each needs. Add your own models here.
- **`.env`**: your API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## For developers

```
app.py                 Streamlit entry point
app_pages/             The four pages: Workspace, Results, Library, Guide
raglab/                The library: RAG pipeline, file loaders, indexes, caches, evaluation, experiments, find_best_rag
raglab_app/            App support: workspaces, background jobs, logging, result views
examples/              find_best_rag.py, quickstart.py, the OrthoFinder example, the FinanceBench benchmark, library/ (example results)
docs/guide.md          The guide shown on the Guide page
tests/                 pytest suite, no API calls needed
```

Run the tests with `python -m pytest -q tests examples/financial_rag_benchmark/tests`.

**How indexes are kept.** An index is identified by the exact files (name and SHA-256) and the settings that change the chunks or vectors: embedding model, chunk size and overlap, and the index-time techniques. A finished index is never modified: the same files and settings reuse it, and anything different builds a new one. Embeddings are cached per model and text, and model answers optionally (*Reuse cached answers*). Long work (index builds, evaluations) runs in background processes, so it survives page changes and browser refreshes.

```
raglab_data/
  collections/<name>/   a workspace: files/, tests/, collection.json
  indexes/<key>/        a Chroma index and index_ready.json (files, settings, chunk count)
  cache/                embeddings.sqlite, responses.sqlite
  jobs/<id>/            a run: status.json, job.log, report.json
  logs/app.log          questions asked in the app
```

## Licence

The RAGLab code is released under the [MIT licence](LICENSE). The example data below keeps its own licence.

PDF reading uses [PyMuPDF](https://pymupdf.readthedocs.io/), installed with the optional `documents` extra, which is licensed under the AGPL-3.0. If you distribute RAGLab together with PyMuPDF, or offer it as a network service, the AGPL applies to that combination.

## Data and licences

- **FinanceBench** ([patronus-ai/financebench](https://github.com/patronus-ai/financebench)) by Patronus AI is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This repository includes 10 of its questions with their gold answers and evidence (`examples/financial_rag_benchmark/data/`), and one example result built from them (`examples/library/financebench-hybrid-vs-vector/`). These files are shared under the same licence: attribution required, non-commercial use only. The filings themselves are not included; the app downloads them from the FinanceBench repository.
- **OrthoFinder paper** (`examples/orthofinder/`): Emms et al., *Nature Methods* (2026), https://doi.org/10.1038/s41592-026-03126-6. Redistributed under the paper's open-access licence.
