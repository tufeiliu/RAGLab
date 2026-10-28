# RAGLab guide

RAGLab answers questions about your own documents with **retrieval-augmented generation (RAG)**, and measures how well that works so you can improve it.

RAG works in two steps:

1. **Search.** Your files are split into short pieces (*chunks*). The question is compared with every chunk, and the *top-K* most related chunks are picked.
2. **Answer.** A language model writes the answer from those chunks only, and is told to say so when they are not enough.

So when an answer is wrong, either the search did not find the right passage, or it did and the model still got it wrong. RAGLab shows you which one, so you know what to change.

## Using the app

The app has four pages:

| Page | What it is for |
|---|---|
| **Workspace** | Where you work: your files, asking questions, test questions and evaluations, on one page. |
| **Results** | Every run in full, **Compare runs** side by side (settings, colour-coded scores, charts, every question per run), and your saved test questions. |
| **Library** | Finished evaluations to learn from, including a FinanceBench example, and benchmarks you can run. |
| **Guide** | This page. |

### The Workspace, top to bottom

A **workspace** is one topic: a set of files, the test questions about them, and their evaluations. Answers are only searched for in the files of the workspace you are in, so use separate workspaces for unrelated material (for example *My papers* and *Company policies*). Switch or create workspaces at the top of the page.

1. **Files.** Upload PDFs, Word, Excel, CSV, text, Markdown, JSON or HTML files. They are saved as soon as you pick them. A file that is already there is skipped, even under another name. The section always lists the files answers come from.
2. **Three tabs**: use whichever fits what you want to do.
   - **Ask a question.** Type a question and, optionally, its answer key (the correct answer, the passage that contains it, or keywords). *Ask and evaluate* returns the answer and the chunks it was based on, then the full evaluation of that answer: scorecards and charts. The first time, the files are prepared for search (a progress bar shows this). *Save as a test question* keeps the question for the other tabs, and *Show on a map* places the question, answer and chunks in a UMAP / t-SNE projection.
   - **Test questions.** Write, edit or upload questions with known answers, then *Run* them all with the current settings. Every run is kept. You get the averaged evaluation, the scores per category, and every question with its answer, its chunks and a diagnosis.
   - *Run several setups in one go* (inside the Test questions tab) is a shortcut for a batch of runs, for example each technique against a baseline, without changing the sidebar each time.
3. **Runs.** Every question you asked and every run of test questions in this workspace, with its status, settings (chunk size, top-K, techniques), headline scores and date. Tick runs to **compare** them (opens the comparison on the Results page), **open** one in full, or **delete** the ones you no longer need. Questions you asked can be compared with each other (the same question with chunk size 200, then 100, then 200 plus re-ranking; or different questions, each shown as a row), and runs of a test set with other runs of that test set.

   Runs continue in the background; their results appear in the tab, and all runs are kept on the Results page.

The **sidebar** holds the RAG settings. Their starting values come from **`settings.yaml`** in the project folder: the answer and embedding models, chunk size and overlap, top-K, techniques, whether the judge is on, and the allowed ranges. Edit it to change the defaults for everyone using this copy of RAGLab; a mistake in it is reported at the top of the sidebar and the built-in default is used instead. They apply to both *Ask* and *Evaluate*, so an evaluation measures exactly the setup you have been trying.

### A typical session

1. Create a workspace and upload your files.
2. Ask a few questions and read the sources. Is the right passage there?
3. For questions where you know the answer, fill in the answer key and *Save as a test question*. Aim for 20 or more, covering the kinds of questions you care about.
4. Run them in the *Test questions* tab. Look at the scores, then at the failing questions.
5. Change one setting in the sidebar (a chunk size, or a technique that addresses those failures such as *LLM re-ranking* or *Include document opening*) and run again. Repeat for each idea.
6. In the Workspace's *Runs* list, tick those runs and press *Compare*: the Results page shows which settings work best. Keep those in the sidebar, and delete the runs you no longer need.

## The answer key and the scores

Each test question can have any of three kinds of answer key. Each one enables different scores.

| Answer key | What it scores | Cost |
|---|---|---|
| **Correct answer** (reference) | The answer: an *LLM judge* (a second model) compares it with the reference. The same call also judges the search: which retrieved passages contain the correct answer's information. | One extra model call per question |
| **Passage that contains the answer** (evidence), copied from the document | The search, precisely: was this passage among the retrieved chunks, and how high? | Free |
| **Keywords** that the right passage contains, e.g. a name or a year | The search, roughly: were passages with these words retrieved? | Free |

**Answer scores** (1–5, from the LLM judge):

- **Accuracy**: is it correct compared with the reference? An answer that declines ("the context does not say") when the reference has an answer scores 1, as does its completeness.
- **Completeness**: does it contain everything the reference says?
- **Relevance**: does it answer the question that was asked?
- **Faithfulness**: is every claim supported by the retrieved passages, i.e. not made up?

**Search scores** (0–1):

- **Evidence recall**: share of the needed passages that were found. The most important search score.
- **Evidence precision**: share of the retrieved chunks that are evidence. Naturally lower: with top-K 5 and one needed passage, a perfect search scores 0.2.
- **Evidence hit rate**: share of questions where any evidence was found.
- **Evidence MRR / nDCG**: how near the top the evidence was. 1 = first.
- **Context recall, precision, hit rate, MRR, nDCG**: the same scores, but the judge decides which passages contain the correct answer's information. You get them with just a correct answer, no passage needed. Shown only when there are no evidence passages, since gold passages are more objective than a model's opinion.
- **Keyword coverage, keyword MRR, keyword nDCG**: the same ideas, using keywords instead of passages. Quicker to write, less exact, because a keyword can appear in an unrelated passage.

**Good, fair, poor.** Scores get a badge: search ≥ 0.9 and answers ≥ 4.5 are good; below 0.75 and 4.0 need work (evidence precision: 0.6 and 0.4). These are rules of thumb, not standards.

A chunk counts as matching an evidence passage when they share enough words, measured against the shorter of the two (and the chunk comes from the passage's file, when that is known). Short texts must share more: 60% for a sentence of 15 words, falling to 30% for passages of 60 words or more, so a short sentence is not matched by chunks that merely use the same topic words. This is a close approximation, not a human judgement.

## Reading results and what to change

The Results page shows, for each evaluation: the averages, the scores **by question category**, and every question with a **diagnosis**:

| Diagnosis | Meaning | What to try |
|---|---|---|
| *Expected keywords not retrieved* / low evidence recall | The search missed the right passage. | Larger chunks; more passages (top-K); *query rewriting* or *query expansion*; another embedding model. |
| Question about the document itself (authors, title, date) fails | The search matches meaning, and an author list never uses the word "author", so it ranks low. | *Include document opening*. |
| *Right passages, wrong answer* | The search worked, the model did not. | *LLM re-ranking*; *better prompt*; a stronger answer model; fewer, more focused passages. |
| *Answer not supported by passages* | The model added things that are not in the passages. | *Better prompt*; fewer passages; a stronger answer model. |
| *Nothing retrieved* | No passages at all. | Check that the files contain text (scanned PDFs need OCR). |

**Compare runs** on the Results page puts runs on the same test questions side by side, for example before and after a change: their settings, colour-coded scores, bar charts, and every question's score per run, with each run's answer and passages one click away.

## Settings and techniques

| Setting | What it does |
|---|---|
| Answer model | Writes the answer (and runs rewriting, re-ranking etc.). |
| Top-K | How many passages the model gets. More gives more context, and more noise. |
| Embedding model | Turns text into vectors for the search. |
| Chunk size and overlap | How documents are cut up. Small chunks are precise but can split a fact; large ones keep context but dilute the match. Papers and reports usually do well with 600–1000 characters (the default is 800). Overlap repeats the end of one chunk at the start of the next, so a sentence on the boundary is not lost; 10–25% of the chunk size is usually enough. Type any values; the range is set in `settings.yaml`. |

| Technique | What it does | Extra cost |
|---|---|---|
| Better prompt | Adds the date, asks the model to separate facts from uncertainty and cite passages. | none |
| Query rewriting | Rewrites the question into a cleaner search query. | 1 call |
| Original + rewritten query | Searches with both and merges the results. | 1 call |
| Query expansion | Writes 2 alternative phrasings and merges what all 3 find. | 1 call |
| LLM re-ranking | Reorders the retrieved passages by relevance before answering. Often the biggest gain. | 1 call |
| Include document opening | Also gives the model the first passage of each file found (title, authors, date, abstract). For questions about the document itself, like "who wrote this paper?", which search alone often misses because the author line never says "author". | none |
| Neighbour expansion | Adds chunks that share many terms with the retrieved ones. | none |
| LLM document clean-up | Rewrites each document section before chunking. Builds a separate index. | 1 call per section |
| Chunk summaries | Adds a summary of every chunk to the index. Builds a separate index. | 1 call per chunk |

## Benchmarks: FinanceBench

A benchmark is a public set of documents with test questions whose answers are already known. **FinanceBench** (Patronus AI) has analyst questions about real company filings (10-K and 10-Q reports): number look-ups, calculations and judgement calls. Each question has the correct answer and the exact supporting passage. The project includes 10 questions over 4 filings; the Library page can download more of the 150-question open sample (licence CC BY-NC 4.0).

*Load as a workspace* on the Library page creates a workspace with the filings and questions. Its Evaluate section offers two pipelines:

- **Generic RAG**: the same RAG as for any files, with your sidebar settings.
- **Finance-tuned**: FinanceBench's own retrieval, which keeps tables intact (line windows), searches only the filing the question names, and mixes keyword (BM25) and vector search.

Both get the same scores, so *Compare with other evaluations* shows how much the finance-specific tricks help.

## Using RAGLab from Python

```bash
pip install -e '.[viz,documents]'
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run app.py                 # the app
```

```python
from raglab import RAGLab, RAGConfig
from raglab.indexes import build_index, open_index, describe_files
from raglab.evaluation import load_tests, evaluate_dataset, LLMJudge

cfg = RAGConfig(chunk_size=800, chunk_overlap=150, top_k=5, rerank=True)
files = ["paper.pdf"]
build_index(cfg, files, "my_indexes")                      # built once, then reused
lab = open_index(cfg, "my_indexes", describe_files(files))
print(lab.ask("Who are the authors?").answer)

report = evaluate_dataset(lab, load_tests("tests.jsonl"), judge=LLMJudge())
print(report["summary"])
```

Test files are JSON Lines, one question per line:

```json
{"question": "Who founded the company?", "reference_answer": "Maria Silva, in 2015.", "keywords": ["Silva"], "evidence": [{"text": "Maria Silva founded the company in 2015.", "source": "about.md"}], "category": "people"}
```

To compare several configurations from code, call `raglab.experiments.run_experiments` with a plan: a YAML file with shared `settings` and a list of named `runs`, each switching on `techniques` and optionally overriding `settings`. Every run the app makes saves its plan in this format (`raglab_data/jobs/<id>/experiments.yaml`), so a run you made in the app is a ready-made template. Model aliases live in `models.yaml` (OpenAI and Anthropic); API keys belong in `.env`.

## Where things are stored

| Folder | Contents |
|---|---|
| `raglab_data/collections/` | Each workspace's files and test questions |
| `raglab_data/indexes/` | Prepared search indexes, one per set of files and chunk settings |
| `raglab_data/cache/` | Cached embeddings (always) and model answers (when *Reuse cached answers* is on) |
| `raglab_data/jobs/` | Each evaluation: settings, timestamped log, results |
| `raglab_data/library/` | Evaluations you saved to the library |
| `raglab_data/logs/app.log` | Questions asked in the app |

Indexes are identified by the exact file contents and the chunk settings, so the same files are never indexed twice and re-uploading a file never duplicates it. Embeddings are cached, so changing the chunk size only pays for text never embedded before.

## Limitations

- Scanned PDFs need OCR before upload; legacy `.doc` files need the `antiword` tool.
- LLM-judge scores are a model's opinion: check important results by hand.
- Evidence and keyword matching are lexical approximations, not human relevance labels.
- This is a local, single-user tool. Do not expose it publicly without authentication and upload limits.
