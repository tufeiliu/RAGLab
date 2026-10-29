# RAGLab guide

RAGLab answers questions about your own documents with **retrieval-augmented generation (RAG)**, and measures how well that works, so you can find the setup that answers your questions best.

## Overview

### How RAG works

RAG answers a question in two steps:

1. **Search.** Your files are split into short pieces (*chunks*), and each chunk is turned into a vector (an *embedding*) that captures its meaning. The question is turned into a vector too, and the *top-K* chunks closest to it are picked.
2. **Answer.** A language model writes the answer from those chunks only, and is told to say so when they are not enough.

```text
question ─▶ [rewrite / expand the query] ─▶ search the index ─▶ candidate chunks ─▶ [re-rank] ─▶ top-K chunks
                                                                                                   │
                                     answer ◀─ answer model ◀─ [+ document opening, neighbours] ◀──┘
```

The steps in brackets are optional **techniques**. So when an answer is wrong, either the search did not find the right passage, or it did and the model still got it wrong. RAGLab scores the two separately, so you know which one to fix.

### What you can use it for

- **Ask questions about a set of documents**: papers, reports, policies, manuals, filings. Every answer shows the passages it came from.
- **Find the best setup before you build a RAG system**: try chunk sizes, top-K, models and techniques on your own documents and questions, and keep the combination that scores best.
- **Check a change**: run the same test questions before and after changing a setting, a model or the documents, and compare.
- **Compare models**: the same questions with different answer or embedding models, with cost and speed side by side.
- **Learn how RAG behaves**: see which passages were found, where they rank, and why an answer failed, question by question.
- **Automate it**: `find_best_rag` in Python searches many setups in one call and returns the winning pipeline (see **Python API**).

### Quick start

1. Install and add a key (see **Python API** for details): `pip install -e '.[viz,documents]'`, put `OPENAI_API_KEY=sk-...` in `.env`, then `streamlit run app.py`.
2. On the **Library** page, open *Try it yourself → OrthoFinder paper → Load as a workspace*. This gives you one research paper and 11 test questions.
3. In the **Workspace**, ask *"Who is the second author of this paper?"* and look at the passages it used.
4. Open the *Test questions* tab and press *Run*. Read the scores and the failing questions.
5. Switch on *Include document opening* in the sidebar, run again, and compare the two runs in the *Runs* list.

### Words used in this guide

| Word | Meaning |
|---|---|
| **Chunk** / passage | A piece of a document, a few hundred characters long. The unit that search finds and the model reads. |
| **Embedding** | A vector representing the meaning of a text. Search compares the question's embedding with the chunks'. |
| **Index** | All chunks of a set of files with their embeddings, ready to search. Built once per set of files, chunk settings and embedding model. |
| **Top-K** | How many chunks the answer model gets. |
| **Answer key** | What you know about the correct answer: the answer itself, the passage containing it, or keywords. |
| **Reference answer** | The correct answer, written by you. |
| **Evidence** | The passage from the document that contains the answer. |
| **Judge** | A language model that scores each answer, 1 to 5. |
| **Run** | One evaluation: a set of questions answered with one setup. |
| **Setup** | One combination of settings and techniques. |
| **Workspace** | One topic: its files, test questions and runs. |

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
   - **Ask a question.** Type a question and, optionally, its answer key. *Ask and evaluate* returns the answer and the chunks it was based on, then the evaluation of that answer: scorecards and charts. The first time, the files are prepared for search (a progress bar shows this). *Save as a test question* keeps the question for the other tabs, and *Show on a map* places the question, answer and chunks in a UMAP / t-SNE projection.
   - **Test questions.** Write, edit or upload questions with known answers, then *Run* them all with the current settings. Every run is kept. You get the averaged evaluation, the scores per category, and every question with its answer, its chunks and a diagnosis.
   - *Run several setups in one go* (inside the Test questions tab) is a shortcut for a batch of runs, for example each technique against a baseline, without changing the sidebar each time.
3. **Runs.** Every question you asked and every run of test questions in this workspace, with its status, settings, headline scores and date. Tick runs to **compare** them (opens the comparison on the Results page), **open** one in full, or **delete** the ones you no longer need. Questions you asked can be compared with each other, and runs of a test set with other runs of that test set.

   Runs continue in the background; their results appear in the tab, and all runs are kept on the Results page.

### The sidebar

The **sidebar** holds the RAG settings and techniques (see **Settings** and **Techniques**). They apply to both *Ask* and the test runs, so an evaluation measures exactly the setup you have been trying. Their starting values come from **`settings.yaml`** in the project folder: the models, chunk size and overlap, top-K, techniques, whether the judge is on, and the allowed ranges. Edit it to change the defaults for everyone using this copy of RAGLab; a mistake in it is reported at the top of the sidebar and the built-in default is used instead.

### A typical session

1. Create a workspace and upload your files.
2. Ask a few questions and read the sources. Is the right passage there?
3. For questions where you know the answer, fill in the answer key and *Save as a test question*. Aim for 20 or more, covering the kinds of questions you care about.
4. Run them in the *Test questions* tab. Look at the scores, then at the failing questions.
5. Change one setting in the sidebar (a chunk size, or a technique that addresses those failures) and run again. Repeat for each idea.
6. In the *Runs* list, tick those runs and press *Compare*: the Results page shows which settings work best. Keep those in the sidebar, and delete the runs you no longer need.

## Test questions

Scores are only as good as the questions behind them. A test question is a question you know something about the answer to: the **answer key**.

### The three kinds of answer key

Each question can have any of three, and each one enables different scores. Use as many as you can; they check different things.

| Answer key | Field | What it scores | Cost |
|---|---|---|---|
| **Correct answer** | `reference_answer` | The answer: the judge compares it with yours (accuracy, completeness). The same call also judges the search: which retrieved passages contain the answer's information (the *context* scores). | One judge call per question |
| **Passage that contains the answer** | `evidence` | The search, precisely: was this passage among the retrieved chunks, and how high? (the *evidence* scores) | Free |
| **Keywords** a relevant passage must contain, such as a name or a year | `keywords` | The search, roughly: were passages with these words retrieved? (the *keyword* scores) | Free |

Relevance and faithfulness are judged for every question that has a correct answer. In Python, questions with no answer key at all can still be judged (see **Python API**).

### File format

Test files are JSON Lines: one JSON object per line.

| Field | Required | Used for |
|---|---|---|
| `question` | yes | The question to ask |
| `reference_answer` | for answer scores | What a correct answer says |
| `keywords` | for quick search scores | A list, or a comma-separated string: `["2021", "Lisbon"]` |
| `evidence` | for precise search scores | A text, a list of texts, or `[{"text": "...", "source": "file name"}]` |
| `category` | no | A label to group questions; results are broken down by it |

```json
{"question": "Who founded the company?", "reference_answer": "Maria Silva, in 2015.", "keywords": ["Silva"], "evidence": [{"text": "Maria Silva founded the company in 2015 after ten years at a consultancy in Porto.", "source": "about.md"}], "category": "people"}
```

### Writing good test questions

- **Ask what your users will ask**, in their words. Mix the kinds: single facts, numbers, questions about the document itself (authors, dates), questions that need two passages, and summaries.
- **Use 20 or more.** With 10 questions, one question is 10% of the score; differences between setups smaller than that are often chance. Put at least 5 in each `category` you want to read separately.
- **Write reference answers as plain statements** containing the facts that matter ("Yi Liu."), not long essays. Completeness is measured against everything the reference says.
- **Copy evidence from the document**, at least one full sentence (8 content words or more; shorter passages are ignored). Add `source` with the file name when several files could contain similar text.
- **Pick distinctive keywords**: names, numbers, rare terms. A common word like "method" appears everywhere and makes the keyword scores look better than they are.
- **Avoid questions whose correct answer is "the documents don't say".** The judge counts any answer that declines as wrong when a reference answer exists, so these questions always score 1.

## Settings

| Setting | What it does | Needs a new index |
|---|---|---|
| **Answer model** | Writes the answer, and runs the techniques that use a model (rewriting, re-ranking, ...). It also judges the answers in the app. | no |
| **Embedding model** | Turns text into vectors for the search. | yes |
| **Chunk size** | The maximum number of characters per chunk. | yes |
| **Chunk overlap** | Characters repeated at the start of the next chunk. | yes |
| **Top-K** | How many chunks the answer model gets. | no |
| **Judge** | Score answers with the judge (one extra call per question with a correct answer). | no |
| **Reuse cached answers** | Return the stored model answer when exactly the same prompt was sent before. Makes reruns free and repeatable, but hides run-to-run variation. Embeddings are always cached. | no |

Building a new index costs embedding calls for text never embedded before, and only once: indexes and embeddings are kept and reused.

### How to choose

- **Chunk size.** Small chunks (200–400 characters) match a question precisely but can cut a fact or a table row apart. Large chunks (1000+) keep context but dilute the match with unrelated text. Papers and reports usually do well with 600–1000; the default is 800. Try two values on either side of where you are.
- **Chunk overlap.** Keeps a sentence on a chunk boundary in one piece. 10–25% of the chunk size is usually enough; more makes the index bigger without helping.
- **Top-K.** More chunks give the answer more chances to contain the right passage (higher evidence recall), but also more noise for the model to ignore, and a longer, slower prompt. 3–8 is typical. Raise it when evidence recall is low; lower it, or add re-ranking, when the right passage is found but the answer is still wrong.
- **Embedding model.** A larger embedding model can find passages that use different words from the question. Compare on evidence recall and MRR.
- **Answer model.** A stronger model reads long or messy context better and follows "say so if the context is not enough" more reliably. Compare on accuracy and faithfulness, together with latency and cost.

Model names in the app and in Python are aliases from `models.yaml` (OpenAI and Anthropic models are listed); add your own there. API keys go in `.env`.

## Techniques

Techniques are optional steps added to the basic search-then-answer pipeline. Switch them on in the sidebar, in *Run several setups in one go*, or by name in Python.

| Technique | Python name | Stage | Extra cost per question | Main use |
|---|---|---|---|---|
| Better prompt | `improved_prompt` | answer | none | Hedged, cited answers |
| Query rewriting | `query_rewrite` | search | 1 call | Vague or chatty questions |
| Original + rewritten query | `query_fusion` | search | 1 call | Rewriting without losing the original wording |
| Query expansion | `query_expansion` | search | 1 call | Documents worded differently from the question |
| LLM re-ranking | `rerank` | search | 1 call | Right passage found but ranked too low |
| Include document opening | `opening` | context | none | Questions about the document itself |
| Neighbour expansion | `graph` | context | none | Answers spread over related passages |
| LLM document clean-up | `preprocess` | index | 1 call per document section, once | Messy extracted text |
| Chunk summaries | `hierarchical` | index | 1 call per chunk, once | Long, dense passages |

Techniques can be combined. Re-ranking pairs well with query expansion and fusion: those find more candidates, and re-ranking picks the best of them.

**Search scores only count the first K passages.** Document opening and neighbour expansion add passages *after* the top K, so their effect shows in the answer scores, not in evidence recall or MRR.

### Better prompt

**What it does.** Adds today's date to the instructions, asks the model to separate confirmed facts from uncertainty, and to cite passage numbers like [1].

**Helps when** answers state guesses as facts, or you want to check an answer against its passages. Dates help with "how long ago" questions.

**Watch out:** answers become longer, which the judge may score lower on relevance.

### Query rewriting

**What it does.** The model rewrites the question into one standalone search query, and the search uses only that query. The answer is still written for the original question.

**Helps when** questions are vague, conversational or full of filler ("so I was wondering, what did they find about...").

**Watch out:** the rewrite can drop a detail that mattered. If evidence recall drops, use *Original + rewritten query* instead.

### Original + rewritten query

**What it does.** Rewrites the question as above, then searches with both the original and the rewritten query. The two result lists are merged by *reciprocal rank fusion*: passages that rank high in both come first.

**Helps when** rewriting finds some passages but loses others. The safer choice of the two rewriting techniques.

### Query expansion

**What it does.** The model writes 2 alternative phrasings of the question. All 3 queries are searched, and the results are merged by reciprocal rank fusion, so passages that several phrasings find come first.

**Helps when** the documents use different words from the question ("revenue" vs "net sales", "author" vs a list of names). Raises evidence recall.

**Watch out:** an unhelpful phrasing can pull in off-topic passages. Combine with re-ranking to filter them.

### LLM re-ranking

**What it does.** The search returns twice as many candidates as top-K. The model reads them (the first 1200 characters of each) and orders them by relevance to the question, and the best K are kept.

**Helps when** the right passage is retrieved but not near the top (low MRR), or the top-K includes passages on the same topic that don't answer the question. Often the biggest single gain in accuracy.

**Watch out:** one more call per question; slower. If evidence recall is low, re-ranking cannot help: the passage is not among the candidates.

### Include document opening

**What it does.** For each file among the retrieved passages (up to 3), also gives the model that file's first chunk, which is usually the title, authors, date and abstract.

**Helps when** questions are about the document itself: "who wrote this paper?", "when was it published?". Search alone misses these, because the author line never contains the word "author". In the OrthoFinder example this turns the second-author question from wrong to right.

**Watch out:** adds up to 3 passages to every prompt, even when they aren't needed.

### Neighbour expansion

**What it does.** Adds up to 3 more chunks from the whole index that share the most words (4 letters or more) with the retrieved ones. A light stand-in for graph RAG: no knowledge graph is built.

**Helps when** an answer needs related passages that the question itself doesn't match, such as the next part of a method description.

**Watch out:** in a document on one topic, many chunks share words, so the added chunks can be off-topic. Check faithfulness and accuracy against a run without it.

### LLM document clean-up

**What it does.** At index time, the model rewrites each document section (for a PDF, each page) to clean the text for search, keeping all facts, numbers and row labels. Builds a separate index.

**Helps when** extracted text is messy: broken lines, hyphenation, scrambled tables.

**Watch out:** costly on large files (one call per section), and the model may drop or change details. Evidence passages copied from the original may no longer match the cleaned chunks, so evidence scores can fall even when answers improve.

### Chunk summaries

**What it does.** At index time, the model writes a summary of every chunk and indexes it next to the chunk. A question can then match a chunk's summary when it doesn't match its wording. Builds a separate index.

**Helps when** chunks are long and dense, and questions are about their gist.

**Watch out:** one call per chunk: a 10-page paper with small chunks means over a thousand calls and many minutes. Summaries count as passages, so they take places in the top-K and don't match evidence passages.

### Switching techniques on

- **App:** tick them in the sidebar, or list several setups in *Run several setups in one go*.
- **`find_best_rag`:** `"techniques": [[], ["rerank"], ["rerank", "opening"]]` tries each set (here: none, re-ranking, re-ranking plus opening).
- **`RAGConfig`:** `RAGConfig(rerank=True, opening=True, query_expansion=3)`. The fields have the Python names above, except that query expansion takes the total number of queries (3 = the question plus 2 phrasings), and *Original + rewritten query* is `query_rewrite=True, query_fusion=True`.

## Scores

Every question gets its own scores; a run shows their **average** over the questions that have them. A question that failed with an error (for example an API error) is counted as *failed* and left out of the averages.

### Answer scores

The **judge** reads the question, the answer, the retrieved passages and the reference answer, and scores 1 (bad) to 5 (excellent). In the app the judge is the answer model; in Python you can choose another one.

| Score | The judge asks | Needs |
|---|---|---|
| **Accuracy** | Is the answer correct compared with the reference? Factually wrong gives 1. | Correct answer |
| **Completeness** | Does it contain everything the reference says? | Correct answer |
| **Relevance** | Does it directly answer the question that was asked? | (judged for every judged question) |
| **Faithfulness** | Is every claim supported by the retrieved passages, i.e. not made up? Judged independently of correctness. | Retrieved passages |
| **Answered** | Share of questions that got a real answer rather than "the documents don't say" (0–1). Shown in the Python leaderboard. | (judged for every judged question) |

An answer that declines ("the context does not say") when the reference does give an answer scores 1 for accuracy and completeness, even if the passages really lacked the information: from the user's point of view the question went unanswered.

**How to read them.** Accuracy is the headline: is the system right? Faithfulness is the safety score: a low value means the model adds things that are not in your documents. A correct but unfaithful answer came from the model's own knowledge; that may be fine for general facts, but not for your documents' specifics. Relevance below 4 usually means the answer dodges the question or buries it in other text.

### Search scores

Search scores (0–1, higher is better) measure whether the **first K** retrieved passages contain the answer, before the model writes anything. There are three families, one per kind of answer key, with the same ideas:

| Score | Meaning | Example |
|---|---|---|
| **Recall** | Share of the needed passages that were found. | 2 evidence passages, 1 found: 0.5 |
| **Precision** | Share of the retrieved passages that are needed. | Top-K 5, 1 relevant: 0.2 |
| **Hit rate** | Share of questions where anything needed was found. | 8 of 10 questions: 0.8 |
| **MRR** | 1 divided by the rank of the first relevant passage, averaged. 1 = first, 0.5 = second, 0 = not found. | First relevant at rank 3: 0.33 |
| **nDCG** | Like MRR, but counts every relevant passage, discounted the lower it ranks. 1 = all relevant passages at the top. | |

- **Evidence scores** (`evidence_recall`, `evidence_precision`, `evidence_hit`, `evidence_mrr`, `evidence_ndcg`) compare the retrieved chunks with your evidence passages. The most objective search scores; **evidence recall** is the most important one.
- **Context scores** (`context_recall`, `context_precision`, `context_hit`, `context_mrr`, `context_ndcg`) use the judge's view of which retrieved passages contain the correct answer's information. You get them with just a correct answer, no evidence needed. The app shows them only when there is no evidence, since gold passages are more objective than a model's opinion. Context recall is the judge's 1–5 rating, mapped to 0–1.
- **Keyword scores** (`keyword_coverage`, `mrr`, `ndcg`) use keywords: coverage is the share of keywords found in any retrieved passage, MRR and nDCG are averaged over keywords. Quick to set up, but less exact: a keyword can appear in an unrelated passage.

**Precision is naturally low.** With top-K 5 and one needed passage, a perfect search scores 0.2. Compare precision between runs with the same top-K, not against 1.

### How evidence matching works

A chunk counts as matching an evidence passage when they share enough content words (numbers count too, and "1,577.0" equals "1577"), measured against the shorter of the two, and when the chunk comes from the passage's file, if that is given. Short texts must share more: 60% for 15 words or fewer, falling to 30% for 60 words or more, so a short sentence isn't matched by chunks that merely use the same topic words. Texts with fewer than 8 content words are not matched at all. This is a close approximation, not a human judgement: on FinanceBench it agrees with the benchmark's own labels on 19 of 20 questions.

### Good, fair, poor

Scores get a badge: search ≥ 0.9 and answers ≥ 4.5 are *good*; below 0.75 and 4.0 *need work* (evidence precision: 0.6 and 0.4). These are rules of thumb, not standards. What counts as good depends on your documents and questions.

### Reading scores together

| What you see | What it means | What to try |
|---|---|---|
| Low evidence recall | The search does not find the passage. Nothing downstream can fix this. | See **Improving results**: chunking, top-K, query techniques, embedding model |
| High recall, low MRR | Found, but ranked low; with a smaller top-K it would be lost. | Re-ranking |
| High recall, low accuracy | The model had the passage and still got it wrong. | Re-ranking, better prompt, stronger answer model, smaller top-K |
| High accuracy, low faithfulness | Right answers from the model's own knowledge, not your documents. | Check recall; better prompt |
| High faithfulness, low accuracy | The answer follows passages that are wrong for the question. | Check which passages were used; improve the search |
| Low answered share | The model often says the documents don't say. | Usually the search: check recall, raise top-K |
| Good scores overall, poor in one category | That kind of question needs its own fix. | Look at that category's questions, e.g. *document opening* for questions about the paper |

### Is a difference real?

- With 10 questions, one question changing is 0.1 in a search score and up to 0.4 in an average answer score. Treat smaller differences as noise.
- Model answers vary between runs. Run the best two setups again (with *Reuse cached answers* off) before deciding.
- Look at the questions that changed, not only the averages: one fixed question and one broken question average out to no change.
- The judge is a model, not a ground truth. Models tend to be lenient when judging their own answers; for decisions that matter, judge with a different, stronger model (Python: `judge="claude_sonnet"`) and read the failing questions yourself.

### Which score to optimise

- You have **correct answers**: accuracy, then faithfulness. This is what `find_best_rag` ranks by.
- You only have **evidence passages or keywords**: evidence MRR (or recall), then check answers by hand. This costs no judge calls.
- You have **no answer key**: relevance, faithfulness and the answered share (Python), which show whether answers are on-topic and grounded, not whether they are correct.

## Improving results

### The loop

1. Run your test questions with the default settings. This is the baseline.
2. Sort out **search** first: if evidence recall (or context recall) is low, no answer model can compensate.
3. Then the **answers**: accuracy and faithfulness, once the right passages arrive.
4. Change **one thing per run**, so you know what helped. Compare each run with the baseline.
5. Keep what helped, and try combinations of the winners. Or let `find_best_rag` try the combinations for you.

### Diagnoses

Every question on the Results page has a diagnosis:

| Diagnosis | Meaning | What to try |
|---|---|---|
| *Expected keywords not retrieved* / low evidence recall | The search missed the right passage. | Larger chunks; a higher top-K; *query expansion* or *original + rewritten query*; another embedding model. |
| Question about the document itself (authors, title, date) fails | The search matches meaning, and an author list never uses the word "author", so it ranks low. | *Include document opening*. |
| *Wrong answer* | The answer disagrees with the reference. | Check the passages: if the right one is there, see the next row; if not, the first row. |
| *Right passages, wrong answer* | The search worked, the model did not. | *LLM re-ranking*; *better prompt*; a stronger answer model; fewer, more focused passages. |
| *Answer not supported by passages* | The model added things that are not in the passages. | *Better prompt*; fewer passages; a stronger answer model. |
| *Nothing retrieved* | No passages at all. | Check that the files contain text (scanned PDFs need OCR). |

### Common situations

- **Tables and numbers come back wrong.** Chunks cut tables apart: try larger chunks. For financial filings, see the *Finance-tuned* pipeline in **Benchmarks**.
- **Long answers that list several things miss some.** Raise top-K; consider query expansion. Completeness shows this.
- **Answers are right but slow.** Compare latency in *Compare runs*: re-ranking and query techniques each add a model call; a smaller answer model may score the same.
- **Everything scores well.** Your questions may be too easy. Add harder ones: two-passage questions, numbers, questions about specific sections.

### Comparing runs

**Compare runs** on the Results page puts runs on the same test questions side by side: their settings, colour-coded scores, bar charts, and every question's score per run, with each run's answer and passages one click away. Tick runs in the Workspace's *Runs* list, or on the Results page, then press *Compare*.

## Python API

```bash
pip install -e '.[viz,documents]'
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run app.py                 # the app
```

### Find the best setup in one call

`find_best_rag` runs every combination of the settings you list on your test questions, scores them exactly as the app does, and ranks them:

```python
from raglab import find_best_rag

search = find_best_rag(
    documents=["paper.pdf"],                 # files, or a folder
    tests="tests.jsonl",                     # or one question, or a list of questions
    search={"chunk_size": [400, 800],
            "top_k": [3, 6],
            "techniques": [[], ["rerank"], ["rerank", "opening"]]},
)
print(search.leaderboard)                    # DataFrame, best first
print(search.best)                           # name, settings, techniques, scores
rag = search.best_rag()                      # the winning pipeline
print(rag.ask("Who are the authors?").answer)
search.save("report.json")                   # every answer and retrieved passage
```

To try it on the OrthoFinder paper, run `python examples/find_best_rag.py`:

- `--dry-run` lists the setups, with no API calls;
- `--quick` searches 2 setups, about 45 model calls;
- `--tests one` or `--tests list` uses a single question or a plain list instead of the answer-key file.

| Argument | Meaning | Default |
|---|---|---|
| `documents` | File paths, or a folder (every supported file in it) | required |
| `tests` | Test questions, see below | required |
| `search` | The values to try, see below. `None` runs the defaults once | `None` |
| `chat`, `embedding` | Model aliases from `models.yaml`, unless `search` varies them | `openai_gpt5_mini`, `openai_small` |
| `judge` | `True`: the answer model judges; a model alias: that model judges; `False`: no judge | `True` |
| `metric` | Rank by this summary score instead of the automatic choice, e.g. `"faithfulness"` | automatic |
| `workdir` | Where indexes and caches are kept; the app's folder, so the cache is shared | `raglab_data` |
| `cache_responses` | Reuse stored model answers for identical prompts | `False` |
| `progress` | A callback receiving progress events (`stage`, `current_run`, `question_done`, ...) | none |

**What can be searched.** `chunk_size`, `chunk_overlap` (default: 20% of each chunk size), `top_k`, `chat` and `embedding` (lists of aliases), and `techniques`: a list of technique sets, each a list of technique names from **Techniques**. Every combination runs, up to 64 setups.

**Test questions** can be:
- a JSONL file (see **Test questions**);
- a single question: `"Who wrote the paper?"`;
- a list mixing plain questions, dicts (`{"question": ..., "reference_answer": ..., "evidence": ..., "keywords": ...}`) and `TestCase` objects.

**Questions without an answer key** are judged without one. The judge rates whether the answer is relevant, whether it is faithful to the retrieved passages, and which passages help answer it (context precision). There is no correctness score, so a confident wrong answer that the passages support can still score well: add correct answers for the questions that matter most. An answer that only says "the documents do not say" gets relevance 1, and *answered* reports the share of questions actually answered. With `judge=False`, every question needs an answer key.

**One or two questions** work, but the winner is then the best setup for those questions only. With fewer than 5 a warning is logged; ten or more give a ranking you can trust.

**How the best is chosen.** The first of these scores that every setup has decides:
- *accuracy* (needs correct answers and the judge);
- *evidence MRR*, then *evidence recall* (needs evidence);
- *context recall*;
- keyword *MRR*;
- *relevance*, then *faithfulness*, then *context precision* (questions judged without an answer key).

Setups where questions failed rank last. Ties go to the next score, then to the faster setup.

**Cost:** setups × questions answers, plus as many judge calls, plus the calls of the techniques used. Indexes and embeddings are cached under `workdir`, so searching again over the same files only pays for answers. `plan_search(search)` returns the plan without running anything, to check what will run.

### The result

| Attribute | Contents |
|---|---|
| `leaderboard` | A pandas DataFrame, one row per setup, best first: settings, techniques, main scores, latency, failures |
| `best` | The winning setup: name, models, settings, techniques and summary scores |
| `ranking` | The same for every setup, best first |
| `metric` | The score the setups were ranked by |
| `best_rag()` | The winning pipeline, ready to `.ask()`; its index is already built |
| `rag(n)` | Any setup's pipeline, by rank (0 = best) or by name |
| `save(path)` | Writes the full report as JSON: every answer, retrieved passage and score |
| `report` | The full report as a dict |

### Building blocks

```python
from raglab import RAGLab, RAGConfig
from raglab.indexes import build_index, open_index, describe_files
from raglab.evaluation import load_tests, evaluate_dataset, LLMJudge

cfg = RAGConfig(chunk_size=800, chunk_overlap=150, top_k=5, rerank=True)
files = ["paper.pdf"]
build_index(cfg, files, "my_indexes")                      # built once, then reused
lab = open_index(cfg, "my_indexes", describe_files(files))
result = lab.ask("Who are the authors?")
print(result.answer)                                       # the answer
print(result.documents)                                    # the passages it was given
print(result.trace)                                        # what each technique did

report = evaluate_dataset(lab, load_tests("tests.jsonl"), judge=LLMJudge())
print(report["summary"])                                   # averaged scores
```

`LLMJudge(without_reference=True)` also judges questions without a correct answer.

For full control, call `raglab.experiments.run_experiments` with a plan (what `find_best_rag` does underneath): a YAML file or dict with shared `settings` and a list of named `runs`, each switching on `techniques` and optionally overriding `settings`. Every run the app makes saves its plan in this format (`raglab_data/jobs/<id>/experiments.yaml`), so a run you made in the app is a ready-made template.

## Benchmarks

A benchmark is a public set of documents with test questions whose answers are already known. **FinanceBench** (Patronus AI) has analyst questions about real company filings (10-K and 10-Q reports): number look-ups, calculations and judgement calls. Each question has the correct answer and the exact supporting passage. The project includes 10 questions over 4 filings; the Library page can download more of the 150-question open sample (licence CC BY-NC 4.0).

*Load as a workspace* on the Library page creates a workspace with the filings and questions. Its Evaluate section offers two pipelines:

- **Generic RAG**: the same RAG as for any files, with your sidebar settings.
- **Finance-tuned**: FinanceBench's own retrieval, which keeps tables intact (line windows), searches only the filing the question names, and mixes keyword (BM25) and vector search.

Both get the same scores, so comparing them shows how much the finance-specific tricks help.

The Library also has the **OrthoFinder paper**: one research paper and 11 questions about it, with correct answers and evidence. It is small and quick, and a good first test.

## Storage and limits

### Where things are stored

| Folder | Contents |
|---|---|
| `raglab_data/collections/` | Each workspace's files and test questions |
| `raglab_data/indexes/` | Prepared search indexes, one per set of files, chunk settings and embedding model |
| `raglab_data/cache/` | Cached embeddings (always) and model answers (when *Reuse cached answers* is on) |
| `raglab_data/jobs/` | Each evaluation: settings, timestamped log, results |
| `raglab_data/library/` | Evaluations you saved to the library |
| `raglab_data/logs/app.log` | Questions asked in the app |

Indexes are identified by the exact file contents and the chunk settings, so the same files are never indexed twice and re-uploading a file never duplicates it. Embeddings are cached, so changing the chunk size only pays for text never embedded before.

### Limitations

- Scanned PDFs need OCR before upload; legacy `.doc` files need the `antiword` tool.
- Judge scores are a model's opinion: check important results by hand.
- Evidence and keyword matching are word-overlap approximations, not human relevance labels.
- In the app, the answer model also judges; use a separate judge from Python when it matters.
- This is a local, single-user tool. Do not expose it publicly without authentication and upload limits.
