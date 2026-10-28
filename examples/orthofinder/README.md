# OrthoFinder paper example

A small, ready-made test for RAGLab: one research paper and 11 questions about it.

- `orthofinder_nature_methods_2026.pdf`: Emms, D. M., Liu, Y., Belcher, L., Holmes, J. & Kelly, S.
  *OrthoFinder: improved phylogenetic orthology inference with enhanced accuracy and scalability.*
  Nature Methods (2026). https://doi.org/10.1038/s41592-026-03126-6
- `tests.jsonl`: questions with a correct answer, the evidence passage from the paper, keywords and a category
  (*about the paper*, *results*, *methods*).

## Use it

In the app: **Library → Try it yourself → OrthoFinder paper → Load as a workspace**. Then ask a question in the
Workspace, or run all 11 in its *Test questions* tab.

From Python:

```python
from raglab.indexes import build_index, open_index, describe_files
from raglab.evaluation import load_tests, evaluate_dataset, LLMJudge
from raglab import RAGConfig

files = ["examples/orthofinder/orthofinder_nature_methods_2026.pdf"]
cfg = RAGConfig(chunk_size=800, chunk_overlap=150, top_k=5)
build_index(cfg, files, "my_indexes")
lab = open_index(cfg, "my_indexes", describe_files(files))
report = evaluate_dataset(lab, load_tests("examples/orthofinder/tests.jsonl"), judge=LLMJudge())
print(report["summary"])
```

## What to expect

Questions about the paper itself, such as *"Who is the second author?"*, are hard for plain RAG: the author line
never uses the word "author", so the search ranks other passages (like the author contributions section) higher.
Compare a run with and without the *Include document opening* technique to see the difference.

The evidence passages were checked to match the text RAGLab extracts from the PDF. With very small chunks
(around 300 characters) the list of comparator methods is split across two chunks, so that question's evidence
cannot be matched by a single chunk.
