"""Plain-language descriptions of every RAG setting, shared by all pages."""
from raglab.experiments import TECHNIQUE_FLAGS

# key -> (label, explanation, RAGConfig overrides from raglab.experiments.TECHNIQUE_FLAGS)
TECHNIQUES = {
    'improved_prompt': ('Better prompt',
        'Gives the model today\'s date and asks it to separate facts from uncertainty and cite passage numbers.',
        TECHNIQUE_FLAGS['improved_prompt']),
    'query_rewrite': ('Query rewriting',
        'An LLM rewrites the question into a cleaner search query before retrieval. Helps with vague or chatty questions.',
        TECHNIQUE_FLAGS['query_rewrite']),
    'query_fusion': ('Original + rewritten query',
        'Searches with both the original and the rewritten question, then merges the results.',
        TECHNIQUE_FLAGS['query_fusion']),
    'query_expansion': ('Query expansion',
        'An LLM writes 2 alternative phrasings; passages found by any of the 3 queries are merged. Finds passages that use different wording.',
        TECHNIQUE_FLAGS['query_expansion']),
    'rerank': ('LLM re-ranking',
        'After retrieval, an LLM reorders the candidate passages by relevance and keeps the best K. Often the biggest accuracy gain, at one extra call per question.',
        TECHNIQUE_FLAGS['rerank']),
    'opening': ('Include document opening',
        'Also gives the model the first passage of each file the search found: usually the title, authors, '
        'date and abstract. Helps with questions about the document itself, such as "who wrote this paper?".',
        TECHNIQUE_FLAGS['opening']),
    'graph': ('Neighbour expansion',
        'Adds up to 3 extra chunks that share many terms with the retrieved ones. A light stand-in for graph RAG.',
        TECHNIQUE_FLAGS['graph']),
    'preprocess': ('LLM document clean-up',
        'Index time: an LLM cleans each document section before it is chunked. Costly (one call per section) and builds a separate index.',
        TECHNIQUE_FLAGS['preprocess']),
    'hierarchical': ('Chunk summaries',
        'Index time: an LLM writes a summary of every chunk and indexes it alongside the chunk. One model call per '
        'chunk: a 10-page paper with small chunks means over a thousand calls and many minutes. Builds a separate index.',
        TECHNIQUE_FLAGS['hierarchical']),
}
INDEX_TIME = {'preprocess', 'hierarchical'}
QUERY_TIME = [k for k in TECHNIQUES if k not in INDEX_TIME]

HELP = {
    'embedding': 'Turns text into vectors. Questions are matched to chunks by vector similarity. '
                 'Changing it needs a new index.',
    'chat': 'Writes the answer from the retrieved passages (and runs rewriting, re-ranking etc.).',
    'judge': 'A separate model that scores each answer against the reference answer (1–5). '
             'One extra call per question.',
    'chunk_size': 'Maximum characters per chunk. Small chunks are precise but can cut facts apart; '
                  'large chunks keep context but dilute the match. Changing it needs a new index.',
    'chunk_overlap': 'Characters shared between neighbouring chunks, so a sentence on a boundary is '
                     'not lost. Changing it needs a new index.',
    'top_k': 'How many passages are given to the answer model. More gives more context but also more noise.',
    'cache_responses': 'Reuse a stored model answer when exactly the same prompt was sent before. '
                       'Makes reruns free and repeatable, but hides run-to-run variation. '
                       'Embeddings are always cached.',
}

TEST_FORMAT = '''One JSON object per line:

| Field | Required | Used for |
|---|---|---|
| `question` | yes | The question to ask |
| `reference_answer` | for answer scores | What a correct answer says; the LLM judge compares against it |
| `keywords` | for quick search scores | Words that a relevant passage must contain, e.g. `["2021", "Lisbon"]` |
| `evidence` | for precise search scores | The passage(s) containing the answer: a text, a list of texts, or `[{"text": "...", "source": "file name"}]` |
| `category` | no | Free label to group questions; results are broken down by it |
'''
TEST_EXAMPLE = (
    '{"question": "When was the company founded?", "reference_answer": "It was founded in 2015 in Lisbon.", '
    '"keywords": ["2015", "Lisbon"], "category": "facts"}\n'
    '{"question": "Who is the CEO?", "reference_answer": "Maria Silva", '
    '"evidence": [{"text": "Maria Silva has led the company as CEO since 2019.", "source": "about.md"}]}\n'
)
