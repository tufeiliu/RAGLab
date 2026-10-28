import numpy as np
import pytest
from raglab.visualization import project_embeddings, map_chroma


def test_invalid_method():
    with pytest.raises(ValueError, match="method"):
        project_embeddings(np.random.default_rng(42).random((20, 5)), method="pca")


def test_too_few_points():
    with pytest.raises(ValueError, match="At least"):
        project_embeddings(np.ones((3, 6)))


def test_tsne_shape():
    pytest.importorskip("sklearn")
    x = np.random.default_rng(1).normal(size=(25, 8))
    out = project_embeddings(x, method="tsne", perplexity=5)
    assert out.shape == (25, 2)
    assert np.isfinite(out).all()


def test_umap_shape():
    pytest.importorskip("umap")
    x = np.random.default_rng(1).normal(size=(25, 8))
    out = project_embeddings(x, method="umap", n_neighbors=5)
    assert out.shape == (25, 2)


def test_map_chroma_fake():
    class FakeCollection:
        def __init__(self):
            self.data = [(str(i), [float(i), float(i % 3), float(i % 5), 0.1],
                          f"Document {i}", {"source": f"s{i % 2}"}) for i in range(24)]
        def count(self): return len(self.data)
        def get(self, ids=None, limit=None, offset=0, include=None):
            result = [r for r in self.data if r[0] in ids] if ids else self.data[offset:offset+limit]
            return {"ids": [r[0] for r in result], "embeddings": [r[1] for r in result],
                    "documents": [r[2] for r in result], "metadatas": [r[3] for r in result]}
    result = map_chroma(FakeCollection(), method="tsne", perplexity=5, max_points=8,
                        selected_ids=["23"])
    assert "23" in result.ids
    assert result.selected[result.ids.index("23")]
    assert result.coordinates.shape[1] == 2


def test_map_places_question_and_answers_and_ranks_passages():
    class FakeCollection:
        def __init__(self):
            self.data = [(str(i), [float(i), float(i % 3), float(i % 5), 0.1],
                          f"Document {i}", {"source": f"s{i % 2}"}) for i in range(24)]
        def count(self): return len(self.data)
        def get(self, ids=None, limit=None, offset=0, include=None):
            result = [r for r in self.data if r[0] in ids] if ids else self.data[offset:offset+limit]
            return {"ids": [r[0] for r in result], "embeddings": [r[1] for r in result],
                    "documents": [r[2] for r in result], "metadatas": [r[3] for r in result]}

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[float(len(t)), 1.0, 2.0, 0.3] for t in texts]

    result = map_chroma(FakeCollection(), method="tsne", perplexity=5, max_points=10, selected_ids=["7", "3"],
                        extra_points={"Question": "q?", "Answer": "an answer", "Correct answer": ""},
                        embedding_function=FakeEmbedder())
    assert set(result.extras) == {"Question", "Answer"}  # empty texts are skipped
    assert result.texts[result.extras["Answer"]] == "an answer"
    assert result.ranks == {"7": 1, "3": 2}
    figure = result.figure(color_by="source")
    names = [trace.name for trace in figure.data]
    assert "Question" in names and "Answer" in names
    retrieved = next(trace for trace in figure.data if trace.name == "Retrieved")
    assert sorted(retrieved.text) == ["#1", "#2"]
