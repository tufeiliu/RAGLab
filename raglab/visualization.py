"""Explore stored RAG embeddings in 2D/3D with UMAP or t-SNE.

Install: pip install 'raglab-experiments[viz]'
No LLM requests are needed to visualize existing vectors. Query overlays embed
one query using the configured embedder, which may incur an API request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


@dataclass
class EmbeddingMap:
    coordinates: np.ndarray
    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    selected: list[bool]
    clusters: list[str] | None = None
    query_index: int | None = None
    method: str = "umap"
    extras: dict[str, int] = field(default_factory=dict)   # label -> row of an extra text (e.g. the answer)
    ranks: dict[str, int] = field(default_factory=dict)    # selected id -> retrieval rank (1 = best)

    def dataframe(self):
        import pandas as pd
        data = {
            "id": self.ids,
            "x": self.coordinates[:, 0],
            "y": self.coordinates[:, 1],
            "source": [str(m.get("source", "unknown")) for m in self.metadatas],
            "type": [str(m.get("type", m.get("text_type", "unknown"))) for m in self.metadatas],
            "selected": self.selected,
            "cluster": self.clusters if self.clusters is not None else ["unclustered"] * len(self.ids),
            "text": self.texts,
        }
        if self.coordinates.shape[1] == 3:
            data["z"] = self.coordinates[:, 2]
        return pd.DataFrame(data)

    def figure(self, color_by: str = "source", title: str | None = None):
        """Return an interactive Plotly figure highlighting retrieved IDs and query."""
        import plotly.express as px
        import plotly.graph_objects as go
        df = self.dataframe()
        if color_by not in {"source", "type", "selected", "cluster"}:
            raise ValueError("color_by must be source, type, selected, or cluster")
        df["preview"] = df["text"].str.slice(0, 220)
        # Display regular points separately so selected points remain visible.
        special = {"__query__", *(f"__extra__{label}" for label in self.extras)}
        ordinary = df.loc[~df["selected"] & ~df["id"].isin(special)]
        labels = {"x": "Dimension 1", "y": "Dimension 2", "z": "Dimension 3"}
        args = dict(data_frame=ordinary, x="x", y="y", color=color_by,
                    hover_data={"id": True, "source": True, "type": True, "preview": True,
                                "x": False, "y": False, "selected": False},
                    labels=labels, title=title or f"Embedding space — {self.method.upper()}")
        if self.coordinates.shape[1] == 3:
            args["z"] = "z"
            args["hover_data"]["z"] = False
            fig = px.scatter_3d(**args)
            fig.update_traces(marker=dict(size=3, opacity=0.55))
        else:
            fig = px.scatter(**args)
            fig.update_traces(marker=dict(size=7, opacity=0.55))
        selected = df.loc[df["selected"] & ~df["id"].isin(special)]
        hover = [f"ID: {r.id}<br>Source: {r.source}<br>{r.preview}" for r in selected.itertuples()]
        rank_labels = [f"#{self.ranks[i]}" if i in self.ranks else "" for i in selected["id"]]
        if len(selected):
            if self.coordinates.shape[1] == 3:
                fig.add_trace(go.Scatter3d(x=selected.x, y=selected.y, z=selected.z,
                    mode="markers", name="Retrieved", text=hover, hoverinfo="text",
                    marker=dict(size=7, symbol="diamond", color="#e15759")))
            else:
                fig.add_trace(go.Scatter(x=selected.x, y=selected.y,
                    mode="markers+text" if self.ranks else "markers", name="Retrieved",
                    text=rank_labels if self.ranks else hover, hovertext=hover, hoverinfo="text",
                    textposition="top center", textfont=dict(size=13),
                    marker=dict(size=13, symbol="diamond", color="#e15759", line=dict(width=1, color="black"))))
        if self.query_index is not None:
            point = self.coordinates[self.query_index]
            if self.coordinates.shape[1] == 3:
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                    mode="markers", name="Query", marker=dict(size=10, color="#17a398", symbol="cross")))
            else:
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]],
                    mode="markers", name="Query", marker=dict(size=16, color="#17a398", symbol="star")))
        styles = {"Question": ("star", "#17a398"), "Answer": ("square", "#f28e2b"),
                  "Correct answer": ("triangle-up", "#59a14f")}
        for label, row in self.extras.items():
            point = self.coordinates[row]
            symbol, colour = styles.get(label, ("x", "#555555"))
            hover_text = f"{label}: {self.texts[row][:220]}"
            if self.coordinates.shape[1] == 3:
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers", name=label,
                    hovertext=[hover_text], hoverinfo="text", marker=dict(size=10, color=colour)))
            else:
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name=label,
                    hovertext=[hover_text], hoverinfo="text",
                    marker=dict(size=18, symbol=symbol, color=colour, line=dict(width=1, color="black"))))
        fig.update_layout(legend_title_text=color_by, margin=dict(l=20, r=20, t=55, b=20))
        return fig


def project_embeddings(vectors: Sequence[Sequence[float]] | np.ndarray, *, method: str = "umap",
                       dimensions: int = 2, n_neighbors: int = 15, min_dist: float = 0.1,
                       perplexity: float = 30.0, metric: str = "cosine", random_state: int = 42) -> np.ndarray:
    """Fit a projection. Embeddings must all come from the SAME vector space."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite, rectangular 2D embedding matrix")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if len(x) < dimensions + 2:
        raise ValueError(f"At least {dimensions + 2} vectors are required")
    method = method.lower()
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP requires: pip install umap-learn") from exc
        if not 2 <= n_neighbors < len(x):
            raise ValueError("UMAP n_neighbors must be >= 2 and less than vector count")
        if not 0 <= min_dist <= 1:
            raise ValueError("UMAP min_dist must be between 0 and 1")
        return umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors,
                         min_dist=min_dist, metric=metric, random_state=random_state).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        if not 0 < perplexity < len(x):
            raise ValueError("t-SNE perplexity must be greater than zero and less than sample count")
        return TSNE(n_components=dimensions, perplexity=perplexity, init="pca",
                    learning_rate="auto", metric=metric, random_state=random_state).fit_transform(x)
    raise ValueError("method must be 'umap' or 'tsne'")


def map_chroma(store: Any, *, method: str = "umap", dimensions: int = 2,
               query: str | None = None, embedding_function: Any = None,
               selected_ids: Sequence[str] = (), max_points: int = 1500,
               n_neighbors: int = 15, min_dist: float = 0.1,
               perplexity: float = 30.0, random_state: int = 42,
               cluster_method: str = "none", cluster_count: int = 5,
               cluster_eps: float = 0.25, extra_points: dict[str, str] | None = None) -> EmbeddingMap:
    """Project sampled Chroma vectors and optional query in the same 2D/3D fit.

    Selected IDs are included ahead of a deterministic sample (when stored); their
    order is shown as rank labels. extra_points ({label: text}, e.g. the question and
    the answer) are embedded and placed in the same projection.
    Requires Chroma IDs for exact retrieval highlighting; use metadata-based
    selection only if IDs are unavailable. The map is not a ranking metric.
    """
    if max_points < dimensions + 2:
        raise ValueError("max_points too small")
    collection = getattr(store, "_collection", store)
    count = collection.count()
    if count == 0:
        raise ValueError("The Chroma collection is empty. Ingest documents first.")
    selected_set = set(selected_ids)
    # Chroma collection.get(limit, offset) supports bounded paging.
    # Ensure selected documents are included even if corpus is large.
    records = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    selected_records = []
    if selected_set:
        found = collection.get(ids=list(selected_set), include=["embeddings", "documents", "metadatas"])
        for i, doc_id in enumerate(found["ids"]):
            selected_records.append((doc_id, found["embeddings"][i], found["documents"][i], found["metadatas"][i]))
    rows = {r[0]: r for r in selected_records}
    stride = max(1, count // max_points)
    batch = 500
    offset = 0
    # Walk through the collection once, sampling positions deterministically.
    while offset < count and len(rows) < max_points:
        take = min(batch, count - offset)
        fetched = collection.get(limit=take, offset=offset, include=["embeddings", "documents", "metadatas"])
        for j, doc_id in enumerate(fetched["ids"]):
            if (offset + j) % stride == 0 and doc_id not in rows:
                rows[doc_id] = (doc_id, fetched["embeddings"][j], fetched["documents"][j], fetched["metadatas"][j])
                if len(rows) >= max_points:
                    break
        offset += take
    if len(rows) < dimensions + 2:
        raise ValueError(f"At least {dimensions + 2} stored vectors are required for visualization")
    ids = list(rows)
    vectors = np.asarray([rows[i][1] for i in ids], dtype=np.float64)
    texts = [rows[i][2] or "" for i in ids]
    metas = [rows[i][3] or {} for i in ids]
    query_index = None
    if query and query.strip():
        if embedding_function is None:
            raise ValueError("embedding_function is required to display a query")
        qvec = np.asarray(embedding_function.embed_query(query), dtype=np.float64)
        if qvec.shape != (vectors.shape[1],):
            raise ValueError("Query embedding dimension does not match indexed document vectors")
        # Fit query in same projection; don't compare separately fitted coordinates.
        query_index = len(ids)
        vectors = np.vstack([vectors, qvec])
        ids.append("__query__")
        texts.append(query)
        metas.append({"source": "query", "type": "query"})
    documents = len(vectors) - (query_index is not None)
    extras = {}
    if extra_points:
        if embedding_function is None:
            raise ValueError("embedding_function is required to display extra points")
        labels = [label for label, text in extra_points.items() if text and text.strip()]
        for label, vec in zip(labels, embedding_function.embed_documents([extra_points[l] for l in labels])):
            vec = np.asarray(vec, dtype=np.float64)
            if vec.shape != (vectors.shape[1],):
                raise ValueError("Embedding dimension does not match indexed document vectors")
            extras[label] = len(ids)
            vectors = np.vstack([vectors, vec])
            ids.append(f"__extra__{label}")
            texts.append(extra_points[label])
            metas.append({"source": label, "type": label})
    # Cluster BEFORE projection. KMeans forces k groups; DBSCAN permits noise (-1).
    cluster_vectors = vectors[:documents]
    if cluster_method == "none":
        clusters = ["unclustered"] * len(cluster_vectors)
    elif cluster_method == "kmeans":
        from sklearn.cluster import KMeans
        if not 2 <= cluster_count <= len(cluster_vectors):
            raise ValueError("cluster_count must be between 2 and number of document vectors")
        labs = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(cluster_vectors)
        clusters = [f"Cluster {label + 1}" for label in labs]
    elif cluster_method == "dbscan":
        from sklearn.cluster import DBSCAN
        if cluster_eps <= 0:
            raise ValueError("cluster_eps must be positive")
        labs = DBSCAN(eps=cluster_eps, min_samples=3, metric="cosine").fit_predict(cluster_vectors)
        clusters = ["Noise / unassigned" if label == -1 else f"Cluster {label + 1}" for label in labs]
    else:
        raise ValueError("cluster_method must be none, kmeans, or dbscan")
    if query_index is not None:
        clusters.append("Query")
    clusters.extend(extras)
    projected = project_embeddings(vectors, method=method, dimensions=dimensions,
          n_neighbors=min(n_neighbors, len(vectors) - 1), min_dist=min_dist,
          perplexity=min(perplexity, len(vectors) - 1), random_state=random_state)
    return EmbeddingMap(projected, ids, texts, metas,
                        [i in selected_set for i in ids], clusters=clusters,
                        query_index=query_index, method=method, extras=extras,
                        ranks={doc_id: rank for rank, doc_id in enumerate(selected_ids, 1)})
