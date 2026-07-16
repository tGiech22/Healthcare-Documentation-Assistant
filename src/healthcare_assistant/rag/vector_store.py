"""A tiny in-memory vector store: hold chunk vectors and search them by cosine.

This is the "vector database" of the pipeline, kept deliberately simple. Because
every stored vector is L2-normalized (see :mod:`embeddings`), cosine similarity
is just a dot product, and searching the whole corpus is a single matrix-vector
multiply -- more than fast enough for thousands of chunks.

When the corpus outgrows memory, replace this class with a Chroma or pgvector
backend behind the same ``search`` signature; nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from healthcare_assistant.rag.chunking import Chunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and how similar it was to the query (1.0 = identical)."""

    chunk: Chunk
    score: float


class VectorStore:
    """Holds one vector per chunk and returns the closest ones to a query."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        self.vectors = vectors
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``query_vector``.

        ``query_vector`` is expected to be a single L2-normalized row (shape
        ``(1, dim)`` or ``(dim,)``), so ``vectors @ query`` yields one cosine
        score per stored chunk.
        """
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self.vectors @ query  # cosine similarity, one per chunk

        k = min(k, len(self.chunks))
        # argpartition finds the top-k cheaply, then we sort just those by score.
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> Path:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectors": self.vectors, "chunks": self.chunks}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        import joblib

        payload = joblib.load(path)
        return cls(payload["vectors"], payload["chunks"])
