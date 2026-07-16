"""Orchestration: build a searchable index from notes, then retrieve for a query.

This is the public entry point for Phase 2. It wires the three pieces together:

    notes DataFrame -> chunk (+redact) -> embed -> VectorStore
    query string    -> embed with the *same* embedder -> VectorStore.search

Phase 3 (summarization) will call :meth:`Retriever.retrieve` to gather grounded
context, then pass those chunks to an LLM with a "use only these notes" prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from healthcare_assistant.config import RETRIEVAL_TOP_K, VECTOR_INDEX_PATH
from healthcare_assistant.rag.chunking import chunk_notes
from healthcare_assistant.rag.embeddings import Embedder, TfidfEmbedder
from healthcare_assistant.rag.vector_store import SearchResult, VectorStore


@dataclass
class Retriever:
    """A fitted embedder paired with the vector store it produced."""

    embedder: Embedder
    store: VectorStore

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        *,
        embedder: Embedder | None = None,
        redact: bool = True,
    ) -> "Retriever":
        """Chunk, embed, and index a notes DataFrame.

        Uses the TF-IDF baseline unless another :class:`Embedder` is supplied.
        PHI is redacted before embedding by default.
        """
        embedder = embedder or TfidfEmbedder()
        chunks = chunk_notes(df, redact=redact)
        if not chunks:
            raise ValueError("No chunks produced -- is the notes DataFrame empty?")

        texts = [c.text for c in chunks]
        embedder.fit(texts)
        vectors = embedder.embed(texts)
        return cls(embedder=embedder, store=VectorStore(vectors, chunks))

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> list[SearchResult]:
        """Return the ``k`` chunks most relevant to ``query``."""
        query_vector = self.embedder.embed([query])
        return self.store.search(query_vector, k=k)

    def save(self, path: Path = VECTOR_INDEX_PATH) -> Path:
        """Persist the whole retriever (fitted embedder + store) to one file."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"embedder": self.embedder, "store": self.store}, path)
        return path

    @classmethod
    def load(cls, path: Path = VECTOR_INDEX_PATH) -> "Retriever":
        import joblib

        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Run `python scripts/04_build_index.py` first."
            )
        payload = joblib.load(path)
        return cls(embedder=payload["embedder"], store=payload["store"])
