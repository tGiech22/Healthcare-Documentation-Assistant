"""Turn text into vectors so it can be searched by meaning.

An *embedder* maps text -> a fixed-length vector of numbers, arranged so that
texts with similar meaning land near each other. Retrieval then reduces to "find
the stored vectors closest to the query vector".

We ship a TF-IDF baseline (scikit-learn, already a dependency). TF-IDF is really
lexical, not semantic -- it matches on shared words rather than paraphrases -- but
it is fast, transparent, needs no downloads, and is the honest baseline to beat
before adding a clinical Sentence-Transformers model.

Swapping backends: anything with ``.fit(texts)`` and ``.embed(texts) -> np.ndarray``
of L2-normalized rows satisfies :class:`Embedder` and works with the rest of the
package unchanged. A sketch of the upgrade::

    class SentenceTransformerEmbedder:
        def __init__(self, model="pritamdeka/S-BioBert-snli-multinli-stsb"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model)
        def fit(self, texts): pass  # pretrained; nothing to fit
        def embed(self, texts):
            return self._model.encode(list(texts), normalize_embeddings=True)
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    """Minimal interface the vector store and retriever depend on."""

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized row vector per input text."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity.

    Rows that are all zeros (e.g. a query with no known vocabulary) are left as
    zeros rather than divided by zero -- they simply score 0 against everything.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """TF-IDF vectors as a retrieval baseline."""

    def __init__(self) -> None:
        # Same feature choices as the note-type classifier, for consistency.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,  # small corpora: don't drop rare-but-meaningful terms
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        """Learn the vocabulary and IDF weights from the chunk corpus."""
        self._vectorizer.fit(list(texts))
        self._fitted = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorize ``texts`` with the fitted vocabulary."""
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        dense = self._vectorizer.transform(list(texts)).toarray().astype(np.float32)
        # TfidfVectorizer L2-normalizes by default, but query rows can be all-zero
        # when a word is out-of-vocabulary, so normalize defensively.
        return _l2_normalize(dense)
