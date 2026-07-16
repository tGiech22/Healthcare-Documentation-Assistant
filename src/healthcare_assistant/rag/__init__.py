"""Phase 2: embeddings & retrieval (RAG).

Retrieval-Augmented Generation works in two stages:

    1. INDEX (once):  chunk documents -> embed each chunk into a vector ->
       store the vectors so they can be searched by meaning.
    2. RETRIEVE (per query):  embed the question with the same model -> find the
       chunks whose vectors are most similar -> hand those chunks to a
       summarizer/LLM as grounded context.

This package implements stage 1 and stage 2. The generation step (Phase 3) will
consume :class:`retriever.Retriever` results as its source material.

Design note: we start with a TF-IDF embedder and an in-memory cosine store --
zero new dependencies, fast, and easy to inspect. Both are pluggable, so a
clinical Sentence-Transformers model and a Chroma/pgvector backend can drop in
later without touching the chunking or orchestration code (mirrors how the
note-type classifier starts with TF-IDF before reaching for transformers).
"""
