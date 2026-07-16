"""Split clinical notes into small, overlapping chunks for retrieval.

RAG retrieves *pieces* of documents, not whole documents: smaller chunks let the
similarity search zero in on the sentence that actually answers a question, and
they keep each unit small enough for an LLM to read comfortably.

We use a sliding word-window with overlap. Overlap matters: without it, a fact
that straddles a boundary ("...continue lisinopril | 10mg daily...") could be cut
in half and become unretrievable. A section-aware splitter (by SUBJECTIVE /
OBJECTIVE / ASSESSMENT / PLAN headings) is a natural Phase 2+ upgrade.

Crucially, we redact PHI *before* chunking so that no protected health
information ever reaches the embedder or, later, an LLM API.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from healthcare_assistant.config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS
from healthcare_assistant.models import phi_detector


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a note.

    ``chunk_id`` is unique across the whole corpus; ``note_id`` and ``note_type``
    are carried along so a retrieved chunk can be traced back to its source note
    (essential for grounded, citable summaries).
    """

    chunk_id: str
    note_id: object
    note_type: str
    text: str


def chunk_text(
    text: str,
    *,
    max_words: int = CHUNK_MAX_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split ``text`` into overlapping windows of at most ``max_words`` words."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    words = text.split()
    if not words:
        return []

    step = max_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break  # last window already reached the end
    return chunks


def chunk_notes(
    df: pd.DataFrame,
    *,
    redact: bool = True,
    max_words: int = CHUNK_MAX_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Turn a notes DataFrame into a flat list of :class:`Chunk`.

    Expects the standard schema (``note_id``, ``note_type``, ``text``). When
    ``redact`` is True (the default), each note is de-identified with the Phase 1
    PHI detector before it is split -- keep this on for anything that leaves the
    machine (embeddings sent to an API, LLM prompts, logs).
    """
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        text = phi_detector.redact(row["text"]) if redact else row["text"]
        for i, piece in enumerate(chunk_text(text, max_words=max_words, overlap=overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{row['note_id']}::{i}",
                    note_id=row["note_id"],
                    note_type=row["note_type"],
                    text=piece,
                )
            )
    return chunks
