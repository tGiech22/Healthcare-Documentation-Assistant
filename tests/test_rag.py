"""Smoke tests for the Phase 2 retrieval (RAG) pipeline.

Like test_pipeline.py, these run on a small in-memory dataset -- no files needed.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from healthcare_assistant.data.generate_synthetic import generate_notes
from healthcare_assistant.rag.chunking import chunk_notes, chunk_text
from healthcare_assistant.rag.retriever import Retriever


@pytest.fixture(scope="module")
def notes() -> pd.DataFrame:
    df = generate_notes(n_per_type=40, seed=0)
    df["phi"] = df["phi"].apply(json.loads)
    return df


def test_chunk_text_overlaps_and_covers() -> None:
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), max_words=40, overlap=10)
    # Every window is within the size limit...
    assert all(len(c.split()) <= 40 for c in chunks)
    # ...and no word is dropped (last word appears in the final chunk).
    assert "w99" in chunks[-1]


def test_chunking_redacts_phi(notes: pd.DataFrame) -> None:
    row = notes.iloc[0]
    single = pd.DataFrame([row])
    chunks = chunk_notes(single, redact=True)
    joined = " ".join(c.text for c in chunks)
    for item in row["phi"]:
        assert item["value"] not in joined


def test_retriever_finds_relevant_chunk(notes: pd.DataFrame) -> None:
    retriever = Retriever.build(notes, redact=True)
    results = retriever.retrieve("lisinopril medication daily", k=3)
    assert results
    # Scores are sorted descending and are valid cosine similarities.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.001 <= s <= 1.001 for s in scores)
    # The top hit should actually mention the queried drug.
    assert "lisinopril" in results[0].chunk.text.lower()


def test_save_and_load_roundtrip(notes: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "index.joblib"
    Retriever.build(notes, redact=True).save(path)
    loaded = Retriever.load(path)
    assert len(loaded.store) > 0
    assert loaded.retrieve("MRI spine findings", k=2)
