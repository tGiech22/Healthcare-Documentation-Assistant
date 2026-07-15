"""Smoke tests for the initial parts of the pipeline.

These run on a small in-memory dataset so they're fast and need no files on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from healthcare_assistant.data.generate_synthetic import generate_notes
from healthcare_assistant.data.loader import make_splits
from healthcare_assistant.models import phi_detector
from healthcare_assistant.models.classifier import build_pipeline, evaluate


@pytest.fixture(scope="module")
def notes() -> pd.DataFrame:
    # Parse the JSON phi column into lists, mirroring loader.load_notes.
    import json

    df = generate_notes(n_per_type=40, seed=0)
    df["phi"] = df["phi"].apply(json.loads)
    return df


def test_generation_is_balanced_and_reproducible(notes: pd.DataFrame) -> None:
    counts = notes["note_type"].value_counts()
    assert counts.min() == counts.max() == 40  # balanced
    # Reproducible given the same seed.
    again = generate_notes(n_per_type=40, seed=0)
    assert notes["text"].tolist() == again["text"].tolist()


def test_splits_are_stratified(notes: pd.DataFrame) -> None:
    splits = make_splits(notes, seed=0)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(notes)
    # Every note type appears in every split.
    for part in (splits.train, splits.val, splits.test):
        assert set(part["note_type"]) == set(notes["note_type"])


def test_classifier_learns_something(notes: pd.DataFrame) -> None:
    splits = make_splits(notes, seed=0)
    pipe = build_pipeline()
    pipe.fit(splits.train["text"], splits.train["note_type"])
    result = evaluate(pipe, splits.test["text"], splits.test["note_type"])
    # Data is intentionally noisy now, so we don't expect perfection -- just clearly
    # above the 0.2 chance rate for 5 balanced classes.
    assert result.accuracy > 0.7


def test_phi_detector_has_high_recall(notes: pd.DataFrame) -> None:
    metrics = phi_detector.evaluate(notes)
    # The baseline should catch essentially all structured PHI it was designed for.
    assert metrics.recall > 0.95


def test_redaction_removes_known_phi(notes: pd.DataFrame) -> None:
    row = notes.iloc[0]
    redacted = phi_detector.redact(row["text"])
    for item in row["phi"]:
        assert item["value"] not in redacted
