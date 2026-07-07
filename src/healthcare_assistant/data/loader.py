"""Loads healthcare data i.e. clinical notes and produce reproducible train/val/test splits.

Everything downstream depends only on this module, so swapping the synthetic data
for real Synthea/MIMIC data is a matter of pointing ``load_notes`` at a different
CSV with the same columns (note_id, note_type, text, phi).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from healthcare_assistant.config import NOTES_CSV, RANDOM_SEED


@dataclass
class DataSplit:
    """A stratified train/validation/test split of notes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        return (
            f"train={len(self.train)}  val={len(self.val)}  test={len(self.test)}"
        )


def load_notes(path: Path = NOTES_CSV) -> pd.DataFrame:
    """Load the notes CSV, parsing the JSON-encoded PHI column."""
    if not path.exists():
        raise FileNotFoundError(
            f"No notes found at {path}. Run `python scripts/00_generate_data.py` first."
        )
    df = pd.read_csv(path)
    df["phi"] = df["phi"].apply(_parse_phi)
    return df


def _parse_phi(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if pd.isna(raw):
        return []
    return json.loads(raw)


def make_splits(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> DataSplit:
    """Stratified train/val/test split.

    ``val_size`` is expressed as a fraction of the *training* portion so the final
    proportions are roughly train=(1-test)*(1-val), val=(1-test)*val, test=test.
    Stratifying on ``note_type`` keeps class balance across all three splits.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["note_type"],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["note_type"],
        random_state=seed,
    )
    return DataSplit(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
