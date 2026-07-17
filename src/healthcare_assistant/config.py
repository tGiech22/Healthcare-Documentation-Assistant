"""Central configuration: paths and shared constants.

Keeping paths in one place means scripts and notebooks stay consistent, and the
project can be relocated without editing every file.
"""

from __future__ import annotations

from pathlib import Path

# Project layout ------------------------------------------------------------
# config.py lives at src/healthcare_assistant/config.py, so the repo root is
# three levels up.
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Canonical file locations --------------------------------------------------
NOTES_CSV = RAW_DATA_DIR / "notes.csv"
CLASSIFIER_PATH = MODELS_DIR / "note_type_classifier.joblib"
VECTOR_INDEX_PATH = PROCESSED_DATA_DIR / "note_index.joblib"

# Reproducibility -----------------------------------------------------------
RANDOM_SEED = 42

# RAG / retrieval (Phase 2) -------------------------------------------------
# Our notes are short (a few sentences), so we chunk with small windows and a
# little overlap. Overlap keeps a fact from being split across a chunk boundary.
CHUNK_MAX_WORDS = 40
CHUNK_OVERLAP_WORDS = 10
# How many chunks to retrieve per query by default.
RETRIEVAL_TOP_K = 3

# LLM summarization (Phase 3) -----------------------------------------------
# Claude model used to turn retrieved chunks into a grounded summary. Opus is the
# most capable default; switch to "claude-haiku-4-5" for cheaper/faster runs or
# "claude-sonnet-5" for a middle ground once the pipeline works.
SUMMARIZER_MODEL = "claude-opus-4-8"
# Summaries are short, but adaptive thinking also draws from this budget, so we
# leave generous headroom. Well under the SDK's non-streaming timeout.
SUMMARIZER_MAX_TOKENS = 8192

# The clinical note types we generate and classify.
NOTE_TYPES = [
    "discharge_summary",
    "radiology_report",
    "progress_note",
    "lab_report",
    "pathology_report",
]


def ensure_dirs() -> None:
    """Create the output directories if they don't exist yet."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
