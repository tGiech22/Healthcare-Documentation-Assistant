"""Exploratory data analysis for the clinical notes.

Answers the questions you'd ask before modeling:
  * Is the dataset class-balanced?
  * How long are the notes (token counts)?
  * Which words are most distinctive per note type?

Saves figures to reports/figures/ and prints a text summary.
"""

from __future__ import annotations

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless: write files, don't open windows
import matplotlib.pyplot as plt
import pandas as pd

from healthcare_assistant.config import FIGURES_DIR, ensure_dirs
from healthcare_assistant.data.loader import load_notes

_TOKEN_RE = re.compile(r"[a-z]{3,}")
# Structural words shared by every note type -- uninformative for "distinctive".
_STOPWORDS = {
    "the", "and", "for", "was", "with", "patient", "name", "date", "report",
    "note", "results", "reviewed", "results", "results",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def class_balance(df: pd.DataFrame) -> pd.Series:
    return df["note_type"].value_counts()


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    lengths = df["text"].apply(lambda t: len(tokenize(t)))
    return (
        lengths.groupby(df["note_type"])
        .agg(["mean", "min", "max"])
        .round(1)
    )


def top_tokens_per_type(df: pd.DataFrame, n: int = 8) -> dict[str, list[str]]:
    """Most frequent non-stopword tokens for each note type."""
    result: dict[str, list[str]] = {}
    for note_type, group in df.groupby("note_type"):
        counter: Counter[str] = Counter()
        for text in group["text"]:
            counter.update(t for t in tokenize(text) if t not in _STOPWORDS)
        result[note_type] = [tok for tok, _ in counter.most_common(n)]
    return result


def _plot_class_balance(counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.sort_index().plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Note count by type")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    fig.tight_layout()
    out = FIGURES_DIR / "class_balance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_length_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    df["_len"] = df["text"].apply(lambda t: len(tokenize(t)))
    for note_type, group in df.groupby("note_type"):
        ax.hist(group["_len"], bins=15, alpha=0.5, label=note_type)
    ax.set_title("Note length distribution (tokens)")
    ax.set_xlabel("tokens")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "length_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def run() -> None:
    ensure_dirs()
    df = load_notes()

    print("=== Class balance ===")
    print(class_balance(df).to_string(), "\n")

    print("=== Note length (tokens) by type ===")
    print(length_stats(df).to_string(), "\n")

    print("=== Top distinctive tokens per type ===")
    for note_type, tokens in top_tokens_per_type(df).items():
        print(f"{note_type:20s}: {', '.join(tokens)}")
    print()

    _plot_class_balance(class_balance(df))
    _plot_length_distribution(df)


if __name__ == "__main__":
    run()
