#!/usr/bin/env python
"""Phase 3: summarize a single note end-to-end (retrieval -> Claude).

Requires ANTHROPIC_API_KEY in the environment (or a local .env file) and a built
index (run scripts/04_build_index.py first).

Usage:
    python scripts/06_summarize.py            # summarize the first indexed note
    python scripts/06_summarize.py 158        # summarize a specific note_id
"""

import sys

from healthcare_assistant.rag.generator import summarize_chunks
from healthcare_assistant.rag.retriever import Retriever


def find_note_chunks(retriever, requested):
    """All chunks for a note, matched by string form so a CLI arg (always a str)
    still matches an int note_id from the CSV -- and works for string IDs too."""
    target = str(requested)
    return [c for c in retriever.store.chunks if str(c.note_id) == target]


def main() -> None:
    retriever = Retriever.load()

    # Default to the first indexed note when no id is given.
    requested = sys.argv[1] if len(sys.argv) > 1 else retriever.store.chunks[0].note_id

    chunks = find_note_chunks(retriever, requested)  # one scan, type-agnostic
    if not chunks:
        print(f"No note found with note_id={requested!r}.")
        return

    note_id = chunks[0].note_id  # the real, correctly-typed id
    print(f"=== Summarizing note {note_id} ({len(chunks)} chunk(s)) ===\n")
    for c in chunks:
        print(f"  source: {c.text}")
    print()

    summary = summarize_chunks(chunks)  # reuse what we already fetched
    print(summary)


if __name__ == "__main__":
    main()
