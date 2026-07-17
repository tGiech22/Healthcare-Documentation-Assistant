#!/usr/bin/env python
"""Phase 3: summarize a single note end-to-end (retrieval -> Claude).

Requires ANTHROPIC_API_KEY in the environment (or a local .env file) and a built
index (run scripts/04_build_index.py first).

Usage:
    python scripts/06_summarize.py            # summarize the first indexed note
    python scripts/06_summarize.py 158        # summarize a specific note_id
"""

import sys

from healthcare_assistant.rag.generator import summarize_note
from healthcare_assistant.rag.retriever import Retriever


def main() -> None:
    retriever = Retriever.load()

    if len(sys.argv) > 1:
        note_id = sys.argv[1]
    else:
        # Default to the first indexed note. note_id came from a CSV, so it may
        # be int-like; match the stored type by taking it straight from a chunk.
        note_id = retriever.store.chunks[0].note_id

    chunks = retriever.chunks_for_note(note_id)
    if not chunks:
        # note_ids from the CSV are integers; retry with an int if given a string.
        try:
            note_id = int(note_id)
            chunks = retriever.chunks_for_note(note_id)
        except (ValueError, TypeError):
            pass
    if not chunks:
        print(f"No note found with note_id={note_id!r}.")
        return

    print(f"=== Summarizing note {note_id} ({len(chunks)} chunk(s)) ===\n")
    for c in chunks:
        print(f"  source: {c.text}")
    print()

    summary = summarize_note(retriever, note_id)
    print(summary)


if __name__ == "__main__":
    main()
