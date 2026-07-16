#!/usr/bin/env python
"""Phase 2: build and save the retrieval (RAG) index over all notes."""

from healthcare_assistant.data.loader import load_notes
from healthcare_assistant.rag.retriever import Retriever


def main() -> None:
    df = load_notes()
    print(f"Loaded {len(df)} notes.")

    # PHI is redacted during chunking, so nothing sensitive enters the index.
    retriever = Retriever.build(df, redact=True)
    print(f"Indexed {len(retriever.store)} chunks.")

    path = retriever.save()
    print(f"Saved index to {path}")


if __name__ == "__main__":
    main()
