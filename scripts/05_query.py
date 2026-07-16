#!/usr/bin/env python
"""Phase 2: query the retrieval index from the command line.

Usage:
    python scripts/05_query.py "what medications is the patient on?"
    python scripts/05_query.py            # runs a couple of demo queries
"""

import sys

from healthcare_assistant.config import RETRIEVAL_TOP_K
from healthcare_assistant.rag.retriever import Retriever

DEMO_QUERIES = [
    "what medications is the patient taking?",
    "MRI findings of the spine",
]


def show(retriever: Retriever, query: str) -> None:
    print(f"\n=== Query: {query!r} ===")
    for rank, result in enumerate(retriever.retrieve(query, k=RETRIEVAL_TOP_K), 1):
        chunk = result.chunk
        print(
            f"{rank}. score={result.score:.3f}  "
            f"[{chunk.note_type} note_id={chunk.note_id}]"
        )
        print(f"   {chunk.text}")


def main() -> None:
    retriever = Retriever.load()
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUERIES
    for query in queries:
        show(retriever, query)


if __name__ == "__main__":
    main()
