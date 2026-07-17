"""Phase 3: turn retrieved chunks into a grounded, structured summary.

This is the "G" (generation) in RAG. We take chunks -- already de-identified and
retrieved for relevance -- and ask Claude to extract diagnoses, medications, and
action items *using only that text*. Two things keep it honest:

  * The prompt instructs the model to rely solely on the provided notes and to
    leave a field empty rather than guess. This is "grounding" -- the antidote to
    hallucination, and the reason RAG is the right shape for a clinical tool.
  * ``output_config.format`` constrains the reply to a JSON schema, so we always
    get back a parseable object with the exact fields we asked for (no brittle
    string-parsing of free-form text).

Auth: the Anthropic client reads ``ANTHROPIC_API_KEY`` from the environment. Put
it in a local ``.env`` file (git-ignored) -- never hardcode it. We load ``.env``
automatically if python-dotenv is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from healthcare_assistant.config import SUMMARIZER_MAX_TOKENS, SUMMARIZER_MODEL
from healthcare_assistant.rag.chunking import Chunk

# System prompt: the assistant's standing instructions. Kept stable so it can be
# prompt-cached later; the per-note text goes in the user message.
SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. You read excerpts from a "
    "patient's de-identified clinical notes and produce a short, plain-language "
    "summary for a general audience. Use ONLY the information in the provided "
    "notes. Never infer, assume, or add facts that are not stated. If a category "
    "has nothing in the notes, return an empty list for it -- do not guess."
)

# Constrains Claude's reply to exactly these fields (all required, no extras).
# Numeric/length constraints aren't supported by structured outputs, and we
# don't need them here.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {"type": "array", "items": {"type": "string"}},
        "medications": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnoses", "medications", "action_items"],
    "additionalProperties": False,
}


@dataclass
class Summary:
    """A grounded, structured summary of one or more note chunks."""

    diagnoses: list[str]
    medications: list[str]
    action_items: list[str]

    def __str__(self) -> str:
        def section(title: str, items: list[str]) -> str:
            if not items:
                return f"{title}: (none found in the notes)"
            bullets = "\n".join(f"  - {item}" for item in items)
            return f"{title}:\n{bullets}"

        return "\n".join(
            (
                section("Diagnoses", self.diagnoses),
                section("Medications", self.medications),
                section("Action items", self.action_items),
            )
        )


def build_prompt(chunks: list[Chunk]) -> str:
    """Assemble the user message: the instruction plus the retrieved notes.

    Kept as a separate, side-effect-free function so it can be unit-tested and
    inspected without making an API call.
    """
    notes = "\n\n".join(f"[note {c.note_id}] {c.text}" for c in chunks)
    return (
        "Summarize the patient's diagnoses, medications, and action items using "
        "ONLY the notes below. If a category is not mentioned, leave it empty.\n\n"
        f"NOTES:\n{notes}"
    )


def _load_client(client=None):
    """Return an Anthropic client, loading .env for the API key if available."""
    if client is not None:
        return client
    try:  # optional convenience -- fine if python-dotenv isn't installed
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def summarize_chunks(
    chunks: list[Chunk],
    *,
    client=None,
    model: str = SUMMARIZER_MODEL,
    max_tokens: int = SUMMARIZER_MAX_TOKENS,
) -> Summary:
    """Ask Claude to summarize ``chunks`` into a grounded :class:`Summary`.

    Pass a pre-built ``client`` to reuse a connection or to inject a fake in
    tests; otherwise one is created from the environment.
    """
    if not chunks:
        raise ValueError("No chunks to summarize.")

    client = _load_client(client)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},  # let Claude reason before answering
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(chunks)}],
        output_config={"format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this request.")

    # With the schema set, the first text block is guaranteed to be valid JSON.
    # (Adaptive thinking may add thinking blocks first, so filter by type.)
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return Summary(
        diagnoses=data["diagnoses"],
        medications=data["medications"],
        action_items=data["action_items"],
    )


def summarize_note(retriever, note_id: object, **kwargs) -> Summary:
    """Convenience: gather one note's chunks from a retriever and summarize them."""
    chunks = retriever.chunks_for_note(note_id)
    if not chunks:
        raise ValueError(f"No chunks found for note_id={note_id!r}.")
    return summarize_chunks(chunks, **kwargs)
