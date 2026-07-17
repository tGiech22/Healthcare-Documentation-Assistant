"""Tests for the Phase 3 summarizer.

These run offline with a fake Anthropic client -- no API key, no network, no cost.
They verify the parts we own (prompt assembly, response parsing, grounding
plumbing); the model's actual output quality is checked manually via
scripts/06_summarize.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from healthcare_assistant.rag.chunking import Chunk
from healthcare_assistant.rag.generator import Summary, build_prompt, summarize_chunks


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list
    stop_reason: str = "end_turn"


class _FakeClient:
    """Stands in for anthropic.Anthropic, capturing the request and replying."""

    def __init__(self, payload: dict, stop_reason: str = "end_turn") -> None:
        self._payload = payload
        self._stop_reason = stop_reason
        self.last_kwargs: dict | None = None
        self.messages = self

    def create(self, **kwargs):  # mimics client.messages.create
        self.last_kwargs = kwargs
        block = _FakeTextBlock(text=json.dumps(self._payload))
        return _FakeResponse(content=[block], stop_reason=self._stop_reason)


def _chunks() -> list[Chunk]:
    return [
        Chunk("53::0", 53, "discharge_summary", "Discharge medications include apixaban 5mg BID."),
        Chunk("53::1", 53, "discharge_summary", "Admitting diagnosis: cellulitis. Follow-up scheduled."),
    ]


def test_build_prompt_includes_all_chunks_and_grounding() -> None:
    prompt = build_prompt(_chunks())
    assert "apixaban" in prompt
    assert "cellulitis" in prompt
    assert "ONLY" in prompt  # the grounding instruction


def test_summarize_chunks_parses_structured_reply() -> None:
    payload = {
        "diagnoses": ["cellulitis"],
        "medications": ["apixaban 5mg twice daily"],
        "action_items": ["attend follow-up appointment"],
    }
    client = _FakeClient(payload)
    summary = summarize_chunks(_chunks(), client=client)

    assert isinstance(summary, Summary)
    assert summary.diagnoses == ["cellulitis"]
    assert summary.medications == ["apixaban 5mg twice daily"]
    # The retrieved note text was actually sent to the model.
    sent = client.last_kwargs["messages"][0]["content"]
    assert "apixaban" in sent
    # And we asked for structured output.
    assert client.last_kwargs["output_config"]["format"]["type"] == "json_schema"


def test_summarize_chunks_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_chunks([], client=_FakeClient({}))


def test_refusal_raises() -> None:
    client = _FakeClient({}, stop_reason="refusal")
    with pytest.raises(RuntimeError):
        summarize_chunks(_chunks(), client=client)


def test_summary_str_marks_empty_sections() -> None:
    summary = Summary(diagnoses=["cellulitis"], medications=[], action_items=[])
    rendered = str(summary)
    assert "cellulitis" in rendered
    assert "(none found in the notes)" in rendered
