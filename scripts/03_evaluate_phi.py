#!/usr/bin/env python
"""Phase 1: evaluate the rule-based PHI detector and show a redaction example."""

from healthcare_assistant.data.loader import load_notes
from healthcare_assistant.models.phi_detector import evaluate, redact


def main() -> None:
    df = load_notes()

    metrics = evaluate(df)
    print("=== PHI detector performance (vs. ground truth) ===")
    print(metrics, "\n")

    example = df.iloc[0]["text"]
    print("=== Redaction example ===")
    print("ORIGINAL:\n" + example + "\n")
    print("REDACTED:\n" + redact(example))


if __name__ == "__main__":
    main()
