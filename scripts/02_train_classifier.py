#!/usr/bin/env python
"""Phase 1: train and evaluate the note-type classifier."""

from healthcare_assistant.data.loader import load_notes, make_splits
from healthcare_assistant.models.classifier import (
    build_pipeline,
    cross_validate,
    evaluate,
    save_confusion_matrix,
    save_model,
)


def main() -> None:
    df = load_notes()
    splits = make_splits(df)
    print(f"Splits: {splits.summary()}\n")

    pipeline = build_pipeline()

    # Cross-validate on the training set before touching the test set.
    cv_scores = cross_validate(pipeline, splits.train["text"], splits.train["note_type"])
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n")

    # Fit on train, report validation accuracy (used for tuning decisions).
    pipeline.fit(splits.train["text"], splits.train["note_type"])
    val_result = evaluate(pipeline, splits.val["text"], splits.val["note_type"])
    print(f"Validation accuracy: {val_result.accuracy:.3f}\n")

    # Final unbiased estimate on the held-out test set.
    test_result = evaluate(pipeline, splits.test["text"], splits.test["note_type"])
    print("=== Test set performance ===")
    print(f"Accuracy: {test_result.accuracy:.3f}\n")
    print(test_result.report)

    fig_path = save_confusion_matrix(test_result)
    print(f"Saved confusion matrix to {fig_path}")

    model_path = save_model(pipeline)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
