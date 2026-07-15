# Healthcare Documentation Assistant

An ML project that analyzes healthcare documentation and summarizes it into clear,
digestible points for users. Built as a hands-on machine learning project using
**synthetic and de-identified public data** — no real patient data (PHI) is used.

## Overview

Clinical documents (discharge summaries, notes, lab reports) are long, dense, and hard
to parse quickly. This project ingests such documents and produces structured,
plain-language summaries — key diagnoses, medications, and action items — grounded in
the source text.

The goal is both a working summarization tool and a practical walk through the core ML
workflow: data prep, classical modeling, embeddings/retrieval, summarization, and
evaluation.

## Status

🚧 Early development. **Phases 0 & 1 are implemented and runnable** (synthetic data
generation, EDA, note-type classifier, and a PHI-detection baseline). Phases 2–5 are
planned — see the roadmap below.

## Data

This project uses **only** synthetic or de-identified public data:

- **[Synthea](https://synthetichealth.github.io/synthea/)** — synthetic patient records, no credentialing required.
- **[MIMIC-III / MIMIC-IV](https://physionet.org/)** — de-identified clinical notes (requires PhysioNet credentialing + CITI training).
- **n2c2 / i2b2** — labeled clinical NLP challenge datasets, useful for supervised tasks.

> ⚠️ No real Protected Health Information (PHI) is stored or processed in this repository.

## Roadmap

| Phase | Focus | Key skills |
|-------|-------|-----------|
| 0 | Setup & data exploration | Data loading, EDA, dataset versioning |
| 1 | Classical ML (note-type classifier, PHI/NER) | TF-IDF, transformers, supervised eval (F1, confusion matrix) |
| 2 | Embeddings & retrieval (RAG) | Clinical embeddings, vector search, chunking |
| 3 | Summarization | Fine-tuning seq2seq models + LLM-based summarization |
| 4 | Evaluation | ROUGE/BERTScore, faithfulness checks, LLM-as-judge, error analysis |
| 5 | Application & writeup | Streamlit/FastAPI demo, model card |

## Planned Tech Stack

- **Language:** Python
- **ML / NLP:** scikit-learn, Hugging Face Transformers, spaCy / scispaCy, Sentence-Transformers
- **Retrieval:** Chroma or pgvector
- **Summarization:** fine-tuned seq2seq (BART/T5/PEGASUS) and an LLM API (Claude)
- **App:** Streamlit (demo) or FastAPI (service)

## Project Structure

```
src/healthcare_assistant/
  config.py              # paths, constants, random seed
  data/
    generate_synthetic.py  # synthetic clinical notes w/ PHI ground truth
    loader.py              # load notes + stratified train/val/test split
  eda/explore.py          # class balance, note lengths, top tokens
  models/
    classifier.py         # TF-IDF + logistic regression note-type classifier
    phi_detector.py       # rule-based PHI detection + redaction
scripts/
  00_generate_data.py     # Phase 0: create the dataset
  01_eda.py               # Phase 0: exploratory analysis (saves figures)
  02_train_classifier.py  # Phase 1: train + evaluate the classifier
  03_evaluate_phi.py      # Phase 1: score the PHI detector
tests/                    # pytest smoke tests for the whole pipeline
```

## Getting Started

Requires Python 3.10+.

```bash
# 1. Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run the pipeline
python scripts/00_generate_data.py      # -> data/raw/notes.csv
python scripts/01_eda.py                # -> prints stats, saves reports/figures/*.png
python scripts/02_train_classifier.py   # -> trains model, saves confusion matrix
python scripts/03_evaluate_phi.py       # -> PHI precision/recall + redaction demo

# 3. Run the tests
pytest
```

### A note on the synthetic data

The generator is deliberately tuned to be a *realistic* ML challenge rather than a toy.
It applies several difficulty knobs (see `data/generate_synthetic.py`):

- **Class imbalance** — note types have unequal counts (progress notes are most common,
  pathology least), like a real corpus.
- **Text noise** (`noise_level`) — typos, dropped characters, and shared abbreviations
  (pt, hx, dx, w/o …).
- **Generic headers** — the giveaway type titles are replaced with generic ones most of
  the time, so the model can't just read the header.
- **Content bleed** — notes cross-reference other note types, overlapping features
  between classes.
- **Label noise** (`label_noise`, default 0.12 for the shipped dataset) — a fraction of
  labels are flipped to simulate imperfect annotation. This is what makes the task
  non-trivial: distinct clinical vocabularies are otherwise perfectly separable. The
  original label is preserved in a `true_type` column for analysis.

With these on, the TF-IDF + logistic-regression baseline lands around **0.87 test
accuracy** with a genuinely informative confusion matrix — a real baseline to improve on
(e.g. with a transformer in a later step). For an even more authentic exercise, swap in
real de-identified data (Synthea/MIMIC) via `data/loader.py`, the only seam downstream
code depends on.

## Disclaimer

This is an educational project. It is **not** a medical device and must not be used for
clinical decision-making. Summaries generated by ML models may contain errors or
omissions.
