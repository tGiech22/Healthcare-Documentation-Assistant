"""Generates synthetic clinical notes for local development.

This lets the whole pipeline run end-to-end without downloading MIMIC (which needs
credentialing) or running Synthea (which needs Java). The notes are intentionally
fake but structured so that:

  * each note type has distinctive vocabulary a classifier can learn, and
  * each note contains embedded fake PHI (names, MRNs, dates, phones, SSNs) with a
    recorded ground truth, so the PHI detector can be evaluated with real metrics.

Swap this out for a real loader once you have Synthea/MIMIC data — everything
downstream only depends on the CSV schema produced here:

    note_id, note_type, text, phi (JSON list of {"type", "value"})
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import pandas as pd

from healthcare_assistant.config import NOTES_CSV, RANDOM_SEED, ensure_dirs

# --- Fake PHI building blocks ---------------------------------------------
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "Maria", "Ahmed", "Wei", "Sofia", "Omar",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Nguyen", "Patel", "Kim", "Okafor", "Rossi",
]
STREETS = ["Oak St", "Maple Ave", "Main St", "Elm Rd", "Cedar Ln", "Park Blvd"]
CITIES = ["Springfield", "Riverton", "Fairview", "Franklin", "Greenville"]


@dataclass
class PHIItem:
    """A single piece of protected health information and its category."""

    type: str
    value: str


def _rand_date(rng: random.Random) -> str:
    year = rng.randint(2018, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{month:02d}/{day:02d}/{year}"


def _make_phi(rng: random.Random) -> dict[str, str]:
    """Build a bundle of PHI values for one patient/note."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return {
        "name": f"{first} {last}",
        "mrn": f"MRN{rng.randint(1000000, 9999999)}",
        "date": _rand_date(rng),
        "phone": f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
        "ssn": f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
        "address": f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
    }


# --- Note templates --------------------------------------------------------
# Each template is a format string that consumes PHI plus some clinical detail.
# The distinctive clinical vocabulary is what the classifier learns from.
TEMPLATES: dict[str, list[str]] = {
    "discharge_summary": [
        (
            "DISCHARGE SUMMARY\nPatient: {name}  MRN: {mrn}\nDischarge Date: {date}\n"
            "Admitting Diagnosis: {dx}. The patient was admitted for management and "
            "responded well to treatment. Discharge medications include {med}. "
            "Follow-up appointment scheduled. Contact: {phone}."
        ),
        (
            "DISCHARGE SUMMARY\nName: {name} (SSN {ssn})\nDate of Discharge: {date}\n"
            "Hospital course was uncomplicated. Diagnosis at discharge: {dx}. "
            "Discharged home in stable condition on {med}. Home address: {address}."
        ),
    ],
    "radiology_report": [
        (
            "RADIOLOGY REPORT\nPatient: {name}  MRN: {mrn}\nExam Date: {date}\n"
            "EXAMINATION: {imaging}. FINDINGS: No acute intracranial abnormality. "
            "IMPRESSION: {impression}. Radiologist reviewed images at station."
        ),
        (
            "IMAGING REPORT\nName: {name}\nStudy Date: {date}\nMODALITY: {imaging}. "
            "COMPARISON: prior study. FINDINGS: {impression}. Callback: {phone}."
        ),
    ],
    "progress_note": [
        (
            "PROGRESS NOTE\nPatient: {name}  MRN: {mrn}\nDate: {date}\n"
            "SUBJECTIVE: Patient reports feeling better today. OBJECTIVE: Vitals "
            "stable, afebrile. ASSESSMENT: {dx}, improving. PLAN: Continue {med}, "
            "reassess in the morning."
        ),
        (
            "DAILY PROGRESS NOTE\nName: {name}\nDate: {date}\nS: No new complaints. "
            "O: Exam unremarkable. A: {dx}. P: Continue current management with {med}."
        ),
    ],
    "lab_report": [
        (
            "LABORATORY REPORT\nPatient: {name}  MRN: {mrn}\nCollected: {date}\n"
            "COMPLETE BLOOD COUNT: WBC {wbc}, Hemoglobin {hgb} g/dL. "
            "BASIC METABOLIC PANEL: Sodium {na} mmol/L, Potassium {k} mmol/L. "
            "Results reviewed and flagged where abnormal."
        ),
        (
            "LAB RESULTS\nName: {name} (SSN {ssn})\nSpecimen Date: {date}\n"
            "Glucose {glu} mg/dL, Creatinine {creat} mg/dL. Reference ranges applied. "
            "Ordering provider notified at {phone}."
        ),
    ],
    "pathology_report": [
        (
            "SURGICAL PATHOLOGY REPORT\nPatient: {name}  MRN: {mrn}\nDate: {date}\n"
            "SPECIMEN: {specimen}. GROSS DESCRIPTION: Tissue submitted in formalin. "
            "MICROSCOPIC: {micro}. DIAGNOSIS: {path_dx}."
        ),
        (
            "PATHOLOGY REPORT\nName: {name}\nAccession Date: {date}\n"
            "SPECIMEN: {specimen}. HISTOLOGY: {micro}. FINAL DIAGNOSIS: {path_dx}. "
            "Report signed electronically."
        ),
    ],
}

# Clinical vocabulary pools keyed by the placeholders used above.
CLINICAL_DETAIL: dict[str, list[str]] = {
    "dx": ["community-acquired pneumonia", "acute exacerbation of CHF",
           "type 2 diabetes mellitus", "cellulitis", "atrial fibrillation"],
    "med": ["lisinopril 10mg daily", "metformin 500mg BID", "furosemide 40mg daily",
            "amoxicillin 500mg TID", "apixaban 5mg BID"],
    "imaging": ["CT head without contrast", "chest radiograph, 2 views",
                "MRI lumbar spine", "abdominal ultrasound"],
    "impression": ["no acute cardiopulmonary process", "mild degenerative changes",
                   "small pleural effusion", "unremarkable study"],
    "wbc": ["6.2", "11.4", "8.8", "14.1"],
    "hgb": ["13.5", "9.8", "15.1", "11.0"],
    "na": ["138", "134", "141", "129"],
    "k": ["4.1", "3.3", "5.2", "4.8"],
    "glu": ["95", "142", "180", "88"],
    "creat": ["0.9", "1.4", "2.1", "1.0"],
    "specimen": ["colon, biopsy", "skin, punch biopsy", "breast, core needle biopsy",
                 "lymph node, excision"],
    "micro": ["benign glandular tissue", "chronic inflammation without malignancy",
              "atypical cells present", "fibroadipose tissue"],
    "path_dx": ["benign, no malignancy identified", "low-grade dysplasia",
                "reactive changes", "negative for carcinoma"],
}


# --- Difficulty knobs ------------------------------------------------------
# Boilerplate that appears on EVERY note type. Shared vocabulary like this blurs
# the class boundaries, so the classifier can't rely on a few giveaway words.
BOILERPLATE = [
    "Electronically signed by the attending physician.",
    "Please contact the care team with any questions.",
    "This document was generated from the electronic health record.",
    "Reviewed and verified by clinical staff.",
    "Confidential patient health information.",
    "Follow-up as clinically indicated.",
]

# Generic headers that replace the giveaway type-named headers most of the time,
# so the classifier can't just read the title. Real EHR exports are often
# inconsistently or generically titled.
GENERIC_HEADERS = [
    "CLINICAL NOTE",
    "PATIENT DOCUMENTATION",
    "MEDICAL RECORD ENTRY",
    "CLINICAL DOCUMENTATION",
]

# One characteristic sentence per type. We append a snippet from a *different*
# type to some notes -- real notes cross-reference labs, imaging, etc., and this
# deliberately overlaps features between classes.
BLEED_SNIPPETS = {
    "discharge_summary": "Discharged home in stable condition with follow-up arranged.",
    "radiology_report": "Impression: no acute abnormality on imaging.",
    "progress_note": "Subjective: patient resting comfortably overnight.",
    "lab_report": "Sodium and potassium remain within normal limits.",
    "pathology_report": "Specimen was submitted for histologic review.",
}

# Abbreviation substitutions shared across all note types -- more overlap, plus
# the informal register real clinical text is full of.
ABBREVIATIONS = {
    "patient": "pt",
    "history": "hx",
    "diagnosis": "dx",
    "treatment": "tx",
    "without": "w/o",
    "follow-up": "f/u",
    "management": "mgmt",
    "results": "rslts",
}


def _inject_noise(text: str, rng: random.Random, level: float) -> str:
    """Add realistic noise: abbreviations, typos, and casing jitter.

    ``level`` (0..1) scales how aggressively noise is applied. Only lowercase
    alphabetic words are touched, so PHI values (names are capitalized; MRN/date/
    phone/SSN/address all contain digits) pass through untouched and stay matchable
    against the ground truth.
    """
    if level <= 0:
        return text

    def transform(word: str) -> str:
        core = word.lower()
        # Abbreviate known words.
        if core in ABBREVIATIONS and rng.random() < level:
            return ABBREVIATIONS[core]
        # Perturb plain lowercase OR ALL-CAPS words (section keywords), never the
        # Title-case names that make up PHI, and never anything with digits.
        if not word.isalpha() or len(word) < 5 or not (word.islower() or word.isupper()):
            return word
        r = rng.random()
        if r < level * 0.4:  # swap two adjacent characters (transposition typo)
            i = rng.randint(0, len(word) - 2)
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if r < level * 0.6:  # drop a character
            i = rng.randint(0, len(word) - 1)
            return word[:i] + word[i + 1 :]
        if r < level * 0.7:  # random uppercasing
            return word.upper()
        return word

    return " ".join(transform(w) for w in text.split(" "))


def _fill_template(
    template: str,
    phi: dict[str, str],
    rng: random.Random,
    note_type: str,
    noise_level: float = 0.0,
) -> str:
    """Fill a template with PHI, clinical detail, shared boilerplate, and noise."""
    fields = dict(phi)
    for key, pool in CLINICAL_DETAIL.items():
        fields[key] = rng.choice(pool)
    # Only the placeholders present in this template are used by format_map.
    text = template.format_map(_DefaultDict(fields))

    # Replace the giveaway type header with a generic one most of the time.
    if rng.random() < 0.85:
        lines = text.split("\n", 1)
        rest = lines[1] if len(lines) > 1 else ""
        text = rng.choice(GENERIC_HEADERS) + ("\n" + rest if rest else "")

    # Bleed in a sentence characteristic of a *different* note type.
    if rng.random() < 0.5:
        other = rng.choice([t for t in BLEED_SNIPPETS if t != note_type])
        text = text + " " + BLEED_SNIPPETS[other]

    # Sprinkle in shared boilerplate to blur class boundaries further.
    if rng.random() < 0.8:
        text = text + " " + rng.choice(BOILERPLATE)

    return _inject_noise(text, rng, noise_level)


class _DefaultDict(dict):
    """format_map helper: leave unknown placeholders untouched instead of raising."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


def _phi_items_for(text: str, phi: dict[str, str]) -> list[PHIItem]:
    """Record which PHI values actually appear in the rendered note."""
    items: list[PHIItem] = []
    for phi_type, value in phi.items():
        if value in text:
            items.append(PHIItem(type=phi_type, value=value))
    return items


# Deliberately imbalanced defaults -- real clinical corpora never have equal
# numbers of each note type, and learning to cope with that is part of the point.
DEFAULT_COUNTS: dict[str, int] = {
    "discharge_summary": 180,
    "radiology_report": 140,
    "progress_note": 220,
    "lab_report": 90,
    "pathology_report": 60,
}


def generate_notes(
    counts: dict[str, int] | None = None,
    *,
    n_per_type: int | None = None,
    noise_level: float = 0.35,
    label_noise: float = 0.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a DataFrame of synthetic notes with PHI ground truth.

    Pass ``counts`` for a per-type count mapping (imbalanced), or ``n_per_type``
    for a balanced dataset. ``noise_level`` (0..1) controls typo/abbreviation noise.

    ``label_noise`` (0..1) flips that fraction of ``note_type`` labels to a random
    other class. Real clinical annotation is imperfect, and a bit of label noise is
    what makes the classification task non-trivial: distinct clinical vocabularies
    are otherwise perfectly separable, so without it the model scores ~1.0 and the
    confusion matrix is uninformative. The original label is kept in ``true_type``.
    """
    if n_per_type is not None:
        counts = {note_type: n_per_type for note_type in TEMPLATES}
    elif counts is None:
        counts = DEFAULT_COUNTS

    rng = random.Random(seed)
    rows = []
    note_id = 0
    for note_type, n in counts.items():
        templates = TEMPLATES[note_type]
        for _ in range(n):
            phi = _make_phi(rng)
            template = rng.choice(templates)
            text = _fill_template(
                template, phi, rng, note_type, noise_level=noise_level
            )
            phi_items = _phi_items_for(text, phi)
            # Optionally corrupt the label to simulate imperfect annotation.
            label = note_type
            if label_noise > 0 and rng.random() < label_noise:
                label = rng.choice([t for t in TEMPLATES if t != note_type])
            rows.append(
                {
                    "note_id": note_id,
                    "note_type": label,
                    "true_type": note_type,
                    "text": text,
                    "phi": json.dumps([item.__dict__ for item in phi_items]),
                }
            )
            note_id += 1
    df = pd.DataFrame(rows)
    # Shuffle so note types aren't grouped together on disk.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    df = generate_notes(label_noise=0.12)
    df.to_csv(NOTES_CSV, index=False)
    print(f"Wrote {len(df)} synthetic notes to {NOTES_CSV}")
    print(df["note_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
