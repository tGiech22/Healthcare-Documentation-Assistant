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


def _fill_template(template: str, phi: dict[str, str], rng: random.Random) -> str:
    """Fill a template with PHI and randomly-chosen clinical detail."""
    fields = dict(phi)
    for key, pool in CLINICAL_DETAIL.items():
        fields[key] = rng.choice(pool)
    # Only the placeholders present in this template are used by format_map.
    return template.format_map(_DefaultDict(fields))


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


def generate_notes(n_per_type: int = 120, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a balanced DataFrame of synthetic notes with PHI ground truth."""
    rng = random.Random(seed)
    rows = []
    note_id = 0
    for note_type, templates in TEMPLATES.items():
        for _ in range(n_per_type):
            phi = _make_phi(rng)
            template = rng.choice(templates)
            text = _fill_template(template, phi, rng)
            phi_items = _phi_items_for(text, phi)
            rows.append(
                {
                    "note_id": note_id,
                    "note_type": note_type,
                    "text": text,
                    "phi": json.dumps([item.__dict__ for item in phi_items]),
                }
            )
            note_id += 1
    df = pd.DataFrame(rows)
    # Shuffle so note types aren't grouped together on disk.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(n_per_type: int = 120) -> None:
    ensure_dirs()
    df = generate_notes(n_per_type=n_per_type)
    df.to_csv(NOTES_CSV, index=False)
    print(f"Wrote {len(df)} synthetic notes to {NOTES_CSV}")
    print(df["note_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
