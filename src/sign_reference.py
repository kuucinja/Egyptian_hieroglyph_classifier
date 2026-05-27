from __future__ import annotations

import csv
from pathlib import Path


REFERENCE_FILE = Path("data/sign_reference.csv")


def normalize_sign(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_sign_reference(path: Path = REFERENCE_FILE) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[normalize_sign(row["sign"])] = row
    return records


def lookup_sign(value: str, reference: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    reference = reference or load_sign_reference()
    normalized = normalize_sign(value)
    row = reference.get(normalized)
    if row:
        return row

    compact = normalized.split("_")[-1]
    row = reference.get(compact)
    if row:
        return row

    return {
        "sign": value,
        "display_name": value or "unknown",
        "gardiner_code": "",
        "phonetic_value": "",
        "logographic_reading": "",
        "meaning": "",
        "notes": "No reference entry yet. Add this sign to data/sign_reference.csv.",
    }
