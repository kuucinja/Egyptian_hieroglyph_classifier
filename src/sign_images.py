from __future__ import annotations

import csv
from pathlib import Path


MANIFEST_FILE = Path("data/sign_image_manifest.csv")


def normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_image_manifest(path: Path = MANIFEST_FILE) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[row["gardiner_code"]] = row
            records[normalize(row["sign"])] = row
            records[normalize(row["display_name"])] = row
    return records


def image_for_token(token: str, manifest: dict[str, dict[str, str]] | None = None) -> str | None:
    manifest = manifest or load_image_manifest()
    row = manifest.get(token) or manifest.get(normalize(token))
    if not row:
        return None
    path = Path(row["image_path"])
    return str(path) if path.exists() else None
