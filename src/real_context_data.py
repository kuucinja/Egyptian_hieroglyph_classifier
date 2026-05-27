from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from src.sign_reference import load_sign_reference


DEFAULT_URL = "https://zenodo.org/records/7991241/files/RamsesTrainingSetModel.json?download=1"
SIGN_TOKEN_RE = re.compile(r"\b[A-Z][0-9]{1,3}[A-Z]?\b")


def download(url: str, output_path: Path, refresh: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not refresh:
        print(f"Using existing file: {output_path}")
        return
    print(f"Downloading real corpus data from {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved download to {output_path}")


def walk_json(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        text_fields = {str(key): item for key, item in value.items() if isinstance(item, str)}
        if text_fields:
            joined = " ".join(text_fields.values())
            if SIGN_TOKEN_RE.search(joined) or likely_transliteration(joined):
                rows.append(text_fields)
        for item in value.values():
            if isinstance(item, (dict, list)):
                rows.extend(walk_json(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(walk_json(item))
    return rows


def likely_transliteration(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z][A-Za-z_.\-]{2,}\b", text))


def choose_field(row: dict[str, str], hints: list[str]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for key, value in row.items():
        key_lower = key.lower()
        if any(hint in key_lower for hint in hints):
            return value
    return ""


def extract_raw_context(download_path: Path, output_file: Path, limit: int) -> None:
    data = json.loads(download_path.read_text(encoding="utf-8"))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        for index, word in enumerate(data["words"]):
            full_context = str(word.get("encoding", "")).strip()
            interpretations = word.get("interpretations", [])
            if not full_context or not interpretations:
                continue
            best = max(interpretations, key=lambda item: float(item.get("relFreq", 0)))
            transliteration = str(best.get("transliteration", "")).strip()
            if not transliteration:
                continue
            rows.append(
                {
                    "id": f"real_{index:06d}",
                    "source": "Zenodo 10.5281/zenodo.7991241 RamsesTrainingSetModel",
                    "full_context": full_context,
                    "transliteration": transliteration,
                    "translation": "",
                }
            )
            if len(rows) >= limit:
                break
    else:
        records = walk_json(data)
        for index, record in enumerate(records):
            full_context = choose_field(record, ["mdc", "hieroglyph", "input", "sign"])
            transliteration = choose_field(record, ["transliteration", "transcription", "output", "trl"])
            translation = choose_field(record, ["translation", "gloss"])

            if not full_context:
                full_context = " ".join(value for value in record.values() if SIGN_TOKEN_RE.search(value))
            if not transliteration:
                transliteration = " ".join(value for value in record.values() if likely_transliteration(value))

            if not full_context and not transliteration:
                continue

            rows.append(
                {
                    "id": f"real_{index:06d}",
                    "source": "Zenodo 10.5281/zenodo.7991241 RamsesTrainingSetModel",
                    "full_context": full_context,
                    "transliteration": transliteration,
                    "translation": translation,
                }
            )
            if len(rows) >= limit:
                break

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source", "full_context", "transliteration", "translation"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} raw real context rows to {output_file}")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def split_values(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[/;,|]", value) if part.strip()]


def meaning_hits(meaning: str, translation: str) -> bool:
    translation_norm = normalize_text(translation)
    for item in split_values(meaning):
        item_norm = normalize_text(item)
        if len(item_norm) >= 3 and item_norm in translation_norm:
            return True
    return False


def phonetic_hits(phonetic_value: str, transliteration: str) -> bool:
    translit_norm = normalize_text(transliteration).replace(" ", "")
    for item in split_values(phonetic_value):
        item_norm = normalize_text(item).replace(" ", "")
        if item_norm and item_norm in translit_norm:
            return True
    return False


def context_window(tokens: list[str], index: int, size: int = 3) -> tuple[str, str]:
    before = " ".join(tokens[max(0, index - size) : index])
    after = " ".join(tokens[index + 1 : index + 1 + size])
    return before, after


def infer_role(
    sign: dict[str, str],
    transliteration: str,
    translation: str,
    before: str,
    after: str,
    token_count: int,
) -> tuple[str, str, float]:
    phonetic = phonetic_hits(sign["phonetic_value"], transliteration)
    logographic = bool(sign["logographic_reading"]) and (
        phonetic_hits(sign["logographic_reading"], transliteration) or meaning_hits(sign["meaning"], translation)
    )
    semantic = meaning_hits(sign["meaning"], translation)

    if logographic and (semantic or token_count == 1 or after.startswith("Z1")):
        return "logographic", "logographic reading is supported by word spelling/context", 0.9
    if phonetic and not semantic:
        return "phonetic", "phonetic value appears in transliteration without meaning evidence", 0.85
    if not phonetic and not logographic and semantic:
        return "determinative", "meaning appears in translation but sign is not reflected phonetically", 0.8
    if not sign["phonetic_value"] and not sign["logographic_reading"] and (before or after):
        return "determinative", "known semantic sign appears in a larger real word spelling without phonetic value", 0.8
    if "Z1" in after and semantic:
        return "logographic", "following stroke plus meaning evidence suggests word-sign use", 0.8
    return "", "weak or conflicting evidence", 0.0


def weak_label(raw_file: Path, output_file: Path, min_confidence: float) -> None:
    references = load_sign_reference()
    gardiner_to_sign = {
        row["gardiner_code"]: row
        for row in references.values()
        if row.get("gardiner_code")
    }

    labeled: list[dict[str, str]] = []
    with raw_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tokens = SIGN_TOKEN_RE.findall(row["full_context"])
            for index, token in enumerate(tokens):
                sign = gardiner_to_sign.get(token)
                if not sign:
                    continue
                before, after = context_window(tokens, index)
                role, reason, confidence = infer_role(
                    sign,
                    row["transliteration"],
                    row["translation"],
                    before,
                    after,
                    len(tokens),
                )
                if role and confidence >= min_confidence:
                    labeled.append(
                        {
                            "id": f"{row['id']}_{index:03d}",
                            "source": row["source"],
                            "target_sign": sign["sign"],
                            "context_before": before,
                            "context_after": after,
                            "transliteration": row["transliteration"],
                            "translation": row["translation"],
                            "role": role,
                            "notes": f"Weak label confidence={confidence:.2f}: {reason}. Full context: {row['full_context']}",
                        }
                    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "id",
            "source",
            "target_sign",
            "context_before",
            "context_after",
            "transliteration",
            "translation",
            "role",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)
    print(f"Wrote {len(labeled)} weakly labeled real context examples to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and weak-label real contextual Egyptian data.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--download-path", type=Path, default=Path("data/raw/RamsesTrainingSetModel.json"))
    parser.add_argument("--raw-file", type=Path, default=Path("data/real_context_raw.csv"))
    parser.add_argument("--output-file", type=Path, default=Path("data/contextual_examples_real_weak.csv"))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.url, args.download_path, args.refresh)
    extract_raw_context(args.download_path, args.raw_file, args.limit)
    weak_label(args.raw_file, args.output_file, args.min_confidence)


if __name__ == "__main__":
    main()
