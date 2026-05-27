from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoModel, AutoTokenizer

from src.context_features import ContextExample, build_context_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_cpu(max_threads: int) -> None:
    thread_count = max(1, max_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
    os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
    torch.set_num_threads(thread_count)


def read_context_csv(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["role"].strip().lower()
            if role in {"", "unknown", "uncertain"}:
                continue
            example = ContextExample(
                target_sign=row.get("target_sign", "").strip(),
                context_before=row.get("context_before", "").strip(),
                context_after=row.get("context_after", "").strip(),
                transliteration=row.get("transliteration", "").strip(),
                translation=row.get("translation", "").strip(),
                notes=row.get("notes", "").strip(),
            )
            texts.append(build_context_text(example))
            labels.append(role)

    if not texts:
        raise ValueError(f"No labeled context examples found in {path}")
    return texts, labels


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
            outputs = model(**inputs)
            features = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            vectors.append(features.cpu().numpy())
    return np.vstack(vectors)


def train(
    data_file: Path,
    model_dir: Path,
    embedding_model_name: str,
    batch_size: int,
    test_size: float,
    cpu_threads: int,
) -> None:
    configure_cpu(cpu_threads)
    texts, labels = read_context_csv(data_file)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    if len(label_encoder.classes_) < 2:
        raise ValueError("Training needs at least two role classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
    x = embed_texts(texts, tokenizer, embedding_model, device, batch_size)

    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    classifier = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "classifier.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
    metadata = {
        "embedding_model": embedding_model_name,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "task_type": "contextual_hieroglyph_role_classification",
        "classes": list(label_encoder.classes_),
        "data_file": str(data_file),
        "embedding_dimension": int(x.shape[1]),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "classification_report": report}, indent=2),
        encoding="utf-8",
    )
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Context model written to {model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a context-aware hieroglyph role classifier.")
    parser.add_argument("--data-file", type=Path, default=Path("data/contextual_examples.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/hieroglyph-context-role-logreg"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.data_file, args.model_dir, args.embedding_model, args.batch_size, args.test_size, args.cpu_threads)


if __name__ == "__main__":
    main()

