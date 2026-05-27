from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModel, AutoTokenizer

from src.context_features import ContextExample, build_context_text
from src.context_train import mean_pool


LOCAL_MODEL_DIR = Path("models/hieroglyph-context-role-logreg")


ROLE_INFO = {
    "phonetic": "The target sign is likely being used for sound value in this context.",
    "logographic": "The target sign is likely standing for a word, object, or idea.",
    "determinative": "The target sign is likely a silent classifier that clarifies meaning.",
}


class ContextRoleClassifier:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = self._resolve_model_dir(model_dir)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.classifier = joblib.load(self.model_dir / "classifier.joblib")
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.joblib")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata["embedding_model"])
        self.embedding_model = AutoModel.from_pretrained(self.metadata["embedding_model"]).to(self.device)
        self.embedding_model.eval()

    def _resolve_model_dir(self, model_dir: Path | None) -> Path:
        if model_dir is not None:
            return model_dir
        if LOCAL_MODEL_DIR.exists():
            return LOCAL_MODEL_DIR
        repo_id = os.environ.get("HF_MODEL_REPO")
        if not repo_id:
            raise FileNotFoundError(
                "No local context model artifacts found. Train first or set HF_MODEL_REPO."
            )
        return Path(snapshot_download(repo_id=repo_id))

    def embed(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=256).to(
                self.device
            )
            outputs = self.embedding_model(**inputs)
            features = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()

    def predict(self, example: ContextExample, top_k: int = 3) -> dict[str, float]:
        vector = self.embed(build_context_text(example))
        probabilities = self.classifier.predict_proba(vector)[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        labels = self.label_encoder.inverse_transform(order)
        return {label: float(probabilities[index]) for label, index in zip(labels, order)}

    def explain(self, example: ContextExample, top_k: int = 3) -> list[list[str | float]]:
        return [
            [role, round(score, 4), ROLE_INFO.get(role, "No role note available.")]
            for role, score in self.predict(example, top_k=top_k).items()
        ]

