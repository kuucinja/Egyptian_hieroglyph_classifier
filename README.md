---
title: Contextual Hieroglyph Role Classifier
colorFrom: yellow
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Contextual Hieroglyph Role Classifier

A machine-learning demo that classifies Egyptian hieroglyphs by their **functional role in context** — not by their visual appearance. The same hieroglyph sign can represent a sound (*phonetic*), stand for a whole word (*logographic*), or silently clarify meaning (*determinative*). This project uses surrounding text and transliteration context to make that distinction automatically.

**Live demo:** https://huggingface.co/spaces/Kucina111/Egyptian_hieroglyph_classifier  
**Dataset:** `data/contextual_examples_real_weak.csv` (7,870 weak-labelled corpus rows)  
**Model:** Logistic regression on sentence-transformer embeddings — see `models/`

---

## How It Works

1. A target hieroglyph sign and its neighbours (context before, context after, transliteration, English gloss) are concatenated into a single text string.
2. The string is embedded with [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions, mean-pooled).
3. A Logistic Regression classifier predicts one of three roles: `phonetic`, `logographic`, or `determinative`.

The model achieves **100% accuracy** on a stratified 25% hold-out test set. This reflects strong consistency within the weak-labelling heuristics used to generate the training data; further evaluation against expert annotations would be needed to assess broader generalisation.

---

## Repository Structure

```
├── app.py                          # Gradio demo entry point (HF Spaces)
├── requirements.txt                # Python dependencies
├── REPORT.md                       # Assignment report
├── DATASET_CARD.md                 # Dataset documentation
├── MODEL_CARD.md                   # Model documentation
│
├── src/
│   ├── context_features.py         # Builds the text input from a training row
│   ├── context_train.py            # Training pipeline (embeddings + LogReg)
│   ├── context_predict.py          # Inference: load model, embed, predict
│   ├── sign_reference.py           # Gardiner code lookup for the UI
│   ├── sign_images.py              # Loads hieroglyph images for the UI
│   └── real_context_data.py        # Downloads corpus and generates weak labels
│
├── data/
│   ├── contextual_examples_real_weak.csv   # 7,870-row weak-labelled dataset
│   ├── contextual_examples.csv             # 12-row hand-annotated starter set
│   ├── sign_reference.csv                  # Gardiner code reference table
│   └── sign_image_manifest.csv             # Image paths for UI sign display
│
├── models/
│   └── hieroglyph-context-role-logreg/
│       ├── classifier.joblib       # Trained LogisticRegression
│       ├── label_encoder.joblib    # Sklearn LabelEncoder
│       ├── metadata.json           # Embedding model name, classes, dimensions
│       └── metrics.json            # Accuracy and per-class F1 on test split
│
└── assets/
    └── signs/                      # PNG images of individual hieroglyph signs
```

---

## Dataset

The dataset (`data/contextual_examples_real_weak.csv`) was produced from the **Ramses/AES corpus** ([Zenodo 10.5281/zenodo.7991241](https://doi.org/10.5281/zenodo.7991241)).

Each row represents one hieroglyph token in its textual context:

| Column | Description |
|---|---|
| `target_sign` | Gardiner code of the hieroglyph being classified |
| `context_before` | Signs preceding the target in the sequence |
| `context_after` | Signs following the target |
| `transliteration` | Egyptian transliteration of the phrase |
| `translation` | English gloss |
| `role` | Weak label: `phonetic`, `logographic`, or `determinative` |
| `notes` | Full original corpus encoding and confidence note |

**Weak-labelling rules:**
- `phonetic` — the sign's phonetic value appears in the transliteration without meaning evidence
- `logographic` — the sign's reading and meaning both appear in context
- `determinative` — the sign's meaning appears in the translation but is not reflected phonetically

Rows with conflicting or ambiguous evidence are discarded. Final dataset: **7,870 high-confidence rows** across three classes.

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

The app opens at `http://localhost:7860`. The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded from Hugging Face Hub on first run; the trained classifier is loaded from `models/` directly.

**To retrain the model on the real corpus:**

```bat
run_real_context_pipeline.bat
```

This downloads the Zenodo corpus, extracts contextual rows, applies the weak-labelling rules, and retrains the classifier. Use `--refresh` to force a fresh download.

---

## Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Interactive web demo |
| `torch` + `transformers` | Text embedding (all-MiniLM-L6-v2) |
| `scikit-learn` | Logistic regression classifier |
| `joblib` | Model serialisation |
| `numpy` | Vector operations |
| `huggingface-hub` | Model download fallback |
