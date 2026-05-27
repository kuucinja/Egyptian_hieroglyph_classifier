---
title: Contextual Hieroglyph Role Classifier
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Contextual Hieroglyph Role Classifier

This project trains a contextual classifier that helps learners decide whether an Egyptian hieroglyph sign is being used as:

- `phonetic`: represents sound
- `logographic`: represents a word, object, or idea
- `determinative`: silent sign that clarifies meaning/category

The model uses text/context embeddings from `sentence-transformers/all-MiniLM-L6-v2` and a logistic regression classifier.

## Why Context Matters

A single hieroglyph image is not enough to know usage. The same sign can behave differently depending on neighboring signs, transliteration, and meaning. This project therefore uses contextual fields rather than isolated images.

Each training row includes:

```text
target_sign
context_before
context_after
transliteration
translation
role
notes
```

## Active Files

```text
.
+-- app_context.py                 # Gradio contextual demo
+-- run_context_pipeline.bat       # Windows one-command runner
+-- data/
|   +-- contextual_examples.csv    # Context role training data
|   +-- sign_reference.csv         # Meaning and reading hints for target signs
+-- src/
|   +-- context_features.py        # Builds model input text
|   +-- context_train.py           # Trains context embedding classifier
|   +-- context_predict.py         # Loads model and predicts roles
|   +-- sign_reference.py          # Looks up meaning/readings for the UI
|   +-- __init__.py
+-- requirements.txt
+-- DATASET_CARD.md
+-- MODEL_CARD.md
+-- SPACE_README.md
+-- archive_image_pipeline/        # Previous image-based pipeline and source image archive
```

## Quickstart

Install dependencies:

```bat
pip install -r requirements.txt
```

Run the real-data weak-supervision pipeline:

```bat
run_real_context_pipeline.bat
```

This downloads real Ramses/AES-derived contextual data from Zenodo, extracts contextual rows, weak-labels high-confidence examples using transliteration/translation/sign-reference evidence, trains the model, and starts the demo.

To force a fresh download and relabel:

```bat
run_real_context_pipeline.bat --refresh
```

You can still run the small starter dataset pipeline:

```bat
run_context_pipeline.bat
```

This trains the model into:

```text
models/hieroglyph-context-role-logreg
```

Then it starts the Gradio app:

```text
app_context.py
```

## Dataset

The active dataset is:

```text
data/contextual_examples_real_weak.csv
```

It is produced from real corpus data by:

```text
src/real_context_data.py
```

The script downloads a real Ramses/AES-derived model-data file from Zenodo record `10.5281/zenodo.7991241`, extracts contextual text/transliteration rows, and creates weak labels only when the evidence is high-confidence.

Weak-label rules include:

- `phonetic`: the target sign's phonetic value appears in transliteration without meaning evidence.
- `logographic`: the sign's logographic reading and meaning appear in context.
- `determinative`: the sign's meaning appears in translation/gloss but the sign is not reflected phonetically.

Uncertain or conflicting rows are skipped.

The small hand-written starter file remains at:

```text
data/contextual_examples.csv
```

## Old Pipeline

The older image-only pipeline has been moved to:

```text
archive_image_pipeline/
```

It is kept for reference, but it is not needed for the contextual role classifier.
