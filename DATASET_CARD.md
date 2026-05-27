---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- hieroglyphs
- egyptology
- contextual-classification
- embeddings
---

# Contextual Hieroglyph Role Dataset

This dataset contains real contextual examples for classifying the role of a target Egyptian hieroglyph sign.

The main training file is `data/contextual_examples_real_weak.csv`. It is built from real Ramses/AES-derived corpus data and weak-labeled using transliteration, translation/gloss, and sign-reference evidence.

Each row includes the target sign, neighboring signs, transliteration, translation/gloss, notes, and an inferred role label:

- `phonetic`
- `logographic`
- `determinative`

Uncertain rows are skipped. The smaller `data/contextual_examples.csv` file is only a starter/demo file.
