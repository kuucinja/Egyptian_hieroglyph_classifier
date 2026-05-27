---
license: mit
library_name: sklearn
pipeline_tag: text-classification
tags:
- embeddings
- hieroglyphs
- contextual-classification
- sklearn
---

# Contextual Hieroglyph Role Classifier

This model predicts whether a target hieroglyph is being used as `phonetic`, `logographic`, or `determinative` from contextual fields. Training uses real corpus rows with weakly inferred labels.

## Model

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Classifier: `sklearn.linear_model.LogisticRegression`
- Input: target sign, neighboring signs, transliteration, translation/gloss, annotation notes
- Output: ranked role probabilities

## Train

Use `run_real_context_pipeline.bat` to download real data, weak-label it, train, and start the demo.

## Limitation

Weak labels are useful for scale, but they are not perfect. The model should be evaluated manually on a smaller expert-checked sample before being treated as reliable.
