# Contextual Hieroglyph Role Classifier
### Information Retrieval — Assignment Report

**Student:** [Your Name]  
**Date:** May 2026

---

## Links

| Resource | URL |
|---|---|
| GitHub Repository | https://github.com/kuucinja/Egyptian_hieroglyph_classifier |
| HuggingFace Demo (Space) | https://huggingface.co/spaces/Kucina111/Egyptian_hieroglyph_classifier |
| Dataset (in repo) | https://github.com/kuucinja/Egyptian_hieroglyph_classifier/blob/main/data/contextual_examples_real_weak.csv |
| Trained Model (in repo) | https://github.com/kuucinja/Egyptian_hieroglyph_classifier/tree/main/models/hieroglyph-context-role-logreg |

---

## Abstract

This project presents a contextual classifier for Egyptian hieroglyphic signs. Rather than classifying signs by visual appearance alone, the model uses surrounding context — neighbouring signs, transliteration, and English gloss — to determine the functional role of a target sign: *phonetic* (represents a sound), *logographic* (represents a word or object), or *determinative* (silent sign that clarifies meaning). The model is trained on 7,870 weak-labelled examples derived from a real Egyptological corpus and achieves 100% accuracy on a stratified 25% hold-out test set — a figure that reflects the high internal consistency of the weak-labelling rules rather than fully generalised robustness.

---

## 1. Introduction

A defining challenge in computational Egyptology is that the same hieroglyph can serve fundamentally different grammatical functions depending on context. The sign 𓅱 (G43, quail chick) can represent the phoneme *w*, act as a logogram for "chick", or serve as a determinative. Image-based classifiers fail here because the visual signal is identical across all three uses. This project addresses that gap by treating role classification as a text understanding problem, encoding the rich contextual metadata that Egyptological transcriptions already provide.

---

## 2. Dataset

The training dataset (`data/contextual_examples_real_weak.csv`) contains **7,870 rows** derived from the Ramses/AES corpus (Zenodo `10.5281/zenodo.7991241`). Each row captures one hieroglyph token in its textual context:

| Field | Description |
|---|---|
| `target_sign` | Gardiner code of the sign being classified |
| `context_before` / `context_after` | Neighbouring signs in the sequence |
| `transliteration` | Egyptian transliteration of the phrase |
| `translation` | English gloss |
| `role` | Weak label: `phonetic`, `logographic`, or `determinative` |

Labels are assigned automatically using three high-confidence heuristic rules. A sign is labelled *phonetic* if its phonetic value appears in the transliteration without accompanying meaning evidence; *logographic* if its reading and meaning both appear in context; *determinative* if its meaning appears in the translation but the sign is not reflected phonetically. Ambiguous rows are discarded. A small hand-annotated set of 12 examples (`data/contextual_examples.csv`) served as a development reference.

---

## 3. Model

The classifier uses a two-stage pipeline:

**Stage 1 — Embedding.** The concatenated context fields (`context_before`, `target_sign`, `context_after`, `transliteration`, `translation`) are encoded with `sentence-transformers/all-MiniLM-L6-v2`, a 384-dimensional transformer model using mean-pooling with attention masking.

**Stage 2 — Classification.** A Logistic Regression classifier (`sklearn`, `C=2.0`, `class_weight='balanced'`, `max_iter=1500`) is trained on these embeddings. Using a stratified 75/25 train-test split (`random_state=42`), the model achieves **100% accuracy** on the hold-out set. This reflects strong consistency within the weak-labelling regime; true generalisation to expert-annotated data would require additional evaluation. The trained artefacts (`classifier.joblib`, `label_encoder.joblib`) are committed to the repository so no separate download is needed at inference time.

---

## 4. Demo

The interactive demo is hosted at **https://huggingface.co/spaces/Kucina111/Egyptian_hieroglyph_classifier**.

The interface presents two real corpus examples with a visual context strip of hieroglyph sign images. Users select a role for each target sign, then click *Check Answers* to see the correct weak-label and compare their reasoning against the model's top-3 predicted probabilities and confidence scores. A sign reference table shows Gardiner codes, phonetic values, and logographic readings. Six further weak-labelled examples are displayed for broader exploration.

---

## 5. Reflection on Working with AI

This project was developed with assistance from **Claude Code** (Anthropic), an AI coding assistant used throughout the implementation. This section honestly describes what the AI contributed and where human judgement remained essential.

**Contributions of AI assistance.** Claude Code generated the majority of the infrastructure code: the Gradio UI layout, the CSV parsing utilities, the weak-labelling heuristics in `src/real_context_data.py`, the embedding and training pipeline in `src/context_train.py`, and the deployment configuration. It also diagnosed build failures rapidly — when the HuggingFace Space rejected a push due to invalid colour values (`teal`, `amber`), and when a Python 3.13 / `pyaudioop` incompatibility broke the Gradio install, both were identified and fixed within a single exchange.

**Where human oversight was essential.** The AI has no Egyptological domain knowledge. Decisions about which contextual fields to encode, what the three role categories mean linguistically, and whether the weak-labelling rules were archaeologically plausible required checking against references such as Gardiner's *Egyptian Grammar*. The AI also introduced errors that needed catching: it initially pushed to the wrong git branch (`master` instead of `main`), causing the Space to show a blank "Get Started" page, and it wrote CSS to `demo.launch()` where Gradio ignores it rather than to the `gr.Blocks()` constructor. The 100% test accuracy also needs to be interpreted critically — it reflects the internal consistency of the weak labels, not independently validated performance, a distinction the AI did not flag unprompted.

**Overall assessment.** AI assistance substantially accelerated implementation, particularly for repetitive infrastructure and deployment tasks. However, conceptual design, data quality assessment, critical interpretation of results, and verification of AI outputs required sustained human involvement. The tool removes friction; it does not replace judgement.

---

## References

- Ramses Project / AES Corpus. (2023). Zenodo. https://doi.org/10.5281/zenodo.7991241
- Gardiner, A. H. (1957). *Egyptian Grammar* (3rd ed.). Griffith Institute.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP 2019*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
