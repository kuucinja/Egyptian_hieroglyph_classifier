from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.context_features import ContextExample
from src.context_predict import ContextRoleClassifier
from src.sign_images import image_for_token, load_image_manifest
from src.sign_reference import lookup_sign


ROLE_CHOICES = ["phonetic", "logographic", "determinative"]
REAL_DATA_FILE = Path("data/contextual_examples_real_weak.csv")
PRIMARY_EXAMPLE_IDS = ["real_000014_001", "real_000564_005"]
EXTRA_EXAMPLE_IDS = [
    "real_000032_001",
    "real_000033_001",
    "real_000639_000",
    "real_000641_000",
    "real_000023_000",
    "real_000024_000",
]

CUSTOM_CSS = """
.sign-strip {
    align-items: flex-end;
    gap: 8px;
    overflow-x: auto;
    padding: 8px 0 14px 0;
}
.context-sign, .target-sign {
    min-width: 96px;
}
.target-sign {
    border: 3px solid #d11f1f !important;
    border-radius: 8px !important;
    padding: 6px !important;
    background: #fff5f5 !important;
}
.missing-sign textarea {
    min-height: 72px !important;
    text-align: center;
    font-weight: 700;
}
"""


def load_real_rows() -> dict[str, dict[str, str]]:
    if not REAL_DATA_FILE.exists():
        return {}
    with REAL_DATA_FILE.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def full_context_from_notes(notes: str) -> str:
    match = re.search(r"Full context:\s*(.+)$", notes)
    return match.group(1).strip() if match else ""


def row_to_example(row: dict[str, str]) -> ContextExample:
    return ContextExample(
        target_sign=row["target_sign"],
        context_before=row["context_before"],
        context_after=row["context_after"],
        transliteration=row["transliteration"],
        translation=row["translation"],
        notes=row["notes"],
    )


def target_code(row: dict[str, str]) -> str:
    return lookup_sign(row["target_sign"])["gardiner_code"] or row["target_sign"]


def sign_reference_row(row: dict[str, str]) -> list[str]:
    sign = lookup_sign(row["target_sign"])
    return [
        sign["display_name"],
        sign["gardiner_code"],
        sign["phonetic_value"] or "not usually pronounced",
        sign["logographic_reading"] or "context dependent",
        sign["meaning"],
        sign["notes"],
    ]


@lru_cache(maxsize=1)
def get_classifier() -> ContextRoleClassifier:
    return ContextRoleClassifier()


def model_prediction_rows(rows: list[dict[str, str]]) -> list[list[str | float]]:
    classifier = get_classifier()
    output: list[list[str | float]] = []
    for index, row in enumerate(rows, start=1):
        prediction = classifier.predict(row_to_example(row), top_k=3)
        best_role = next(iter(prediction))
        output.append([f"Example {index}", best_role, round(prediction[best_role], 4), row["role"]])
    return output


def check_answers(answer_one: str, answer_two: str) -> tuple[str, list[list[str | float]]]:
    answers = [answer_one, answer_two]
    rows = [EXAMPLES[example_id] for example_id in PRIMARY_EXAMPLE_IDS if example_id in EXAMPLES]
    lines: list[str] = []
    for index, row in enumerate(rows):
        selected = answers[index]
        correct = row["role"]
        if selected == correct:
            result = "Correct"
        elif not selected:
            result = "Choose an answer"
        else:
            result = f"Not quite. Correct answer: {correct}"
        lines.append(f"**Example {index + 1}**: {result}\n\n{row['notes']}")
    return "\n\n".join(lines), model_prediction_rows(rows)


def token_sequence(row: dict[str, str]) -> list[tuple[str, bool]]:
    before = row["context_before"].split()
    after = row["context_after"].split()
    return [(token, False) for token in before] + [(target_code(row), True)] + [(token, False) for token in after]


def add_token_visual(token: str, is_target: bool) -> None:
    manifest = load_image_manifest()
    image_path = image_for_token(token, manifest)
    label = f"TARGET: {token}" if is_target else token
    elem_classes = ["target-sign"] if is_target else ["context-sign"]
    if image_path:
        gr.Image(
            value=image_path,
            label=label,
            height=130 if is_target else 105,
            interactive=False,
            elem_classes=elem_classes,
        )
    else:
        gr.Textbox(
            value=f"Image missing: {token}",
            label=label,
            interactive=False,
            elem_classes=elem_classes + ["missing-sign"],
        )


def add_context_strip(row: dict[str, str]) -> None:
    gr.Markdown("**Visual context sequence**")
    with gr.Row(equal_height=True, elem_classes=["sign-strip"]):
        for token, is_target in token_sequence(row):
            add_token_visual(token, is_target)


def format_example(row: dict[str, str], number: int) -> str:
    full_context = full_context_from_notes(row["notes"])
    translation = row["translation"] or "English translation unavailable in this corpus row"
    return (
        f"### Example {number}: target `{row['target_sign']}`\n"
        f"Real corpus encoding: `{full_context or '[not available]'}`  \n"
        f"Transliteration: `{row['transliteration'] or '[not available]'}`\n\n"
        f"English translation/gloss: {translation}\n\n"
        f"Weak corpus label: `{row['role']}`"
    )


def extra_examples_table() -> list[list[str]]:
    rows: list[list[str]] = []
    for example_id in EXTRA_EXAMPLE_IDS:
        row = EXAMPLES.get(example_id)
        if not row:
            continue
        sign = lookup_sign(row["target_sign"])
        rows.append(
            [
                row["id"],
                row["target_sign"],
                sign["gardiner_code"],
                full_context_from_notes(row["notes"]),
                row["transliteration"],
                row["role"],
            ]
        )
    return rows


EXAMPLES = load_real_rows()
PRIMARY_ROWS = [EXAMPLES[example_id] for example_id in PRIMARY_EXAMPLE_IDS if example_id in EXAMPLES]


with gr.Blocks(title="Contextual Hieroglyph Role Classifier") as demo:
    gr.Markdown(
        "# Contextual Hieroglyph Role Classifier\n"
        "A hieroglyph's shape alone does not always tell you how it is being used. "
        "The same sign can represent a sound, stand for a whole word, or silently clarify meaning. "
        "Context is what lets a learner decide whether a sign is phonetic, logographic, or determinative."
    )

    gr.Markdown("## Real Corpus Practice")
    with gr.Row(equal_height=False):
        with gr.Column():
            add_context_strip(PRIMARY_ROWS[0])
            gr.Markdown(format_example(PRIMARY_ROWS[0], 1))
            answer_one = gr.Radio(ROLE_CHOICES, label="What is the target sign doing here?")
        with gr.Column():
            add_context_strip(PRIMARY_ROWS[1])
            gr.Markdown(format_example(PRIMARY_ROWS[1], 2))
            answer_two = gr.Radio(ROLE_CHOICES, label="What is the target sign doing here?")

    check_button = gr.Button("Check Answers", variant="primary")
    feedback = gr.Markdown()
    model_output = gr.Dataframe(
        headers=["example", "model_prediction", "confidence", "weak_label"],
        label="Model comparison",
        interactive=False,
    )

    gr.Markdown("## Target Sign Reference")
    sign_output = gr.Dataframe(
        value=[sign_reference_row(row) for row in PRIMARY_ROWS],
        headers=[
            "name",
            "gardiner_code",
            "phonetic_sound",
            "logographic_reading",
            "what_it_depicts",
            "description",
        ],
        interactive=False,
    )

    gr.Markdown("## More Real Weak-Labeled Examples")
    gr.Dataframe(
        value=extra_examples_table(),
        headers=["id", "target", "gardiner_code", "real_encoding", "transliteration", "weak_label"],
        interactive=False,
    )

    check_button.click(check_answers, inputs=[answer_one, answer_two], outputs=[feedback, model_output])


demo.launch(server_name="0.0.0.0", css=CUSTOM_CSS)
