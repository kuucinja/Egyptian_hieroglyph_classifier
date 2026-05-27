from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextExample:
    target_sign: str
    context_before: str
    context_after: str
    transliteration: str
    translation: str
    notes: str = ""


def build_context_text(example: ContextExample) -> str:
    return (
        f"before signs: {example.context_before or '[none]'}\n"
        f"target sign: {example.target_sign}\n"
        f"after signs: {example.context_after or '[none]'}\n"
        f"transliteration: {example.transliteration or '[unknown]'}\n"
        f"translation or gloss: {example.translation or '[unknown]'}\n"
        f"annotation notes: {example.notes or '[none]'}"
    )

