"""Fill-in-the-blank questions built from a vocabulary card, with no model call.

TOPIK's blank questions ask which word goes with the rest ("비가 ___" — 오다, 불다
or 들다?), so the blank is the *node word* of a chunk and the wrong choices are
words that look right to a Vietnamese learner: the card's own `distractors`
first, then the node words of the other chunks in its family (they are real
Korean words, and wrong *here*).

A card without a node word can still be asked when its example sentence holds the
word verbatim ("오늘 날씨가 좋아요" for 날씨): the word is blanked and the wrong
choices are other words of the same part of speech from the same lesson.

Anything that cannot give at least three choices returns None and the card is
simply reviewed the ordinary way.
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

BLANK = "____"
MIN_CHOICES = 3
MAX_CHOICES = 4
_KEY_STRIP_RE = re.compile(r"[\s·\-~()（）.,:;!?]")


class Card(Protocol):
    id: int
    hangul: str
    pos: str | None
    example_ko: str | None
    family: str | None
    node_word: str | None
    distractors: list[str] | None
    lesson_id: int | None


@dataclass(frozen=True)
class Cloze:
    prompt_ko: str
    answer: str
    choices: tuple[str, ...]


def _key(text: str) -> str:
    return _KEY_STRIP_RE.sub("", text).lower()


def _stem(word: str) -> str | None:
    """The part of a dictionary-form verb/adjective that survives conjugation,
    when it is long enough not to match by accident."""
    return word[:-1] if word.endswith("다") and len(word) >= 3 else None


def _pick(answer: str, pool: Sequence[str], rng: random.Random) -> tuple[str, ...] | None:
    seen = {_key(answer)}
    wrong: list[str] = []
    for word in pool:
        if word and _key(word) not in seen:
            seen.add(_key(word))
            wrong.append(word)
    if len(wrong) + 1 < MIN_CHOICES:
        return None
    chosen = wrong[: MAX_CHOICES - 1]
    choices = [answer, *chosen]
    rng.shuffle(choices)
    return tuple(choices)


def build_cloze(card: Card, mates: Sequence[Card], seed: str) -> Cloze | None:
    """A blank question for `card`, or None when it cannot be asked well.
    `mates` are the cards of the same lesson (the card itself may be among them).
    `seed` makes the choices' order stable for one sitting."""
    rng = random.Random(seed)
    others = [m for m in mates if m.id != card.id]

    node = card.node_word
    if node and node in card.hangul and node != card.hangul:
        pool = list(card.distractors or [])
        pool += [
            m.node_word
            for m in others
            if m.node_word and card.family is not None and m.family == card.family
        ]
        choices = _pick(node, pool, rng)
        if choices is None:
            return None
        return Cloze(prompt_ko=card.hangul.replace(node, BLANK, 1), answer=node, choices=choices)

    example = card.example_ko or ""
    if not example or not card.pos:  # without a part of speech the wrong choices could be any word
        return None
    if card.hangul in example:
        answer = card.hangul
        pool = [m.hangul for m in others if m.pos == card.pos and m.hangul not in example]
    else:
        stem = _stem(card.hangul)
        if stem is None or len(stem) < 2 or stem not in example:
            return None
        answer = stem
        pool = [s for m in others if m.pos == card.pos and (s := _stem(m.hangul)) and s not in example]
    rng.shuffle(pool)
    choices = _pick(answer, pool, rng)
    if choices is None:
        return None
    return Cloze(prompt_ko=example.replace(answer, BLANK, 1), answer=answer, choices=choices)
