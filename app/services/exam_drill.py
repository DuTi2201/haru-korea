"""Mini-drills made of *real* exam questions that a human has confirmed.

Nothing here is generated: a question is only offered when it came from an
imported exam paper and its answer was read from an answer key (`answer_source`
"editor"). A question that needs a picture, a chart or audio the app does not
keep (listening questions, charts without text) is never offered, because it
could not be answered fairly.

Pure functions only — the router loads the rows and the learner's attempts and
hands them over — so the rules are unit-tested without a database.
"""
from __future__ import annotations

import random
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_DRILL_SIZE = 5
MAX_DRILL_SIZE = 10
MIN_OPTIONS = 2
MAX_OPTIONS = 5
RIGHT_AGAIN_AFTER = timedelta(days=7)  # a question answered right is left alone this long
WRONG_AGAIN_AFTER = timedelta(hours=12)  # ...one answered wrong may come back after this
JUST_ASKED = timedelta(minutes=30)  # nothing is asked twice in one sitting

# Question types that cannot be read without the passage they belong to.
NEEDS_PASSAGE = {"read_short_passage", "read_long_passage", "read_title_topic", "read_chart_info"}
READING_SKILL = "đọc"
LISTENING_PASSAGE_KIND = "nghe"


@dataclass(frozen=True)
class Candidate:
    item_id: uuid.UUID
    number: int
    qtype_code: str
    stem_ko: str
    options: object  # the JSON column as stored; usable() checks its shape
    answer: int | None
    answer_source: str | None
    skill: str
    passage_kind: str | None
    passage_ko: str | None


def clean_options(options: object) -> list[str] | None:
    """The answer choices as a list of non-empty strings, or None when the stored
    shape is not one a learner can answer."""
    if not isinstance(options, list):
        return None
    cleaned = [o.strip() for o in options if isinstance(o, str) and o.strip()]
    if len(cleaned) != len(options) or not MIN_OPTIONS <= len(cleaned) <= MAX_OPTIONS:
        return None
    return cleaned


def usable(c: Candidate) -> bool:
    """Can this question be answered fairly and graded against a trusted key?"""
    if c.answer_source != "editor" or c.answer is None:
        return False
    if c.skill != READING_SKILL or c.passage_kind == LISTENING_PASSAGE_KIND:
        return False
    options = clean_options(c.options)
    if options is None or not 1 <= c.answer <= len(options):
        return False
    if not c.stem_ko.strip():
        return False
    if c.qtype_code in NEEDS_PASSAGE and not (c.passage_ko or "").strip():
        return False
    return True


@dataclass(frozen=True)
class Attempt:
    item_id: uuid.UUID
    correct: bool
    at: datetime


def pick(
    candidates: Iterable[Candidate],
    attempts: Iterable[Attempt],
    accuracy_by_type: Mapping[str, int],
    now: datetime,
    *,
    n: int = DEFAULT_DRILL_SIZE,
    seed: str = "",
) -> list[Candidate]:
    """Which questions to ask now.

    A question asked in the last half hour, or answered right in the last week,
    is skipped. Of the rest, those the learner got wrong before come first (a
    mistake is worth meeting again), then those never asked; within each group the
    question types with the lowest accuracy so far come first, and ties are broken
    by a shuffle that is stable for `seed` (so a page reload shows the same set).
    """
    last: dict[uuid.UUID, Attempt] = {}
    for a in sorted(attempts, key=lambda a: a.at):
        last[a.item_id] = a

    pool: list[tuple[int, int, Candidate]] = []
    for c in candidates:
        if not usable(c):
            continue
        previous = last.get(c.item_id)
        if previous is None:
            group = 1
        else:
            age = now - previous.at
            if age < JUST_ASKED:
                continue
            if previous.correct:
                if age < RIGHT_AGAIN_AFTER:
                    continue
                group = 2
            else:
                if age < WRONG_AGAIN_AFTER:
                    continue
                group = 0
        pool.append((group, accuracy_by_type.get(c.qtype_code, 50), c))

    rng = random.Random(seed)
    rng.shuffle(pool)  # the stable tie-break
    pool.sort(key=lambda row: (row[0], row[1]))
    return [c for _, _, c in pool[: max(0, min(n, MAX_DRILL_SIZE))]]


def accuracy_by_type(attempts: Iterable[tuple[str, bool]]) -> dict[str, int]:
    """qtype code -> percent right, from (qtype code, correct) pairs."""
    seen: dict[str, list[bool]] = defaultdict(list)
    for code, correct in attempts:
        seen[code].append(correct)
    return {code: round(100 * sum(v) / len(v)) for code, v in seen.items()}


def grade(candidate: Candidate, chosen: int) -> bool:
    """Whether choice `chosen` (1-based) is the key. The caller has checked usable()."""
    return candidate.answer is not None and chosen == candidate.answer


def option_count(candidate: Candidate) -> int:
    return len(clean_options(candidate.options) or [])


__all__ = [
    "Attempt",
    "Candidate",
    "DEFAULT_DRILL_SIZE",
    "MAX_DRILL_SIZE",
    "accuracy_by_type",
    "clean_options",
    "grade",
    "option_count",
    "pick",
    "usable",
]

