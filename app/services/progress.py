"""Per-learner progress helpers that need no schema of their own.

`ItemState.strength` (0..1, nudged +/-0.2 by /progress/reviews) is the only
per-learner memory signal the spec defines (SRS §5 ITEM_STATE) — this module
only *reads* it. It is deliberately pure (plain dicts in, a small dataclass
out, no DB session) so the selection rule can be unit-tested without
Postgres; the router does the two bulk queries and hands the rows over.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

# An item counts as "learned" once four correct nudges in a row have lifted it
# to 0.8. Compared with a small epsilon: 0.2 steps accumulate float noise.
MASTERY_THRESHOLD = 0.8
_EPS = 1e-9

ItemKey = tuple[str, int]  # ("vocab_item" | "grammar_point", content id)


def is_mastered(strength: float | None) -> bool:
    return strength is not None and strength >= MASTERY_THRESHOLD - _EPS


@dataclass(frozen=True)
class LessonProgress:
    lesson_id: int
    total: int
    mastered: int

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.mastered >= self.total


def count_mastered(items: Sequence[ItemKey], states: Mapping[ItemKey, tuple[float, datetime]]) -> int:
    """How many of `items` the learner has mastered (items never reviewed count as not)."""
    return sum(1 for key in items if (state := states.get(key)) is not None and is_mastered(state[0]))


def pick_next_lesson(
    lessons: Mapping[int, Sequence[ItemKey]],
    states: Mapping[ItemKey, tuple[float, datetime]],
) -> LessonProgress | None:
    """Which lesson should this learner see next?

    * `lessons`: lesson id -> the vocab/grammar items that belong to it.
      Lessons without any item are skipped (nothing to study in them).
    * `states`: this learner's ItemState rows as (strength, last_seen),
      keyed by (item_type, item_id). Items with no row are simply "not
      started yet".

    Rule: the lowest-numbered lesson that still has an item the learner has
    not mastered — so a learner who finished lesson 1 is moved on to lesson 2
    instead of being handed lesson 1 forever, and one who is halfway through
    lesson 2 resumes there. If every lesson is mastered, return the one whose
    least-recently-seen item is oldest (the one most due for a refresher).
    Returns None when there is no lesson with any item at all.
    """
    progress: list[tuple[LessonProgress, datetime | None]] = []
    for lesson_id in sorted(lessons):
        items = lessons[lesson_id]
        if not items:
            continue
        mastered = 0
        oldest: datetime | None = None
        for key in items:
            state = states.get(key)
            if state is None:
                continue
            strength, last_seen = state
            if is_mastered(strength):
                mastered += 1
            if oldest is None or last_seen < oldest:
                oldest = last_seen
        progress.append((LessonProgress(lesson_id, len(items), mastered), oldest))

    if not progress:
        return None
    for entry, _ in progress:
        if not entry.complete:
            return entry

    # Everything mastered: refresh the lesson that has gone longest unseen.
    # `oldest` is never None here — a mastered item has a state row.
    return min(progress, key=lambda p: (p[1], p[0].lesson_id))[0]
