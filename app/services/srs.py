"""A small spaced-repetition schedule, and the daily review queue built on it.

`ItemState.strength` (0..1, +/-0.2 per answer) stays the readiness signal. This
module adds *when an item should come back*, in the way an SM-2 deck does but
for a plain right/wrong answer:

* a correct answer moves the item one step up a ladder of gaps — 1 day, 3 days,
  then each gap is the last one times `ease` (about 1-3-7-15-34-75 days);
* a wrong answer is a lapse: the ladder restarts, the item is due again in ten
  minutes (so it comes back in the same sitting) and its `ease` drops a little,
  so an item that keeps failing is seen more often than one that doesn't;
* a correct answer to an item that is not due yet changes nothing in the
  schedule — re-running a deck the same day must not push every card out to
  next month. (Strength still moves, as it always did.)

Pure functions only (no database, no clock of its own): the router loads the rows,
hands them over and stores what comes back, so the rules are unit-tested without
Postgres.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

START_EASE = 2.2
MIN_EASE = 1.3
LAPSE_EASE_PENALTY = 0.2
MAX_INTERVAL_DAYS = 180.0
FIRST_INTERVAL_DAYS = 1.0
SECOND_INTERVAL_DAYS = 3.0
RELEARN_DELAY = timedelta(minutes=10)
STRENGTH_STEP = 0.2

# An item counts as due this long before its due time. People study at about the
# same hour each day, so "due in exactly 24 h" would otherwise mean "tomorrow,
# an hour after you open the app".
DUE_GRACE = timedelta(hours=6)

DEFAULT_REVIEW_LIMIT = 20  # due cards in one sitting
DEFAULT_NEW_PER_DAY = 8  # new cards per rolling 24 h
NEW_WINDOW = timedelta(hours=24)
GRAMMAR_SHARE = 4  # one new card in four is a grammar point (when there is one)

ItemKey = tuple[str, int]  # ("vocab_item" | "grammar_point", content id)


@dataclass(frozen=True)
class Schedule:
    strength: float = 0.0
    reps: int = 0  # correct answers in a row since the last lapse
    lapses: int = 0
    ease: float = START_EASE
    interval_days: float = 0.0
    due_at: datetime | None = None
    introduced_at: datetime | None = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def review(prev: Schedule | None, correct: bool, now: datetime) -> Schedule:
    """The schedule after one answer given at `now`."""
    base = prev or Schedule(introduced_at=now)
    strength = _clamp(base.strength + (STRENGTH_STEP if correct else -STRENGTH_STEP))
    introduced = base.introduced_at or now

    if not correct:
        return Schedule(
            strength=strength,
            reps=0,
            lapses=base.lapses + 1,
            ease=max(MIN_EASE, base.ease - LAPSE_EASE_PENALTY),
            interval_days=0.0,
            due_at=now + RELEARN_DELAY,
            introduced_at=introduced,
        )

    if prev is not None and prev.due_at is not None and now < prev.due_at - DUE_GRACE:
        return replace(base, strength=strength, introduced_at=introduced)  # early: no credit

    reps = base.reps + 1
    if reps == 1:
        interval = FIRST_INTERVAL_DAYS
    elif reps == 2:
        interval = SECOND_INTERVAL_DAYS
    else:
        interval = max(base.interval_days + 1, round(base.interval_days * base.ease))
    interval = min(MAX_INTERVAL_DAYS, float(interval))
    return Schedule(
        strength=strength,
        reps=reps,
        lapses=base.lapses,
        ease=base.ease,
        interval_days=interval,
        due_at=now + timedelta(days=interval),
        introduced_at=introduced,
    )


def is_due(schedule: Schedule, now: datetime) -> bool:
    return schedule.due_at is not None and schedule.due_at <= now + DUE_GRACE


@dataclass
class Queue:
    due: list[ItemKey] = field(default_factory=list)  # most overdue first, cut at `limit`
    due_total: int = 0  # everything due, before the cut
    new: list[ItemKey] = field(default_factory=list)
    new_today: int = 0  # started in the last 24 h
    new_budget: int = 0  # how many new cards today's cap still allows


def _spread(main: list[ItemKey], extra: list[ItemKey]) -> list[ItemKey]:
    """`extra` spaced evenly among `main` (grammar in between the chunks, not
    all at the end when the learner is tired)."""
    if not extra:
        return list(main)
    out: list[ItemKey] = []
    step = max(1, round(len(main) / (len(extra) + 1)))
    pending = list(extra)
    for i, key in enumerate(main, start=1):
        out.append(key)
        if pending and i % step == 0:
            out.append(pending.pop(0))
    out.extend(pending)
    return out


def build_queue(
    schedules: Mapping[ItemKey, Schedule],
    lessons: Mapping[int, Sequence[ItemKey]],
    now: datetime,
    *,
    limit: int = DEFAULT_REVIEW_LIMIT,
    new_limit: int = DEFAULT_NEW_PER_DAY,
) -> Queue:
    """What to study now.

    `schedules` is this learner's ItemState rows; `lessons` is lesson id -> the
    items in it (only items that exist are in there, so a row whose lesson was
    rolled back is never offered). Due items come first, most overdue first. New
    items come from the lowest-numbered lessons that still have unseen items,
    capped at `new_limit` per rolling 24 h, about one grammar point to three
    chunks, and every item of a family stays next to its mates because the
    lesson's own order is kept.
    """
    in_lessons = {key for items in lessons.values() for key in items}
    due = sorted(
        (key for key, s in schedules.items() if key in in_lessons and is_due(s, now)),
        key=lambda key: (schedules[key].due_at, key),  # type: ignore[arg-type,return-value]
    )

    introduced = sum(
        1 for s in schedules.values() if s.introduced_at is not None and s.introduced_at > now - NEW_WINDOW
    )
    budget = max(0, new_limit - introduced)

    vocab_pool: list[ItemKey] = []
    grammar_pool: list[ItemKey] = []
    for lesson_id in sorted(lessons):
        unseen = sorted(key for key in lessons[lesson_id] if key not in schedules)
        vocab_pool += [key for key in unseen if key[0] == "vocab_item"]
        grammar_pool += [key for key in unseen if key[0] == "grammar_point"]
    grammar_n = min(len(grammar_pool), budget // GRAMMAR_SHARE)
    vocab_n = min(len(vocab_pool), budget - grammar_n)
    grammar_n = min(len(grammar_pool), budget - vocab_n)  # no vocab left: grammar fills the day
    new = _spread(vocab_pool[:vocab_n], grammar_pool[:grammar_n])

    return Queue(due=due[:limit], due_total=len(due), new=new, new_today=introduced, new_budget=budget)
