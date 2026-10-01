"""Where a learner keeps going wrong — the "điểm yếu" report.

`error_log` holds every wrong answer (a review card, an exam question, a writing
slip) and `exam_attempt` every answer to an exam question. This module only
*classifies and counts*; the router does the queries and hands plain rows over,
so the rules are unit-tested without a database.

The point of the report is to be actionable and honest: a count of what went
wrong in the last N days, the cards that went wrong more than once, the sets
("họ từ") the mistakes cluster in, and per question type how many exam answers
were right — never a prediction of an exam score.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

REPORT_DAYS = 30
MIN_REPEATS = 2  # a card is "hay sai" once it went wrong this many times in the window
MAX_TOP_ITEMS = 6
MAX_FAMILIES = 4

# error_type -> what the learner reads. Exam rows use the question type's own
# Vietnamese name instead (it comes from content.question_type).
ERROR_LABELS: dict[str, str] = {
    "collocation": "Chọn sai từ đi cùng trong cụm",
    "word_in_context": "Điền sai từ vào câu",
    "vocab_recall": "Quên nghĩa của từ/cụm",
    "grammar_recall": "Quên cách dùng ngữ pháp",
    "writing_grammar": "Viết sai ngữ pháp",
    "writing_spelling": "Viết sai chính tả",
    "writing_register": "Sai mức độ lịch sự (-요 / -ㅂ니다 / 반말)",
    "writing_word_choice": "Chọn từ chưa tự nhiên",
    "writing_content": "Câu chưa đúng ý của chỗ trống",
}


def error_type_for_review(item_type: str, mode: str, has_node_word: bool) -> str:
    """The kind of mistake a wrong review answer is. A fill-in-the-blank on a chunk
    that has a node word tests the collocation; on a plain word it tests the word
    in a sentence; a card judged "chưa nhớ" is a failure to recall."""
    if item_type == "grammar_point":
        return "grammar_recall"
    if mode == "cloze":
        return "collocation" if has_node_word else "word_in_context"
    return "vocab_recall"


def label_for(error_type: str, qtype_names: Mapping[str, str] | None = None) -> str:
    if qtype_names and error_type in qtype_names:
        return qtype_names[error_type]
    return ERROR_LABELS.get(error_type, error_type)


ItemKey = tuple[str, int]


@dataclass(frozen=True)
class ErrorRow:
    skill: str
    error_type: str
    item_type: str | None = None
    item_id: int | None = None


@dataclass(frozen=True)
class TypeCount:
    error_type: str
    skill: str
    label: str
    count: int


@dataclass(frozen=True)
class RepeatItem:
    key: ItemKey
    errors: int


@dataclass(frozen=True)
class FamilyCount:
    family: str
    errors: int


@dataclass(frozen=True)
class QtypeAccuracy:
    qtype_code: str
    name: str
    attempts: int
    correct: int

    @property
    def accuracy_pct(self) -> int:
        return round(100 * self.correct / self.attempts) if self.attempts else 0


def count_by_type(rows: Iterable[ErrorRow], qtype_names: Mapping[str, str] | None = None) -> list[TypeCount]:
    """Most frequent kind of mistake first (ties: by name, so the order is stable)."""
    counter: Counter[tuple[str, str]] = Counter((r.error_type, r.skill) for r in rows)
    out = [
        TypeCount(error_type=t, skill=skill, label=label_for(t, qtype_names), count=n)
        for (t, skill), n in counter.items()
    ]
    out.sort(key=lambda c: (-c.count, c.error_type))
    return out


def repeat_items(rows: Iterable[ErrorRow], *, minimum: int = MIN_REPEATS, limit: int = MAX_TOP_ITEMS) -> list[RepeatItem]:
    """The cards that went wrong more than once, worst first."""
    counter: Counter[ItemKey] = Counter(
        (r.item_type, r.item_id) for r in rows if r.item_type is not None and r.item_id is not None
    )
    out = [RepeatItem(key=key, errors=n) for key, n in counter.items() if n >= minimum]
    out.sort(key=lambda r: (-r.errors, r.key))
    return out[:limit]


def count_by_family(
    rows: Iterable[ErrorRow], family_of: Mapping[ItemKey, str | None], *, limit: int = MAX_FAMILIES
) -> list[FamilyCount]:
    """The sets ("Động từ đi với thời tiết") the mistakes gather in. A set only shows
    once at least two mistakes fell in it — one slip is not a pattern."""
    counter: Counter[str] = Counter()
    for r in rows:
        if r.item_type is None or r.item_id is None:
            continue
        family = family_of.get((r.item_type, r.item_id))
        if family:
            counter[family] += 1
    out = [FamilyCount(family=f, errors=n) for f, n in counter.items() if n >= 2]
    out.sort(key=lambda f: (-f.errors, f.family))
    return out[:limit]


def exam_accuracy(
    attempts: Iterable[tuple[str, bool]], qtype_names: Mapping[str, str]
) -> list[QtypeAccuracy]:
    """Per question type, how many of the learner's exam answers were right —
    weakest type first. Types with a single answer are kept (the count is shown,
    so a 0/1 is not mistaken for a trend)."""
    attempted: Counter[str] = Counter()
    right: Counter[str] = Counter()
    for code, correct in attempts:
        attempted[code] += 1
        if correct:
            right[code] += 1
    out = [
        QtypeAccuracy(qtype_code=code, name=qtype_names.get(code, code), attempts=n, correct=right[code])
        for code, n in attempted.items()
    ]
    out.sort(key=lambda a: (a.accuracy_pct, -a.attempts, a.qtype_code))
    return out
