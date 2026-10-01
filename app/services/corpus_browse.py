"""Browsing the listening corpus: de-duplication, register detection, facets
and a stable, repeat-free order. Everything here is a pure function over
plain values (no DB, no I/O), so it is unit-tested without a database — the
router loads one light snapshot of corpus.corpus_item and hands it in.

Why in Python and not SQL: "the same sentence" needs a normaliser that treats
punctuation/spacing the way a reader does, and a Postgres regex class for
Hangul depends on the server's locale. One Python implementation is also the
single definition used by the importer (so a line that is already in the
corpus never reaches the reviewer twice) and by the learner endpoints.
"""
from __future__ import annotations

import hashlib
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, replace

REGISTER_POLITE = "존댓말"
REGISTER_CASUAL = "반말"
REGISTER_MIXED = "hỗn hợp"

# U+AC00 is the first precomposed Hangul syllable; a syllable is
# 0xAC00 + (lead * 21 + vowel) * 28 + final, with final 0 = "no batchim".
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_FINAL_BIEUP = 17  # ㅂ as a final consonant: 합|니다, 습|니다, 입|니까


@dataclass(frozen=True)
class Sentence:
    """One corpus sentence, as light as the browsing endpoints need."""

    id: uuid.UUID
    film_id: int
    text_ko: str
    kind: str
    level: int
    register: str
    topic_ids: tuple[int, ...] = ()
    grammar_ids: tuple[int, ...] = ()


# ------------------------------------------------------------ normalising --
def normalize_key(text: str) -> str:
    """Identity of a sentence for de-duplication: NFKC, case-folded, with every
    space, punctuation mark, symbol and control character removed — so
    "아니요, 저는 괜찮아요." and "아니요 저는 괜찮아요" are one sentence. Different
    word forms (해요 vs 합니다) stay different: that is a variant, not a copy."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in ("L", "N", "M"))


# -------------------------------------------------------------- register ---
def _syllable_final(ch: str) -> int | None:
    code = ord(ch)
    if not (_HANGUL_BASE <= code <= _HANGUL_LAST):
        return None
    return (code - _HANGUL_BASE) % 28


def _sentences(text: str) -> list[str]:
    """The sentences of a subtitle line (a cue often holds two or three)."""
    parts: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch in ".?!…~\n":
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p for p in (part.strip().strip("\"'“”‘’()[]-– ") for part in parts) if p]


# Plain-style (반말 / 해라체) endings that are safe to call on their own. Left
# out on purpose: "는데" and "고" (clause connectives that also end polite
# sentences), "까" (-ㄹ까 vs the -니까 connective) — better unknown than wrong.
_CASUAL_ENDINGS = (
    "다", "야", "어", "아", "지", "냐", "니", "자", "라", "해", "봐", "줘",
    "래", "걸", "네", "군", "가", "거든", "잖아",
)  # fmt: skip
# Nouns that end in 요 and are not a 해요체 sentence ("그게 필요.").
_NOT_POLITE_YO = ("필요", "중요", "수요", "주요")


def _piece_register(piece: str) -> str | None:
    """존댓말 / 반말 for one clause, or None when its ending says nothing (a
    fragment such as "더 좋은 계약으로", an interjection such as "예")."""
    # Formal: -ㅂ니다 / -ㅂ니까 / -ㅂ시다. The syllable before carries a final ㅂ
    # (합|니다, 습|니까, 갑|시다); that is what keeps "아니다", a plain verb, out.
    for tail in ("니다", "니까", "시다"):
        if piece.endswith(tail) and len(piece) > 2 and _syllable_final(piece[-3]) == _FINAL_BIEUP:
            return REGISTER_POLITE
    if piece.endswith("시오") or piece.endswith("죠"):
        return REGISTER_POLITE
    if piece.endswith("요") and not piece.endswith(_NOT_POLITE_YO):
        return REGISTER_POLITE  # 해요체: -아/어요, -예요, -세요, -네요, -(으)ㄹ까요
    if len(piece) >= 2 and piece.endswith(_CASUAL_ENDINGS):
        return REGISTER_CASUAL
    return None


def _sentence_register(sentence: str) -> str | None:
    # The ending of the LAST clause decides ("안녕하세요, 공씨" has no ending after
    # the comma, so fall back to the clause before it).
    for piece in reversed([p.strip() for p in sentence.split(",")]):
        if piece and (reg := _piece_register(piece)):
            return reg
    return None


def detect_register(text: str) -> str | None:
    """Speech level read from the sentence endings, or None when it cannot be
    told. Several sentences in one line that disagree make it "hỗn hợp"."""
    found = {r for sent in _sentences(text) if (r := _sentence_register(sent))}
    if not found:
        return None
    if len(found) > 1:
        return REGISTER_MIXED
    return next(iter(found))


def effective_register(text: str, stored: str) -> str:
    """The register shown to learners. The importer's label comes from a
    language model and is sometimes wrong ("합니다" lines tagged 반말); the
    ending itself is deterministic, so it wins whenever it is conclusive."""
    return detect_register(text) or stored


def with_effective_register(s: Sentence) -> Sentence:
    reg = effective_register(s.text_ko, s.register)
    return s if reg == s.register else replace(s, register=reg)


# ------------------------------------------------------------ de-duplicating --
def dedupe(sentences: list[Sentence]) -> list[Sentence]:
    """One entry per distinct sentence (see normalize_key). Which copy stays
    is deterministic — the lowest (film_id, id) — so a learner never sees the
    list reshuffle between requests."""
    seen: dict[str, Sentence] = {}
    for s in sorted(sentences, key=lambda x: (x.film_id, str(x.id))):
        key = normalize_key(s.text_ko)
        if key and key not in seen:
            seen[key] = s
    return list(seen.values())


# ----------------------------------------------------------------- filtering --
def matches(
    s: Sentence,
    *,
    level: int | None = None,
    topic_id: int | None = None,
    register: str | None = None,
    grammar_id: int | None = None,
    film_id: int | None = None,
) -> bool:
    return (
        (level is None or s.level == level)
        and (topic_id is None or topic_id in s.topic_ids)
        and (register is None or s.register == register)
        and (grammar_id is None or grammar_id in s.grammar_ids)
        and (film_id is None or s.film_id == film_id)
    )


def seeded_order(sentences: list[Sentence], seed: str = "") -> list[Sentence]:
    """A shuffle that is a pure function of (seed, sentence): the same seed
    gives the same order on every request, so offset/limit paging walks the
    whole set once with no repeats and no gaps — a random sample per request
    cannot promise that. An empty seed is simply a fixed order."""
    return sorted(sentences, key=lambda s: hashlib.sha1(f"{seed}:{s.id}".encode()).hexdigest())


# -------------------------------------------------------------------- facets --
@dataclass(frozen=True)
class FacetCounts:
    total: int
    levels: list[tuple[int, int]]
    registers: list[tuple[str, int]]
    topics: list[tuple[int, int]]
    grammar: list[tuple[int, int]]
    films: list[tuple[int, int]]


def _by_count(counter: Counter) -> list[tuple]:
    # biggest first, ties by key, so the order is stable between requests
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


def count_facets(sentences: list[Sentence]) -> FacetCounts:
    levels: Counter[int] = Counter()
    registers: Counter[str] = Counter()
    topics: Counter[int] = Counter()
    grammar: Counter[int] = Counter()
    films: Counter[int] = Counter()
    for s in sentences:
        levels[s.level] += 1
        registers[s.register] += 1
        films[s.film_id] += 1
        topics.update(set(s.topic_ids))
        grammar.update(set(s.grammar_ids))
    return FacetCounts(
        total=len(sentences),
        levels=sorted(levels.items()),
        registers=_by_count(registers),
        topics=_by_count(topics),
        grammar=_by_count(grammar),
        films=_by_count(films),
    )


# ------------------------------------------------------------------- import --
def corpus_keys(texts: list[str]) -> set[str]:
    """normalize_key of every text already in the corpus (empty keys dropped)."""
    return {k for t in texts if (k := normalize_key(t))}
