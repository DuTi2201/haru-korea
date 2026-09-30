"""Read-along: when is each sentence spoken in the "Nghe toàn văn" audio?

The reader highlights the sentence being read, so the audio needs a map from
sentence to time. The TTS services only return audio, never word/sentence
timestamps, so the map is built here:

* the text is cut into *units* — one per sentence of the original article
  (the very sentences the study pack and the reader use, see
  study_pack.split_sentences), or one per paragraph of the simplified text;
* units are packed into small *chunks* (a few sentences each) that are
  synthesized one request at a time, exactly like before;
* the length of every returned clip is known to the sample, so each chunk's
  start time is EXACT. Only the few sentences inside one chunk share the
  chunk's duration in proportion to how long they take to say (a character
  weight), which is an estimate — but one that can drift by at most a
  fraction of a second, because it resets at every chunk.

Pure functions, no network and no audio: the planning and the timeline are
unit-tested with a fake voice.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from app.services.article_extract import speakable, split_paragraphs, tts_title
from app.services.study_pack import split_sentences

# Part of the article-audio cache key: bumping it regenerates every recording
# (old rows have no timeline, so there would be nothing to follow).
READ_ALONG_VERSION = "ra1"
TIMINGS_FORMAT = 1
TITLE_PARAGRAPH = -1

_PARAGRAPH_GAP_WEIGHT = 3.0  # a paragraph break inside one clip ~ 3 syllables of silence


class Unit(NamedTuple):
    """One thing the reader can highlight."""

    p: int  # body paragraph index; TITLE_PARAGRAPH for the headline
    s: int  # sentence index inside that paragraph (0 for the title / a simplified paragraph)
    text: str


# --------------------------------------------------------------------- units --
def original_units(title: str | None, cleaned_body: str) -> list[Unit]:
    """The headline, then every sentence of every body paragraph. `cleaned_body`
    must be the same cleaned text the reader shows, so (p, s) lines up with the
    study pack's paragraphs/sentences. A sentence that is only a URL voices
    nothing and is skipped, but keeps its index."""
    units: list[Unit] = []
    head = tts_title(title)
    if head:
        units.append(Unit(TITLE_PARAGRAPH, 0, head))
    for p, paragraph in enumerate(split_paragraphs(cleaned_body)):
        for s, sentence in enumerate(split_sentences(paragraph)):
            text = speakable(sentence)
            if text:
                units.append(Unit(p, s, text))
    return units


def easy_units(title: str | None, pack: dict[str, Any]) -> list[Unit]:
    """The headline, then one unit per paragraph of the simplified Korean (a
    paragraph the model left without one falls back to its original sentences,
    as the reader does). Indexed by the pack's paragraph number."""
    units: list[Unit] = []
    head = tts_title(title)
    if head:
        units.append(Unit(TITLE_PARAGRAPH, 0, head))
    for p, para in enumerate(pack.get("paragraphs", [])):
        text = (para.get("easy_ko") or "").strip()
        if not text:
            text = " ".join(s["ko"] for s in para.get("sentences", []))
        text = speakable(re.sub(r"\s+", " ", text))
        if text:
            units.append(Unit(p, 0, text))
    return units


def units_to_text(units: Sequence[Unit]) -> str:
    """What the voice will say: one paragraph per line (sentences of a
    paragraph separated by a space). This is the string the cache key hashes."""
    lines: list[str] = []
    last_p: int | None = None
    for u in units:
        if lines and u.p == last_p:
            lines[-1] += " " + u.text
        else:
            lines.append(u.text)
        last_p = u.p
    return "\n".join(lines)


# ------------------------------------------------------------------ planning --
@dataclass(frozen=True)
class Piece:
    unit: int  # index into the units list
    para: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """What one TTS request says, plus the silence inserted after it."""

    pieces: tuple[Piece, ...]
    pause_after_ms: int

    @property
    def text(self) -> str:
        out = ""
        for i, pc in enumerate(self.pieces):
            if i:
                out += " " if pc.para == self.pieces[i - 1].para else "\n\n"
            out += pc.text
        return out


def _split_long(text: str, limit: int) -> list[str]:
    """A single sentence longer than `limit`: cut at the last space before it
    (hard cut if there is none). Rare — the limit is the request size cap."""
    parts: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = rest.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def _pack(pieces: Sequence[Piece], limit: int) -> list[list[Piece]]:
    groups: list[list[Piece]] = []
    cur: list[Piece] = []
    size = 0
    for pc in pieces:
        add = len(pc.text) + (1 if cur else 0)
        if cur and size + add > limit:
            groups.append(cur)
            cur, size, add = [], 0, len(pc.text)
        cur.append(pc)
        size += add
    if cur:
        groups.append(cur)
    return groups


def plan_chunks(
    units: Sequence[Unit],
    *,
    pack_to: int,
    split_over: int,
    max_total: int,
    sentence_gap_ms: int,
    paragraph_gap_ms: int,
    cross_paragraph: bool,
) -> list[Chunk]:
    """Units -> the chunks to synthesize, in order.

    * `pack_to`: aim for chunks of at most this many characters; sentences are
      never split to reach it (a longer sentence just gets a chunk of its own).
    * `split_over`: the request size cap — only a sentence above it is cut.
    * `max_total`: the tail of a very long text is not voiced (the units past
      the cap simply get no timing).
    * `cross_paragraph`: False → a chunk never spans two paragraphs, so the
      paragraph pause is always real silence (small chunks, precise timing);
      True → whole paragraphs are packed together the way the Gemini path
      always did, to keep its request count (= daily quota) low.
    """
    # group consecutive units by paragraph, then apply the length cap
    paragraphs: list[list[int]] = []
    for i, u in enumerate(units):
        if paragraphs and units[paragraphs[-1][0]].p == u.p:
            paragraphs[-1].append(i)
        else:
            paragraphs.append([i])
    kept: list[list[int]] = []
    total = 0
    for idxs in paragraphs:
        size = sum(len(units[i].text) for i in idxs)
        if kept and total + size > max_total:
            break
        total += size
        kept.append(idxs)

    chunks: list[Chunk] = []
    cur: list[Piece] = []
    cur_size = 0

    def flush(pause_ms: int) -> None:
        nonlocal cur, cur_size
        if cur:
            chunks.append(Chunk(tuple(cur), pause_ms))
            cur, cur_size = [], 0

    for idxs in kept:
        pieces: list[Piece] = []
        for i in idxs:
            u = units[i]
            parts = [u.text] if len(u.text) <= split_over else _split_long(u.text, split_over)
            pieces.extend(Piece(i, u.p, part) for part in parts)
        text_len = sum(len(pc.text) for pc in pieces) + max(0, len(pieces) - 1)

        if cross_paragraph and text_len <= pack_to:
            if cur and cur_size + text_len + 2 > pack_to:
                flush(paragraph_gap_ms)
            cur.extend(pieces)
            cur_size += text_len + 2
            continue

        flush(paragraph_gap_ms)
        groups = _pack(pieces, pack_to)
        for gi, group in enumerate(groups):
            last = gi == len(groups) - 1
            chunks.append(Chunk(tuple(group), paragraph_gap_ms if last else sentence_gap_ms))
    flush(0)
    if chunks:
        chunks[-1] = replace(chunks[-1], pause_after_ms=0)
    return chunks


# ------------------------------------------------------------------ timeline --
def _weight(text: str) -> float:
    """Rough "how long does this take to say": a syllable/letter ~ 1, a digit a
    bit more (numbers are read out in full), a sentence end a short pause."""
    w = 0.0
    for ch in text:
        if ch.isspace():
            continue
        if ch in ".!?…":
            w += 1.0
        elif ch in ",;:、，":
            w += 0.5
        elif ch.isdigit():
            w += 1.3
        elif ch.isalnum():
            w += 1.0
        else:
            w += 0.2
    return max(w, 0.5)


def build_timeline(
    units: Sequence[Unit],
    chunks: Sequence[Chunk],
    chunk_start_ms: Sequence[float],
    chunk_speech_ms: Sequence[float],
) -> dict[str, Any]:
    """{"v": 1, "items": [[p, s, start_ms], ...]} in reading order.

    `chunk_start_ms[i]` / `chunk_speech_ms[i]` are where chunk i begins in the
    final audio and how long its clip is (both known exactly from the PCM).
    Inside a chunk the pieces split the clip in proportion to their weight.
    An item's end is the next item's start (the last one ends with the audio),
    so the highlight never flickers off during a pause."""
    first_start: dict[int, float] = {}
    for chunk, t0, dur in zip(chunks, chunk_start_ms, chunk_speech_ms, strict=True):
        weights: list[float] = []
        for k, pc in enumerate(chunk.pieces):
            w = _weight(pc.text)
            nxt = chunk.pieces[k + 1] if k + 1 < len(chunk.pieces) else None
            if nxt is not None and nxt.para != pc.para:
                w += _PARAGRAPH_GAP_WEIGHT
            weights.append(w)
        total = sum(weights)
        acc = 0.0
        for pc, w in zip(chunk.pieces, weights, strict=True):
            first_start.setdefault(pc.unit, t0 + dur * acc / total)
            acc += w
    items = [[units[i].p, units[i].s, int(round(first_start[i]))] for i in sorted(first_start)]
    return {"v": TIMINGS_FORMAT, "items": items}
