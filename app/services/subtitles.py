"""Minimal, dependency-free SRT/VTT parser for the corpus/film ingestion
pipeline (SRS §5 FILM/CORPUS_ITEM, SDD FR-17..FR-21, Gate G6). Only pulls
out what the classification step needs — a stable `source_ref` (the cue's
timecode) and the cue's cleaned Korean text — not a full subtitle-editing
library, so no new dependency was added to requirements.txt for this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TAG_RE = re.compile(r"<[^>]+>")  # <i>, <b>, <font ...> etc — VTT/SRT both allow these
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[-–—]\s*")
_SRT_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)
_MUSIC_ONLY_RE = re.compile(r"^[♪\[\(].*[♪\]\)]?$")


@dataclass
class Cue:
    source_ref: str  # "start_ms-end_ms", stable regardless of re-parsing/re-numbering
    text: str


def _clean_line(line: str) -> str:
    line = _TAG_RE.sub("", line)
    line = _SPEAKER_PREFIX_RE.sub("", line)
    return line.strip()


def _timecode_to_ms(tc: str) -> int:
    tc = tc.replace(",", ".")
    h, m, rest = tc.split(":")
    s, ms = rest.split(".")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_subtitles(raw: str) -> list[Cue]:
    """Parses both .srt and .vtt (the WEBVTT header, if present, is just
    skipped — the cue block shape below is otherwise identical to SRT)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())

    cues: list[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        if lines[0].strip().upper().startswith("WEBVTT"):
            continue

        time_line_idx = None
        for i, ln in enumerate(lines[:2]):  # timecode is line 0 (VTT) or line 1 (SRT, after the index)
            if _SRT_TIME_RE.search(ln):
                time_line_idx = i
                break
        if time_line_idx is None:
            continue  # not a cue block (stray index/number, cue settings-only line, etc.)

        m = _SRT_TIME_RE.search(lines[time_line_idx])
        start_ms, end_ms = _timecode_to_ms(m.group(1)), _timecode_to_ms(m.group(2))

        text_lines = [_clean_line(ln) for ln in lines[time_line_idx + 1 :]]
        merged = " ".join(ln for ln in text_lines if ln)
        if not merged or _MUSIC_ONLY_RE.match(merged):
            continue

        cues.append(Cue(source_ref=f"{start_ms}-{end_ms}", text=merged))

    return cues


def parse_plain_lines(raw: str) -> list[Cue]:
    """Fallback for corpus input that isn't real timestamped .srt/.vtt — a
    plain-text paste, or the raw text Gemini-OCR recovers from a photographed
    /screenshotted subtitle list (neither carries real timecodes). Every
    non-empty, non-music-only line becomes its own cue with a synthetic
    sequential source_ref, so this kind of upload doesn't silently stage
    zero items just because parse_subtitles found no timecode blocks."""
    cues: list[Cue] = []
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for i, raw_line in enumerate(text.split("\n")):
        line = _clean_line(raw_line)
        if not line or line.upper().startswith("WEBVTT"):
            continue
        if _MUSIC_ONLY_RE.match(line):
            continue
        cues.append(Cue(source_ref=f"line-{i}", text=line))
    return cues


def chunk_cues(cues: list[Cue], chunk_size: int) -> list[list[Cue]]:
    """Fixed-size chunks — see config.CORPUS_CHUNK_SIZE: this is what
    keeps each Gemini call's prompt a constant size regardless of how
    long the film's subtitle file is (FR-19 / Gate G6)."""
    return [cues[i : i + chunk_size] for i in range(0, len(cues), chunk_size)]
