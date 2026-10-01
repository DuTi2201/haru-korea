"""Cut a lecture script into runs by language, so each run can be voiced by a
voice that actually speaks that language.

The "Bài giảng tổng hợp" script is Vietnamese explanation with Korean phrases
and examples inside it ("... Mình cùng nói chậm nhé. 이거 얼마예요... nghĩa là
..."). A Google Chirp 3 HD voice belongs to ONE locale: a ko-KR voice reading
Vietnamese (or a vi-VN voice reading Hangul) gets the pronunciation wrong,
which is exactly the part that must be right for Korean. So the script is cut
wherever the language changes and each piece is sent to the matching voice.

Pure text processing, no network: the voicing lives in app/services/tts.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KO = "ko"
VI = "vi"

# Silence (ms) inserted after a run. The learner is meant to repeat Korean
# phrases after hearing them, so a Korean run that ends on an ellipsis gets a
# real gap, not just the model's own pause.
GAP_LANGUAGE_SWITCH_MS = 120
GAP_AFTER_PARAGRAPH_MS = 500
GAP_AFTER_ELLIPSIS_KO_MS = 700
GAP_AFTER_ELLIPSIS_VI_MS = 350

_ELLIPSIS_END = re.compile(r"(\.{2,}|…)\s*[\"'”’)\]]*\s*$")


@dataclass(frozen=True)
class Run:
    lang: str  # KO or VI
    text: str
    pause_after_ms: int


def _is_hangul(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F or 0xA960 <= o <= 0xA97F


def _lang_of(ch: str) -> str | None:
    """KO for Hangul, VI for any other letter, None for spaces, digits and
    punctuation (those stay with whatever run they follow)."""
    if _is_hangul(ch):
        return KO
    if ch.isalpha():
        return VI
    return None


def _has_speech(text: str) -> bool:
    return any(ch.isalpha() or ch.isdigit() for ch in text)


def _split_paragraph(line: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, list[str]]] = []
    lead: list[str] = []  # neutral characters before the first letter
    for ch in line:
        lang = _lang_of(ch)
        if lang is None:
            (runs[-1][1] if runs else lead).append(ch)
        elif runs and runs[-1][0] == lang:
            runs[-1][1].append(ch)
        else:
            runs.append((lang, lead + [ch] if not runs else [ch]))
            lead = []
    return [(lang, "".join(chars).strip()) for lang, chars in runs]


def split_runs(script: str) -> list[Run]:
    """script -> language runs in reading order, each with the silence to put
    after it. A paragraph is one line. Runs with nothing to say (only
    punctuation) are dropped, and the paragraph's last run always carries at
    least the paragraph gap; the very last run of the script carries none."""
    out: list[Run] = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = [(lang, text) for lang, text in _split_paragraph(line) if _has_speech(text)]
        for i, (lang, text) in enumerate(pieces):
            last = i == len(pieces) - 1
            pause = GAP_AFTER_PARAGRAPH_MS if last else GAP_LANGUAGE_SWITCH_MS
            if _ELLIPSIS_END.search(text):
                pause = max(pause, GAP_AFTER_ELLIPSIS_KO_MS if lang == KO else GAP_AFTER_ELLIPSIS_VI_MS)
            out.append(Run(lang, text, pause))
    if out:
        out[-1] = Run(out[-1].lang, out[-1].text, 0)
    return out
