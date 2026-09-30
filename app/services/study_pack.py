"""Beginner "reading ladder" for an editorial article.

Korean news is written at TOPIK 5-6. A learner who has just finished the
beginner vocabulary cannot read it, and a glossary of a few words does not
change that. So each article gets a *study pack*, generated ONCE (by a
Celery task, then stored) and served with the article:

- overview      — Vietnamese summary, key points, key terms ("Hiểu nhanh"),
                  read BEFORE the article so the learner knows what it is about;
- per sentence  — Vietnamese translation, the words a beginner may not know
                  (with base form + meaning) and short notes on the hard
                  endings/grammar in that sentence;
- per paragraph — the same content rewritten in simple Korean (TOPIK 1-2:
                  short sentences, basic words) — a "bản dễ" that can be read
                  and listened to before tackling the original.

Nothing here talks to Gemini directly: `build_study_pack` receives a
`generate(prompt, schema) -> dict` callable, so the sentence splitting,
batching, validation and assembly are unit-tested with a fake. Sentence
boundaries are computed here (deterministically) and sent to the model
numbered, so translations always line up with the article's own sentences —
the model is never trusted to split text.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.services.article_extract import TEXT_CLEAN_VERSION, article_tts_text

STUDY_VERSION = "study-v1"

# One Gemini call handles a batch of whole paragraphs. Small enough that the
# per-sentence JSON (translation + word list + notes) fits comfortably in the
# output limit and one bad answer only costs a re-ask of ~1.5k characters.
_BATCH_CHARS = 1500
_BATCH_SENTENCES = 14
_MAX_WORDS_PER_SENTENCE = 8
_BATCH_ATTEMPTS = 2
_PAUSE_BETWEEN_CALLS_SEC = 0.6

Generate = Callable[[str, dict[str, Any]], dict[str, Any]]


# --------------------------------------------------------- sentence splitting --
_TERMINATORS = ".!?…"
_CLOSERS = "\"”’'」』)]〉》"


def split_sentences(paragraph: str) -> list[str]:
    """Splits one paragraph into sentences at . ! ? … (plus any closing
    quote/bracket that follows) when a space or the end comes next — so
    "18.4GW" and "5.3%를" stay whole, and `"…주세요." 지난 17일…` splits
    AFTER the closing quote. Fragments of ≤ 3 characters ("1.") are merged
    into the next sentence. Pure and deterministic: the same paragraph always
    yields the same sentences, which is what lets stored translations line up
    with the text the reader shows."""
    text = re.sub(r"\s+", " ", paragraph or "").strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in _TERMINATORS:
            j = i + 1
            while j < n and text[j] in _TERMINATORS:
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j >= n or text[j] == " ":
                parts.append(text[start:j].strip())
                start = j
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)

    merged: list[str] = []
    carry = ""
    for part in parts:
        part = (carry + " " + part).strip() if carry else part
        carry = ""
        if len(part) <= 3:
            carry = part
            continue
        merged.append(part)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def study_text_sig(cleaned_body: str) -> str:
    """Identifies the exact text a pack was written for. A pack whose sig no
    longer matches the article's current cleaned body (text backfilled,
    cleaning rules changed, prompt version bumped) is treated as absent and
    regenerated, never shown against the wrong paragraphs."""
    return hashlib.sha256(f"{TEXT_CLEAN_VERSION}\n{STUDY_VERSION}\n{cleaned_body}".encode()).hexdigest()[:16]


# ------------------------------------------------------------ Gemini schemas --
OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary_vi": {"type": "STRING"},
        "key_points_vi": {"type": "ARRAY", "items": {"type": "STRING"}},
        "key_terms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"ko": {"type": "STRING"}, "vi": {"type": "STRING"}},
                "required": ["ko", "vi"],
            },
        },
    },
    "required": ["summary_vi", "key_points_vi", "key_terms"],
}

BATCH_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "paragraphs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "easy_ko": {"type": "STRING"},
                    "sentences": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "i": {"type": "INTEGER"},
                                "vi": {"type": "STRING"},
                                "words": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "surface": {"type": "STRING"},
                                            "base": {"type": "STRING", "nullable": True},
                                            "pos": {"type": "STRING", "nullable": True},
                                            "meaning_vi": {"type": "STRING"},
                                        },
                                        "required": ["surface", "meaning_vi"],
                                    },
                                },
                                "grammar_notes_vi": {"type": "ARRAY", "items": {"type": "STRING"}},
                            },
                            "required": ["i", "vi"],
                        },
                    },
                },
                "required": ["index", "easy_ko", "sentences"],
            },
        }
    },
    "required": ["paragraphs"],
}


# ------------------------------------------------------- lenient parse models --
class _Word(BaseModel):
    surface: str
    base: str | None = None
    pos: str | None = None
    meaning_vi: str


class _Sentence(BaseModel):
    i: int
    vi: str = ""
    words: list[_Word] = Field(default_factory=list)
    grammar_notes_vi: list[str] = Field(default_factory=list)


class _Paragraph(BaseModel):
    index: int
    easy_ko: str = ""
    sentences: list[_Sentence] = Field(default_factory=list)


class _Batch(BaseModel):
    paragraphs: list[_Paragraph] = Field(default_factory=list)


class _KeyTerm(BaseModel):
    ko: str
    vi: str


class _Overview(BaseModel):
    summary_vi: str = ""
    key_points_vi: list[str] = Field(default_factory=list)
    key_terms: list[_KeyTerm] = Field(default_factory=list)


# ------------------------------------------------------------------- prompts --
_AUDIENCE = (
    "Bạn là giáo viên tiếng Hàn cho người Việt mới học (sơ cấp, TOPIK 1-2). "
    "Người học vừa học xong từ vựng cơ bản nhưng muốn đọc báo tiếng Hàn thật; "
    "bài báo này khó hơn trình độ của họ rất nhiều, nên nhiệm vụ của bạn là "
    "giúp họ HIỂU được bài, không phải dạy lại toàn bộ ngữ pháp."
)


def build_overview_prompt(title: str | None, paragraphs: list[str]) -> str:
    body = "\n".join(paragraphs)
    return f"""{_AUDIENCE}

Đọc bài báo dưới đây và viết bằng TIẾNG VIỆT dễ hiểu (không dịch từng câu):
- summary_vi: 3-4 câu tóm tắt — chuyện gì đang xảy ra / vấn đề gì, liên quan đến
  ai, vì sao đáng quan tâm.
- key_points_vi: 3-5 ý chính, mỗi ý một câu ngắn.
- key_terms: 4-8 thuật ngữ/khái niệm then chốt để hiểu bài (ko = từ hoặc cụm từ
  đúng như trong bài; vi = giải thích ngắn gọn bằng tiếng Việt, kèm bối cảnh nếu cần).
Chỉ dùng thông tin có trong bài, không bịa thêm.

Tiêu đề: {title or "(không có)"}

Bài viết:
{body}"""


def build_batch_prompt(
    title: str | None,
    summary_vi: str,
    batch: list[tuple[int, list[str]]],
) -> str:
    blocks: list[str] = []
    for index, sentences in batch:
        lines = "\n".join(f"  [{i}] {s}" for i, s in enumerate(sentences))
        blocks.append(f"Đoạn {index}:\n{lines}")
    listing = "\n\n".join(blocks)
    return f"""{_AUDIENCE}

Bài báo: {title or "(không có tiêu đề)"}
Tóm tắt (để hiểu ngữ cảnh): {summary_vi or "(chưa có)"}

Dưới đây là một số đoạn của bài, mỗi đoạn đã được tách sẵn thành các câu đánh số.
Với MỖI đoạn, trả về:
1. easy_ko: viết lại cả đoạn bằng tiếng Hàn ĐƠN GIẢN cho trình độ TOPIK 1-2 —
   câu ngắn (mỗi câu một ý, tối đa khoảng 12 từ), từ vựng cơ bản, dạng lịch sự
   -아요/어요 hoặc -ㅂ니다; thay từ Hán-Hàn/từ khó bằng từ dễ hơn hoặc giải
   thích ngắn trong ngoặc. Giữ nguyên số liệu, tên riêng, đúng sự thật; không
   thêm ý mới, không bỏ ý chính. Nếu đoạn chỉ là tên tác giả hoặc tiêu đề phụ
   thì giữ gần nguyên.
2. sentences: cho TỪNG câu gốc, đúng số thứ tự i, đủ số câu, không gộp/tách câu:
   - vi: bản dịch tiếng Việt tự nhiên, sát nghĩa.
   - words: các từ/cụm từ đáng chú ý mà người học sơ cấp có thể CHƯA biết (tối
     đa {_MAX_WORDS_PER_SENTENCE} từ mỗi câu; bỏ từ rất cơ bản và trợ từ).
     surface = viết ĐÚNG như trong câu (giữ nguyên đuôi chia); base = dạng từ
     điển (động từ/tính từ kết thúc bằng 다); pos = từ loại tiếng Việt (danh
     từ, động từ, tính từ, trạng từ…); meaning_vi = nghĩa ngắn theo ngữ cảnh
     (từ Hán-Hàn thì thêm âm Hán Việt trong ngoặc).
   - grammar_notes_vi: 0-2 ghi chú ngắn bằng tiếng Việt về đuôi câu/cấu trúc
     ngữ pháp khó trong câu, dạng "-는다고: trích dẫn gián tiếp, nghĩa là
     'nói rằng…'". Để mảng rỗng nếu câu đơn giản.

Các đoạn:

{listing}"""


# ------------------------------------------------------------------ batching --
def plan_batches(sentence_lists: list[list[str]]) -> list[list[int]]:
    """Groups consecutive paragraph indices into batches under the size
    budgets. A paragraph bigger than a whole batch gets a batch of its own."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    cur_sents = 0
    for idx, sentences in enumerate(sentence_lists):
        chars = sum(len(s) for s in sentences)
        if cur and (cur_chars + chars > _BATCH_CHARS or cur_sents + len(sentences) > _BATCH_SENTENCES):
            batches.append(cur)
            cur, cur_chars, cur_sents = [], 0, 0
        cur.append(idx)
        cur_chars += chars
        cur_sents += len(sentences)
    if cur:
        batches.append(cur)
    return batches


# ---------------------------------------------------------------- validation --
def _stem(base_or_surface: str) -> str:
    s = base_or_surface.strip()
    return s[:-1] if len(s) > 2 and s.endswith("다") else s


def _clean_words(ko: str, words: list[_Word]) -> list[dict[str, Any]]:
    """Keeps only words the sentence really contains (the model sometimes
    lists words from a neighbouring sentence or invents a base form), no
    duplicates, capped per sentence."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for w in words:
        surface = w.surface.strip()
        meaning = w.meaning_vi.strip()
        if not surface or not meaning or surface in seen:
            continue
        present = surface in ko or (w.base and _stem(w.base) and _stem(w.base) in ko)
        if not present:
            continue
        seen.add(surface)
        out.append(
            {
                "surface": surface,
                "base": (w.base or "").strip() or None,
                "pos": (w.pos or "").strip() or None,
                "meaning_vi": meaning,
            }
        )
        if len(out) >= _MAX_WORDS_PER_SENTENCE:
            break
    return out


def _assemble_paragraph(sentences: list[str], parsed: _Paragraph | None) -> tuple[dict[str, Any], int]:
    """One paragraph's pack entry + how many of its sentences got a
    translation. Sentences are matched by their number `i`, so a model that
    skips or reorders one cannot shift the others' translations."""
    by_i = {s.i: s for s in (parsed.sentences if parsed else [])}
    entries: list[dict[str, Any]] = []
    translated = 0
    for i, ko in enumerate(sentences):
        got = by_i.get(i)
        vi = (got.vi.strip() if got else "") or None
        if vi:
            translated += 1
        entries.append(
            {
                "ko": ko,
                "vi": vi,
                "words": _clean_words(ko, got.words) if got else [],
                "grammar_notes_vi": [g.strip() for g in (got.grammar_notes_vi if got else []) if g.strip()][:2],
            }
        )
    easy = (parsed.easy_ko.strip() if parsed else "") or None
    return {"easy_ko": easy, "sentences": entries}, translated


def _loads(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    return json.loads(cleaned)


# ------------------------------------------------------------------- builder --
def build_study_pack(
    title: str | None,
    paragraphs: list[str],
    generate: Generate,
    *,
    text_sig: str,
    on_progress: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Generates the whole pack for one article. Raises RuntimeError if the
    model produced no usable translation at all (so the caller marks the
    pack failed and can retry later); partial gaps inside an otherwise good
    pack are tolerated — the reader simply shows those sentences untranslated."""
    sentence_lists = [split_sentences(p) for p in paragraphs]
    batches = plan_batches(sentence_lists)
    total_steps = len(batches) + 1
    if on_progress:
        on_progress(0, total_steps)

    overview = _Overview()
    try:
        overview = _Overview.model_validate(
            generate(build_overview_prompt(title, paragraphs), OVERVIEW_SCHEMA)
        )
    except Exception as exc:  # noqa: BLE001 — a missing summary must not sink the translations
        print(f"[study-pack] overview failed: {exc}", flush=True)
    if on_progress:
        on_progress(1, total_steps)

    out_paragraphs: list[dict[str, Any] | None] = [None] * len(paragraphs)
    translated_total = 0
    sentence_total = sum(len(s) for s in sentence_lists)

    for n, batch_idx in enumerate(batches):
        sleep(_PAUSE_BETWEEN_CALLS_SEC)
        batch_input = [(i, sentence_lists[i]) for i in batch_idx]
        prompt = build_batch_prompt(title, overview.summary_vi, batch_input)
        want = sum(len(s) for _, s in batch_input)
        best: dict[int, tuple[dict[str, Any], int]] = {}
        best_translated = -1
        for attempt in range(_BATCH_ATTEMPTS):
            try:
                parsed = _Batch.model_validate(generate(prompt, BATCH_SCHEMA))
            except Exception as exc:  # noqa: BLE001
                print(f"[study-pack] batch {n + 1}/{len(batches)} attempt {attempt + 1}: {exc}", flush=True)
                sleep(_PAUSE_BETWEEN_CALLS_SEC * 3)
                continue
            by_index = {p.index: p for p in parsed.paragraphs}
            attempt_result: dict[int, tuple[dict[str, Any], int]] = {}
            translated = 0
            for i in batch_idx:
                entry, t = _assemble_paragraph(sentence_lists[i], by_index.get(i))
                attempt_result[i] = (entry, t)
                translated += t
            if translated > best_translated:
                best, best_translated = attempt_result, translated
            if translated >= 0.9 * want:
                break
            sleep(_PAUSE_BETWEEN_CALLS_SEC * 3)
        for i in batch_idx:
            if i in best:
                out_paragraphs[i], t = best[i]
                translated_total += t
            else:
                entry, _ = _assemble_paragraph(sentence_lists[i], None)
                out_paragraphs[i] = entry
        if on_progress:
            on_progress(n + 2, total_steps)

    if sentence_total and translated_total < max(1, int(0.3 * sentence_total)):
        raise RuntimeError(f"study pack unusable: {translated_total}/{sentence_total} sentences translated")

    return {
        "version": STUDY_VERSION,
        "text_sig": text_sig,
        "summary_vi": overview.summary_vi.strip(),
        "key_points_vi": [p.strip() for p in overview.key_points_vi if p.strip()][:6],
        "key_terms": [{"ko": t.ko.strip(), "vi": t.vi.strip()} for t in overview.key_terms if t.ko.strip() and t.vi.strip()][:8],
        "paragraphs": [p if p is not None else {"easy_ko": None, "sentences": []} for p in out_paragraphs],
    }


def make_gemini_generate(model: str, gemini_generate_structured: Callable[..., dict[str, Any]]) -> Generate:
    """Adapts gemini_client.generate_structured to the `generate` callable."""

    def generate(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        result = gemini_generate_structured(
            model=model, prompt=prompt, response_schema=schema, prompt_version=STUDY_VERSION
        )
        return _loads(result["text"])

    return generate


# --------------------------------------------------------------- easy audio --
def easy_paragraphs(pack: dict[str, Any]) -> list[str]:
    """The simplified-Korean text of each paragraph; a paragraph the model
    left without one falls back to its original sentences."""
    out: list[str] = []
    for p in pack.get("paragraphs", []):
        easy = (p.get("easy_ko") or "").strip()
        if not easy:
            easy = " ".join(s["ko"] for s in p.get("sentences", []))
        if easy:
            out.append(easy)
    return out


def easy_tts_text(title: str | None, pack: dict[str, Any]) -> str:
    """Exactly what "Nghe bản dễ" voices: the title, then the simplified
    paragraphs, one per line (same shape as article_tts_text)."""
    return article_tts_text(title, "\n".join(easy_paragraphs(pack)))
