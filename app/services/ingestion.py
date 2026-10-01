"""Content-ingestion pipeline: Gemini extraction/classification schemas +
prompts, and the "confirm" step that writes reviewed import_items into the
canonical content/corpus tables. This is the implementation of SDD FR-17..
FR-21 (corpus/film) and FR-42..FR-51 (lesson image/PDF), built on the
generic import_batch/import_item review-queue already in app/models.py.

Everything here is plain sync code (SQLAlchemy sync Session) because both
its callers — the Celery extraction tasks AND the confirm/rollback task —
run worker-side (app/workers/tasks.py). The FastAPI side (app/api/routers/
ingest.py) only ever creates the batch/job rows and enqueues; it never
calls into this module directly, so there is no async/sync split to
maintain here.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import feedparser
import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    CorpusItem,
    EditorialArticle,
    EditorialCandidate,
    EditorialSource,
    ExamItem,
    ExamPaper,
    ExamPassage,
    Film,
    GrammarPoint,
    ImportBatch,
    ImportItem,
    Lesson,
    LessonTopic,
    QuestionType,
    Topic,
    VocabItem,
)
from app.services import (
    article_extract,
    corpus_browse,
    corpus_enrich,
    exam_extract,
    gemini_client,
    lesson_extract,
    podcast_script,
)
from app.services.subtitles import Cue, chunk_cues, parse_plain_lines, parse_subtitles

# ============================================================ extraction ==
# Response schemas as plain Gemini-Schema dicts (uppercase `type`s, no
# `$ref`/`$defs`) rather than Pydantic .model_json_schema() output — the
# pinned google-genai==0.3.0 SDK does not resolve nested-model refs before
# sending them (it only upper-cases `type` and strips `title`), so a
# schema with nested submodels would reach the API with dangling `$ref`s.
# Hand-written, fully-inlined dicts sidestep that entirely.

LESSON_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "description": "Tên bài học, ngắn gọn"},
        "level": {"type": "INTEGER", "description": "1-6, ước lượng theo cấp TOPIK tương ứng"},
        "content": {
            "type": "STRING",
            "description": "Nội dung bài học dạng văn bản thuần (diễn giải lại những gì đọc được), hiển thị lại cho người học",
        },
        "topics": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Tên các chủ đề liên quan"},
        "vocab": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "hangul": {"type": "STRING"},
                    "pos": {"type": "STRING", "nullable": True, "description": "từ loại, vd 동사/명사/형용사"},
                    "meaning_vi": {"type": "STRING"},
                    "definition_ko": {"type": "STRING", "nullable": True},
                    "level": {"type": "INTEGER"},
                    "hanja": {"type": "STRING", "nullable": True},
                    "sino_vietnamese": {"type": "STRING", "nullable": True, "description": "âm Hán Việt nếu có"},
                    "example_ko": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "NUMBER", "description": "0-1, độ tự tin của mô hình"},
                },
                "required": ["hangul", "meaning_vi", "level", "confidence"],
            },
        },
        "grammar": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "pattern": {
                        "type": "STRING",
                        "description": "Mẫu ngữ pháp viết dạng V/A + hình thái, vd 'V + -(으)ㄹ 뿐만 아니라'",
                    },
                    "meaning_vi": {"type": "STRING"},
                    "level": {"type": "INTEGER"},
                    "example_ko": {"type": "STRING", "nullable": True},
                    "usage_context_vi": {
                        "type": "STRING",
                        "description": "Giải thích bằng tiếng Việt: dùng KHI NÀO, trong HOÀN CẢNH/tình huống nào, với sắc thái gì — không chỉ định nghĩa suông",
                    },
                    "topik_tip_vi": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Nhận xét bằng tiếng Việt (câu khẳng định, không phải lời ra lệnh): mẫu này hay xuất hiện ở dạng bài/ngữ cảnh TOPIK nào và dấu hiệu nhận biết là gì",
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["pattern", "meaning_vi", "level", "usage_context_vi", "confidence"],
            },
        },
        "confidence": {"type": "NUMBER"},
    },
    "required": ["title", "level", "content", "vocab", "grammar", "confidence"],
}

CORPUS_CHUNK_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source_ref": {"type": "STRING", "description": "Echo lại y nguyên source_ref của câu thoại"},
                    "keep": {
                        "type": "BOOLEAN",
                        "description": "false nếu câu này không có giá trị học tập (chỉ tên riêng, thán từ, v.v.)",
                    },
                    "is_crude": {"type": "BOOLEAN", "description": "true nếu chứa ngôn từ thô tục/khiêu dâm/bạo lực"},
                    "kind": {"type": "STRING", "enum": ["câu", "cụm từ", "mẫu ngữ pháp"]},
                    "level": {"type": "INTEGER"},
                    "register": {"type": "STRING", "enum": ["존댓말", "반말", "hỗn hợp"]},
                    "topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "grammar_patterns": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Chỉ chọn từ danh sách mẫu ngữ pháp đã biết được cung cấp, để trống nếu không khớp",
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["source_ref", "keep", "kind", "level", "register", "confidence"],
            },
        },
    },
    "required": ["items"],
}

# Editorial reading (사설/칼럼 luyện đọc + luyện dàn ý cho TOPIK viết câu 54):
# one call per article. body_ko is the admin-reviewable staged text (see
# ingest.py's module docstring on "staged, human confirms") — the model
# reads it as-is, warts (stray scraped nav/ad lines) and all.
EDITORIAL_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "level_estimate": {
            "type": "INTEGER",
            "description": "1-6, ước lượng theo cấp TOPIK tương ứng với độ khó bài viết",
        },
        "topic_tags": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "2-4 chủ đề ngắn gọn bằng tiếng Việt, vd 'già hóa dân số', 'AI', 'môi trường', 'giáo dục'",
        },
        "vocab": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "hangul": {"type": "STRING"},
                    "pos": {"type": "STRING", "nullable": True, "description": "từ loại, vd 동사/명사/형용사"},
                    "meaning_vi": {"type": "STRING"},
                    "definition_ko": {"type": "STRING", "nullable": True},
                    "level": {"type": "INTEGER"},
                    "hanja": {"type": "STRING", "nullable": True},
                    "sino_vietnamese": {"type": "STRING", "nullable": True, "description": "âm Hán Việt nếu có"},
                    "example_ko": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["hangul", "meaning_vi", "level", "confidence"],
            },
        },
        "grammar": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "pattern": {"type": "STRING", "description": "Mẫu ngữ pháp dạng V/A + hình thái"},
                    "meaning_vi": {"type": "STRING"},
                    "level": {"type": "INTEGER"},
                    "example_ko": {"type": "STRING", "nullable": True},
                    "usage_context_vi": {
                        "type": "STRING",
                        "description": "Giải thích bằng tiếng Việt: dùng KHI NÀO, trong HOÀN CẢNH/tình huống nào — không chỉ định nghĩa suông",
                    },
                    "topik_tip_vi": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Nhận xét bằng tiếng Việt (câu khẳng định, không phải lời ra lệnh): mẫu này hay xuất hiện ở dạng bài/ngữ cảnh TOPIK nào (đặc biệt viết câu 54) và cách vận dụng",
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["pattern", "meaning_vi", "level", "usage_context_vi", "confidence"],
            },
        },
        "model_outline": {
            "type": "OBJECT",
            "properties": {
                "phenomenon": {"type": "STRING", "description": "Đoạn hiện tượng (현상), viết bằng tiếng Hàn"},
                "cause": {"type": "STRING", "description": "Đoạn nguyên nhân (원인), tiếng Hàn"},
                "consequence": {"type": "STRING", "description": "Đoạn kết quả/ảnh hưởng (결과), tiếng Hàn"},
                "solution": {"type": "STRING", "description": "Đoạn giải pháp/kiến nghị (해결 방안), tiếng Hàn"},
            },
            "required": ["phenomenon", "cause", "consequence", "solution"],
        },
        "thinking_guide_text": {
            "type": "STRING",
            "description": (
                "3-5 câu hỏi gợi mở bằng tiếng Việt giúp người học tự suy nghĩ trước khi viết dàn ý, "
                "không tiết lộ nội dung dàn ý mẫu"
            ),
        },
        "confidence": {"type": "NUMBER"},
    },
    "required": [
        "level_estimate",
        "topic_tags",
        "vocab",
        "grammar",
        "model_outline",
        "thinking_guide_text",
        "confidence",
    ],
}


class VocabExtraction(BaseModel):
    hangul: str
    pos: str | None = None
    meaning_vi: str
    definition_ko: str | None = None
    level: int = Field(ge=1, le=6)
    hanja: str | None = None
    sino_vietnamese: str | None = None
    example_ko: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)


class GrammarExtraction(BaseModel):
    pattern: str
    meaning_vi: str
    level: int = Field(ge=1, le=6)
    example_ko: str | None = None
    usage_context_vi: str | None = None
    topik_tip_vi: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)


class LessonExtraction(BaseModel):
    title: str
    level: int = Field(ge=1, le=6)
    content: str
    topics: list[str] = Field(default_factory=list)
    vocab: list[VocabExtraction] = Field(default_factory=list)
    grammar: list[GrammarExtraction] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class CorpusCueClassification(BaseModel):
    source_ref: str
    keep: bool
    is_crude: bool = False
    kind: Literal["câu", "cụm từ", "mẫu ngữ pháp"]
    level: int = Field(ge=1, le=6)
    register: Literal["존댓말", "반말", "hỗn hợp"]
    topics: list[str] = Field(default_factory=list)
    grammar_patterns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class CorpusChunkClassification(BaseModel):
    items: list[CorpusCueClassification] = Field(default_factory=list)


class ModelOutline(BaseModel):
    phenomenon: str
    cause: str
    consequence: str
    solution: str


class EditorialExtraction(BaseModel):
    model_config = {"protected_namespaces": ()}

    level_estimate: int = Field(ge=1, le=6)
    topic_tags: list[str] = Field(default_factory=list)
    vocab: list[VocabExtraction] = Field(default_factory=list)
    grammar: list[GrammarExtraction] = Field(default_factory=list)
    model_outline: ModelOutline
    thinking_guide_text: str
    confidence: float = Field(ge=0, le=1, default=0.5)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json(text: str) -> dict[str, Any]:
    """Gemini's JSON-mode output is usually bare JSON, but strip a stray
    ``` fence defensively — cheap, and saves a retry on the rare case."""
    return json.loads(_JSON_FENCE_RE.sub("", text.strip()))


def confidence_status(confidence: float, *, force_flag: bool = False) -> str:
    """Maps a self-reported Gemini confidence to the review-queue's
    traffic-light status. `force_flag` is for content that must always
    get a human look regardless of confidence (e.g. is_crude=True)."""
    if force_flag or confidence < 0.5:
        return "flagged_red"
    if confidence < 0.8:
        return "flagged_yellow"
    return "pending"


LESSON_PROMPT = """Bạn là biên tập viên nội dung học tiếng Hàn cho người Việt.
Đọc ảnh/tài liệu bài học được đính kèm và trích xuất:
- Thông tin bài học: tiêu đề, cấp độ (1-6, ước lượng theo TOPIK), nội dung
  (diễn giải lại toàn bộ bài học bằng văn bản thuần), và các chủ đề liên quan.
- Toàn bộ từ vựng xuất hiện trong bài, mỗi từ kèm nghĩa tiếng Việt, định
  nghĩa tiếng Hàn ngắn gọn (nếu có), từ loại, Hán tự và âm Hán Việt (nếu là
  từ Hán Hàn), câu ví dụ trong bài.
- Toàn bộ mẫu ngữ pháp xuất hiện, viết pattern theo dạng "V/A + hình thái"
  (ví dụ: "V + -(으)ㄹ 뿐만 아니라"), kèm nghĩa tiếng Việt và câu ví dụ. QUAN
  TRỌNG: đừng chỉ liệt kê công thức như sách giáo khoa — với mỗi mẫu ngữ
  pháp, viết THÊM usage_context_vi giải thích bằng tiếng Việt: dùng khi nào,
  trong hoàn cảnh/tình huống nào, với sắc thái/thái độ gì so với các mẫu gần
  nghĩa khác (người học cần biết ÁP DỤNG chứ không chỉ nhớ công thức). Nếu
  mẫu này thường gặp trong đề thi TOPIK, thêm topik_tip_vi: viết như MỘT NHẬN
  XÉT VỀ NGÔN NGỮ (câu khẳng định: mẫu này hay xuất hiện ở dạng bài/ngữ cảnh
  nào, dấu hiệu nhận biết là gì), KHÔNG viết như lời ra lệnh hay lời dặn làm bài
  (tránh "hãy chọn...", "khi gặp câu hỏi... thì...").
Trả về đúng JSON schema đã cho, không thêm giải thích. Với mỗi mục, tự đánh
giá độ tự tin (confidence, 0-1) dựa trên độ rõ ràng của tài liệu gốc."""


def build_lesson_prompt_parts(file_bytes: bytes, mime_type: str) -> list[Any]:
    return [gemini_client.part_from_bytes(file_bytes, mime_type), LESSON_PROMPT]


def build_corpus_chunk_prompt(
    cues: list[Cue], known_topics: list[str], known_grammar_patterns: list[str]
) -> str:
    cue_lines = "\n".join(f'- source_ref="{c.source_ref}": {c.text}' for c in cues)
    topics_hint = ", ".join(known_topics) or "(chưa có chủ đề nào trong hệ thống)"
    grammar_hint = "\n".join(f"- {p}" for p in known_grammar_patterns) or "(chưa có mẫu ngữ pháp nào trong hệ thống)"
    return f"""Bạn đang xây dựng kho câu ví dụ tiếng Hàn từ phụ đề phim, dùng để
người học tra cứu câu mẫu theo chủ đề/trình độ/mẫu ngữ pháp.

Với MỖI câu thoại dưới đây, phân loại:
- keep: false nếu câu không có giá trị học tập (chỉ là tên riêng, thán từ,
  tiếng động, câu quá ngắn/vô nghĩa khi tách khỏi ngữ cảnh).
- is_crude: true nếu câu chứa ngôn từ thô tục, tình dục hoặc bạo lực rõ rệt
  (mặc định hệ thống sẽ lọc/gắn cờ các câu này).
- kind: "câu" (câu hoàn chỉnh), "cụm từ" (cụm chưa đủ thành câu), hoặc
  "mẫu ngữ pháp" (câu minh hoạ rõ một mẫu ngữ pháp cụ thể).
- level: 1-6, ước lượng theo TOPIK.
- register: "존댓말", "반말", hoặc "hỗn hợp".
- topics: 1-3 chủ đề về NỘI DUNG câu nói (nói về cái gì), CHỈ chọn từ danh
  sách sau và viết đúng từng chữ — không tự đặt chủ đề mới, không dùng "khẩu
  ngữ" làm chủ đề (kiểu nói đã có trường register): {topics_hint}.
- grammar_patterns: CHỈ chọn từ danh sách mẫu ngữ pháp đã có trong hệ thống
  dưới đây nếu câu thực sự minh hoạ mẫu đó; để trống nếu không khớp mẫu nào
  (không tự tạo mẫu ngữ pháp mới ở bước này):
{grammar_hint}

Trả về đúng JSON schema đã cho (mảng "items", mỗi phần tử echo lại đúng
source_ref của câu tương ứng), không thêm giải thích.

Danh sách câu thoại:
{cue_lines}"""


CORPUS_OCR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "raw_text": {
            "type": "STRING",
            "description": (
                "Toàn bộ văn bản/phụ đề nhận diện được trong ảnh/tài liệu, giữ "
                "đúng thứ tự xuất hiện, mỗi câu/dòng thoại một dòng riêng."
            ),
        },
    },
    "required": ["raw_text"],
}

CORPUS_OCR_PROMPT = """Bạn đang đọc một ảnh/tài liệu chứa phụ đề hoặc lời thoại phim tiếng Hàn
(có thể là ảnh chụp/màn hình chụp tệp phụ đề, hoặc ảnh chụp cảnh phim có phụ đề).
Hãy nhận diện TOÀN BỘ văn bản tiếng Hàn xuất hiện, giữ đúng thứ tự xuất hiện.
Nếu tài liệu có hiển thị mã thời gian dạng phụ đề chuẩn (vd
"00:00:01,000 --> 00:00:03,000"), hãy giữ nguyên định dạng đó kèm số thứ tự
để tái tạo đúng cấu trúc tệp .srt gốc. Nếu KHÔNG có mã thời gian hiển thị, chỉ
cần liệt kê từng câu thoại theo đúng thứ tự xuất hiện, mỗi câu một dòng —
KHÔNG tự bịa mã thời gian giả.
Trả về đúng JSON schema đã cho, không thêm giải thích."""


def build_corpus_ocr_prompt_parts(file_bytes: bytes, mime_type: str) -> list[Any]:
    return [gemini_client.part_from_bytes(file_bytes, mime_type), CORPUS_OCR_PROMPT]


def extract_corpus_source_text(file_bytes: bytes, mime_type: str | None) -> str:
    """kind="corpus" uploads now come in three shapes: a real .srt/.vtt
    file, plain typed/pasted dialogue text, or an image/PDF (a photographed
    or screenshotted subtitle list) — the Studio upload form's file picker
    already accepted image mime types before this function existed, but the
    old code path just base64-decoded the bytes as UTF-8 text regardless,
    which silently turned an image upload into garbage. An image/PDF now
    gets a Gemini-vision OCR pass first (Part.from_bytes is mime-agnostic,
    same as lesson/exam_paper's multimodal extraction) to recover the raw
    text; anything else is decoded as text exactly as before. Either way,
    run_corpus_extraction parses whatever text comes back the same way."""
    if mime_type and (mime_type.startswith("image/") or mime_type == "application/pdf"):
        result = gemini_client.generate_structured(
            model=settings.GEMINI_MODEL_LESSON_INGEST,
            prompt=build_corpus_ocr_prompt_parts(file_bytes, mime_type),
            response_schema=CORPUS_OCR_SCHEMA,
            prompt_version="corpus-ocr-v1",
        )
        return _parse_json(result["text"]).get("raw_text", "")
    return file_bytes.decode("utf-8", errors="replace")


def run_exam_extraction(
    db: Session,
    batch: ImportBatch,
    files: list[tuple[bytes, str]],
    *,
    session_label: str = "",
    on_progress=None,
) -> tuple[int, int]:
    """Reads the paper (see app/services/exam_extract.py) and stages the result.
    `files` is (bytes, mime_type) per uploaded file; they are the pages of ONE paper."""
    qtypes = [
        (code, name_vi, skill)
        for code, name_vi, skill in db.execute(
            select(QuestionType.code, QuestionType.name_vi, QuestionType.skill).where(QuestionType.active.is_(True))
        ).all()
    ]
    draft = exam_extract.extract(files, qtypes, session_label=session_label, on_progress=on_progress)
    return stage_exam_draft(db, batch, draft)


def stage_exam_draft(db: Session, batch: ImportBatch, draft: exam_extract.Draft) -> tuple[int, int]:
    """One ImportItem per passage (kind="exam_passage") and per question
    (kind="exam_item"), each with the reasons it needs a look in payload["flags"],
    plus one "exam_summary" item that records how the paper was read (pages, the
    section, missing question numbers, where the answers came from) and is not
    itself reviewed. Returns (staged, flagged)."""
    staged = 0
    flagged = 0
    for entry in draft.passages:
        status = exam_extract.status_for(entry["flags"], entry["confidence"])
        flagged += status != "pending"
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="exam_passage",
                status=status,
                confidence=entry["confidence"],
                payload=entry["payload"],
            )
        )
        staged += 1
    for entry in draft.items:
        status = exam_extract.status_for(entry["flags"], entry["confidence"])
        flagged += status != "pending"
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="exam_item",
                status=status,
                confidence=entry["confidence"],
                payload=entry["payload"],
            )
        )
        staged += 1
    db.add(
        ImportItem(
            import_batch_id=batch.id, kind=EXAM_SUMMARY, status="confirmed", confidence=1.0, payload=draft.summary
        )
    )
    db.flush()

    # An answer key printed on the paper's own pages is used like an uploaded one.
    skill = exam_extract.choose_key_section(draft.key, draft.summary.get("section"))
    if skill is not None:
        apply_answers(db, batch, draft.key[skill], skill=skill, source="paper")
    return staged, flagged


EXAM_SUMMARY = "exam_summary"


def _answer_rows(
    items: list[ImportItem], live: list[ExamItem], answers: dict[int, int], *, commit: bool
) -> exam_extract.AnswerReport:
    """Put the answers on the staged questions (and on the questions already in the
    library, if the batch was confirmed). With commit=False nothing changes — it only
    reports what would."""
    questions = [i for i in items if i.kind == "exam_item" and i.status != "rejected"]
    by_number = {i.payload.get("number"): i for i in questions}
    applied: list[int] = []
    changed: list[int] = []
    for number, answer in sorted(answers.items()):
        item = by_number.get(number)
        if item is None:
            continue
        applied.append(number)
        if item.payload.get("answer") not in (None, answer):
            changed.append(number)
        if not commit:
            continue
        flags = [f for f in item.payload.get("flags", []) if f != "no_answer"]
        item.payload = {**item.payload, "answer": answer, "answer_from_key": True, "flags": flags}
        if item.status in ("pending", "flagged_yellow", "flagged_red"):
            item.status = exam_extract.status_for(flags, item.confidence)
    missing = sorted(n for n in by_number if n not in answers)
    if commit:
        for number in missing:
            item = by_number[number]
            flags = list(item.payload.get("flags", []))
            if "no_answer" not in flags:
                flags.append("no_answer")
            item.payload = {**item.payload, "flags": flags}
            if item.status in ("pending", "flagged_yellow", "flagged_red"):
                item.status = exam_extract.status_for(flags, item.confidence)
        for row in live:
            if row.number in answers:
                row.answer = answers[row.number]
                row.answer_source = "editor"
    return exam_extract.AnswerReport(
        skill=None,
        applied=applied,
        changed=changed,
        unmatched=sorted(n for n in answers if n not in by_number),
        missing=missing,
    )


def _record_key(
    summary: ImportItem | None, report: exam_extract.AnswerReport, *, skill: str | None, source: str, conflicts: list[int]
) -> None:
    if summary is None:
        return
    summary.payload = {
        **summary.payload,
        "key": {
            "source": source,
            "skill": skill,
            "applied": len(report.applied),
            "missing": report.missing,
            "unmatched": report.unmatched,
            "conflicts": conflicts,
            "at": exam_extract.now_iso(),
        },
    }


def apply_answers(
    db: Session,
    batch: ImportBatch,
    answers: dict[int, int],
    *,
    skill: str | None,
    source: str,
    conflicts: list[int] | None = None,
    commit: bool = True,
) -> exam_extract.AnswerReport:
    """Set the answers of a paper's questions from a key ("file" = an uploaded
    answer key, "paper" = a key printed on the paper's own pages, "manual" = typed
    by an editor). Questions already in the library get the answer too, so a key can
    arrive after the batch was confirmed."""
    items = db.execute(select(ImportItem).where(ImportItem.import_batch_id == batch.id)).scalars().all()
    live = (
        db.execute(select(ExamItem).where(ExamItem.paper_id == batch.exam_paper_id)).scalars().all()
        if batch.exam_paper_id is not None
        else []
    )
    report = _answer_rows(list(items), list(live), answers, commit=commit)
    report.skill = skill
    report.conflicts = conflicts or []
    if commit:
        summary = next((i for i in items if i.kind == EXAM_SUMMARY), None)
        _record_key(summary, report, skill=skill, source=source, conflicts=report.conflicts)
        paper = db.get(ExamPaper, batch.exam_paper_id) if batch.exam_paper_id is not None else None
        if paper is not None:
            paper.answer_status = _answer_status(items)
        db.flush()
    return report


async def apply_answers_async(
    db: AsyncSession,
    batch: ImportBatch,
    answers: dict[int, int],
    *,
    skill: str | None,
    source: str,
    commit: bool = True,
) -> exam_extract.AnswerReport:
    """The same as apply_answers, for a request handler's async session."""
    items = (await db.execute(select(ImportItem).where(ImportItem.import_batch_id == batch.id))).scalars().all()
    live = (
        (await db.execute(select(ExamItem).where(ExamItem.paper_id == batch.exam_paper_id))).scalars().all()
        if batch.exam_paper_id is not None
        else []
    )
    report = _answer_rows(list(items), list(live), answers, commit=commit)
    report.skill = skill
    if commit:
        summary = next((i for i in items if i.kind == EXAM_SUMMARY), None)
        _record_key(summary, report, skill=skill, source=source, conflicts=[])
        paper = await db.get(ExamPaper, batch.exam_paper_id) if batch.exam_paper_id is not None else None
        if paper is not None:
            paper.answer_status = _answer_status(items)
        await db.commit()
    return report


def _answer_status(items: list[ImportItem]) -> str:
    """content.exam_paper.answer_status: pending (no answers), partial, complete."""
    questions = [i for i in items if i.kind == "exam_item" and i.status != "rejected"]
    have = sum(1 for i in questions if i.payload.get("answer") is not None)
    if not questions or have == 0:
        return "pending"
    return "complete" if have == len(questions) else "partial"


def paper_skill(items: list[ImportItem], session_label: str | None) -> str | None:
    """Which part of the exam this batch is (đọc / nghe / viết): the editor's own
    label first, then what extraction recorded, then the staged question types."""
    from_label = exam_extract.section_from_label(session_label)
    if from_label:
        return from_label
    summary = next((i for i in items if i.kind == EXAM_SUMMARY), None)
    if summary is not None and summary.payload.get("section"):
        return summary.payload["section"]
    return None


def _extract_lesson_one_shot(file_bytes: bytes, mime_type: str) -> LessonExtraction:
    """The original single-call extraction. It lists far fewer items than the
    lesson teaches (the model stops early), so it is only the fallback for a
    document the inventory step finds nothing in — see lesson_extract."""
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=build_lesson_prompt_parts(file_bytes, mime_type),
        response_schema=LESSON_SCHEMA,
        prompt_version="lesson-v2",
    )
    return LessonExtraction.model_validate(_parse_json(result["text"]))


def _draft_from(extraction: LessonExtraction) -> lesson_extract.LessonDraft:
    return lesson_extract.LessonDraft(
        title=extraction.title,
        level=extraction.level,
        content=extraction.content,
        topics=extraction.topics,
        confidence=extraction.confidence,
        vocab=[lesson_extract.VocabCard(**v.model_dump()) for v in extraction.vocab],
        grammar=[lesson_extract.GrammarCard(**g.model_dump()) for g in extraction.grammar],
    )


def extract_lesson(file_bytes: bytes, mime_type: str, on_progress=None) -> lesson_extract.LessonDraft:
    """Lesson (image/PDF/plain text) -> one card per item it teaches: an
    inventory first, then the cards in small groups (see lesson_extract for
    why a single call loses most of a lesson)."""
    return lesson_extract.extract(
        file_bytes,
        mime_type,
        fallback=lambda: _draft_from(_extract_lesson_one_shot(file_bytes, mime_type)),
        on_progress=on_progress,
    )


def classify_corpus_chunk(
    cues: list[Cue], known_topics: list[str], known_grammar_patterns: list[str]
) -> CorpusChunkClassification:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_CORPUS_INGEST,
        prompt=build_corpus_chunk_prompt(cues, known_topics, known_grammar_patterns),
        response_schema=CORPUS_CHUNK_SCHEMA,
        prompt_version="corpus-v2",
    )
    return CorpusChunkClassification.model_validate(_parse_json(result["text"]))


def drop_duplicate_cues(cues: list[Cue], known_keys: set[str]) -> tuple[list[Cue], int]:
    """Cues whose sentence is not already in the corpus (known_keys) and has
    not appeared earlier in this same file. A repeated line is never staged,
    so it neither costs a classification call nor lands in front of the
    reviewer twice. Returns (fresh cues in order, how many repeats were
    dropped); a cue with no letters at all is ignored without being counted."""
    seen = set(known_keys)
    fresh: list[Cue] = []
    dropped = 0
    for cue in cues:
        key = corpus_browse.normalize_key(cue.text)
        if not key:
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        fresh.append(cue)
    return fresh, dropped


@dataclass
class CorpusExtractionStats:
    """What one subtitle import did, for the job result the Studio shows."""

    staged: int = 0
    flagged: int = 0
    duplicates: int = 0
    # lines judged not natural Korean (machine-translated, garbled): never
    # staged, so they cannot reach a learner; a few are kept for the report
    unnatural: int = 0
    unnatural_samples: list[str] = field(default_factory=list)
    # staged lines that have no meaning/usage note yet (their enrichment call
    # failed) — the Studio "Làm giàu kho câu" run fills them in after confirm
    unenriched: int = 0


_UNNATURAL_SAMPLE_LIMIT = 15


def drop_garbled_cues(cues: list[Cue]) -> tuple[list[Cue], list[str]]:
    """Cues that are not garbled by the rule-based pre-check (see
    corpus_enrich.is_garbled), and the texts that were dropped."""
    kept: list[Cue] = []
    dropped: list[str] = []
    for cue in cues:
        (dropped if corpus_enrich.is_garbled(cue.text) else kept).append(cue)
    return kept, dropped


def _enrich_texts(
    texts: list[str],
    known_grammar: list[str],
    generate: corpus_enrich.Generate,
    *,
    chunk_size: int,
    pause: float,
) -> list[corpus_enrich.Enrichment | None]:
    """Enriches `texts` in fixed-size calls. Best effort: a failed call leaves
    its sentences None (un-enriched) instead of failing the import — the
    backfill picks them up later."""
    out: list[corpus_enrich.Enrichment | None] = []
    for start in range(0, len(texts), chunk_size):
        part = texts[start : start + chunk_size]
        if start and pause:
            time.sleep(pause)
        try:
            out.extend(corpus_enrich.enrich_sentences(part, known_grammar, generate))
        except Exception as exc:  # noqa: BLE001 — keep the import going
            print(f"[corpus-enrich] import chunk skipped: {str(exc)[:200]}", flush=True)
            out.extend([None] * len(part))
    return out


def run_corpus_extraction(
    db: Session,
    batch: ImportBatch,
    raw_subtitle_text: str,
    generate: corpus_enrich.Generate | None = None,
    pause: float | None = None,
) -> CorpusExtractionStats:
    """Parses the subtitle file, drops lines the corpus already has (or that
    repeat inside the file) and lines that are plainly garbled, classifies the
    rest in fixed-size chunks (so each Gemini call's prompt stays constant-size
    regardless of film length — FR-19/Gate G6), then enriches the kept lines
    (Vietnamese meaning, usage note, a naturalness verdict, topics). A line the
    model judges unnatural is not staged at all. One ImportItem is staged per
    remaining cue."""
    stats = CorpusExtractionStats()
    cues = parse_subtitles(raw_subtitle_text)
    if not cues:
        # Not real timestamped .srt/.vtt — a plain-text paste or an
        # image/PDF's OCR output (see extract_corpus_source_text), neither
        # of which has timecodes for parse_subtitles to key off. Fall back
        # to one cue per line instead of silently staging nothing.
        cues = parse_plain_lines(raw_subtitle_text)
    known_keys = corpus_browse.corpus_keys([t for (t,) in db.execute(select(CorpusItem.text_ko)).all()])
    cues, stats.duplicates = drop_duplicate_cues(cues, known_keys)
    cues, garbled = drop_garbled_cues(cues)
    stats.unnatural += len(garbled)
    stats.unnatural_samples.extend(garbled[:_UNNATURAL_SAMPLE_LIMIT])
    known_grammar = [p for (p,) in db.execute(select(GrammarPoint.pattern)).all()][:200]
    if generate is None:
        generate = corpus_enrich.make_generate(settings.GEMINI_MODEL_CORPUS_INGEST, gemini_client.generate_structured)
    if pause is None:
        pause = settings.CORPUS_ENRICH_PAUSE_SEC

    for chunk in chunk_cues(cues, settings.CORPUS_CHUNK_SIZE):
        by_ref = {c.source_ref: c for c in chunk}
        try:
            classification = classify_corpus_chunk(chunk, list(corpus_enrich.CORPUS_TOPICS), known_grammar)
        except (ValidationError, json.JSONDecodeError, KeyError):
            # One bad chunk shouldn't sink the whole film — skip it; the
            # reviewer will simply see fewer candidates from this stretch.
            continue

        kept = [
            (by_ref[entry.source_ref], entry)
            for entry in classification.items
            if entry.keep and entry.source_ref in by_ref
        ]
        enrichments = _enrich_texts(
            [cue.text for cue, _ in kept],
            known_grammar,
            generate,
            chunk_size=settings.CORPUS_ENRICH_CHUNK_SIZE,
            pause=pause,
        )

        for (cue, entry), enrichment in zip(kept, enrichments):
            if enrichment is not None and enrichment.naturalness == corpus_enrich.UNNATURAL:
                stats.unnatural += 1
                if len(stats.unnatural_samples) < _UNNATURAL_SAMPLE_LIMIT:
                    stats.unnatural_samples.append(cue.text)
                continue
            status = confidence_status(entry.confidence, force_flag=entry.is_crude)
            if enrichment is not None and enrichment.naturalness == corpus_enrich.AWKWARD and status == "pending":
                status = "flagged_yellow"  # readable but stiff — a human should look
            if status != "pending":
                stats.flagged += 1
            payload: dict[str, Any] = {
                "text_ko": cue.text,  # always the ORIGINAL subtitle line, never Gemini's paraphrase
                "source_ref": cue.source_ref,
                "kind": entry.kind,
                "level": entry.level,
                "register": entry.register,
                "topics": corpus_enrich.normalize_topics(entry.topics),
                "grammar_patterns": entry.grammar_patterns,
                "is_crude": entry.is_crude,
            }
            if enrichment is not None:
                payload.update(
                    topics=enrichment.topics,
                    grammar_patterns=list(dict.fromkeys([*entry.grammar_patterns, *enrichment.grammar_patterns])),
                    meaning_vi=enrichment.meaning_vi,
                    usage_note_vi=enrichment.usage_note_vi,
                    naturalness=enrichment.naturalness,
                    enriched_version=corpus_enrich.ENRICH_VERSION,
                )
            else:
                stats.unenriched += 1
            db.add(
                ImportItem(
                    import_batch_id=batch.id,
                    kind="corpus_item",
                    status=status,
                    confidence=entry.confidence,
                    payload=payload,
                )
            )
            stats.staged += 1
    db.flush()
    return stats


def clear_staged_items(db: Session, batch: ImportBatch) -> int:
    """Drop the unreviewed proposals of a lesson batch that is about to be read
    again (re-uploaded after the extraction improved). Nothing is written to the
    library before the batch is confirmed, so there is nothing to undo; a
    confirmed batch is left alone."""
    if batch.status == "confirmed":
        return 0
    result = db.execute(delete(ImportItem).where(ImportItem.import_batch_id == batch.id))
    return result.rowcount or 0


def run_lesson_extraction(
    db: Session, batch: ImportBatch, file_bytes: bytes, mime_type: str, on_progress=None
) -> tuple[int, int]:
    extraction = extract_lesson(file_bytes, mime_type, on_progress)
    staged = 0
    flagged = 0

    lesson_status = confidence_status(extraction.confidence)
    if lesson_status != "pending":
        flagged += 1
    db.add(
        ImportItem(
            import_batch_id=batch.id,
            kind="lesson",
            status=lesson_status,
            confidence=extraction.confidence,
            payload={
                "title": extraction.title,
                "level": extraction.level,
                "content": extraction.content,
                "topics": extraction.topics,
            },
        )
    )
    staged += 1

    for v in extraction.vocab:
        status = confidence_status(v.confidence)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="vocab_item",
                status=status,
                confidence=v.confidence,
                payload=v.model_dump(exclude={"confidence"}),
            )
        )
        staged += 1

    for g in extraction.grammar:
        status = confidence_status(g.confidence)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="grammar_point",
                status=status,
                confidence=g.confidence,
                payload=g.model_dump(exclude={"confidence"}),
            )
        )
        staged += 1

    db.flush()
    return staged, flagged


_ARTICLE_FETCH_TIMEOUT = 15.0
_ARTICLE_MAX_BODY_CHARS = 20000
_ARTICLE_USER_AGENT = "Mozilla/5.0 (compatible; HaruBot/1.0; +family-use-only, not for redistribution)"

def fetch_article(url: str) -> article_extract.ExtractedArticle:
    """Fetches one editorial/column page and extracts ONLY the article: title,
    clean paragraphs (one per line) and inline images with captions. The
    real work — finding the article container, dropping page chrome/
    captions/ranking widgets — lives in app.services.article_extract (see
    its docstring for why the old "get_text() of the whole <body>" approach
    leaked menus, AI-summary disclaimers and related-article lists into both
    the reader and TTS).

    Never written straight onto EditorialArticle: it becomes part of a
    staged `editorial_meta` ImportItem instead, so an admin can review — and
    hand-fix a messy scrape or a wrong outlet-name guess — before any of it
    ever reaches a learner (same "staged, human confirms" principle as
    everything else here)."""
    resp = httpx.get(
        url,
        timeout=_ARTICLE_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _ARTICLE_USER_AGENT},
    )
    resp.raise_for_status()
    art = article_extract.extract_article(resp.text, str(resp.url))
    if not art.site_name:
        art.site_name = (httpx.URL(url).host or "").removeprefix("www.") or None
    art.body = art.body[:_ARTICLE_MAX_BODY_CHARS]
    return art


def fetch_article_text(url: str) -> tuple[str, str | None, str | None]:
    """Back-compat wrapper: (body_ko, parsed_title, suggested_source_name)."""
    art = fetch_article(url)
    return art.body, art.title, art.site_name


def build_editorial_prompt(body_ko: str) -> str:
    return f"""Bạn là biên tập viên nội dung luyện đọc xã luận/chuyên mục báo tiếng
Hàn cho người Việt học tiếng Hàn trình độ trung-cao cấp (TOPIK II), đang
chuẩn bị cho câu 54 phần viết (bài luận theo cấu trúc hiện tượng – nguyên
nhân – kết quả – giải pháp).

Văn bản dưới đây được lấy tự động từ một trang báo nên có thể còn lẫn vài
dòng menu/quảng cáo/liên quan khác — hãy bỏ qua các dòng đó, chỉ tập trung
vào nội dung bài xã luận/chuyên mục chính.

Đọc bài viết và:
1. Ước lượng cấp độ TOPIK (1-6) và đề xuất 2-4 chủ đề ngắn gọn bằng tiếng Việt.
2. Trích xuất từ vựng và mẫu ngữ pháp đáng học (cùng tiêu chí như trích xuất
   bài học thông thường: nghĩa tiếng Việt, định nghĩa tiếng Hàn ngắn gọn nếu
   có, Hán tự/âm Hán Việt nếu là từ Hán Hàn, câu ví dụ). Với MỖI mẫu ngữ
   pháp: viết usage_context_vi (tiếng Việt) giải thích dùng khi nào/hoàn
   cảnh nào — không chỉ nêu công thức; và topik_tip_vi nếu mẫu này thường
   gặp trong đề thi TOPIK (đặc biệt viết câu 54) — nêu cách vận dụng thực tế
   dưới dạng nhận xét về ngôn ngữ (câu khẳng định), không viết thành lời ra
   lệnh hay lời dặn làm bài.
3. Viết MỘT dàn ý mẫu bằng tiếng Hàn theo đúng cấu trúc 4 phần của câu 54:
   hiện tượng, nguyên nhân, kết quả/ảnh hưởng, giải pháp/kiến nghị — mỗi
   phần 2-4 câu, lấy cảm hứng từ chủ đề bài xã luận này (không cần bám sát
   từng câu chữ gốc).
4. Viết 3-5 câu hỏi gợi mở bằng tiếng Việt để người học tự suy nghĩ TRƯỚC
   khi xem dàn ý mẫu — gợi mở tư duy, không tiết lộ nội dung dàn ý.

Trả về đúng JSON schema đã cho, không thêm giải thích. Tự đánh giá độ tự tin
(confidence, 0-1) cho toàn bộ kết quả.

Bài viết:
{body_ko}"""


def extract_editorial(body_ko: str) -> EditorialExtraction:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=build_editorial_prompt(body_ko),
        response_schema=EDITORIAL_SCHEMA,
        prompt_version="editorial-v1",
    )
    return EditorialExtraction.model_validate(_parse_json(result["text"]))


def run_editorial_extraction(
    db: Session,
    batch: ImportBatch,
    body_ko: str,
    title_ko: str | None,
    suggested_source_name: str | None = None,
    images: list[dict] | None = None,
) -> tuple[int, int]:
    """Stages the fetched body + Gemini's classification as one
    kind="editorial_meta" ImportItem (body_ko included, so an admin can
    review/edit the scraped text before confirm — see fetch_article_text),
    plus one kind="vocab_item"/"grammar_point" ImportItem per extracted
    word/pattern (lesson_id left null at confirm time — see
    apply_editorial_batch). `suggested_source_name` (og:site_name/domain,
    see fetch_article_text) rides along in the same payload so an admin can
    correct the article's real outlet name at review time too — EditorialArticle.
    source_name is admin-typed at submission and otherwise never editable
    again (see find_or_create_editorial_article's docstring), which is
    exactly how a sample article ended up with an author's byline stored
    as source_name. Returns (staged_count, flagged_count)."""
    extraction = extract_editorial(body_ko)
    staged = 0
    flagged = 0

    meta_status = confidence_status(extraction.confidence)
    if meta_status != "pending":
        flagged += 1
    db.add(
        ImportItem(
            import_batch_id=batch.id,
            kind="editorial_meta",
            status=meta_status,
            confidence=extraction.confidence,
            payload={
                "body_ko": body_ko,
                "title_ko": title_ko,
                "source_name": suggested_source_name,
                # Inline photos scraped with the article (url/caption/
                # after_paragraph) — see article_extract.extract_article.
                "images": images or [],
                "level_estimate": extraction.level_estimate,
                "topic_tags": extraction.topic_tags,
                "model_outline": extraction.model_outline.model_dump(),
                "thinking_guide_text": extraction.thinking_guide_text,
            },
        )
    )
    staged += 1

    for v in extraction.vocab:
        status = confidence_status(v.confidence)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="vocab_item",
                status=status,
                confidence=v.confidence,
                payload=v.model_dump(exclude={"confidence"}),
            )
        )
        staged += 1

    for g in extraction.grammar:
        status = confidence_status(g.confidence)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="grammar_point",
                status=status,
                confidence=g.confidence,
                payload=g.model_dump(exclude={"confidence"}),
            )
        )
        staged += 1

    db.flush()
    return staged, flagged


# ================================================================ confirm ==
_VOCAB_FIELDS = {
    "hangul", "pos", "meaning_vi", "definition_ko", "level", "hanja", "sino_vietnamese", "example_ko",
    "family", "node_word", "register", "usage_note_vi", "collocations", "distractors",
}
_GRAMMAR_FIELDS = {
    "pattern", "meaning_vi", "level", "example_ko", "usage_context_vi", "topik_tip_vi",
    "contrast_group", "contrasts",
}


def _row_fields(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The columns a confirmed vocab/grammar proposal becomes. A reviewer can edit
    the payload by hand in Studio, so the chunk layers are tidied again here: a
    node word that is not part of the phrase or a stray register would otherwise
    reach learners (and break the fill-in-the-blank built from them)."""
    data = {k: v for k, v in payload.items() if k in (_VOCAB_FIELDS if kind == "vocab_item" else _GRAMMAR_FIELDS)}
    if kind == "vocab_item":
        lesson_extract.tidy_vocab(data)
    else:
        lesson_extract.tidy_grammar(data)
    return data


def _find_or_create_topic(db: Session, name: str) -> int:
    name = name.strip()
    existing = db.execute(select(Topic).where(Topic.name == name)).scalar_one_or_none()
    if existing:
        return existing.id
    topic = Topic(name=name)
    db.add(topic)
    db.flush()
    return topic.id


def apply_lesson_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    """Writes every CONFIRMED item of a `kind=lesson` batch into
    content.lesson/vocab_item/grammar_point. Idempotent/re-callable: an
    item already applied (a canonical row already carries its
    import_item_id) is skipped, so calling confirm again after a
    reviewer confirms a few more items only writes the new ones.
    """
    items = db.execute(select(ImportItem).where(ImportItem.import_batch_id == batch.id)).scalars().all()
    lesson_item = next((i for i in items if i.kind == "lesson" and i.status == "confirmed"), None)
    if lesson_item is None:
        return {"applied": 0, "reason": "no_confirmed_lesson_item"}

    content_hash = hashlib.sha256(lesson_item.payload["content"].encode("utf-8")).hexdigest()
    lesson = db.execute(select(Lesson).where(Lesson.content_hash == content_hash)).scalar_one_or_none()
    applied = 0
    if lesson is None:
        lesson = Lesson(
            title=lesson_item.payload["title"],
            level=lesson_item.payload["level"],
            content=lesson_item.payload["content"],
            content_hash=content_hash,
            import_item_id=lesson_item.id,
        )
        db.add(lesson)
        db.flush()
        applied += 1

    for name in lesson_item.payload.get("topics", []):
        topic_id = _find_or_create_topic(db, name)
        exists = db.execute(
            select(LessonTopic).where(LessonTopic.lesson_id == lesson.id, LessonTopic.topic_id == topic_id)
        ).scalar_one_or_none()
        if exists is None:
            db.add(LessonTopic(lesson_id=lesson.id, topic_id=topic_id))

    for item in items:
        if item.status != "confirmed" or item.kind not in ("vocab_item", "grammar_point"):
            continue
        model = VocabItem if item.kind == "vocab_item" else GrammarPoint
        already = db.execute(select(model).where(model.import_item_id == item.id)).scalar_one_or_none()
        if already is not None:
            continue
        payload = _row_fields(item.kind, item.payload)
        db.add(model(lesson_id=lesson.id, import_item_id=item.id, **payload))
        applied += 1

    db.flush()
    return {"applied": applied, "lesson_id": lesson.id}


def apply_corpus_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    """Writes every CONFIRMED `corpus_item` proposal into corpus.corpus_item
    (embedding included — the one AI call left in the confirm step, which
    is exactly why confirm runs as a Celery job rather than inline in the
    FastAPI handler). ON CONFLICT DO NOTHING on (film_id, source_ref)
    makes re-running confirm after more items are reviewed safe."""
    if batch.film_id is None:
        return {"applied": 0, "reason": "batch_has_no_film_id"}

    items = (
        db.execute(
            select(ImportItem).where(
                ImportItem.import_batch_id == batch.id,
                ImportItem.kind == "corpus_item",
                ImportItem.status == "confirmed",
            )
        )
        .scalars()
        .all()
    )
    if not items:
        return {"applied": 0, "reason": "no_confirmed_items"}

    grammar_rows = {p: pid for pid, p in db.execute(select(GrammarPoint.id, GrammarPoint.pattern)).all()}
    # The extraction step already skips known lines, but two batches can be
    # reviewed side by side (or a film re-uploaded under another title), so the
    # write step checks again: the same sentence is stored once.
    known_keys = corpus_browse.corpus_keys([t for (t,) in db.execute(select(CorpusItem.text_ko)).all()])

    applied = 0
    duplicates_skipped = 0
    for item in items:
        payload = item.payload
        key = corpus_browse.normalize_key(payload["text_ko"])
        if key in known_keys:
            duplicates_skipped += 1
            continue
        topic_ids = [_find_or_create_topic(db, name) for name in payload.get("topics", [])]
        grammar_point_ids = [
            grammar_rows[p] for p in payload.get("grammar_patterns", []) if p in grammar_rows
        ]
        embedding = gemini_client.embed_text(text=payload["text_ko"])

        stmt = (
            pg_insert(CorpusItem)
            .values(
                film_id=batch.film_id,
                text_ko=payload["text_ko"],
                kind=payload["kind"],
                level=payload["level"],
                register=payload["register"],
                source_ref=payload.get("source_ref"),
                topic_ids=topic_ids,
                grammar_point_ids=grammar_point_ids,
                embedding=embedding,
                # filled at import time by the enrichment step (and editable by
                # the reviewer); absent on older payloads, which the backfill
                # then completes
                meaning_vi=payload.get("meaning_vi") or None,
                usage_note_vi=payload.get("usage_note_vi") or None,
                naturalness=payload.get("naturalness"),
                enriched_version=payload.get("enriched_version"),
            )
            .on_conflict_do_nothing(index_elements=["film_id", "source_ref"])
        )
        result = db.execute(stmt)
        if result.rowcount:
            applied += 1
            known_keys.add(key)

    db.flush()
    outcome: dict[str, Any] = {"applied": applied, "duplicates_skipped": duplicates_skipped}
    if applied == 0 and duplicates_skipped:
        outcome["reason"] = "all_duplicates"
    return outcome


@dataclass
class CorpusEnrichmentStats:
    """Outcome of one backfill run (becomes the job result)."""

    pending_before: int = 0
    processed: int = 0
    enriched: int = 0
    hidden: int = 0  # judged unnatural in this run: now hidden from learners
    awkward: int = 0
    skipped: int = 0  # the model gave nothing usable; retried by the next run
    failed_chunks: int = 0
    stopped_early: bool = False
    error: str | None = None
    hidden_samples: list[str] = field(default_factory=list)


# After this many chunks in a row fail (quota exhausted, outage, bad key) the
# run stops instead of burning through the whole corpus on guaranteed errors.
_MAX_CONSECUTIVE_FAILED_CHUNKS = 3


def corpus_enrichment_pending_filter():
    """SQL condition: sentences the current enrichment prompt has not covered."""
    return or_(
        CorpusItem.enriched_version.is_(None),
        CorpusItem.enriched_version != corpus_enrich.ENRICH_VERSION,
    )


def run_corpus_enrichment(
    db: Session,
    generate: corpus_enrich.Generate,
    *,
    chunk_size: int | None = None,
    pause: float | None = None,
    on_progress: Any = None,
) -> CorpusEnrichmentStats:
    """Adds a Vietnamese meaning, a usage note, a naturalness verdict and
    re-tagged topics to every corpus sentence the current prompt version has
    not covered yet. Safe to run again and again:

    - only un-enriched rows are selected, and each chunk is COMMITTED as it
      finishes, so a crash/redeploy loses at most the chunk in flight and the
      next run continues where this one stopped;
    - a sentence the model returns nothing usable for stays un-enriched (and is
      tried again next run) rather than being marked done with an empty note;
    - a line judged unnatural gets only the verdict (it is hidden, nothing else
      about it is changed) — Studio can restore it;
    - for the rest, `topic_ids` is REPLACED by the fresh tags (the old free-text
      tags like "Khẩu ngữ" were the noise being fixed) and grammar patterns are
      only ever ADDED to, never removed.
    The Korean text itself is never touched."""
    size = chunk_size or settings.CORPUS_ENRICH_CHUNK_SIZE
    wait = settings.CORPUS_ENRICH_PAUSE_SEC if pause is None else pause
    stats = CorpusEnrichmentStats()

    rows = db.execute(
        select(CorpusItem.id, CorpusItem.text_ko, CorpusItem.grammar_point_ids, CorpusItem.naturalness)
        .where(corpus_enrichment_pending_filter())
        .order_by(CorpusItem.id)
    ).all()
    stats.pending_before = len(rows)
    if not rows:
        return stats

    grammar_rows = {p: pid for pid, p in db.execute(select(GrammarPoint.id, GrammarPoint.pattern)).all()}
    known_grammar = list(grammar_rows)[:200]
    topic_ids: dict[str, int] = {}

    def topic_id(name: str) -> int:
        if name not in topic_ids:
            topic_ids[name] = _find_or_create_topic(db, name)
        return topic_ids[name]

    consecutive_failures = 0
    total_chunks = (len(rows) + size - 1) // size
    for number, start in enumerate(range(0, len(rows), size), 1):
        chunk = rows[start : start + size]
        if number > 1 and wait:
            time.sleep(wait)
        try:
            results = corpus_enrich.enrich_sentences([text for _, text, _, _ in chunk], known_grammar, generate)
        except Exception as exc:  # noqa: BLE001 — recorded; the run decides whether to go on
            db.rollback()
            stats.failed_chunks += 1
            consecutive_failures += 1
            stats.error = str(exc)[:200]
            print(f"[corpus-enrich] chunk {number}/{total_chunks} failed: {stats.error}", flush=True)
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILED_CHUNKS:
                stats.stopped_early = True
                break
            continue
        consecutive_failures = 0

        for (row_id, text, grammar_ids, current), result in zip(chunk, results):
            stats.processed += 1
            if result is None:
                stats.skipped += 1
                continue
            approved = current == corpus_enrich.APPROVED  # an editor restored this line: never hide it again
            values: dict[str, Any] = {"enriched_version": corpus_enrich.ENRICH_VERSION}
            if not approved:
                values["naturalness"] = result.naturalness
            if result.naturalness == corpus_enrich.UNNATURAL:
                if approved:
                    # keep the editor's decision; there is no meaning to add
                    db.execute(update(CorpusItem).where(CorpusItem.id == row_id).values(**values))
                    continue
                values.update(meaning_vi=None, usage_note_vi=None)
                stats.hidden += 1
                if len(stats.hidden_samples) < _UNNATURAL_SAMPLE_LIMIT:
                    stats.hidden_samples.append(text)
            else:
                values.update(meaning_vi=result.meaning_vi, usage_note_vi=result.usage_note_vi)
                stats.enriched += 1
                if result.naturalness == corpus_enrich.AWKWARD:
                    stats.awkward += 1
                values["topic_ids"] = [topic_id(name) for name in result.topics]
                added = [grammar_rows[p] for p in result.grammar_patterns if p in grammar_rows]
                if added:
                    values["grammar_point_ids"] = sorted({*(grammar_ids or []), *added})
            db.execute(update(CorpusItem).where(CorpusItem.id == row_id).values(**values))
        db.commit()
        print(
            f"[corpus-enrich] chunk {number}/{total_chunks}: enriched={stats.enriched} "
            f"hidden={stats.hidden} skipped={stats.skipped}",
            flush=True,
        )
        if on_progress is not None:
            on_progress(min(start + size, len(rows)), len(rows))
    return stats


def apply_exam_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    """Writes confirmed exam_passage/exam_item proposals into content.
    exam_passage/exam_item. content.exam_passage is one of the tables
    transcribed VERBATIM from the SDD's CREATE TABLE snippet (see
    models.py's module docstring), so it deliberately does NOT get an
    import_item_id column added just for our own bookkeeping — instead,
    the applied passage's real id is stashed back onto the *ImportItem's*
    own payload (`_applied_id`), which is purely ingest-internal staging,
    not spec-constrained. That's what makes both idempotent re-apply and
    rollback-after-confirm possible without touching the verbatim shape.
    qtype_code is resolved against content.question_type's controlled
    vocabulary; an item whose qtype_code doesn't resolve (model left it
    blank, or picked something stale) is simply left unapplied — the
    reviewer edits the item's payload with a valid code and calls confirm
    again, same re-callable pattern as everything else here.
    """
    if batch.exam_paper_id is None:
        return {"applied": 0, "reason": "batch_has_no_exam_paper_id"}

    items = db.execute(select(ImportItem).where(ImportItem.import_batch_id == batch.id)).scalars().all()
    passage_items = [i for i in items if i.kind == "exam_passage" and i.status == "confirmed"]
    exam_items = [i for i in items if i.kind == "exam_item" and i.status == "confirmed"]

    qtype_by_code = {code: qid for qid, code in db.execute(select(QuestionType.id, QuestionType.code)).all()}
    local_to_real: dict[str, uuid.UUID] = {}
    applied = 0

    for p in passage_items:
        applied_id = p.payload.get("_applied_id")
        if applied_id:
            local_to_real[p.payload["local_ref"]] = uuid.UUID(applied_id)
            continue
        # A passage the exam withholds (copyright notice) is kept as a row with no
        # text, so the questions that rest on it stay linked to it and the drill
        # can tell they cannot be answered.
        withheld = bool(p.payload.get("withheld"))
        passage = ExamPassage(
            paper_id=batch.exam_paper_id,
            kind=p.payload["kind"],
            body_ko=None if withheld else p.payload.get("body_ko"),
            chart_data={"withheld": True} if withheld else None,
            image_key=None,
            audio_key=None,
            source_page=p.payload.get("source_page", 1),
            source_bbox={},
        )
        db.add(passage)
        db.flush()
        local_to_real[p.payload["local_ref"]] = passage.id
        p.payload = {**p.payload, "_applied_id": str(passage.id)}
        db.add(p)
        applied += 1

    for it in exam_items:
        already = db.execute(select(ExamItem).where(ExamItem.import_item_id == it.id)).scalar_one_or_none()
        if already is not None:
            continue
        qtype_id = qtype_by_code.get(it.payload.get("qtype_code"))
        if qtype_id is None:
            continue  # unresolved qtype — reviewer must patch qtype_code, then confirm again
        passage_ref = it.payload.get("passage_ref")
        answer_from_key = bool(it.payload.get("answer_from_key"))
        answer = it.payload.get("answer")
        db.add(
            ExamItem(
                paper_id=batch.exam_paper_id,
                passage_id=local_to_real.get(passage_ref) if passage_ref else None,
                number=it.payload["number"],
                qtype_id=qtype_id,
                instruction_ko=it.payload.get("instruction_ko"),
                stem_ko=it.payload.get("stem_ko") or "",
                options=it.payload.get("options", []),
                answer=answer,
                answer_source=("editor" if answer_from_key else "ai_guess") if answer is not None else None,
                difficulty_est=0.5,
                confidence=it.confidence,
                import_item_id=it.id,
            )
        )
        applied += 1

    db.flush()
    return {"applied": applied, "exam_paper_id": str(batch.exam_paper_id)}


def apply_editorial_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    """Writes a confirmed `editorial_meta` item's body/classification onto
    the pre-created EditorialArticle row exactly once (import_item_id
    guard, same idea as Lesson's content_hash dedup — see
    find_or_create_editorial_article's docstring for why the row already
    exists before this runs), and confirmed vocab/grammar proposals into
    content.vocab_item/grammar_point with lesson_id=None, appending their
    ids onto the article's own vocab_ids/grammar_ids bare arrays — no
    cross-schema FK/JOIN, same module-boundary rule corpus_item's
    topic_ids/grammar_point_ids already follow."""
    if batch.editorial_article_id is None:
        return {"applied": 0, "reason": "batch_has_no_editorial_article_id"}

    article = db.get(EditorialArticle, batch.editorial_article_id)
    if article is None:
        return {"applied": 0, "reason": "editorial_article_not_found"}

    items = db.execute(select(ImportItem).where(ImportItem.import_batch_id == batch.id)).scalars().all()
    applied = 0

    # Unlike vocab_item/grammar_point (individually AI-guessed proposals
    # that genuinely warrant a per-row accept/reject), editorial_meta's
    # body_ko is the real, human-written scraped article text — not an AI
    # invention — bundled with Gemini's classification of it (level/topics/
    # outline) under one status. Gating this on status == "confirmed"
    # meant a reviewer who confirmed the vocab/grammar rows and hit "Xác
    # nhận lô" WITHOUT separately noticing and clicking "Xác nhận" on this
    # one extra row got a batch marked confirmed (vocab/grammar did apply)
    # with the article's own body_ko silently left empty forever — no
    # warning anywhere. Confirmed in production: an article with real
    # vocab+grammar and body_ko == "". Apply it unless a reviewer
    # EXPLICITLY rejected it (they still can, e.g. a garbled scrape) —
    # matches how a scraped article is actually reviewed in practice (read
    # the whole thing once, act on the article, not tick every row).
    meta_item = next((i for i in items if i.kind == "editorial_meta" and i.status != "rejected"), None)
    if meta_item is not None and article.import_item_id is None:
        payload = meta_item.payload
        article.body_ko = payload["body_ko"]
        article.title_ko = payload.get("title_ko") or article.title_ko
        # An admin who noticed/edited a wrong source_name (e.g. an author
        # byline scraped in as the outlet name) in this item's payload
        # before confirming gets that correction applied here — see
        # run_editorial_extraction's docstring.
        article.source_name = payload.get("source_name") or article.source_name
        article.level_estimate = payload["level_estimate"]
        article.topic_tags = payload.get("topic_tags", [])
        article.model_outline = payload["model_outline"]
        article.thinking_guide_text = payload["thinking_guide_text"]
        article.images = payload.get("images") or []
        article.images_fetched_at = datetime.now(timezone.utc)
        article.import_item_id = meta_item.id
        db.add(article)
        applied += 1

    for item in items:
        if item.status != "confirmed" or item.kind not in ("vocab_item", "grammar_point"):
            continue
        model = VocabItem if item.kind == "vocab_item" else GrammarPoint
        already = db.execute(select(model).where(model.import_item_id == item.id)).scalar_one_or_none()
        if already is not None:
            continue
        payload = _row_fields(item.kind, item.payload)
        row = model(lesson_id=None, import_item_id=item.id, **payload)
        db.add(row)
        db.flush()
        if item.kind == "vocab_item":
            article.vocab_ids = [*article.vocab_ids, row.id]
        else:
            article.grammar_ids = [*article.grammar_ids, row.id]
        db.add(article)
        applied += 1

    db.flush()
    return {"applied": applied, "editorial_article_id": str(article.id)}


def apply_import_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    if batch.kind == "lesson":
        outcome = apply_lesson_batch(db, batch)
    elif batch.kind == "corpus":
        outcome = apply_corpus_batch(db, batch)
    elif batch.kind == "exam_paper":
        outcome = apply_exam_batch(db, batch)
    elif batch.kind == "editorial_article":
        outcome = apply_editorial_batch(db, batch)
    else:
        outcome = {"applied": 0, "reason": f"confirm not implemented for kind={batch.kind!r}"}

    # Only flip to "confirmed" once something was actually written — a
    # confirm call against a batch nobody has reviewed yet (all items
    # still pending/flagged) should leave status as "awaiting_review"
    # rather than claim confirmed with zero rows written.
    if outcome.get("applied", 0) > 0:
        batch.status = "confirmed"
        db.add(batch)
    db.flush()
    return outcome


async def rollback_import_batch(db: AsyncSession, batch: ImportBatch) -> dict[str, Any]:
    """Lesson batches carry import_item_id lineage on every canonical row
    they create, so rollback can delete-by-lineage even after confirm.
    Corpus batches deliberately do NOT (SRS §5 CORPUS_ITEM has no
    import_item_id — see models.py CorpusItem docstring), so a corpus
    batch can only be rolled back before it's confirmed. No AI call here
    (pure deletes), so — unlike extraction/confirm — this runs directly
    on the FastAPI request's own AsyncSession rather than via Celery.
    """
    if batch.kind == "lesson":
        rows = await db.execute(select(ImportItem.id).where(ImportItem.import_batch_id == batch.id))
        item_ids = [row[0] for row in rows.all()]
        if item_ids:
            for model in (VocabItem, GrammarPoint, Lesson):
                await db.execute(delete(model).where(model.import_item_id.in_(item_ids)))
    elif batch.kind == "corpus" and batch.status == "confirmed":
        raise ValueError(
            "corpus_rollback_unsupported_after_confirm: corpus_item không lưu import_item_id theo SRS §5 — "
            "chỉ có thể huỷ lô câu phim trước khi xác nhận"
        )
    elif batch.kind == "exam_paper":
        # exam_item carries import_item_id directly (delete-by-lineage,
        # same as lesson). exam_passage is verbatim-SDD and has no such
        # column, so its lineage lives on the *ImportItem's* own payload
        # (`_applied_id`, set by apply_exam_batch) instead — read that
        # back to find which real exam_passage rows to remove.
        rows = await db.execute(
            select(ImportItem.id, ImportItem.kind, ImportItem.payload).where(
                ImportItem.import_batch_id == batch.id
            )
        )
        all_rows = rows.all()
        item_ids = [row[0] for row in all_rows]
        passage_ids = [
            uuid.UUID(row[2]["_applied_id"])
            for row in all_rows
            if row[1] == "exam_passage" and row[2].get("_applied_id")
        ]
        if item_ids:
            await db.execute(delete(ExamItem).where(ExamItem.import_item_id.in_(item_ids)))
        if passage_ids:
            await db.execute(delete(ExamPassage).where(ExamPassage.id.in_(passage_ids)))
    elif batch.kind == "editorial_article":
        # vocab_item/grammar_point carry import_item_id directly, same as
        # lesson. EditorialArticle itself is immediate-creation (like
        # Film/ExamPaper) and never gets deleted — only the enrichment
        # this batch's `editorial_meta` item wrote gets reset back to
        # null/empty, and only if THIS batch is the one that applied it
        # (article.import_item_id guard, mirrors apply_editorial_batch's
        # own idempotency check).
        rows = await db.execute(select(ImportItem.id).where(ImportItem.import_batch_id == batch.id))
        item_ids = [row[0] for row in rows.all()]
        if item_ids:
            for model in (VocabItem, GrammarPoint):
                await db.execute(delete(model).where(model.import_item_id.in_(item_ids)))
        if batch.editorial_article_id is not None:
            article = await db.get(EditorialArticle, batch.editorial_article_id)
            if article is not None and article.import_item_id in item_ids:
                article.body_ko = None
                article.level_estimate = None
                article.topic_tags = []
                article.vocab_ids = []
                article.grammar_ids = []
                article.model_outline = None
                article.thinking_guide_text = None
                article.images = []
                article.images_fetched_at = None
                article.import_item_id = None
                db.add(article)

    batch.status = "rolled_back"
    db.add(batch)
    await db.commit()
    return {"status": "rolled_back"}


def find_or_create_film(db: Session, title: str) -> Film:
    title = title.strip()
    existing = db.execute(select(Film).where(Film.title == title)).scalar_one_or_none()
    if existing:
        return existing
    film = Film(title=title)
    db.add(film)
    db.flush()
    return film


def find_or_create_exam_paper(
    db: Session, owner_id: uuid.UUID, file_hash: str, exam_kind: str, session_label: str
) -> ExamPaper:
    """Same immediate-creation timing as find_or_create_film: exam_kind/
    session_label are admin-typed at upload time, not AI-derived, so this
    row is created up front rather than staged through import_item."""
    existing = db.execute(
        select(ExamPaper).where(ExamPaper.owner_id == owner_id, ExamPaper.file_hash == file_hash)
    ).scalar_one_or_none()
    if existing:
        existing.prompt_version = exam_extract.PROMPT_VERSION  # a re-read by the current extraction
        return existing
    paper = ExamPaper(
        exam_kind=exam_kind.strip(),
        session_label=session_label.strip(),
        owner_id=owner_id,
        file_hash=file_hash,
        prompt_version=exam_extract.PROMPT_VERSION,
    )
    db.add(paper)
    db.flush()
    return paper


def find_or_create_editorial_article(
    db: Session,
    source_url: str,
    source_name: str,
    title_ko: str | None = None,
    published_date: datetime | None = None,
) -> EditorialArticle:
    """Same immediate-creation timing as find_or_create_film/
    find_or_create_exam_paper: source_url/source_name (and title_ko/
    published_date, when already known — e.g. from an RSS candidate) are
    admin/candidate-given, not AI-derived, so this row exists before
    extraction runs. body_ko/level_estimate/topic_tags/model_outline stay
    NULL until apply_editorial_batch fills them in at confirm time —
    until then this row is just the stable id that
    import_batch.editorial_article_id points at."""
    source_url = source_url.strip()
    existing = db.execute(
        select(EditorialArticle).where(EditorialArticle.source_url == source_url)
    ).scalar_one_or_none()
    if existing:
        return existing
    article = EditorialArticle(
        source_url=source_url,
        source_name=source_name.strip(),
        title_ko=title_ko.strip() if title_ko else None,
        published_date=published_date,
    )
    db.add(article)
    db.flush()
    return article


# ---------------------------------------------------- RSS auto-discovery ==
# Phase 2 of the editorial proposal doc, implemented now per owner's
# request: a periodic Celery-beat task pulls each registered source's
# official RSS feed and keyword-filters it — NO Gemini call, no
# publishing. Matching entries just become editorial_candidate rows
# (status="new") for an admin to browse in Studio and pick from; only the
# *search* step is automated (SDD: "nguồn staged, con người xác nhận").
EDITORIAL_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "già hóa dân số": ["고령화", "저출산", "인구 감소", "인구절벽", "인구정책"],
    "AI": ["인공지능", "생성형 AI", "챗GPT", " AI "],
    "môi trường": ["환경", "기후변화", "탄소중립", "미세먼지", "온실가스"],
    "giáo dục": ["교육", "입시", "사교육", "학교폭력", "대학수학능력"],
}


def classify_candidate_topics(text: str) -> list[str]:
    """Plain keyword match, no AI — see module note above on what Phase 2
    does and doesn't automate."""
    matched = []
    for tag, keywords in EDITORIAL_TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(tag)
    return matched


def discover_editorial_candidates_for_source(db: Session, source: EditorialSource) -> int:
    """Fetches one EditorialSource's RSS feed, keeps only entries matching
    a topic keyword, and upserts them into editorial_candidate (status=
    "new"). ON CONFLICT DO NOTHING on source_url means re-seeing an entry
    the feed still lists on a later poll is a no-op — in particular, a
    candidate an admin already dismissed or ingested never gets silently
    reset back to "new". Returns how many rows were newly inserted."""
    if not source.rss_url:
        return 0
    parsed = feedparser.parse(source.rss_url)
    inserted = 0
    for entry in parsed.entries:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        if not title or not link:
            continue

        topics = classify_candidate_topics(f"{title}\n{summary}")
        if not topics:
            continue

        published_date = None
        parsed_time = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if parsed_time:
            published_date = datetime(*parsed_time[:6], tzinfo=timezone.utc)

        stmt = (
            pg_insert(EditorialCandidate)
            .values(
                source_name=source.name,
                source_url=link,
                title_ko=title,
                snippet_ko=(summary.strip()[:500] or None),
                topic_tags=topics,
                published_date=published_date,
                status="new",
            )
            .on_conflict_do_nothing(index_elements=["source_url"])
        )
        result = db.execute(stmt)
        if result.rowcount:
            inserted += 1

    db.flush()
    return inserted


def discover_editorial_candidates(db: Session) -> dict[str, Any]:
    """Runs discovery across every active registered source. One bad feed
    (network error, malformed XML) is recorded in `errors` and does not
    sink the rest of the run.

    IMPORTANT: this is the one place in the whole ingestion module that
    owns its own commit. The caller (app.workers.tasks.
    discover_editorial_candidates) just does `with Session(_sync_engine)
    as db: return discover_editorial_candidates(db)` — no db.commit()
    anywhere in that path, unlike every other task in tasks.py, which all
    commit explicitly before their `with` block exits. Session.__exit__
    only closes the session; it does NOT commit a pending transaction, so
    every row this used to insert (only ever flushed, never committed) was
    silently rolled back the moment the task function returned. That
    produced exactly the two symptoms this was written to fix: the
    candidate queue stayed empty no matter how many times the scan ran
    (nothing was ever actually persisted), AND re-running the scan kept
    "inserting" the identical entries every time (ON CONFLICT DO NOTHING
    on source_url never found a conflict, because the rows it should have
    conflicted with had already been rolled back out of the database).
    """
    sources = db.execute(select(EditorialSource).where(EditorialSource.active.is_(True))).scalars().all()
    per_source: dict[str, int] = {}
    errors: dict[str, str] = {}
    total = 0
    for source in sources:
        try:
            n = discover_editorial_candidates_for_source(db, source)
        except Exception as exc:  # noqa: BLE001 — one feed's failure isn't fatal to the run
            errors[source.name] = str(exc)[:200]
            continue
        per_source[source.name] = n
        total += n
    db.commit()
    return {"total_inserted": total, "per_source": per_source, "errors": errors}


# ==================================================== consolidated podcast ==
# Owner feedback: per-word TTS ("Nghe" buttons on vocab_item/corpus_item)
# misses the point — studying should be "nghe và nhại lại" (listen and
# shadow) from ONE consolidated lecture that actually teaches each word/
# grammar point in context, not flip through isolated flashcards read aloud
# ("như hiện tại thì tôi dùng quizlet cho nhanh chứ build app mới làm gì").
# This generates that lecture script; app.workers.tasks.generate_content_podcast
# then feeds it into the existing tts.synthesize_korean_tts and stores both
# script + audio in audio.lecture_audio (cache_key prefixed "podcast:...").
PODCAST_SCHEMA = podcast_script.PODCAST_SCHEMA
PODCAST_VERSION = podcast_script.PODCAST_VERSION
# Writing the lecture is creative work a flash-lite model does thinly; set
# GEMINI_MODEL_PODCAST on Railway to use a stronger model without a deploy.
_PODCAST_ATTEMPTS = 2


def build_podcast_prompt(
    title: str,
    vocab: list[podcast_script.Vocab],
    grammar: list[podcast_script.Grammar],
    *,
    retry_note: str | None = None,
) -> str:
    """vocab: (hangul, pos, meaning_vi, example_ko, hanja, sino_vietnamese).
    grammar: (pattern, meaning_vi, example_ko, usage_context_vi, topik_tip_vi).
    The prompt and the checks around it live in app/services/podcast_script.py."""
    return podcast_script.build_prompt(title, vocab, grammar, retry_note=retry_note)


def generate_podcast_script(
    title: str,
    vocab: list[podcast_script.Vocab],
    grammar: list[podcast_script.Grammar],
) -> str:
    """Asks Gemini for the lecture, cleans it for the voice and checks it before
    any text-to-speech quota is spent. A script that is empty, far too long or
    that reads out its own instructions is asked for once more with a note about
    what was wrong; after that the cleaned script is used as it is (unless it
    is empty) — a slightly long lecture is better than no lecture."""
    model = settings.GEMINI_MODEL_PODCAST or settings.GEMINI_MODEL_LESSON_INGEST
    note: str | None = None
    script = ""
    for attempt in range(1, _PODCAST_ATTEMPTS + 1):
        result = gemini_client.generate_structured(
            model=model,
            prompt=build_podcast_prompt(title, vocab, grammar, retry_note=note),
            response_schema=PODCAST_SCHEMA,
            prompt_version=PODCAST_VERSION,
        )
        script = podcast_script.clean_script_for_tts(str(_parse_json(result["text"]).get("script") or ""))
        problems = podcast_script.script_problems(script)
        if not problems:
            return script
        print(f"[podcast] attempt {attempt}: script problems={problems} chars={len(script)}", flush=True)
        note = podcast_script.retry_note(problems) or None
    if not script:
        raise RuntimeError("Gemini không viết được kịch bản bài giảng (kịch bản rỗng).")
    return script


# ============================================== editorial outline feedback ==
# Owner feedback: saving a "luyện dàn ý" attempt never actually got
# analyzed — only a static, ingestion-time model_outline was ever shown.
# This is the real per-submission Gemini call app.workers.tasks.
# grade_editorial_outline runs after every save.
OUTLINE_FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "feedback_vi": {
            "type": "STRING",
            "description": "Nhận xét chi tiết bằng tiếng Việt cho dàn ý của học viên",
        },
    },
    "required": ["feedback_vi"],
}


def build_outline_feedback_prompt(
    article_title: str,
    article_body: str,
    reference_outline: dict[str, str],
    learner_outline: dict[str, str | None],
) -> str:
    learner_lines = "\n".join(
        f"- {label}: {learner_outline.get(key) or '(chưa viết)'}"
        for key, label in (
            ("phenomenon_text", "Hiện tượng (현상)"),
            ("cause_text", "Nguyên nhân (원인)"),
            ("consequence_text", "Kết quả/ảnh hưởng (결과)"),
            ("solution_text", "Giải pháp/kiến nghị (해결 방안)"),
        )
    )
    return f"""Bạn là giáo viên chấm bài luyện viết TOPIK II câu 54 (dàn ý theo cấu trúc
hiện tượng - nguyên nhân - kết quả - giải pháp) cho người Việt học tiếng Hàn.

Bài xã luận học viên đang luyện đọc, tựa đề: {article_title}
Trích bài xã luận (để bạn hiểu ngữ cảnh, không cần nhắc lại nguyên văn):
{article_body[:3000]}

Dàn ý THAM KHẢO (chỉ để bạn đối chiếu — KHÔNG chép lại nguyên văn cho học
viên, học viên có thể chưa xem dàn ý này):
- Hiện tượng: {reference_outline.get("phenomenon", "")}
- Nguyên nhân: {reference_outline.get("cause", "")}
- Kết quả: {reference_outline.get("consequence", "")}
- Giải pháp: {reference_outline.get("solution", "")}

Dàn ý CỦA HỌC VIÊN (tiếng Hàn, có thể còn sơ sài hoặc có lỗi):
{learner_lines}

Viết nhận xét bằng tiếng Việt (feedback_vi), khoảng 4-8 câu, thẳng thắn
nhưng khích lệ:
1. Học viên đã bám đúng cấu trúc 4 phần chưa, ý tưởng có hợp lý/liên quan
   bài xã luận không.
2. Chỉ ra 1-2 lỗi ngữ pháp/từ vựng/chính tả tiếng Hàn CỤ THỂ nếu có (trích
   nguyên văn phần sai, kèm cách sửa).
3. Gợi ý CỤ THỂ cách phát triển/làm rõ ý còn thiếu hoặc còn chung chung
   (không chỉ nói "cần chi tiết hơn" mà nói rõ chi tiết hơn LÀ GÌ).
4. Nếu học viên chưa viết phần nào, nhắc nhở nhẹ nhàng, không chê bai.

Trả về đúng JSON schema đã cho, không thêm giải thích."""


def generate_outline_feedback(
    article_title: str,
    article_body: str,
    reference_outline: dict[str, str],
    learner_outline: dict[str, str | None],
) -> str:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=build_outline_feedback_prompt(article_title, article_body, reference_outline, learner_outline),
        response_schema=OUTLINE_FEEDBACK_SCHEMA,
        prompt_version="outline-feedback-v1",
    )
    return _parse_json(result["text"])["feedback_vi"]
