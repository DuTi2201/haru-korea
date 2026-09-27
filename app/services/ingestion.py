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
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    CorpusItem,
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
from app.services import gemini_client
from app.services.subtitles import Cue, chunk_cues, parse_subtitles

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
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["pattern", "meaning_vi", "level", "confidence"],
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

# Exam-paper extraction: one multimodal call per uploaded paper (bounded
# size — an exam paper is a fixed handful of pages, not open-ended like a
# film script, so FR-19/Gate G6's chunking constraint doesn't apply here).
# Passages carry a `local_ref` the model invents (e.g. "P1") purely so it
# can point items at the passage they belong to within the SAME response;
# apply_exam_batch resolves local_ref -> real content.exam_passage.id.
EXAM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "passages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "local_ref": {"type": "STRING", "description": "id tạm, vd 'P1', để items tham chiếu tới"},
                    "kind": {"type": "STRING", "enum": ["đọc hiểu", "nghe", "biểu đồ"]},
                    "body_ko": {"type": "STRING", "nullable": True, "description": "toàn văn đoạn văn/kịch bản nghe"},
                    "source_page": {"type": "INTEGER"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["local_ref", "kind", "source_page", "confidence"],
            },
        },
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "number": {"type": "INTEGER", "description": "số thứ tự câu hỏi trong đề"},
                    "passage_ref": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "local_ref của đoạn văn/bài nghe liên quan, để trống nếu câu hỏi độc lập",
                    },
                    "qtype_code": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "CHỈ chọn từ danh sách mã loại câu hỏi đã cho, để trống nếu không khớp mã nào",
                    },
                    "stem_ko": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "các lựa chọn, thường 4"},
                    "answer": {
                        "type": "INTEGER",
                        "nullable": True,
                        "description": "số thứ tự đáp án đúng (1-based), để trống nếu không xác định được",
                    },
                    "answer_from_key": {
                        "type": "BOOLEAN",
                        "description": "true nếu đáp án đọc được từ bảng đáp án in trong tài liệu, false nếu là suy đoán của mô hình",
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["number", "stem_ko", "options", "answer_from_key", "confidence"],
            },
        },
    },
    "required": ["passages", "items"],
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


class ExamPassageExtraction(BaseModel):
    local_ref: str
    kind: Literal["đọc hiểu", "nghe", "biểu đồ"]
    body_ko: str | None = None
    source_page: int = 1
    confidence: float = Field(ge=0, le=1, default=0.5)


class ExamItemExtraction(BaseModel):
    number: int
    passage_ref: str | None = None
    qtype_code: str | None = None
    stem_ko: str
    options: list[str] = Field(default_factory=list)
    answer: int | None = Field(default=None, ge=1, le=5)
    answer_from_key: bool = False
    confidence: float = Field(ge=0, le=1, default=0.5)


class ExamExtraction(BaseModel):
    passages: list[ExamPassageExtraction] = Field(default_factory=list)
    items: list[ExamItemExtraction] = Field(default_factory=list)


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
  (ví dụ: "V + -(으)ㄹ 뿐만 아니라"), kèm nghĩa tiếng Việt và câu ví dụ.
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
- topics: chọn từ danh sách chủ đề đã có nếu phù hợp ({topics_hint}); có thể
  đề xuất chủ đề mới bằng tên tiếng Việt ngắn gọn nếu không có chủ đề nào khớp.
- grammar_patterns: CHỈ chọn từ danh sách mẫu ngữ pháp đã có trong hệ thống
  dưới đây nếu câu thực sự minh hoạ mẫu đó; để trống nếu không khớp mẫu nào
  (không tự tạo mẫu ngữ pháp mới ở bước này):
{grammar_hint}

Trả về đúng JSON schema đã cho (mảng "items", mỗi phần tử echo lại đúng
source_ref của câu tương ứng), không thêm giải thích.

Danh sách câu thoại:
{cue_lines}"""


def build_exam_prompt(known_qtypes: list[tuple[str, str]]) -> str:
    qtype_hint = "\n".join(f"- {code}: {name_vi}" for code, name_vi in known_qtypes) or (
        "(chưa có loại câu hỏi nào trong hệ thống — để qtype_code trống cho mọi câu)"
    )
    return f"""Bạn là biên tập viên đề thi TOPIK cho người Việt học tiếng Hàn.
Đọc ảnh/tài liệu đề thi được đính kèm (có thể nhiều trang) và trích xuất:

1) Các đoạn văn/bài nghe (passages): mỗi đoạn đọc hiểu, kịch bản nghe, hoặc
   biểu đồ/quảng cáo dùng chung cho một hoặc nhiều câu hỏi. Đặt cho mỗi đoạn
   một local_ref ngắn tự chọn (vd "P1", "P2") để các câu hỏi tham chiếu tới.
2) Từng câu hỏi (items): số thứ tự, đoạn văn liên quan (passage_ref, để
   trống nếu câu hỏi độc lập không cần đoạn văn), đề bài, các lựa chọn.

Với qtype_code, CHỈ chọn từ danh sách mã đã có trong hệ thống dưới đây nếu
đúng khớp; để trống nếu không có mã nào khớp (không tự đặt mã mới):
{qtype_hint}

Với đáp án: nếu tài liệu có in kèm bảng đáp án (answer key), đọc chính xác
đáp án cho từng câu và đánh dấu answer_from_key=true. Nếu KHÔNG có bảng đáp
án trong tài liệu, có thể tự suy luận đáp án khả dĩ nhất (answer_from_key=
false) hoặc để answer trống nếu không đủ căn cứ — không suy đoán bừa.

Trả về đúng JSON schema đã cho, không thêm giải thích. Tự đánh giá độ tự tin
(confidence, 0-1) cho từng đoạn văn và từng câu hỏi."""


def build_exam_prompt_parts(file_bytes: bytes, mime_type: str, known_qtypes: list[tuple[str, str]]) -> list[Any]:
    return [gemini_client.part_from_bytes(file_bytes, mime_type), build_exam_prompt(known_qtypes)]


def extract_exam_paper(file_bytes: bytes, mime_type: str, known_qtypes: list[tuple[str, str]]) -> ExamExtraction:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=build_exam_prompt_parts(file_bytes, mime_type, known_qtypes),
        response_schema=EXAM_SCHEMA,
        prompt_version="exam-v1",
    )
    return ExamExtraction.model_validate(_parse_json(result["text"]))


def run_exam_extraction(db: Session, batch: ImportBatch, file_bytes: bytes, mime_type: str) -> tuple[int, int]:
    """Stages one ImportItem per extracted passage (kind="exam_passage")
    and per extracted question (kind="exam_item"). A question whose answer
    is the model's own guess (no printed answer key found) is ALWAYS
    flagged for review regardless of confidence — getting an exam answer
    wrong is worse than a wrong vocab gloss, so it never slips through on
    a high self-reported confidence alone (same force_flag idea as
    corpus's is_crude)."""
    known_qtypes = db.execute(
        select(QuestionType.code, QuestionType.name_vi).where(QuestionType.active.is_(True))
    ).all()
    extraction = extract_exam_paper(file_bytes, mime_type, known_qtypes)
    staged = 0
    flagged = 0

    for p in extraction.passages:
        status = confidence_status(p.confidence)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="exam_passage",
                status=status,
                confidence=p.confidence,
                payload={"local_ref": p.local_ref, "kind": p.kind, "body_ko": p.body_ko, "source_page": p.source_page},
            )
        )
        staged += 1

    for it in extraction.items:
        needs_human_answer_check = it.answer is not None and not it.answer_from_key
        status = confidence_status(it.confidence, force_flag=needs_human_answer_check)
        if status != "pending":
            flagged += 1
        db.add(
            ImportItem(
                import_batch_id=batch.id,
                kind="exam_item",
                status=status,
                confidence=it.confidence,
                payload={
                    "number": it.number,
                    "passage_ref": it.passage_ref,
                    "qtype_code": it.qtype_code,
                    "stem_ko": it.stem_ko,
                    "options": it.options,
                    "answer": it.answer,
                    "answer_from_key": it.answer_from_key,
                },
            )
        )
        staged += 1

    db.flush()
    return staged, flagged


def extract_lesson(file_bytes: bytes, mime_type: str) -> LessonExtraction:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=build_lesson_prompt_parts(file_bytes, mime_type),
        response_schema=LESSON_SCHEMA,
        prompt_version="lesson-v1",
    )
    return LessonExtraction.model_validate(_parse_json(result["text"]))


def classify_corpus_chunk(
    cues: list[Cue], known_topics: list[str], known_grammar_patterns: list[str]
) -> CorpusChunkClassification:
    result = gemini_client.generate_structured(
        model=settings.GEMINI_MODEL_CORPUS_INGEST,
        prompt=build_corpus_chunk_prompt(cues, known_topics, known_grammar_patterns),
        response_schema=CORPUS_CHUNK_SCHEMA,
        prompt_version="corpus-v1",
    )
    return CorpusChunkClassification.model_validate(_parse_json(result["text"]))


def run_corpus_extraction(db: Session, batch: ImportBatch, raw_subtitle_text: str) -> tuple[int, int]:
    """Parses the subtitle file, classifies it in fixed-size chunks (so
    each Gemini call's prompt stays constant-size regardless of film
    length — FR-19/Gate G6), and stages one ImportItem per kept cue.
    Returns (staged_count, flagged_count)."""
    cues = parse_subtitles(raw_subtitle_text)
    known_topics = [t for (t,) in db.execute(select(Topic.name)).all()]
    known_grammar = [p for (p,) in db.execute(select(GrammarPoint.pattern)).all()][:200]

    staged = 0
    flagged = 0
    for chunk in chunk_cues(cues, settings.CORPUS_CHUNK_SIZE):
        by_ref = {c.source_ref: c for c in chunk}
        try:
            classification = classify_corpus_chunk(chunk, known_topics, known_grammar)
        except (ValidationError, json.JSONDecodeError, KeyError):
            # One bad chunk shouldn't sink the whole film — skip it; the
            # reviewer will simply see fewer candidates from this stretch.
            continue

        for entry in classification.items:
            cue = by_ref.get(entry.source_ref)
            if cue is None or not entry.keep:
                continue
            status = confidence_status(entry.confidence, force_flag=entry.is_crude)
            if status != "pending":
                flagged += 1
            db.add(
                ImportItem(
                    import_batch_id=batch.id,
                    kind="corpus_item",
                    status=status,
                    confidence=entry.confidence,
                    payload={
                        "text_ko": cue.text,  # always the ORIGINAL subtitle line, never Gemini's paraphrase
                        "source_ref": cue.source_ref,
                        "kind": entry.kind,
                        "level": entry.level,
                        "register": entry.register,
                        "topics": entry.topics,
                        "grammar_patterns": entry.grammar_patterns,
                        "is_crude": entry.is_crude,
                    },
                )
            )
            staged += 1
    db.flush()
    return staged, flagged


def run_lesson_extraction(db: Session, batch: ImportBatch, file_bytes: bytes, mime_type: str) -> tuple[int, int]:
    extraction = extract_lesson(file_bytes, mime_type)
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


# ================================================================ confirm ==
_VOCAB_FIELDS = {"hangul", "pos", "meaning_vi", "definition_ko", "level", "hanja", "sino_vietnamese", "example_ko"}
_GRAMMAR_FIELDS = {"pattern", "meaning_vi", "level", "example_ko"}


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
        allowed_fields = _VOCAB_FIELDS if item.kind == "vocab_item" else _GRAMMAR_FIELDS
        already = db.execute(select(model).where(model.import_item_id == item.id)).scalar_one_or_none()
        if already is not None:
            continue
        payload = {k: v for k, v in item.payload.items() if k in allowed_fields}
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

    applied = 0
    for item in items:
        payload = item.payload
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
            )
            .on_conflict_do_nothing(index_elements=["film_id", "source_ref"])
        )
        result = db.execute(stmt)
        if result.rowcount:
            applied += 1

    db.flush()
    return {"applied": applied}


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
        passage = ExamPassage(
            paper_id=batch.exam_paper_id,
            kind=p.payload["kind"],
            body_ko=p.payload.get("body_ko"),
            chart_data=None,
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
                stem_ko=it.payload["stem_ko"],
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


def apply_import_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    if batch.kind == "lesson":
        outcome = apply_lesson_batch(db, batch)
    elif batch.kind == "corpus":
        outcome = apply_corpus_batch(db, batch)
    elif batch.kind == "exam_paper":
        outcome = apply_exam_batch(db, batch)
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
        return existing
    paper = ExamPaper(
        exam_kind=exam_kind.strip(),
        session_label=session_label.strip(),
        owner_id=owner_id,
        file_hash=file_hash,
        prompt_version="exam-v1",
    )
    db.add(paper)
    db.flush()
    return paper
