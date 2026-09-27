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
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    CorpusItem,
    Film,
    GrammarPoint,
    ImportBatch,
    ImportItem,
    Lesson,
    LessonTopic,
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


def apply_import_batch(db: Session, batch: ImportBatch) -> dict[str, Any]:
    if batch.kind == "lesson":
        outcome = apply_lesson_batch(db, batch)
    elif batch.kind == "corpus":
        outcome = apply_corpus_batch(db, batch)
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
