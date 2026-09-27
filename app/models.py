"""SQLAlchemy models — one module, sectioned to match the SDD's module
boundaries (identity / content / audio / corpus / analytics / practice /
writing / ingest). Kept in one file for a first scaffold; split into
app/models/<module>.py once a module's table count grows.

Field names/types follow the SDD verbatim where it gave CREATE TABLE
snippets (audio.lecture_audio, corpus.corpus_item, analytics.learning_event,
content.exam_paper/question_type/exam_passage/exam_item/exam_item_tag).
Tables the SDD only indexed without a full column list (vocab_item,
item_state, import_item, writing_submission, error_log, writing_score,
jobs, profiles) are reasonable extensions inferred from context — revisit
once the corresponding module is actually implemented.
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid_pk():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# ---------------------------------------------------------------- identity --
class Profile(Base):
    """Learner/editor/admin profile. `id` mirrors the auth subject id."""

    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(
        Enum("learner", "editor", "admin", name="profile_role"), default="learner"
    )
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    goal: Mapped[str | None] = mapped_column(
        Enum("talk", "topik1", "topik4", "topik5", "topik6", name="profile_goal"),
        nullable=True,
    )
    exam_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    daily_minutes: Mapped[int] = mapped_column(Integer, default=15)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --------------------------------------------------------------------- jobs --
class Job(Base):
    """Replaces Celery+SSE-over-Redis fan-out with a durable row: worker
    updates status/progress here AND publishes the same payload on the
    `job:{id}:events` Redis channel for live streaming (see
    app/services/job_events.py). Poll this table if you miss the stream.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    type: Mapped[str] = mapped_column(String(64), index=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        Enum("queued", "running", "succeeded", "failed", "cancelled", name="job_status"),
        default="queued",
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    step: Mapped[str | None] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("type", "idempotency_key", name="uq_job_type_idempotency"),
    )


class AppConfig(Base):
    """Versioned config: rubric, gate thresholds, model names, prompt
    versions — "cấu hình thay cho hard-code" (SDD §5, principle 6)."""

    __tablename__ = "app_config"

    name: Mapped[str] = mapped_column(String(120), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    value: Mapped[dict] = mapped_column(JSONB)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ------------------------------------------------------------------- audio --
class LectureAudio(Base):
    __tablename__ = "lecture_audio"
    __table_args__ = {"schema": "audio"}

    id: Mapped[uuid.UUID] = _uuid_pk()
    cache_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    opus_path: Mapped[str] = mapped_column(String(500))
    aac_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(32))
    voice: Mapped[str] = mapped_column(String(64))
    duration_sec: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ------------------------------------------------------------------ corpus --
class Film(Base):
    """SRS §5 FILM — the registry a subtitle upload is filed under. Named
    by the admin at upload time (not AI-derived), so unlike everything
    else this module writes, a row here is created immediately rather
    than staged through import_item/review (SDD's "khu chờ duyệt" rule
    protects against hallucinated *content*, not an admin-typed title).
    """

    __tablename__ = "film"
    __table_args__ = {"schema": "corpus"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), unique=True)


class CorpusItem(Base):
    """SRS §5 CORPUS_ITEM. `film_id`/`topic_ids` stay bare ids with no
    physical FK to `content` (cross-schema — SDD principle: modules never
    JOIN/FK across schema boundaries); `film_id` DOES get a real FK since
    both tables live in `corpus`. `grammar_point_ids` is the
    `CORPUS_ITEM }o--o{ GRAMMAR_POINT : dùng` relationship from the SRS
    ER diagram — present in the diagram but not in the SDD's DDL snippet,
    so it's added here the same bare-array way as `topic_ids`.
    Deliberately has NO `import_item_id` (SRS's "Nguồn dữ liệu" convention
    only lists it for từ/ngữ pháp/bài đọc, not câu) — see ingest.py's
    corpus rollback note for what that means for undo.
    """

    __tablename__ = "corpus_item"
    __table_args__ = (
        UniqueConstraint("film_id", "source_ref", name="uq_corpus_item_film_source_ref"),
        {"schema": "corpus"},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    film_id: Mapped[int] = mapped_column(ForeignKey("corpus.film.id", ondelete="CASCADE"), index=True)
    text_ko: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Enum("câu", "cụm từ", "mẫu ngữ pháp", name="corpus_item_kind"))
    level: Mapped[int] = mapped_column(SmallInteger)
    register: Mapped[str] = mapped_column(Enum("존댓말", "반말", "hỗn hợp", name="corpus_item_register"))
    source_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    topic_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    grammar_point_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    embedding = mapped_column(Vector(768), nullable=True)


# --------------------------------------------------------------- analytics --
class LearningEvent(Base):
    """Partitioned by occurred_at range per SDD. SQLAlchemy declares the
    parent table's shape; the actual `PARTITION BY RANGE (occurred_at)`
    DDL + monthly partitions live in the Alembic migration (raw SQL),
    not here — the ORM layer doesn't need to know about partitions.
    """

    __tablename__ = "learning_event"
    __table_args__ = {"schema": "analytics"}

    learner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id"), primary_key=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(64))
    skill: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


# ------------------------------------------------------------------ content --
class ExamPaper(Base):
    __tablename__ = "exam_paper"
    __table_args__ = {"schema": "content"}

    id: Mapped[uuid.UUID] = _uuid_pk()
    exam_kind: Mapped[str] = mapped_column(String(32))
    session_label: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    file_hash: Mapped[str] = mapped_column(String(64))
    answer_status: Mapped[str] = mapped_column(String(32), default="pending")
    import_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(32))

    __table_args__ = (
        UniqueConstraint("owner_id", "file_hash", name="uq_exam_paper_owner_hash"),
        {"schema": "content"},
    )


class QuestionType(Base):
    __tablename__ = "question_type"
    __table_args__ = {"schema": "content"}

    id: Mapped[uuid.UUID] = _uuid_pk()
    skill: Mapped[str] = mapped_column(String(32))
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name_ko: Mapped[str] = mapped_column(String(120))
    name_vi: Mapped[str] = mapped_column(String(120))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ExamPassage(Base):
    __tablename__ = "exam_passage"
    __table_args__ = {"schema": "content"}

    id: Mapped[uuid.UUID] = _uuid_pk()
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("content.exam_paper.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(32))
    body_ko: Mapped[str | None] = mapped_column(Text, nullable=True)
    chart_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    image_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_page: Mapped[int] = mapped_column(Integer)
    source_bbox: Mapped[dict] = mapped_column(JSONB)
    embedding = mapped_column(Vector(768), nullable=True)


class ExamItem(Base):
    __tablename__ = "exam_item"
    __table_args__ = (
        UniqueConstraint("paper_id", "number", name="uq_exam_item_paper_number"),
        {"schema": "content"},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("content.exam_paper.id", ondelete="CASCADE")
    )
    passage_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("content.exam_passage.id"), nullable=True
    )
    number: Mapped[int] = mapped_column(Integer)
    qtype_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("content.question_type.id"))
    stem_ko: Mapped[str] = mapped_column(Text)
    options: Mapped[dict] = mapped_column(JSONB)
    answer: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    answer_source: Mapped[str | None] = mapped_column(
        Enum("editor", "ai_guess", name="exam_item_answer_source"), nullable=True
    )
    difficulty_est: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    import_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    embedding = mapped_column(Vector(768), nullable=True)


class ExamItemTag(Base):
    __tablename__ = "exam_item_tag"
    __table_args__ = {"schema": "content"}

    item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("content.exam_item.id", ondelete="CASCADE"), primary_key=True
    )
    tag_kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    ref_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32))


# --------------------------------------------------------- content: lessons --
# SRS §5: LESSON, TOPIC, VOCAB_ITEM, GRAMMAR_POINT — all lesson-scoped,
# all in `content` (same schema as the exam_* tables above), field lists
# transcribed verbatim from the SRS ER diagram (pages 12-14).
class Topic(Base):
    __tablename__ = "topic"
    __table_args__ = {"schema": "content"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    quizlet_url: Mapped[str | None] = mapped_column(String(500), nullable=True)


class Lesson(Base):
    __tablename__ = "lesson"
    __table_args__ = {"schema": "content"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255))
    level: Mapped[int] = mapped_column(SmallInteger)  # 1-6, TOPIK-tương ứng, ước lượng — not an exam score
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    quizlet_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    import_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LessonTopic(Base):
    """LESSON }o--o{ TOPIC : thuộc — real FK join table (both sides are
    same-schema `content`, unlike corpus_item's bare topic_ids array,
    which crosses into `corpus`)."""

    __tablename__ = "lesson_topic"
    __table_args__ = {"schema": "content"}

    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("content.lesson.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("content.topic.id", ondelete="CASCADE"), primary_key=True
    )


class VocabItem(Base):
    __tablename__ = "vocab_item"
    __table_args__ = {"schema": "content"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("content.lesson.id", ondelete="CASCADE"), index=True)
    hangul: Mapped[str] = mapped_column(String(120))
    pos: Mapped[str | None] = mapped_column(String(32), nullable=True)
    meaning_vi: Mapped[str] = mapped_column(String(255))
    definition_ko: Mapped[str | None] = mapped_column(Text, nullable=True)
    level: Mapped[int] = mapped_column(SmallInteger)
    hanja: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sino_vietnamese: Mapped[str | None] = mapped_column(String(120), nullable=True)
    example_ko: Mapped[str | None] = mapped_column(Text, nullable=True)
    import_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


class GrammarPoint(Base):
    __tablename__ = "grammar_point"
    __table_args__ = {"schema": "content"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("content.lesson.id", ondelete="CASCADE"), index=True)
    pattern: Mapped[str] = mapped_column(String(255))  # V/A + hình thái, vd "V + -(으)ㄹ 뿐만 아니라"
    meaning_vi: Mapped[str] = mapped_column(String(255))
    level: Mapped[int] = mapped_column(SmallInteger)
    example_ko: Mapped[str | None] = mapped_column(Text, nullable=True)
    import_item_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


# ----------------------------------------------------------------- practice --
class ItemState(Base):
    """SRS §5 ITEM_STATE — a single decay-style `strength` per learner
    per (vocab_item|grammar_point), nudged by quick in-app checks; NOT a
    full SM-2 state machine (no srs_stage/ease/next_review_at — those
    were this scaffold's own pre-SRS guess, dropped now that the spec is
    in hand). `item_id` is a bare id (polymorphic across two `content`
    tables), so — same as corpus_item.topic_ids — no physical FK.
    """

    __tablename__ = "item_state"

    id: Mapped[uuid.UUID] = _uuid_pk()
    learner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), index=True)
    item_type: Mapped[str] = mapped_column(Enum("vocab_item", "grammar_point", name="item_state_type"))
    item_id: Mapped[int] = mapped_column(Integer)
    strength: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("learner_id", "item_type", "item_id", name="uq_item_state_learner_item"),
    )


class ErrorLog(Base):
    __tablename__ = "error_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    learner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    skill: Mapped[str] = mapped_column(String(32))
    error_type: Mapped[str] = mapped_column(String(64))
    example_ko: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ------------------------------------------------------------------ writing --
class WritingSubmission(Base):
    """State machine (SDD §11):
    uploaded ↔ image_rejected → transcribing → awaiting_confirmation →
    grading → graded → revising → revised → rewriting → grading (loop)
    """

    __tablename__ = "writing_submission"

    id: Mapped[uuid.UUID] = _uuid_pk()
    learner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    prompt_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum(
            "uploaded",
            "image_rejected",
            "transcribing",
            "awaiting_confirmation",
            "grading",
            "graded",
            "revising",
            "revised",
            "rewriting",
            name="writing_submission_status",
        ),
        default="uploaded",
    )
    image_key: Mapped[str] = mapped_column(String(500))
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WritingScore(Base):
    __tablename__ = "writing_score"

    id: Mapped[uuid.UUID] = _uuid_pk()
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("writing_submission.id", ondelete="CASCADE")
    )
    criteria: Mapped[dict] = mapped_column(JSONB)
    score_min: Mapped[int] = mapped_column(Integer)
    score_max: Mapped[int] = mapped_column(Integer)
    priority_fixes: Mapped[dict] = mapped_column(JSONB, default=list)

    __table_args__ = (CheckConstraint("score_min <= score_max", name="ck_writing_score_range"),)


# ------------------------------------------------------------------- ingest --
class ImportBatch(Base):
    """queued → extracting → validating → awaiting_review →
    confirmed|cancelled → rolled_back (extracting/validating can → failed)
    """

    __tablename__ = "import_batch"

    id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(Enum("lesson", "corpus", "exam_paper", name="import_kind"))
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    status: Mapped[str] = mapped_column(
        Enum(
            "queued",
            "extracting",
            "validating",
            "awaiting_review",
            "confirmed",
            "cancelled",
            "rolled_back",
            "failed",
            name="import_batch_status",
        ),
        default="queued",
    )
    flagged_count: Mapped[int] = mapped_column(Integer, default=0)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Bare id, no FK — only meaningful for kind="corpus"; crosses into
    # `corpus` schema, same bare-id convention as corpus_item.topic_ids.
    film_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Bare id, no FK — only meaningful for kind="exam_paper"; crosses into
    # `content` schema. Set inside extract_exam_paper_import (same
    # find-or-create-immediately timing as film_id, since exam_kind/
    # session_label are admin-typed, not AI-derived).
    exam_paper_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("kind", "file_hash", name="uq_import_batch_kind_file_hash"),
    )


class ImportItem(Base):
    __tablename__ = "import_item"

    id: Mapped[uuid.UUID] = _uuid_pk()
    import_batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("import_batch.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(
        Enum("pending", "flagged_yellow", "flagged_red", "confirmed", "rejected", name="import_item_status"),
        default="pending",
    )
    payload: Mapped[dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
