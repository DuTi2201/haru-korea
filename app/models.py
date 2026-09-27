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
class CorpusItem(Base):
    __tablename__ = "corpus_item"
    __table_args__ = {"schema": "corpus"}

    id: Mapped[uuid.UUID] = _uuid_pk()
    film_id: Mapped[str] = mapped_column(String(64))
    text_ko: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32))
    level: Mapped[str] = mapped_column(String(16))
    register: Mapped[str] = mapped_column(String(32))
    topic_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
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


# ----------------------------------------------------------------- practice --
class VocabItem(Base):
    __tablename__ = "vocab_item"

    id: Mapped[uuid.UUID] = _uuid_pk()
    learner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    term_ko: Mapped[str] = mapped_column(String(120))
    meaning_vi: Mapped[str] = mapped_column(String(255))
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    level: Mapped[str] = mapped_column(String(16))


class ItemState(Base):
    """Spaced-repetition state (SM-2-style) per vocab item per learner."""

    __tablename__ = "item_state"

    id: Mapped[uuid.UUID] = _uuid_pk()
    learner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    vocab_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("vocab_item.id"))
    srs_stage: Mapped[int] = mapped_column(Integer, default=0)
    next_review_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ease: Mapped[float] = mapped_column(Float, default=2.5)

    __table_args__ = (UniqueConstraint("learner_id", "vocab_item_id", name="uq_item_state_learner_item"),)


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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
