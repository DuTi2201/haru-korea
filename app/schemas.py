"""Pydantic schemas — the contract the Lovable client's TypeScript types
must mirror exactly (see initial Lovable message: "Khai báo TypeScript
types đồng bộ chính xác với Pydantic schemas của backend FastAPI").
"""
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
Role = Literal["learner", "editor", "admin"]


# --------------------------------------------------------------------- auth --
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class ProfileOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    role: Role
    display_name: str | None
    goal: str | None
    exam_date: datetime | None
    daily_minutes: int

    model_config = {"from_attributes": True}


# --------------------------------------------------------------------- jobs --
class JobOut(BaseModel):
    id: uuid.UUID
    type: str
    status: JobStatus
    progress: float
    step: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobAccepted(BaseModel):
    """The SDD's "202 Accepted + Job ID" envelope."""

    job_id: uuid.UUID
    status: JobStatus = "queued"
    poll_url: str
    events_url: str


class ProblemDetail(BaseModel):
    """RFC 9457 problem+json, as SDD §7 requires for all error responses."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    code: str | None = None
    retryable: bool | None = None


# ------------------------------------------------------------------- audio --
class LectureAudioRequest(BaseModel):
    lesson_id: str
    text_ko: str
    voice: str = "ko-female-1"
    prompt_version: str = "v1"


# ----------------------------------------------------------------- writing --
class WritingSubmissionOut(BaseModel):
    id: uuid.UUID
    prompt_id: str
    status: str
    transcript: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class WritingConfirmTranscriptRequest(BaseModel):
    transcript: str


class WritingScoreOut(BaseModel):
    submission_id: uuid.UUID
    criteria: dict[str, Any]
    score_min: int
    score_max: int
    priority_fixes: list[dict[str, Any]]

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------- ingest --
class ImportBatchOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    flagged_count: int
    source_file: str | None
    film_id: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ImportBatchAccepted(BaseModel):
    """POST /imports's 202 envelope: same job_id/poll_url/events_url shape
    as JobAccepted, plus the batch id — the caller needs it immediately
    (to list/patch/confirm items) and can't wait for the job to finish."""

    import_batch_id: uuid.UUID
    job_id: uuid.UUID
    status: JobStatus = "queued"
    poll_url: str
    events_url: str


class ImportItemOut(BaseModel):
    id: uuid.UUID
    import_batch_id: uuid.UUID
    kind: str
    status: str
    payload: dict[str, Any]
    confidence: float

    model_config = {"from_attributes": True}


class ImportItemPatch(BaseModel):
    payload: dict[str, Any] | None = None
    status: Literal["confirmed", "rejected"] | None = None
