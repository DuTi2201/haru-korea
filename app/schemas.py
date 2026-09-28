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


class RefreshRequest(BaseModel):
    refresh_token: str


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


class CorpusAudioRequest(BaseModel):
    """No corpus_item_id field — it's the path param on POST
    /corpus/{corpus_item_id}/audio (mirrors LectureAudioRequest's shape,
    minus the redundant id)."""

    text_ko: str
    voice: str = "ko-female-1"
    prompt_version: str = "v1"


class VocabAudioRequest(BaseModel):
    """Same shape as CorpusAudioRequest — text_ko is the vocab item's own
    `hangul` (the frontend sends it, same as the corpus/listening screen
    does, so the backend never needs to re-look up the word itself)."""

    text_ko: str
    voice: str = "ko-female-1"
    prompt_version: str = "v1"


class PodcastRequest(BaseModel):
    """POST /lessons/{id}/podcast and /editorials/{id}/podcast: no text
    from the caller at all — unlike lecture/corpus/vocab audio, THIS
    endpoint generates its own script (Gemini synthesizes the lesson's/
    article's vocab+grammar into one consolidated teaching script) before
    handing it to TTS. See app.workers.tasks.generate_content_podcast."""

    voice: str = "ko-female-1"
    prompt_version: str = "podcast-v1"


class ArticleAudioRequest(BaseModel):
    """POST /editorials/{id}/audio: reads the article's OWN body_ko text
    aloud verbatim (native-length sentences, real pacing/pauses) — no
    Gemini script-writing step at all, unlike PodcastRequest. Deliberately
    separate from the podcast feature: a learner listening to the actual
    news article and a learner listening to a teaching script ABOUT its
    vocab/grammar are two different study modes, cached independently
    (see app.workers.tasks.generate_article_audio)."""

    voice: str = "ko-female-1"
    prompt_version: str = "article-v1"


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


# ------------------------------------------------------------------ content --
class TopicOut(BaseModel):
    id: int
    name: str
    quizlet_url: str | None

    model_config = {"from_attributes": True}


class VocabItemOut(BaseModel):
    id: int
    lesson_id: int | None
    hangul: str
    pos: str | None
    meaning_vi: str
    definition_ko: str | None
    level: int
    hanja: str | None
    sino_vietnamese: str | None
    example_ko: str | None

    model_config = {"from_attributes": True}


class GrammarPointOut(BaseModel):
    id: int
    lesson_id: int | None
    pattern: str
    meaning_vi: str
    level: int
    example_ko: str | None
    usage_context_vi: str | None = None
    topik_tip_vi: str | None = None

    model_config = {"from_attributes": True}


class LessonOut(BaseModel):
    id: int
    title: str
    level: int
    content: str
    created_at: datetime
    topics: list[TopicOut]
    vocab: list[VocabItemOut]
    grammar: list[GrammarPointOut]


class CorpusItemOut(BaseModel):
    id: uuid.UUID
    film_id: int
    film_title: str
    text_ko: str
    kind: str
    level: int
    register: str
    topics: list[str]
    grammar_patterns: list[str]

    model_config = {"from_attributes": True}


class ItemStateReviewRequest(BaseModel):
    """Records a learner's quick in-app check on one vocab/grammar item —
    SRS §5 ITEM_STATE's `strength` nudge, not a full SM-2 scheduler (see
    ItemState's docstring in app/models.py)."""

    item_type: Literal["vocab_item", "grammar_point"]
    item_id: int
    correct: bool


class ItemStateOut(BaseModel):
    item_type: str
    item_id: int
    strength: float
    last_seen: datetime

    model_config = {"from_attributes": True}


class TodayPlanTask(BaseModel):
    """One real, clickable suggestion for /me/plan — never a fabricated
    stat. `status` is only ever computed from data that actually exists;
    a task whose progress genuinely isn't tracked yet (e.g. listening)
    always reports "todo" rather than faking a checkmark."""

    kind: Literal["vocab_review", "listening", "reading"]
    title: str
    subtitle: str
    status: Literal["todo", "in_progress", "done"]
    lesson_id: int | None = None
    article_id: uuid.UUID | None = None


class TodayPlanOut(BaseModel):
    streak_days: int
    readiness_pct: float | None
    goal: str | None
    exam_date: datetime | None
    tasks: list[TodayPlanTask]


# ------------------------------------------------------------------- ingest --
class ImportBatchOut(BaseModel):
    id: uuid.UUID
    kind: str
    status: str
    flagged_count: int
    source_file: str | None
    film_id: int | None
    exam_paper_id: uuid.UUID | None
    editorial_article_id: uuid.UUID | None
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


# ---------------------------------------------------------------- editorial --
class EditorialSourceCreate(BaseModel):
    name: str
    base_url: str
    rss_url: str | None = None
    license_note: str | None = None
    active: bool = True


class EditorialSourceUpdate(BaseModel):
    base_url: str | None = None
    rss_url: str | None = None
    license_note: str | None = None
    active: bool | None = None


class EditorialSourceOut(BaseModel):
    id: int
    name: str
    base_url: str
    rss_url: str | None
    license_note: str | None
    active: bool

    model_config = {"from_attributes": True}


class EditorialArticleSummaryOut(BaseModel):
    """GET /editorials list card — no body_ko (too big for a list) and no
    model_outline (that's revealed only after the learner submits their
    own outline attempt — see EditorialOutlineOut)."""

    id: uuid.UUID
    source_name: str
    title_ko: str | None
    level_estimate: int | None
    topic_tags: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class EditorialArticleOut(BaseModel):
    id: uuid.UUID
    source_name: str
    source_url: str
    title_ko: str | None
    level_estimate: int | None
    topic_tags: list[str]
    body_ko: str
    vocab: list[VocabItemOut]
    grammar: list[GrammarPointOut]
    thinking_guide_text: str | None
    created_at: datetime


class EditorialOutlineSubmitRequest(BaseModel):
    phenomenon_text: str | None = None
    cause_text: str | None = None
    consequence_text: str | None = None
    solution_text: str | None = None


class EditorialOutlineOut(BaseModel):
    """The learner's own outline attempt. `model_outline` is included only
    once revision_count > 0 — the point of "luyện dàn ý" is thinking it
    through first, so the reference answer stays hidden until the learner
    has actually written and saved their own attempt at least once."""

    editorial_article_id: uuid.UUID
    phenomenon_text: str | None
    cause_text: str | None
    consequence_text: str | None
    solution_text: str | None
    revision_count: int
    updated_at: datetime
    model_outline: dict[str, Any] | None = None
    # Real Gemini analysis of the learner's OWN outline (not the static
    # model_outline reference) — "none" until the first save triggers
    # grade_editorial_outline, "pending" while it runs, then "ready"/"failed".
    feedback_status: str = "none"
    feedback_text: str | None = None

    model_config = {"protected_namespaces": ()}


class EditorialCandidateOut(BaseModel):
    id: uuid.UUID
    source_name: str
    source_url: str
    title_ko: str
    snippet_ko: str | None
    topic_tags: list[str]
    published_date: datetime | None
    status: str
    discovered_at: datetime

    model_config = {"from_attributes": True}
