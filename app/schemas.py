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
    # Part of the audio cache key. "v2" = the first generation voiced by Google
    # Chirp 3 HD; clips cached under "v1" (Gemini voice) are therefore not reused.
    prompt_version: str = "v2"


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
    # Accepted for compatibility but ignored: the server decides which prompt
    # version writes the lecture (app.services.podcast_script.PODCAST_VERSION),
    # so improving the prompt regenerates lectures without a client release.
    prompt_version: str = "podcast-v2"


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
    # "original" = the article's own text; "easy" = the simplified-Korean
    # (TOPIK 1-2) rewrite from the study pack, for beginners who can't follow
    # the original at native speed yet. Cached independently per text.
    variant: Literal["original", "easy"] = "original"


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


class CollocationOut(BaseModel):
    ko: str
    vi: str


class ContrastOut(BaseModel):
    pattern: str
    diff_vi: str


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
    # chunk layers (lesson-v4); null on cards extracted before they existed
    family: str | None = None
    node_word: str | None = None
    register: str | None = None
    usage_note_vi: str | None = None
    collocations: list[CollocationOut] | None = None
    distractors: list[str] | None = None

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
    contrast_group: str | None = None
    contrasts: list[ContrastOut] | None = None

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


class LessonSummaryOut(BaseModel):
    """One row of the lesson picker (GET /lessons): enough to choose a lesson
    without fetching its whole content."""

    id: int
    title: str
    level: int
    topics: list[str]
    vocab_count: int
    grammar_count: int
    # Items of this lesson the signed-in learner has mastered; None when the
    # caller is anonymous (nothing to report — not "0").
    mastered: int | None = None
    # True for the lesson /me/plan would hand this learner next (never for anonymous callers).
    is_next: bool = False


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
    # Vietnamese meaning of the line and how/when to use it; null until the
    # enrichment step has covered the sentence (the app shows the Korean alone).
    meaning_vi: str | None = None
    usage_note_vi: str | None = None

    model_config = {"from_attributes": True}


class CorpusSimilarOut(CorpusItemOut):
    """A sentence close in meaning to another one (pgvector cosine distance on
    the stored embedding; 0 = identical, larger = further apart) — typically
    the same idea in another speech level or with other word forms."""

    distance: float


class CorpusLevelCount(BaseModel):
    level: int
    count: int


class CorpusRegisterCount(BaseModel):
    register: str
    count: int


class CorpusTopicCount(BaseModel):
    id: int
    name: str
    count: int


class CorpusGrammarCount(BaseModel):
    id: int
    pattern: str
    count: int


class CorpusFilmCount(BaseModel):
    id: int
    title: str
    count: int


class CorpusFacetsOut(BaseModel):
    """What the listening picker can filter by, with how many distinct
    sentences each choice holds (repeated sentences counted once)."""

    total: int
    levels: list[CorpusLevelCount]
    registers: list[CorpusRegisterCount]
    topics: list[CorpusTopicCount]
    grammar: list[CorpusGrammarCount]
    films: list[CorpusFilmCount]


class CorpusPageOut(BaseModel):
    """One page of a listening session: `total` is the size of the whole
    filtered set, so the client knows how far it can keep paging."""

    total: int
    offset: int
    items: list[CorpusItemOut]


class CorpusEnrichmentStatusOut(BaseModel):
    """Studio: how far the corpus is from having a Vietnamese meaning, usage
    note and naturalness verdict on every sentence. Counts are stored rows
    (a sentence imported twice counts twice here, once for learners)."""

    version: str
    total: int
    pending: int  # not yet covered by the current enrichment prompt
    enriched: int
    hidden: int  # judged unnatural: not shown to learners
    awkward: int  # readable but stiff: shown
    active_job_id: uuid.UUID | None = None  # a run in progress, if any


class HiddenCorpusItemOut(BaseModel):
    id: uuid.UUID
    film_title: str
    text_ko: str


class ItemStateReviewRequest(BaseModel):
    """Records a learner's quick in-app check on one vocab/grammar item: nudges
    SRS §5 ITEM_STATE's `strength` and moves its review schedule
    (app.services.srs)."""

    item_type: Literal["vocab_item", "grammar_point"]
    item_id: int
    correct: bool
    # How the card was asked and, for a wrong fill-in-the-blank, what was picked.
    # Optional (older clients send neither); they only enrich the error log.
    mode: Literal["recognize", "cloze"] | None = None
    chosen: str | None = Field(default=None, max_length=120)


class ItemStateOut(BaseModel):
    item_type: str
    item_id: int
    strength: float
    last_seen: datetime
    # the spaced-review schedule (app.services.srs): when it comes back, how many
    # right answers in a row, how often it was forgotten
    due_at: datetime | None = None
    reps: int = 0
    lapses: int = 0
    interval_days: float = 0.0

    model_config = {"from_attributes": True}


class ClozeOut(BaseModel):
    """A fill-in-the-blank question. The answer is sent along so the app can mark
    it at once, without a round trip per card."""

    prompt_ko: str
    answer: str
    choices: list[str]


class ReviewQueueItem(BaseModel):
    item_type: Literal["vocab_item", "grammar_point"]
    item_id: int
    is_new: bool
    mode: Literal["recognize", "cloze"]
    reps: int
    lapses: int
    leech: bool = False  # forgotten again and again: shown first, flagged "hay sai"
    due_at: datetime | None = None
    lesson_title: str | None = None
    # A word that comes from a news article rather than a lesson.
    article_id: uuid.UUID | None = None
    article_title: str | None = None
    vocab: VocabItemOut | None = None
    grammar: GrammarPointOut | None = None
    cloze: ClozeOut | None = None


class ReviewQueueOut(BaseModel):
    """Today's sitting: what is due, then what is new."""

    due_total: int  # everything due now, which can be more than the due cards returned
    new_available: int  # new cards today's cap still allows
    new_today: int  # new cards started in the last 24 hours
    items: list[ReviewQueueItem]
    focus: Literal["due", "weak"] = "due"  # "weak": the cards the learner keeps forgetting, due or not
    article_waiting: int = 0  # article words not started yet that fit the learner's level
    article_above_level: int = 0  # article words held back because they are above the goal level


# ------------------------------------------------------- practice: weak spots --
class WeakTypeOut(BaseModel):
    error_type: str
    label: str
    skill: str
    count: int


class WeakItemOut(BaseModel):
    item_type: Literal["vocab_item", "grammar_point"]
    item_id: int
    title: str  # the chunk or the grammar pattern
    meaning_vi: str
    family: str | None = None
    errors: int


class WeakFamilyOut(BaseModel):
    family: str
    errors: int


class ExamAccuracyOut(BaseModel):
    qtype_code: str
    name: str
    attempts: int
    correct: int
    accuracy_pct: int


class WeaknessesOut(BaseModel):
    """Where the learner keeps going wrong in the last `days` days — counts of what
    happened, never a prediction of an exam score."""

    days: int
    total_errors: int
    by_type: list[WeakTypeOut]
    top_items: list[WeakItemOut]  # cards that went wrong more than once
    families: list[WeakFamilyOut]  # sets ("họ từ") the mistakes gather in
    exam: list[ExamAccuracyOut]  # per question type, weakest first
    weak_cards: int  # cards forgotten and not yet back on their feet ("Ôn thẻ hay sai")
    leeches: int  # of those, forgotten twice or more
    exam_ready: int  # confirmed reading questions available for the mini-drill


# ------------------------------------------------------ practice: exam drill --
class ExamDrillQuestion(BaseModel):
    id: uuid.UUID
    number: int
    qtype_code: str
    qtype_name_vi: str
    instruction_ko: str | None = None  # the group instruction above the question
    stem_ko: str
    options: list[str]
    passage_ko: str | None = None
    paper_label: str


class ExamDrillOut(BaseModel):
    items: list[ExamDrillQuestion]
    available: int  # usable questions in the bank, whether or not they are offered now


class ExamAnswerRequest(BaseModel):
    item_id: uuid.UUID
    chosen: int = Field(ge=1, le=5)  # 1-based, like the key


class ExamAnswerOut(BaseModel):
    item_id: uuid.UUID
    chosen: int
    answer: int
    correct: bool


# ---------------------------------------------------- practice: writing drill --
class WritingBlankView(BaseModel):
    label: str
    intent_vi: str
    uses: list[str] = Field(default_factory=list)


class WritingTargetOut(BaseModel):
    item_id: int
    hangul: str
    meaning_vi: str


class WritingPromptOut(BaseModel):
    """The exercise as the learner sees it. The model answers are not here — they
    come with the result."""

    text_type: str | None = None
    title_ko: str | None = None
    body_ko: str
    register: str | None = None
    blanks: list[WritingBlankView]
    targets: list[WritingTargetOut] = Field(default_factory=list)


class WritingFixOut(BaseModel):
    original: str
    corrected: str = ""
    category: str
    reason_vi: str
    source: Literal["ai", "auto"]


class WritingBlankResultOut(BaseModel):
    label: str
    answer: str
    verdict: Literal["good", "minor", "off", "unchecked"]
    comment_vi: str = ""
    fixes: list[WritingFixOut] = Field(default_factory=list)
    model_answer: str
    alt_answers: list[str] = Field(default_factory=list)
    intent_vi: str


class WritingResultOut(BaseModel):
    blanks: list[WritingBlankResultOut]
    ai_checked: bool
    note: str


class WritingDrillOut(BaseModel):
    id: uuid.UUID
    status: Literal["pending", "ready", "failed"]  # the exercise being made
    grade_status: Literal["none", "pending", "ready", "failed"]  # the answers being checked
    job_id: uuid.UUID | None = None
    prompt: WritingPromptOut | None = None
    answers: dict[str, str] | None = None
    result: WritingResultOut | None = None
    created_at: datetime


class WritingDrillStartOut(BaseModel):
    drill_id: uuid.UUID
    status: Literal["pending", "ready", "failed"]
    job_id: uuid.UUID | None = None  # follow it with the jobs API while status is "pending"


class WritingAnswersRequest(BaseModel):
    answers: dict[Literal["㉠", "㉡"], str]  # one sentence per blank; each at most 200 characters


class TodayPlanTask(BaseModel):
    """One real, clickable suggestion for /me/plan — never a fabricated
    stat. `status` is only ever computed from data that actually exists;
    a task whose progress genuinely isn't tracked yet (e.g. listening)
    always reports "todo" rather than faking a checkmark."""

    kind: Literal["review", "vocab_review", "listening", "reading"]
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


class ExamAnswerTextRequest(BaseModel):
    """Answers an editor typed or pasted for an exam batch: "1-2, 2-1, …" pairs, or an
    unbroken run of digits taken as the answers from question `start`."""

    text: str = Field(min_length=1, max_length=4000)
    start: int = Field(default=1, ge=1, le=200)
    dry_run: bool = False  # only report what would change


class ExamAnswerReportOut(BaseModel):
    dry_run: bool
    skill: str | None = None
    answers: dict[int, int]  # what was read from the text / key
    applied: list[int]  # question numbers that got an answer
    changed: list[int]  # ...of which replaced a different answer
    unmatched: list[int]  # given, but no such question in the paper
    missing: list[int]  # questions of the paper with no answer given
    conflicts: list[int] = []  # numbers a key could not be read reliably for


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
    own outline attempt — see EditorialOutlineOut). `cover_image_url` is the
    article's first scraped photo, for the card thumbnail."""

    id: uuid.UUID
    source_name: str
    title_ko: str | None
    level_estimate: int | None
    topic_tags: list[str]
    created_at: datetime
    cover_image_url: str | None = None

    model_config = {"from_attributes": True}


class ArticleImageOut(BaseModel):
    """One photo from the source article. `after_paragraph` = how many body
    paragraphs precede it (0 = above the first paragraph), so the reader
    can put it back where it sat in the original."""

    url: str
    caption: str | None = None
    after_paragraph: int = 0


class StudyWordOut(BaseModel):
    surface: str
    base: str | None = None
    pos: str | None = None
    meaning_vi: str


class StudySentenceOut(BaseModel):
    ko: str
    vi: str | None = None
    words: list[StudyWordOut] = []
    grammar_notes_vi: list[str] = []


class StudyParagraphOut(BaseModel):
    """One body paragraph, aligned by index with the paragraphs of `body_ko`
    (split on newlines): its sentences with translations + word breakdowns,
    and the paragraph rewritten in simple Korean."""

    easy_ko: str | None = None
    sentences: list[StudySentenceOut] = []


class StudyKeyTermOut(BaseModel):
    ko: str
    vi: str


class StudyPackOut(BaseModel):
    """Beginner "reading ladder" — see app.services.study_pack."""

    summary_vi: str = ""
    key_points_vi: list[str] = []
    key_terms: list[StudyKeyTermOut] = []
    paragraphs: list[StudyParagraphOut] = []


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
    images: list[ArticleImageOut] = []
    # True when this article's photos have never been fetched (imported
    # before photos existed) and a background fetch was just queued — the
    # client refetches once after a few seconds instead of showing none.
    images_pending: bool = False
    # none = not requested yet, pending = being generated (client polls),
    # ready = `study` is present, failed = generation failed (retried later).
    study_status: Literal["none", "pending", "ready", "failed"] = "none"
    study: StudyPackOut | None = None


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
