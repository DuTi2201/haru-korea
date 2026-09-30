"""Editorial reading module (사설/칼럼 luyện đọc + luyện dàn ý cho TOPIK
viết câu 54). Two audiences share this file:

- Learner-facing (public read, no login): GET /editorials, GET
  /editorials/{id}. Only articles an admin has actually confirmed show up
  here (`body_ko IS NOT NULL` — see apply_editorial_batch in
  app/services/ingestion.py); nothing staged or rejected ever leaks out.
  GET/POST /editorials/{id}/outline DO require login — this is the
  learner's own "luyện dàn ý" attempt, tied to their profile.
- Admin-facing (editor/admin role): the RSS-discovery review queue
  (GET /editorial-candidates, POST .../dismiss, POST .../ingest). Nothing
  here is ever auto-published — discover_editorial_candidates (Celery
  beat) only ever proposes a candidate; an admin picks one, which routes
  through the exact same POST /imports?kind=editorial_article flow as a
  manually-typed URL (see app.api.routers.ingest.start_editorial_import).
"""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile, require_role
from app.api.routers.audio import _request_podcast
from app.api.routers.ingest import start_editorial_import
from app.db import get_db
from app.models import (
    EditorialArticle,
    EditorialCandidate,
    EditorialOutlineSubmission,
    EditorialSource,
    GrammarPoint,
    Job,
    LectureAudio,
    Profile,
    VocabItem,
)
from app.schemas import (
    ArticleAudioRequest,
    ArticleImageOut,
    EditorialArticleOut,
    EditorialArticleSummaryOut,
    EditorialCandidateOut,
    EditorialOutlineOut,
    EditorialOutlineSubmitRequest,
    EditorialSourceCreate,
    EditorialSourceOut,
    EditorialSourceUpdate,
    GrammarPointOut,
    ImportBatchAccepted,
    JobAccepted,
    PodcastRequest,
    StudyPackOut,
    VocabItemOut,
)
from app.services.article_extract import (
    TEXT_CLEAN_VERSION,
    article_tts_text,
    clean_article_text,
    clean_images,
    split_paragraphs,
)
from app.services.study_pack import STUDY_VERSION, easy_tts_text, study_text_sig
from app.workers.tasks import (
    discover_editorial_candidates,
    generate_article_audio,
    generate_study_pack,
    grade_editorial_outline,
    refresh_editorial_images,
)

router = APIRouter(tags=["editorial"])

_editor_or_admin = require_role("editor", "admin")

_STALE_JOB_AFTER = timedelta(minutes=8)
# A study pack still "pending" with no worker heartbeat for this long is
# dead (worker killed mid-way) and is re-queued; a "failed" one is retried
# after the (longer) cool-down so a persistently failing article doesn't
# hit Gemini on every page view.
_STUDY_STALE_AFTER = timedelta(minutes=10)
_STUDY_RETRY_AFTER = timedelta(minutes=20)


def _cover_image_url(images: list[dict] | None) -> str | None:
    """First usable photo (captions cleaned, author head-shots dropped) —
    the same filtering the detail view applies, so the list thumbnail never
    shows a picture the article page then hides."""
    cleaned = clean_images(images)
    return cleaned[0]["url"] if cleaned else None


def _valid_pack(article: EditorialArticle, cleaned_body: str) -> dict | None:
    """The stored study pack, only if it was written for exactly this text and
    prompt version — otherwise it is treated as absent (never shown against
    paragraphs it doesn't belong to)."""
    pack = article.study_pack
    if not pack or pack.get("version") != STUDY_VERSION:
        return None
    if pack.get("text_sig") != study_text_sig(cleaned_body):
        return None
    if len(pack.get("paragraphs", [])) != len(split_paragraphs(cleaned_body)):
        return None
    return pack


def _study_state(article: EditorialArticle, cleaned_body: str, now: datetime) -> tuple[str, dict | None, bool]:
    """(status to report, valid pack or None, whether to queue generation)."""
    pack = _valid_pack(article, cleaned_body)
    if pack is not None:
        return "ready", pack, False
    if not split_paragraphs(cleaned_body):
        return "none", None, False
    status_now = article.study_status or "none"
    beat = article.study_updated_at
    if status_now == "pending" and beat is not None and now - beat < _STUDY_STALE_AFTER:
        return "pending", None, False
    if status_now == "failed" and beat is not None and now - beat < _STUDY_RETRY_AFTER:
        return "failed", None, False
    return "pending", None, True


# ------------------------------------------------------------ learner-facing --
@router.get("/editorials", response_model=list[EditorialArticleSummaryOut])
async def list_editorials(
    db: Annotated[AsyncSession, Depends(get_db)],
    level: Annotated[int | None, Query(ge=1, le=6)] = None,
    topic: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
):
    """Public read, same as GET /corpus/items — only articles an admin has
    confirmed (body_ko set) ever show up here."""
    query = (
        select(EditorialArticle)
        .where(EditorialArticle.body_ko.isnot(None))
        .order_by(EditorialArticle.created_at.desc())
        .limit(limit)
    )
    if level is not None:
        query = query.where(EditorialArticle.level_estimate == level)
    if topic:
        query = query.where(EditorialArticle.topic_tags.any(topic))
    articles = (await db.execute(query)).scalars().all()
    return [
        EditorialArticleSummaryOut(
            id=a.id,
            source_name=a.source_name,
            title_ko=a.title_ko,
            level_estimate=a.level_estimate,
            topic_tags=a.topic_tags,
            created_at=a.created_at,
            cover_image_url=_cover_image_url(a.images),
        )
        for a in articles
    ]


@router.get("/editorials/{article_id}", response_model=EditorialArticleOut)
async def get_editorial(article_id: uuid.UUID, db: Annotated[AsyncSession, Depends(get_db)]):
    article = await db.get(EditorialArticle, article_id)
    if article is None or article.body_ko is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Editorial article not found", "not_found")

    # No cross-schema JOIN (module-boundary rule) — vocab_ids/grammar_ids
    # are bare arrays into content.*, resolved with their own bulk query,
    # same as content.py's corpus items resolving film/topic/grammar names.
    vocab = (
        (await db.execute(select(VocabItem).where(VocabItem.id.in_(article.vocab_ids)))).scalars().all()
        if article.vocab_ids
        else []
    )
    grammar = (
        (await db.execute(select(GrammarPoint).where(GrammarPoint.id.in_(article.grammar_ids)))).scalars().all()
        if article.grammar_ids
        else []
    )

    # Articles imported before the page-chrome filter existed still have the
    # menu / AI-summary / "지금 많이 보는 기사" lines stored in body_ko —
    # clean_article_text is idempotent, so applying it on every read fixes
    # them retroactively (and is a no-op on already-clean text).
    images_pending = False
    if article.images_fetched_at is None:
        # Imported before article photos existed: fetch them once in the
        # background. Stamp the time NOW so concurrent views don't each queue
        # a fetch; the task overwrites it when done.
        try:
            article.images_fetched_at = datetime.now(timezone.utc)
            db.add(article)
            await db.commit()
            refresh_editorial_images.delay(str(article.id))
            images_pending = True
        except Exception:  # noqa: BLE001 — reading must never fail because a background fetch could not be queued
            await db.rollback()

    cleaned_body = clean_article_text(article.body_ko)
    now = datetime.now(timezone.utc)
    study_status, study_pack, queue_study = _study_state(article, cleaned_body, now)
    if queue_study:
        # First read of an article without a (current) study pack: generate
        # it once in the background; the client polls while it is pending.
        try:
            article.study_status = "pending"
            article.study_updated_at = now
            db.add(article)
            await db.commit()
            generate_study_pack.delay(str(article.id))
        except Exception:  # noqa: BLE001 — reading must never fail because a background job could not be queued
            await db.rollback()
            study_status = "none"

    return EditorialArticleOut(
        id=article.id,
        source_name=article.source_name,
        source_url=article.source_url,
        title_ko=article.title_ko,
        level_estimate=article.level_estimate,
        topic_tags=article.topic_tags,
        body_ko=cleaned_body,
        vocab=[VocabItemOut.model_validate(v) for v in vocab],
        grammar=[GrammarPointOut.model_validate(g) for g in grammar],
        thinking_guide_text=article.thinking_guide_text,
        created_at=article.created_at,
        images=[ArticleImageOut(**im) for im in clean_images(article.images)],
        images_pending=images_pending,
        study_status=study_status,
        study=StudyPackOut.model_validate(study_pack) if study_pack else None,
    )


@router.post("/editorials/{article_id}/podcast", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_editorial_podcast(
    article_id: uuid.UUID,
    body: PodcastRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Same consolidated "bài giảng" podcast as POST /lessons/{id}/podcast
    (see app.workers.tasks.generate_content_podcast), but for one editorial
    article's vocab/grammar instead of a lesson's. No login required —
    same rationale as GET /editorials/{id} itself."""
    article = await db.get(EditorialArticle, article_id)
    if article is None or article.body_ko is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Editorial article not found", "not_found")
    return await _request_podcast(
        db, "article", str(article_id), body, request, article.vocab_ids, article.grammar_ids
    )


@router.post("/editorials/{article_id}/audio", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_article_audio(
    article_id: uuid.UUID,
    body: ArticleAudioRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Reads the article's OWN body_ko text aloud verbatim — completely
    independent of POST .../podcast (that endpoint has Gemini WRITE a
    teaching script about the article's vocab/grammar; this one just hands
    the article's real scraped sentences straight to TTS, so a learner can
    hear native-length pacing/pauses on the actual news text). No login
    required — same rationale as GET /editorials/{id} itself.

    cache_key is keyed off a hash of body_ko itself (not just article_id)
    so a later re-ingest/backfill that changes the article's text (e.g. the
    apply_editorial_batch fix that recovers a previously-empty body_ko)
    naturally busts any stale cached recording instead of serving audio for
    text that no longer matches what's on screen."""
    article = await db.get(EditorialArticle, article_id)
    if article is None or article.body_ko is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Editorial article not found", "not_found")

    # Voice the CLEANED text (title + body, page chrome/captions/ranking
    # widgets removed, URLs stripped) — the raw scrape used to be read aloud
    # verbatim, menu items included. Same function the reader screen's
    # body goes through, so what you hear is what you see.
    cleaned_body = clean_article_text(article.body_ko)
    if not cleaned_body.strip():
        # (checked on the BODY, not the final text: a title alone must never
        # be voiced as if it were the article)
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Article has no body text yet",
            "no_body_text",
            "This article hasn't been fully imported yet — its full text isn't available to read aloud.",
        )

    if body.variant == "easy":
        # The simplified-Korean rewrite from the study pack — for beginners.
        pack = _valid_pack(article, cleaned_body)
        easy_text = easy_tts_text(article.title_ko, pack) if pack else ""
        if not easy_text.strip():
            raise _problem(
                status.HTTP_409_CONFLICT,
                "Easy version not ready",
                "study_not_ready",
                "The simplified version of this article is still being prepared — try again in a minute.",
            )
        tts_text = easy_text
    else:
        tts_text = article_tts_text(article.title_ko, cleaned_body)

    # Keyed on the exact text voiced + the cleaning-rules version, so a
    # body backfill OR a cleaning-rule change regenerates the recording
    # instead of serving audio for text that no longer matches the screen.
    content_sig = hashlib.sha256(f"{TEXT_CLEAN_VERSION}\n{tts_text}".encode("utf-8")).hexdigest()[:16]
    cache_key = f"article-audio:{article_id}:{body.voice}:{body.prompt_version}:{content_sig}"
    if body.variant == "easy":
        cache_key += ":easy"

    cached = await db.execute(select(LectureAudio).where(LectureAudio.cache_key == cache_key))
    hit = cached.scalar_one_or_none()

    idempotency_key = request.headers.get("Idempotency-Key", cache_key)
    existing_job = await db.execute(
        select(Job).where(Job.type == "generate_article_audio", Job.idempotency_key == idempotency_key)
    )
    job = existing_job.scalar_one_or_none()
    if job is not None and (
        job.status not in ("queued", "running", "succeeded")
        or (job.status in ("queued", "running") and datetime.now(timezone.utc) - job.updated_at > _STALE_JOB_AFTER)
    ):
        # Same reasoning as _request_podcast's identical guard: a stale
        # "failed" row under this idempotency key would otherwise wedge
        # every retry behind a unique-constraint 500 forever. Also covers a
        # job whose worker died mid-way (a Railway redeploy during a
        # multi-minute generation): it would sit "running" for ever and
        # every later play would poll it endlessly. The worker refreshes
        # updated_at after every chunk, so no update for _STALE_JOB_AFTER
        # means it is dead, not slow.
        await db.delete(job)
        await db.flush()
        job = None

    if job is None:
        job = Job(
            type="generate_article_audio",
            owner_id=None,
            idempotency_key=idempotency_key,
            status="succeeded" if hit else "queued",
            progress=1.0 if hit else 0.0,
            result={
                "cache_key": hit.cache_key,
                "opus_path": hit.opus_path,
                "aac_path": hit.aac_path,
                "duration_sec": hit.duration_sec,
            }
            if hit
            else None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        if not hit:
            generate_article_audio.delay(str(job.id), tts_text, body.voice, body.prompt_version, cache_key)

    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


async def _get_or_init_outline(
    db: AsyncSession, article: EditorialArticle, learner_id: uuid.UUID
) -> EditorialOutlineSubmission | None:
    return (
        await db.execute(
            select(EditorialOutlineSubmission).where(
                EditorialOutlineSubmission.learner_id == learner_id,
                EditorialOutlineSubmission.editorial_article_id == article.id,
            )
        )
    ).scalar_one_or_none()


@router.get("/editorials/{article_id}/outline", response_model=EditorialOutlineOut)
async def get_editorial_outline(
    article_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """The learner's own draft. `model_outline` stays hidden (null) until
    they've saved at least one attempt — "luyện dàn ý" means thinking it
    through first, not reading the reference answer first."""
    article = await db.get(EditorialArticle, article_id)
    if article is None or article.body_ko is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Editorial article not found", "not_found")

    submission = await _get_or_init_outline(db, article, profile.id)
    if submission is None:
        return EditorialOutlineOut(
            editorial_article_id=article_id,
            phenomenon_text=None,
            cause_text=None,
            consequence_text=None,
            solution_text=None,
            revision_count=0,
            updated_at=datetime.now(timezone.utc),
            model_outline=None,
            feedback_status="none",
            feedback_text=None,
        )

    return EditorialOutlineOut(
        editorial_article_id=article_id,
        phenomenon_text=submission.phenomenon_text,
        cause_text=submission.cause_text,
        consequence_text=submission.consequence_text,
        solution_text=submission.solution_text,
        revision_count=submission.revision_count,
        updated_at=submission.updated_at,
        model_outline=article.model_outline if submission.revision_count > 0 else None,
        feedback_status=submission.feedback_status,
        feedback_text=submission.feedback_text,
    )


@router.post("/editorials/{article_id}/outline", response_model=EditorialOutlineOut)
async def submit_editorial_outline(
    article_id: uuid.UUID,
    body: EditorialOutlineSubmitRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Upsert — one row per (learner, article); re-saving bumps
    revision_count instead of creating a new row (unique constraint
    uq_editorial_outline_learner_article backs this)."""
    article = await db.get(EditorialArticle, article_id)
    if article is None or article.body_ko is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Editorial article not found", "not_found")

    submission = await _get_or_init_outline(db, article, profile.id)
    if submission is None:
        submission = EditorialOutlineSubmission(
            learner_id=profile.id,
            editorial_article_id=article_id,
            phenomenon_text=body.phenomenon_text,
            cause_text=body.cause_text,
            consequence_text=body.consequence_text,
            solution_text=body.solution_text,
            revision_count=1,
        )
    else:
        submission.phenomenon_text = body.phenomenon_text
        submission.cause_text = body.cause_text
        submission.consequence_text = body.consequence_text
        submission.solution_text = body.solution_text
        submission.revision_count += 1
    # Every save gets a REAL Gemini pass over what the learner actually
    # wrote (grade_editorial_outline) — previously this endpoint only ever
    # persisted the text and handed back the static model_outline, with no
    # analysis of the learner's own attempt at all.
    submission.feedback_status = "pending"
    submission.feedback_text = None
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    job = Job(type="grade_editorial_outline", owner_id=profile.id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    grade_editorial_outline.delay(str(job.id), str(submission.id))

    return EditorialOutlineOut(
        editorial_article_id=article_id,
        phenomenon_text=submission.phenomenon_text,
        cause_text=submission.cause_text,
        consequence_text=submission.consequence_text,
        solution_text=submission.solution_text,
        revision_count=submission.revision_count,
        updated_at=submission.updated_at,
        model_outline=article.model_outline,  # always revealed after a save — revision_count is now >= 1
        feedback_status=submission.feedback_status,
        feedback_text=submission.feedback_text,
    )


# --------------------------------------------------------------- admin-facing --
@router.get("/editorial-sources", response_model=list[EditorialSourceOut])
async def list_editorial_sources(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """Registered outlets discover_editorial_candidates polls (KBS/Chosun/
    Naver/...). No official RSS URL ships pre-seeded — an admin registers
    each source's real, verified feed URL here first; a source with no
    rss_url or active=false is simply skipped by the periodic task."""
    return (await db.execute(select(EditorialSource).order_by(EditorialSource.name))).scalars().all()


@router.post("/editorial-sources", response_model=EditorialSourceOut, status_code=status.HTTP_201_CREATED)
async def create_editorial_source(
    body: EditorialSourceCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    existing = await db.execute(select(EditorialSource).where(EditorialSource.name == body.name))
    if existing.scalar_one_or_none() is not None:
        raise _problem(status.HTTP_409_CONFLICT, "Source name already exists", "name_taken")
    source = EditorialSource(**body.model_dump())
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


@router.patch("/editorial-sources/{source_id}", response_model=EditorialSourceOut)
async def update_editorial_source(
    source_id: int,
    body: EditorialSourceUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    source = await db.get(EditorialSource, source_id)
    if source is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Source not found", "not_found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(source, field, value)
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


@router.get("/editorial-candidates", response_model=list[EditorialCandidateOut])
async def list_editorial_candidates(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    status_filter: Annotated[str | None, Query(alias="status")] = "new",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """Backs the Studio "candidate" review queue — RSS auto-discovery only
    ever proposes here (discover_editorial_candidates, Celery beat);
    nothing is published without an admin picking it via .../ingest."""
    query = select(EditorialCandidate).order_by(EditorialCandidate.discovered_at.desc()).limit(limit)
    if status_filter:
        query = query.where(EditorialCandidate.status == status_filter)
    return (await db.execute(query)).scalars().all()


@router.post("/editorial-candidates/discover", status_code=status.HTTP_202_ACCEPTED)
async def trigger_editorial_discovery(profile: Annotated[Profile, Depends(_editor_or_admin)]):
    """Manually kicks off the same RSS-discovery pass discover_editorial_
    candidates (Celery beat) otherwise only runs every 6h — the "Làm mới"
    button used to just re-fetch the candidate list (a no-op if the beat
    scheduler hadn't ticked yet, which — until this deploy — it never had:
    the worker process ran with no beat scheduler at all, so this task had
    literally never executed once). Fire-and-forget, same as the beat
    schedule: no Job row (per the task's own docstring, "không ai chờ kết
    quả trực tiếp"), just enqueues and returns immediately; refresh the
    candidate list a few seconds later to see what it found."""
    discover_editorial_candidates.delay()
    return {"status": "queued"}


@router.post("/editorial-candidates/{candidate_id}/dismiss", response_model=EditorialCandidateOut)
async def dismiss_editorial_candidate(
    candidate_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    candidate = await db.get(EditorialCandidate, candidate_id)
    if candidate is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Candidate not found", "not_found")
    candidate.status = "dismissed"
    db.add(candidate)
    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.post(
    "/editorial-candidates/{candidate_id}/ingest",
    response_model=ImportBatchAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_editorial_candidate(
    candidate_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """Same underlying flow as POST /imports?kind=editorial_article — this
    just pre-fills source_url/source_name/title_ko/published_date from the
    candidate instead of requiring the admin to type them, and marks the
    candidate "ingested" so it drops off the review queue."""
    candidate = await db.get(EditorialCandidate, candidate_id)
    if candidate is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Candidate not found", "not_found")

    accepted = await start_editorial_import(
        db,
        profile,
        candidate.source_url,
        candidate.source_name,
        candidate.title_ko,
        candidate.published_date,
    )
    candidate.status = "ingested"
    db.add(candidate)
    await db.commit()
    return accepted
