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
import uuid
from datetime import datetime, timezone
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
    Profile,
    VocabItem,
)
from app.schemas import (
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
    VocabItemOut,
)
from app.workers.tasks import discover_editorial_candidates, grade_editorial_outline

router = APIRouter(tags=["editorial"])

_editor_or_admin = require_role("editor", "admin")


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
    return (await db.execute(query)).scalars().all()


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

    return EditorialArticleOut(
        id=article.id,
        source_name=article.source_name,
        source_url=article.source_url,
        title_ko=article.title_ko,
        level_estimate=article.level_estimate,
        topic_tags=article.topic_tags,
        body_ko=article.body_ko,
        vocab=[VocabItemOut.model_validate(v) for v in vocab],
        grammar=[GrammarPointOut.model_validate(g) for g in grammar],
        thinking_guide_text=article.thinking_guide_text,
        created_at=article.created_at,
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
