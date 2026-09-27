"""Xưởng viết (writing workshop) — 5 steps, state machine on
writing_submission.status:
uploaded ↔ image_rejected → transcribing → awaiting_confirmation →
grading → graded → revising → revised → rewriting → grading (loop)
"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile
from app.db import get_db
from app.models import Job, Profile, WritingScore, WritingSubmission
from app.schemas import (
    JobAccepted,
    WritingConfirmTranscriptRequest,
    WritingScoreOut,
    WritingSubmissionOut,
)
from app.workers.tasks import grade_writing_submission

router = APIRouter(prefix="/writing", tags=["writing"])


@router.post("/submissions", response_model=WritingSubmissionOut, status_code=status.HTTP_201_CREATED)
async def create_submission(
    prompt_id: str,
    photo: UploadFile,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Bước 2 (chụp bài): stores the photo (Storage integration TODO —
    placeholder key below) and starts in `uploaded`. A real image-quality
    check (the 3-light indicator in the mockup) belongs in a Celery task
    that flips status to `image_rejected` on failure."""
    image_key = f"writing/{profile.id}/{uuid.uuid4()}_{photo.filename}"
    submission = WritingSubmission(
        learner_id=profile.id, prompt_id=prompt_id, image_key=image_key, status="uploaded"
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)
    return submission


@router.get("/submissions/{submission_id}", response_model=WritingSubmissionOut)
async def get_submission(
    submission_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    submission = await db.get(WritingSubmission, submission_id)
    if submission is None or submission.learner_id != profile.id:
        raise _problem(status.HTTP_404_NOT_FOUND, "Submission not found", "not_found")
    return submission


@router.post("/submissions/{submission_id}/transcript", response_model=WritingSubmissionOut)
async def confirm_transcript(
    submission_id: uuid.UUID,
    body: WritingConfirmTranscriptRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Bước 3 (đối chiếu bản chép): learner confirms/edits the OCR
    transcript -> awaiting_confirmation -> grading."""
    submission = await db.get(WritingSubmission, submission_id)
    if submission is None or submission.learner_id != profile.id:
        raise _problem(status.HTTP_404_NOT_FOUND, "Submission not found", "not_found")

    submission.transcript = body.transcript
    submission.status = "grading"
    await db.commit()

    job = Job(type="grade_writing_submission", owner_id=profile.id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    grade_writing_submission.delay(str(job.id), str(submission.id))

    return submission


@router.get("/submissions/{submission_id}/score", response_model=WritingScoreOut)
async def get_score(
    submission_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Bước 4 (kết quả): score always comes back as a range + priority
    fixes, never a single "true" number — SDD principle: "trung thực về
    độ chắc chắn của AI"."""
    result = await db.execute(select(WritingScore).where(WritingScore.submission_id == submission_id))
    score = result.scalar_one_or_none()
    if score is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Score not ready", "not_found")
    return WritingScoreOut(
        submission_id=score.submission_id,
        criteria=score.criteria,
        score_min=score.score_min,
        score_max=score.score_max,
        priority_fixes=score.priority_fixes,
    )
