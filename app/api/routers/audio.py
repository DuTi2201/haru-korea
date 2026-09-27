"""Audio module — the ONE fully-wired example of the SDD's async-job
pattern end to end: client POSTs -> 202 + job_id -> Celery task runs ->
Redis publishes progress -> SSE streams it -> row lands in
audio.lecture_audio. Every other AI-backed endpoint (writing grading,
exam ingestion, placement scoring) should follow this exact shape.
"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_profile
from app.db import get_db
from app.models import Job, LectureAudio, Profile
from app.schemas import JobAccepted, LectureAudioRequest
from app.workers.tasks import generate_lecture_audio

router = APIRouter(prefix="/lessons", tags=["audio"])


@router.post("/{lesson_id}/lecture", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_lecture_audio(
    lesson_id: str,
    body: LectureAudioRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Content-addressed cache: `cache_key` = f(lesson_id, voice,
    prompt_version). If audio already exists, return it as an
    already-succeeded job instead of re-running Gemini+ffmpeg
    (SDD: "bất biến theo nội dung: khóa cache là hash").
    """
    cache_key = f"{lesson_id}:{body.voice}:{body.prompt_version}"

    cached = await db.execute(select(LectureAudio).where(LectureAudio.cache_key == cache_key))
    hit = cached.scalar_one_or_none()

    idempotency_key = request.headers.get("Idempotency-Key", cache_key)

    existing_job = await db.execute(
        select(Job).where(
            Job.type == "generate_lecture_audio",
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running", "succeeded"]),
        )
    )
    job = existing_job.scalar_one_or_none()

    if job is None:
        job = Job(
            type="generate_lecture_audio",
            owner_id=profile.id,
            idempotency_key=idempotency_key,
            status="succeeded" if hit else "queued",
            progress=1.0 if hit else 0.0,
            result={"cache_key": hit.cache_key, "opus_path": hit.opus_path, "duration_sec": hit.duration_sec}
            if hit
            else None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        if not hit:
            generate_lecture_audio.delay(
                str(job.id), lesson_id, body.text_ko, body.voice, body.prompt_version
            )

    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )
