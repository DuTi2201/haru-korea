"""Generic job read endpoints — GET /jobs/{id} (poll fallback) and
GET /jobs/{id}/events (SSE), per SDD §8 "jobs" group. Job CREATION
endpoints live in each owning module's router (e.g. POST
/lessons/{id}/lecture in audio.py) since the request body differs per
job type; this router only reads.
"""
import asyncio
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile, get_current_profile_sse
from app.core.redis_client import async_redis, job_channel
from app.db import get_db
from app.models import Job, Profile
from app.schemas import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


async def _get_owned_job(job_id: uuid.UUID, db: AsyncSession, profile: Profile) -> Job:
    job = await db.get(Job, job_id)
    if job is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Job not found", "not_found")
    if job.owner_id is not None and job.owner_id != profile.id and profile.role not in ("editor", "admin"):
        raise _problem(status.HTTP_403_FORBIDDEN, "Not your job", "forbidden")
    return job


@router.get("/{job_id}", response_model=JobOut)
async def get_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    return await _get_owned_job(job_id, db, profile)


@router.get("/{job_id}/events")
async def stream_job_events(
    job_id: uuid.UUID,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile_sse)],
):
    """SSE stream. Sends the current state immediately (in case the job
    already finished before the client subscribed), then relays every
    Redis-published update until job.succeeded/job.failed or disconnect.
    """
    job = await _get_owned_job(job_id, db, profile)

    async def event_stream():
        yield _sse(
            "job.progress" if job.status in ("queued", "running") else f"job.{job.status}",
            {
                "job_id": str(job.id),
                "status": job.status,
                "progress": job.progress,
                "step": job.step,
                "result": job.result,
                "error": job.error,
            },
        )
        if job.status in ("succeeded", "failed", "cancelled"):
            return

        pubsub = async_redis.pubsub()
        await pubsub.subscribe(job_channel(str(job_id)))
        try:
            async for message in pubsub.listen():
                if await request.is_disconnected():
                    break
                if message["type"] != "message":
                    continue
                envelope = json.loads(message["data"])
                yield _sse(envelope["event"], envelope["data"])
                if envelope["event"] in ("job.succeeded", "job.failed"):
                    break
        finally:
            await pubsub.unsubscribe(job_channel(str(job_id)))
            await pubsub.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
