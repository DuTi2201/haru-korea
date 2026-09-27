"""Job progress fan-out: Postgres `jobs` row is the source of truth;
Redis pub/sub is the live-push side channel the SSE endpoint reads from.
This is the direct replacement for the SDD's Celery+Redis SSE design —
same shape (job.progress / job.succeeded / job.failed events), different
transport (Redis pub/sub instead of an in-process broker sitting beside
FastAPI), because here the publisher (Celery worker) and the subscriber
(FastAPI SSE handler) are different Railway services/processes.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.redis_client import job_channel, sync_redis
from app.models import Job


def publish_job_event(db: Session, job_id: uuid.UUID, **fields: Any) -> None:
    """Call from a Celery task (sync context). Updates the `jobs` row AND
    publishes the same delta on `job:{id}:events` for anyone subscribed.
    """
    job = db.get(Job, job_id)
    if job is None:
        return
    for key, value in fields.items():
        setattr(job, key, value)
    job.updated_at = datetime.now(timezone.utc)
    db.add(job)
    db.commit()

    event_type = {
        "succeeded": "job.succeeded",
        "failed": "job.failed",
    }.get(fields.get("status", ""), "job.progress")

    payload = {
        "job_id": str(job_id),
        "status": job.status,
        "progress": job.progress,
        "step": job.step,
        "result": job.result,
        "error": job.error,
    }
    sync_redis.publish(job_channel(str(job_id)), json.dumps({"event": event_type, "data": payload}))


def acquire_job_lock(db: Session, job_type: str, idempotency_key: str, owner_id: uuid.UUID | None) -> Job | None:
    """Single-flight guard — the Postgres-table equivalent of a Redis
    SETNX lock (SDD §5, principle: "AI vừa đắt vừa chậm ... không tạo
    job trùng"). Returns the EXISTING job if one with the same
    (type, idempotency_key) is already queued/running, else None meaning
    "safe to create a new one". The unique constraint on
    (type, idempotency_key) in the DB is the actual race-proof guarantee;
    this lookup just gives a fast, friendly 200-with-existing-job path.
    """
    existing = db.execute(
        select(Job).where(
            Job.type == job_type,
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running"]),
        )
    ).scalar_one_or_none()
    return existing
