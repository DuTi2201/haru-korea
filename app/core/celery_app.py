"""Celery app — the async-worker replacement the SDD calls for: every
Gemini call and every heavy transform (audio transcode, exam ingestion
pass 1-3, writing OCR+grading) runs here, never inline in a FastAPI
request handler.

Railway topology: the "api" service runs uvicorn (app.main:app); the
"worker" service runs this module (`celery -A app.core.celery_app worker`).
Both read the same REDIS_URL and DATABASE_URL, so they share the queue and
the `jobs` table.
"""
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "haru",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Single-flight: a task id derived from a content hash (cache_key) is
    # rejected if already queued/running — see app/services/job_events.py
    # `acquire_job_lock`, the Postgres-table equivalent of a Redis lock.
    beat_schedule={
        "reset-daily-ai-quota": {
            "task": "app.workers.tasks.reset_daily_ai_quota",
            "schedule": 3600.0,  # hourly tick; task itself checks UTC midnight
        },
    },
)
