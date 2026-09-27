"""Celery tasks — where every AI call and heavy transform actually runs.
Sync SQLAlchemy session (Celery workers are sync by default); the API
process uses the async session instead (see app/db.py AsyncSessionLocal).
"""
import time
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.job_events import publish_job_event

# Separate sync engine for worker-side DB access.
_sync_engine = create_engine(settings.SYNC_DATABASE_URL, pool_pre_ping=True)


@celery_app.task(name="app.workers.tasks.generate_lecture_audio", bind=True, max_retries=3)
def generate_lecture_audio(self, job_id: str, lesson_id: str, text_ko: str, voice: str, prompt_version: str):
    """Example end-to-end async job, wired to prove the 202+job_id+SSE
    pattern: text -> Gemini TTS/script (stub) -> ffmpeg transcode to
    opus+aac -> row in audio.lecture_audio -> job.succeeded.

    Swap the two `time.sleep` placeholders for the real Gemini call and
    an actual ffmpeg subprocess once the audio module is implemented.
    """
    jid = uuid.UUID(job_id)
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tạo lời thoại")
            time.sleep(1)  # placeholder for the Gemini call

            publish_job_event(db, jid, progress=0.5, step="Đang chuyển đổi âm thanh (ffmpeg)")
            time.sleep(1)  # placeholder for the ffmpeg opus/aac transcode

            publish_job_event(db, jid, progress=0.9, step="Đang lưu file")
            cache_key = f"{lesson_id}:{voice}:{prompt_version}"
            opus_path = f"audio/{cache_key}.opus"

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={"cache_key": cache_key, "opus_path": opus_path, "duration_sec": 0},
            )
        except Exception as exc:  # noqa: BLE001 — report to job row, then re-raise for Celery retry bookkeeping
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.grade_writing_submission")
def grade_writing_submission(job_id: str, submission_id: str):
    """Stub for the Xưởng viết grading step (OCR transcript already
    confirmed by the learner -> Gemini grading -> writing_score row)."""
    jid = uuid.UUID(job_id)
    with Session(_sync_engine) as db:
        publish_job_event(db, jid, status="running", progress=0.3, step="Đang chấm điểm")
        publish_job_event(
            db,
            jid,
            status="succeeded",
            progress=1.0,
            step="Hoàn tất chấm điểm",
            result={"submission_id": submission_id},
        )


@celery_app.task(name="app.workers.tasks.reset_daily_ai_quota")
def reset_daily_ai_quota():
    """pg_cron-equivalent scheduled task (Celery beat), replaces the
    SDD's Redis TTL-based daily quota reset. No-op scaffold: wire up the
    actual per-learner counter reset once the quota table is added.
    """
    return {"ok": True}
