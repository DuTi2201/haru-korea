"""Celery tasks — where every AI call and heavy transform actually runs.
Sync SQLAlchemy session (Celery workers are sync by default); the API
process uses the async session instead (see app/db.py AsyncSessionLocal).
"""
import base64
import time
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.config import settings
from app.models import ImportBatch
from app.services import ingestion
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


@celery_app.task(name="app.workers.tasks.extract_lesson_import", bind=True, max_retries=2)
def extract_lesson_import(self, job_id: str, batch_id: str, file_b64: str, mime_type: str):
    """Bài học ảnh/PDF -> Gemini (đa phương thức) -> đề xuất lesson/vocab/
    grammar vào khu chờ duyệt (import_item). Không ghi bảng chính ở đây —
    chỉ POST /imports/{id}/confirm mới ghi (SDD: "khu chờ duyệt")."""
    jid, bid = uuid.UUID(job_id), uuid.UUID(batch_id)
    with Session(_sync_engine) as db:
        batch = db.get(ImportBatch, bid)
        if batch is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "batch not found"})
            return
        try:
            batch.status = "extracting"
            db.add(batch)
            db.commit()
            publish_job_event(db, jid, status="running", progress=0.2, step="Đang đọc tài liệu với Gemini")

            file_bytes = base64.b64decode(file_b64)
            staged, flagged = ingestion.run_lesson_extraction(db, batch, file_bytes, mime_type)

            batch.status = "awaiting_review"
            batch.flagged_count = flagged
            db.add(batch)
            db.commit()
            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất trích xuất",
                result={"import_batch_id": str(bid), "staged": staged, "flagged": flagged},
            )
        except Exception as exc:  # noqa: BLE001
            batch.status = "failed"
            db.add(batch)
            db.commit()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.extract_corpus_import", bind=True, max_retries=2)
def extract_corpus_import(self, job_id: str, batch_id: str, file_b64: str, film_title: str):
    """Phụ đề phim -> phân đoạn theo lô cố định (chunk_cues) -> mỗi lô gọi
    Gemini phân loại câu/cụm/mẫu ngữ pháp riêng, KHÔNG dồn cả kịch bản vào
    một prompt (FR-19, Gate G6: kích thước prompt không phình theo độ dài
    kịch bản) -> đề xuất corpus_item vào khu chờ duyệt."""
    jid, bid = uuid.UUID(job_id), uuid.UUID(batch_id)
    with Session(_sync_engine) as db:
        batch = db.get(ImportBatch, bid)
        if batch is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "batch not found"})
            return
        try:
            film = ingestion.find_or_create_film(db, film_title)
            batch.film_id = film.id
            batch.status = "extracting"
            db.add(batch)
            db.commit()
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tách câu thoại")

            raw_text = base64.b64decode(file_b64).decode("utf-8", errors="replace")
            staged, flagged = ingestion.run_corpus_extraction(db, batch, raw_text)

            batch.status = "awaiting_review"
            batch.flagged_count = flagged
            db.add(batch)
            db.commit()
            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất phân loại kho câu",
                result={"import_batch_id": str(bid), "film_id": film.id, "staged": staged, "flagged": flagged},
            )
        except Exception as exc:  # noqa: BLE001
            batch.status = "failed"
            db.add(batch)
            db.commit()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.apply_import_batch", bind=True, max_retries=2)
def apply_import_batch_task(self, job_id: str, batch_id: str):
    """Ghi các import_item đã confirmed vào bảng chính (content.lesson/
    vocab_item/grammar_point, hoặc corpus.corpus_item kèm embedding —
    lệnh gọi AI duy nhất ở bước confirm, nên bước này chạy qua Celery chứ
    không ghi trực tiếp trong handler FastAPI)."""
    jid, bid = uuid.UUID(job_id), uuid.UUID(batch_id)
    with Session(_sync_engine) as db:
        batch = db.get(ImportBatch, bid)
        if batch is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "batch not found"})
            return
        try:
            publish_job_event(db, jid, status="running", progress=0.3, step="Đang ghi dữ liệu đã duyệt")
            outcome = ingestion.apply_import_batch(db, batch)
            db.commit()
            publish_job_event(
                db, jid, status="succeeded", progress=1.0, step="Hoàn tất xác nhận", result=outcome
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.reset_daily_ai_quota")
def reset_daily_ai_quota():
    """pg_cron-equivalent scheduled task (Celery beat), replaces the
    SDD's Redis TTL-based daily quota reset. No-op scaffold: wire up the
    actual per-learner counter reset once the quota table is added.
    """
    return {"ok": True}
