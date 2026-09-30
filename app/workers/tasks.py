"""Celery tasks — where every AI call and heavy transform actually runs.
Sync SQLAlchemy session (Celery workers are sync by default); the API
process uses the async session instead (see app/db.py AsyncSessionLocal).
"""
import base64
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.models import (
    CorpusItemAudio,
    EditorialArticle,
    EditorialOutlineSubmission,
    GrammarPoint,
    ImportBatch,
    Job,
    LectureAudio,
    Lesson,
    VocabItem,
    VocabItemAudio,
)
from app.services import article_extract, ingestion, read_along, study_pack, tts
from app.services import gemini_client
from app.services.job_events import publish_job_event

# Separate sync engine for worker-side DB access.
_sync_engine = create_engine(settings.SYNC_DATABASE_URL, pool_pre_ping=True)


@celery_app.task(name="app.workers.tasks.generate_lecture_audio", bind=True, max_retries=3)
def generate_lecture_audio(self, job_id: str, lesson_id: str, text_ko: str, voice: str, prompt_version: str):
    """text -> Gemini TTS -> ffmpeg transcode to opus+aac -> row in
    audio.lecture_audio -> job.succeeded. Real pipeline (app/services/tts.py);
    this task is now just the job/cache bookkeeping around it — the ONE
    fully-wired example of the SDD's async-job pattern end to end, for
    real this time."""
    jid = uuid.UUID(job_id)
    cache_key = f"{lesson_id}:{voice}:{prompt_version}"
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tạo giọng đọc")
            opus_bytes, aac_bytes, duration_sec = tts.synthesize_korean_tts_chirp_first(text_ko, voice)

            publish_job_event(db, jid, progress=0.8, step="Đang lưu âm thanh")
            row = LectureAudio(
                cache_key=cache_key,
                opus_path=f"/api/v1/lessons/audio/{cache_key}.opus",
                aac_path=f"/api/v1/lessons/audio/{cache_key}.aac",
                opus_data=opus_bytes,
                aac_data=aac_bytes,
                prompt_version=prompt_version,
                voice=voice,
                duration_sec=duration_sec,
            )
            db.add(row)
            db.commit()

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={"cache_key": cache_key, "opus_path": row.opus_path, "duration_sec": duration_sec},
            )
        except Exception as exc:  # noqa: BLE001 — report to job row, then re-raise for Celery retry bookkeeping
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.generate_corpus_audio", bind=True, max_retries=3)
def generate_corpus_audio(self, job_id: str, corpus_item_id: str, text_ko: str, voice: str, prompt_version: str):
    """Same pattern as generate_lecture_audio, keyed to one corpus_item
    (a single listening sentence) instead of a whole lesson — backs the
    /listening screen's playback button."""
    jid = uuid.UUID(job_id)
    cache_key = f"{corpus_item_id}:{voice}:{prompt_version}"
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tạo giọng đọc")
            opus_bytes, aac_bytes, duration_sec = tts.synthesize_korean_tts_chirp_first(text_ko, voice)

            publish_job_event(db, jid, progress=0.8, step="Đang lưu âm thanh")
            row = CorpusItemAudio(
                corpus_item_id=uuid.UUID(corpus_item_id),
                cache_key=cache_key,
                opus_data=opus_bytes,
                aac_data=aac_bytes,
                prompt_version=prompt_version,
                voice=voice,
                duration_sec=duration_sec,
            )
            db.add(row)
            db.commit()

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={
                    "cache_key": cache_key,
                    "opus_path": f"/api/v1/corpus/audio/{cache_key}.opus",
                    "aac_path": f"/api/v1/corpus/audio/{cache_key}.aac",
                    "duration_sec": duration_sec,
                },
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.generate_vocab_audio", bind=True, max_retries=3)
def generate_vocab_audio(self, job_id: str, vocab_item_id: str, text_ko: str, voice: str, prompt_version: str):
    """Same pattern as generate_corpus_audio, keyed to one content.vocab_item
    (a single flashcard's hangul) instead of a listening sentence — backs
    the "nghe" (listen) button on the vocab-study screen (both the lesson
    flashcards and an editorial article's vocab list use the same
    VocabItem rows, so this one endpoint covers both)."""
    jid = uuid.UUID(job_id)
    cache_key = f"{vocab_item_id}:{voice}:{prompt_version}"
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tạo giọng đọc")
            opus_bytes, aac_bytes, duration_sec = tts.synthesize_korean_tts_chirp_first(text_ko, voice)

            publish_job_event(db, jid, progress=0.8, step="Đang lưu âm thanh")
            row = VocabItemAudio(
                vocab_item_id=int(vocab_item_id),
                cache_key=cache_key,
                opus_data=opus_bytes,
                aac_data=aac_bytes,
                prompt_version=prompt_version,
                voice=voice,
                duration_sec=duration_sec,
            )
            db.add(row)
            db.commit()

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={
                    "cache_key": cache_key,
                    "opus_path": f"/api/v1/vocab-items/audio/{cache_key}.opus",
                    "aac_path": f"/api/v1/vocab-items/audio/{cache_key}.aac",
                    "duration_sec": duration_sec,
                },
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.generate_content_podcast", bind=True, max_retries=3)
def generate_content_podcast(
    self, job_id: str, owner_kind: str, owner_id: str, voice: str, prompt_version: str, cache_key: str
):
    """Gemini synthesizes a lesson's/article's ENTIRE vocab+grammar into
    one consolidated "bài giảng" script (ingestion.generate_podcast_script)
    -> TTS -> row in audio.lecture_audio (script_text set, cache_key
    prefixed "podcast:lesson:..."/"podcast:article:..." so it shares that
    table/route instead of a parallel one — see LectureAudio's docstring).
    Reuses the SAME "content-addressed job" pattern as generate_lecture_audio,
    except the cache_key here is computed by the API route from a hash of
    the owner's CURRENT vocab/grammar ids (see app/api/routers/audio.py and
    editorial.py), so adding a word later naturally busts the cache instead
    of serving a stale lecture.

    `owner_kind` is "lesson" or "article" — re-queries vocab/grammar fresh
    from the DB rather than trusting caller-supplied payloads, same as
    every other extraction task here."""
    jid = uuid.UUID(job_id)
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tổng hợp nội dung bài giảng")

            if owner_kind == "lesson":
                lesson = db.get(Lesson, int(owner_id))
                if lesson is None:
                    raise RuntimeError(f"lesson {owner_id} not found")
                title = lesson.title
                vocab_rows = db.execute(select(VocabItem).where(VocabItem.lesson_id == lesson.id)).scalars().all()
                grammar_rows = (
                    db.execute(select(GrammarPoint).where(GrammarPoint.lesson_id == lesson.id)).scalars().all()
                )
            else:
                article = db.get(EditorialArticle, uuid.UUID(owner_id))
                if article is None:
                    raise RuntimeError(f"editorial_article {owner_id} not found")
                title = article.title_ko or article.source_name
                vocab_rows = (
                    db.execute(select(VocabItem).where(VocabItem.id.in_(article.vocab_ids))).scalars().all()
                    if article.vocab_ids
                    else []
                )
                grammar_rows = (
                    db.execute(select(GrammarPoint).where(GrammarPoint.id.in_(article.grammar_ids))).scalars().all()
                    if article.grammar_ids
                    else []
                )

            vocab = [(v.hangul, v.pos, v.meaning_vi, v.example_ko) for v in vocab_rows]
            grammar = [
                (g.pattern, g.meaning_vi, g.example_ko, g.usage_context_vi, g.topik_tip_vi) for g in grammar_rows
            ]

            publish_job_event(db, jid, progress=0.3, step="Đang viết kịch bản với Gemini")
            script = ingestion.generate_podcast_script(title, vocab, grammar)

            # The script alternates Vietnamese explanation with Korean examples, so it
            # stays on Gemini (a ko-KR Chirp voice cannot read Vietnamese).
            publish_job_event(db, jid, progress=0.6, step="Đang tạo giọng đọc với Gemini")
            opus_bytes, aac_bytes, duration_sec = tts.synthesize_korean_tts(script, voice)

            publish_job_event(db, jid, progress=0.9, step="Đang lưu bài giảng")
            row = LectureAudio(
                cache_key=cache_key,
                opus_path=f"/api/v1/lessons/audio/{cache_key}.opus",
                aac_path=f"/api/v1/lessons/audio/{cache_key}.aac",
                opus_data=opus_bytes,
                aac_data=aac_bytes,
                prompt_version=prompt_version,
                voice=voice,
                duration_sec=duration_sec,
                script_text=script,
            )
            db.add(row)
            db.commit()

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={
                    "cache_key": cache_key,
                    "opus_path": row.opus_path,
                    "aac_path": row.aac_path,
                    "duration_sec": duration_sec,
                    "script_text": script,
                },
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


def _timings_for_key(db: Session, cache_key: str) -> dict | None:
    """The read-along timeline of an earlier recording, taken from the result of
    the job that made it (the timeline is stored with the job result — there is
    no column for it on the audio row). None when that job is gone or predates
    read-along."""
    result = db.execute(
        select(Job.result)
        .where(
            Job.type == "generate_article_audio",
            Job.status == "succeeded",
            Job.result["cache_key"].astext == cache_key,
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    timings = (result or {}).get("timings")
    return timings if isinstance(timings, dict) else None


def _voice_article(
    db: Session,
    tts_text: str,
    voice: str,
    cache_key: str,
    fallback_cache_key: str | None,
    on_progress,
    units: list | None = None,
):
    """Voice an article. Returns (reuse_row, opus, aac, duration_sec,
    stored_key, timings).

    `units` (read_along.Unit tuples) turns on read-along: the article is voiced
    sentence by sentence and `timings` says when each one starts. Without it
    (an old queued task) the article is voiced as before and `timings` is None.

    `fallback_cache_key` is None when the API did not have Google Chirp on:
    plain Gemini, stored under `cache_key`, exactly as before.

    Otherwise Chirp 3 HD is tried first and its recording is stored under
    `cache_key` (which embeds the voice+pace spec). If Chirp fails, the
    WHOLE article is redone with Gemini (never two voices in one recording)
    and stored under `fallback_cache_key`, so a later play retries Chirp
    instead of being stuck with the fallback voice forever. A fallback
    recording that already exists is reused rather than spending Gemini's
    small daily quota again while Chirp is still down."""
    timed = [read_along.Unit(*u) for u in units] if units else None

    def gemini():
        if timed:
            return tts.synthesize_article_tts_timed(timed, voice, on_progress)
        opus, aac, dur = tts.synthesize_article_tts(tts_text, voice, on_progress)
        return opus, aac, dur, None

    def chirp():
        if timed:
            return tts.synthesize_article_chirp_timed(timed, on_progress)
        opus, aac, dur = tts.synthesize_article_chirp(tts_text, on_progress)
        return opus, aac, dur, None

    if fallback_cache_key is None:
        opus, aac, dur, timings = gemini()
        return None, opus, aac, dur, cache_key, timings

    try:
        if not tts.chirp_enabled():
            raise RuntimeError("TTS_GG_Chirp is not set on the worker service")
        opus, aac, dur, timings = chirp()
        return None, opus, aac, dur, cache_key, timings
    except Exception as chirp_exc:  # noqa: BLE001 - ANY Chirp failure (incl. an unexpected client error) must fall back, not lose the audio
        chirp_exc = RuntimeError(tts.describe_error(chirp_exc))
        print(f"[tts] article: Google Chirp failed ({chirp_exc}); using Gemini fallback", flush=True)
        existing = db.execute(select(LectureAudio).where(LectureAudio.cache_key == fallback_cache_key)).scalar_one_or_none()
        if existing is not None:
            return existing, b"", b"", existing.duration_sec, fallback_cache_key, _timings_for_key(db, fallback_cache_key)
        try:
            opus, aac, dur, timings = gemini()
        except RuntimeError as gemini_exc:
            raise RuntimeError(f"Google Chirp lỗi ({chirp_exc}); Gemini dự phòng cũng lỗi: {gemini_exc}") from gemini_exc
        return None, opus, aac, dur, fallback_cache_key, timings


@celery_app.task(name="app.workers.tasks.generate_article_audio", bind=True, max_retries=3)
def generate_article_audio(
    self,
    job_id: str,
    tts_text: str,
    voice: str,
    prompt_version: str,
    cache_key: str,
    fallback_cache_key: str | None = None,
    units: list | None = None,
):
    """Reads an editorial article's OWN cleaned text aloud verbatim —
    deliberately independent of generate_content_podcast (which has Gemini
    WRITE a teaching script about the article's vocab/grammar, never the
    article's actual words). `tts_text` is built by the API route
    (article_extract.article_tts_text: cleaned title + body, one paragraph
    per line) and is exactly what the cache key hashes.

    Voice: Google Chirp 3 HD first, Gemini as the fallback (see _voice_article).
    Both paths use paragraph-aligned chunks with real silence between them,
    per-chunk retry on transient errors, and a progress callback so the client
    can show "2/4" during the (long) first generation. Every later play of
    the same text is a cache hit and never reaches this task."""
    jid = uuid.UUID(job_id)
    with Session(_sync_engine) as db:
        try:
            publish_job_event(db, jid, status="running", progress=0.05, step="Đang chuẩn bị giọng đọc")

            def on_progress(done: int, total: int) -> None:
                if total <= 0:
                    return
                publish_job_event(
                    db,
                    jid,
                    progress=round(0.05 + 0.85 * done / total, 3),
                    step=f"Đang tạo giọng đọc ({done}/{total})" if done < total else "Đang ghép âm thanh",
                )

            reuse, opus_bytes, aac_bytes, duration_sec, stored_key, timings = _voice_article(
                db, tts_text, voice, cache_key, fallback_cache_key, on_progress, units
            )

            if reuse is not None:
                row = reuse
            else:
                publish_job_event(db, jid, progress=0.95, step="Đang lưu âm thanh")
                row = LectureAudio(
                    cache_key=stored_key,
                    opus_path=f"/api/v1/lessons/audio/{stored_key}.opus",
                    aac_path=f"/api/v1/lessons/audio/{stored_key}.aac",
                    opus_data=opus_bytes,
                    aac_data=aac_bytes,
                    prompt_version=prompt_version,
                    voice=voice,
                    duration_sec=duration_sec,
                )
                db.add(row)
                db.commit()

            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất",
                result={
                    "cache_key": stored_key,
                    "opus_path": row.opus_path,
                    "aac_path": row.aac_path,
                    "duration_sec": duration_sec,
                    # read-along: when each sentence starts (None for a
                    # recording made without it)
                    "timings": timings,
                },
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.grade_editorial_outline", bind=True, max_retries=3)
def grade_editorial_outline(self, job_id: str, submission_id: str):
    """Real Gemini feedback on the learner's OWN "luyện dàn ý" attempt —
    previously POST /editorials/{id}/outline only ever persisted the text
    and returned the static, ingestion-time model_outline (a reference
    answer, not feedback on what the learner actually wrote). Runs after
    every save (see app/api/routers/editorial.py submit_editorial_outline).
    """
    jid, sid = uuid.UUID(job_id), uuid.UUID(submission_id)
    with Session(_sync_engine) as db:
        submission = db.get(EditorialOutlineSubmission, sid)
        if submission is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "submission not found"})
            return
        try:
            article = db.get(EditorialArticle, submission.editorial_article_id)
            if article is None:
                raise RuntimeError(f"editorial_article {submission.editorial_article_id} not found")

            publish_job_event(db, jid, status="running", progress=0.2, step="Đang phân tích dàn ý với Gemini")
            feedback = ingestion.generate_outline_feedback(
                article_title=article.title_ko or article.source_name,
                article_body=article.body_ko or "",
                reference_outline=article.model_outline or {},
                learner_outline={
                    "phenomenon_text": submission.phenomenon_text,
                    "cause_text": submission.cause_text,
                    "consequence_text": submission.consequence_text,
                    "solution_text": submission.solution_text,
                },
            )

            submission.feedback_status = "ready"
            submission.feedback_text = feedback
            db.add(submission)
            db.commit()

            publish_job_event(
                db, jid, status="succeeded", progress=1.0, step="Hoàn tất",
                result={"submission_id": submission_id, "feedback_text": feedback},
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            submission.feedback_status = "failed"
            db.add(submission)
            db.commit()
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
def extract_corpus_import(self, job_id: str, batch_id: str, file_b64: str, film_title: str, mime_type: str | None = None):
    """Phụ đề phim -> phân đoạn theo lô cố định (chunk_cues) -> mỗi lô gọi
    Gemini phân loại câu/cụm/mẫu ngữ pháp riêng, KHÔNG dồn cả kịch bản vào
    một prompt (FR-19, Gate G6: kích thước prompt không phình theo độ dài
    kịch bản) -> đề xuất corpus_item vào khu chờ duyệt.

    `mime_type` decides how the raw bytes become text: a real .srt/.vtt file
    or plain text is just decoded, but an image/PDF (a photographed or
    screenshotted subtitle list) goes through a Gemini-vision OCR pass first
    — see ingestion.extract_corpus_source_text. `mime_type=None` (older
    enqueued jobs / callers that don't pass it) falls back to the previous
    plain-decode-only behavior, so this stays backward compatible."""
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
            is_image = bool(mime_type) and (mime_type.startswith("image/") or mime_type == "application/pdf")
            publish_job_event(
                db, jid, status="running", progress=0.1,
                step="Đang nhận diện văn bản từ ảnh" if is_image else "Đang tách câu thoại",
            )

            file_bytes = base64.b64decode(file_b64)
            raw_text = ingestion.extract_corpus_source_text(file_bytes, mime_type)
            if is_image:
                publish_job_event(db, jid, progress=0.4, step="Đang tách câu thoại")
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


@celery_app.task(name="app.workers.tasks.extract_exam_paper_import", bind=True, max_retries=2)
def extract_exam_paper_import(
    self, job_id: str, batch_id: str, files_b64: list[dict], exam_kind: str, session_label: str
):
    """Đề thi ảnh/PDF -> Gemini (đa phương thức) -> đề xuất exam_passage/
    exam_item vào khu chờ duyệt. exam_paper được tạo ngay (giống film,
    exam_kind/session_label do admin gõ chứ không phải AI suy ra) trước khi
    trích xuất; không ghi content.exam_passage/exam_item ở đây — chỉ POST
    /imports/{id}/confirm mới ghi.

    `files_b64` is 1-3 files (Studio's exam_paper upload now accepts a
    reading-passage file, a listening/writing file, and/or a separate
    answer-key file for the SAME đề thi — see app/api/routers/ingest.py),
    each `{"data": <base64>, "mime_type": <str>}`. All of them go into ONE
    Gemini call as separate multimodal parts (ingestion.build_exam_prompt_parts)
    so it can cross-reference an answer key against the actual questions."""
    jid, bid = uuid.UUID(job_id), uuid.UUID(batch_id)
    with Session(_sync_engine) as db:
        batch = db.get(ImportBatch, bid)
        if batch is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "batch not found"})
            return
        try:
            paper = ingestion.find_or_create_exam_paper(db, batch.owner_id, batch.file_hash, exam_kind, session_label)
            batch.exam_paper_id = paper.id
            batch.status = "extracting"
            db.add(batch)
            db.commit()
            publish_job_event(db, jid, status="running", progress=0.2, step="Đang đọc đề thi với Gemini")

            files = [(base64.b64decode(f["data"]), f["mime_type"]) for f in files_b64]
            staged, flagged = ingestion.run_exam_extraction(db, batch, files)

            batch.status = "awaiting_review"
            batch.flagged_count = flagged
            db.add(batch)
            db.commit()
            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất trích xuất đề thi",
                result={"import_batch_id": str(bid), "exam_paper_id": str(paper.id), "staged": staged, "flagged": flagged},
            )
        except Exception as exc:  # noqa: BLE001
            batch.status = "failed"
            db.add(batch)
            db.commit()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.extract_editorial_import", bind=True, max_retries=2)
def extract_editorial_import(
    self,
    job_id: str,
    batch_id: str,
    source_url: str,
    source_name: str,
    title_ko: str | None,
    published_date_iso: str | None = None,
):
    """URL bài xã luận/chuyên mục -> tải + làm sạch HTML (không gọi AI ở
    bước này) -> Gemini phân loại từ vựng/ngữ pháp/cấp độ + soạn dàn ý mẫu
    -> đề xuất vào khu chờ duyệt (import_item, kind=editorial_meta/
    vocab_item/grammar_point). editorial_article được tạo ngay (giống
    film/exam_paper — source_url/source_name do admin/candidate cung cấp
    chứ không phải AI suy ra); body_ko/level_estimate/... vẫn NULL cho
    tới khi POST /imports/{id}/confirm ghi vào (app/services/ingestion.py
    apply_editorial_batch)."""
    jid, bid = uuid.UUID(job_id), uuid.UUID(batch_id)
    with Session(_sync_engine) as db:
        batch = db.get(ImportBatch, bid)
        if batch is None:
            publish_job_event(db, jid, status="failed", error={"code": "not_found", "message": "batch not found"})
            return
        try:
            publish_job_event(db, jid, status="running", progress=0.1, step="Đang tải bài viết")
            fetched = ingestion.fetch_article(source_url)
            body_ko, parsed_title, suggested_source_name = fetched.body, fetched.title, fetched.site_name
            resolved_title = title_ko or parsed_title
            published_date = datetime.fromisoformat(published_date_iso) if published_date_iso else None

            article = ingestion.find_or_create_editorial_article(
                db,
                source_url=source_url,
                source_name=source_name,
                title_ko=resolved_title,
                published_date=published_date,
            )
            batch.editorial_article_id = article.id
            batch.status = "extracting"
            db.add(batch)
            db.commit()
            publish_job_event(db, jid, progress=0.4, step="Đang phân tích với Gemini")

            staged, flagged = ingestion.run_editorial_extraction(
                db, batch, body_ko, resolved_title, suggested_source_name, fetched.images
            )

            batch.status = "awaiting_review"
            batch.flagged_count = flagged
            db.add(batch)
            db.commit()
            publish_job_event(
                db,
                jid,
                status="succeeded",
                progress=1.0,
                step="Hoàn tất phân tích bài xã luận",
                result={
                    "import_batch_id": str(bid),
                    "editorial_article_id": str(article.id),
                    "staged": staged,
                    "flagged": flagged,
                },
            )
        except Exception as exc:  # noqa: BLE001
            batch.status = "failed"
            db.add(batch)
            db.commit()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.refresh_editorial_images", bind=True, max_retries=1)
def refresh_editorial_images(self, article_id: str):
    """Re-fetches one already-published article's source page and stores ONLY
    its photos (+captions). Never touches body_ko/vocab/grammar — those were
    reviewed by an admin. Queued lazily by GET /editorials/{id} the first
    time an article imported before the `images` column existed is read
    (images_fetched_at IS NULL). images_fetched_at is stamped even when the
    fetch fails or finds no photo, so a broken source page is tried once, not
    on every page view; an admin can still force a full re-import."""
    aid = uuid.UUID(article_id)
    with Session(_sync_engine) as db:
        article = db.get(EditorialArticle, aid)
        if article is None:
            return {"ok": False, "reason": "not_found"}
        images: list[dict] = []
        try:
            images = ingestion.fetch_article(article.source_url).images
        except Exception as exc:  # noqa: BLE001 — a dead source page must not retry-loop
            print(f"[refresh-images] {article.source_url}: {exc}", flush=True)
        article.images = images
        article.images_fetched_at = datetime.now(timezone.utc)
        db.add(article)
        db.commit()
        return {"ok": True, "images": len(images)}


@celery_app.task(name="app.workers.tasks.generate_study_pack", bind=True, max_retries=0)
def generate_study_pack(self, article_id: str):
    """Builds the beginner study pack (summary, per-sentence translation +
    word breakdown, simplified-Korean paragraphs) for one article and stores
    it on the row. Queued lazily by GET /editorials/{id} (and after an
    import is applied); runs ONCE per (article text, prompt version) —
    every reader afterwards just reads the stored JSON, so Gemini is never
    called per view. `study_updated_at` is touched after every Gemini batch:
    the API treats a 'pending' pack with no heartbeat for a while as dead
    (worker killed by a redeploy) and re-queues it."""
    aid = uuid.UUID(article_id)
    with Session(_sync_engine) as db:
        article = db.get(EditorialArticle, aid)
        if article is None or not (article.body_ko or "").strip():
            return {"ok": False, "reason": "not_found"}

        cleaned = article_extract.clean_article_text(article.body_ko)
        paragraphs = article_extract.split_paragraphs(cleaned)
        if not paragraphs:
            article.study_status = "failed"
            article.study_updated_at = datetime.now(timezone.utc)
            db.add(article)
            db.commit()
            return {"ok": False, "reason": "no_body"}

        article.study_status = "pending"
        article.study_updated_at = datetime.now(timezone.utc)
        db.add(article)
        db.commit()

        def heartbeat(done: int, total: int) -> None:
            article.study_updated_at = datetime.now(timezone.utc)
            db.add(article)
            db.commit()

        try:
            pack = study_pack.build_study_pack(
                article.title_ko,
                paragraphs,
                study_pack.make_gemini_generate(settings.GEMINI_MODEL_STUDY, gemini_client.generate_structured),
                text_sig=study_pack.study_text_sig(cleaned),
                on_progress=heartbeat,
            )
            article.study_pack = pack
            article.study_status = "ready"
        except Exception as exc:  # noqa: BLE001 — recorded, retried later by the API
            print(f"[study-pack] {article.source_url}: {exc}", flush=True)
            db.rollback()
            article = db.get(EditorialArticle, aid)
            article.study_status = "failed"
        article.study_updated_at = datetime.now(timezone.utc)
        db.add(article)
        db.commit()
        return {"ok": article.study_status == "ready", "status": article.study_status}


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
            # Prepare the beginner study pack right away, so the first learner
            # to open a freshly published article doesn't wait for it.
            if batch.kind == "editorial_article" and batch.editorial_article_id is not None:
                try:
                    generate_study_pack.delay(str(batch.editorial_article_id))
                except Exception as exc:  # noqa: BLE001 — the API also queues it lazily on first read
                    print(f"[study-pack] could not queue after import: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            publish_job_event(
                db, jid, status="failed", error={"code": "internal_error", "message": str(exc), "retryable": True}
            )
            raise


@celery_app.task(name="app.workers.tasks.discover_editorial_candidates")
def discover_editorial_candidates():
    """Celery-beat periodic task (Phase 2 của tính năng đọc xã luận): kéo
    RSS của các nguồn đã đăng ký (editorial.editorial_source), lọc theo
    từ khóa chủ đề, và tạo sẵn editorial_candidate ở trạng thái "new" để
    admin duyệt trong Studio — không bao giờ tự động publish, chỉ tự động
    hoá bước tìm kiếm (xem app/services/ingestion.py discover_editorial_
    candidates). Không có Job/job_id — cùng dạng "chạy nền, không ai chờ
    kết quả trực tiếp" như reset_daily_ai_quota, vì không gọi Gemini."""
    with Session(_sync_engine) as db:
        return ingestion.discover_editorial_candidates(db)


@celery_app.task(name="app.workers.tasks.reset_daily_ai_quota")
def reset_daily_ai_quota():
    """pg_cron-equivalent scheduled task (Celery beat), replaces the
    SDD's Redis TTL-based daily quota reset. No-op scaffold: wire up the
    actual per-learner counter reset once the quota table is added.
    """
    return {"ok": True}
