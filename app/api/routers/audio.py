"""Audio module — the ONE fully-wired example of the SDD's async-job
pattern end to end: client POSTs -> 202 + job_id -> Celery task runs ->
Redis publishes progress -> SSE streams it -> row lands in
audio.lecture_audio (or audio.corpus_item_audio). Every other AI-backed
endpoint (writing grading, exam ingestion, placement scoring) should
follow this exact shape.

Two resources share this pattern:
- lecture audio: whole-lesson narration, gated behind login (Job.owner_id
  is required) — matches the existing /lessons/{id} learner-progress flow.
- corpus-item audio: one listening-screen sentence. Playback itself needs
  no account, same as GET /corpus/items — only the "đã nhớ/chưa nhớ"
  progress buttons require login, not hearing a sentence read aloud.

Both are content-addressed (`cache_key = f"{id}:{voice}:{prompt_version}"`)
so a repeat request short-circuits to the already-succeeded job instead of
re-billing Gemini, and both are served back out through the GET streaming
routes below — the frontend just points an <audio> tag at `opus_path`/
`aac_path`, no separate fetch-then-blob dance needed.
"""
import hashlib
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_profile
from app.db import get_db
from app.models import CorpusItemAudio, GrammarPoint, Job, LectureAudio, Lesson, Profile, VocabItem, VocabItemAudio
from app.schemas import CorpusAudioRequest, JobAccepted, LectureAudioRequest, PodcastRequest, VocabAudioRequest
from app.workers.tasks import (
    generate_content_podcast,
    generate_corpus_audio,
    generate_lecture_audio,
    generate_vocab_audio,
)

router = APIRouter(prefix="/lessons", tags=["audio"])
corpus_router = APIRouter(prefix="/corpus", tags=["audio"])
vocab_router = APIRouter(prefix="/vocab-items", tags=["audio"])

_CACHE_CONTROL = "public, max-age=31536000, immutable"  # cache_key is content-addressed — never changes once written


async def _podcast_cache_key(
    db: AsyncSession, owner_kind: str, owner_id: str, voice: str, prompt_version: str, vocab_ids: list[int], grammar_ids: list[int]
) -> str:
    """Includes a short hash of the CURRENT vocab/grammar id set so the
    cache naturally busts if an admin adds/removes a word later — unlike
    plain lecture/corpus/vocab audio, a podcast's content can change out
    from under a stable owner id (a lesson/article gets edited in Studio),
    so `{owner_id}:{voice}:{prompt_version}` alone would keep serving a
    stale recording forever."""
    sig_src = "v:" + ",".join(str(i) for i in sorted(vocab_ids)) + "|g:" + ",".join(str(i) for i in sorted(grammar_ids))
    content_sig = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    return f"podcast:{owner_kind}:{owner_id}:{voice}:{prompt_version}:{content_sig}"


async def _request_podcast(
    db: AsyncSession,
    owner_kind: str,
    owner_id: str,
    body: PodcastRequest,
    request: Request,
    vocab_ids: list[int],
    grammar_ids: list[int],
) -> JobAccepted:
    cache_key = await _podcast_cache_key(db, owner_kind, owner_id, body.voice, body.prompt_version, vocab_ids, grammar_ids)

    cached = await db.execute(select(LectureAudio).where(LectureAudio.cache_key == cache_key))
    hit = cached.scalar_one_or_none()

    idempotency_key = request.headers.get("Idempotency-Key", cache_key)
    existing_job = await db.execute(
        select(Job).where(
            Job.type == "generate_content_podcast",
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running", "succeeded"]),
        )
    )
    job = existing_job.scalar_one_or_none()

    if job is None:
        job = Job(
            type="generate_content_podcast",
            owner_id=None,
            idempotency_key=idempotency_key,
            status="succeeded" if hit else "queued",
            progress=1.0 if hit else 0.0,
            result={
                "cache_key": hit.cache_key,
                "opus_path": hit.opus_path,
                "aac_path": hit.aac_path,
                "duration_sec": hit.duration_sec,
                "script_text": hit.script_text,
            }
            if hit
            else None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        if not hit:
            generate_content_podcast.delay(
                str(job.id), owner_kind, owner_id, body.voice, body.prompt_version, cache_key
            )

    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


def _split_cache_filename(filename: str) -> tuple[str, str]:
    """"{cache_key}.opus" / "{cache_key}.aac" -> (cache_key, media_type)."""
    if filename.endswith(".opus"):
        return filename[: -len(".opus")], "audio/ogg; codecs=opus"
    if filename.endswith(".aac"):
        return filename[: -len(".aac")], "audio/aac"
    raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unsupported audio format")


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


@router.post("/{lesson_id}/podcast", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_lesson_podcast(
    lesson_id: int,
    body: PodcastRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Gemini writes ONE consolidated teaching script covering every vocab
    word + grammar point of this lesson, then TTS's it — see
    app.workers.tasks.generate_content_podcast and ingestion.
    generate_podcast_script. No login required, same rationale as vocab/
    corpus audio (hearing content is a public read)."""
    lesson = await db.get(Lesson, lesson_id)
    if lesson is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Lesson not found")
    vocab_ids = [i for (i,) in (await db.execute(select(VocabItem.id).where(VocabItem.lesson_id == lesson_id))).all()]
    grammar_ids = [
        i for (i,) in (await db.execute(select(GrammarPoint.id).where(GrammarPoint.lesson_id == lesson_id))).all()
    ]
    return await _request_podcast(db, "lesson", str(lesson_id), body, request, vocab_ids, grammar_ids)


@router.get("/audio/{filename}")
async def stream_lecture_audio(filename: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """Serves the bytes behind `LectureAudio.opus_path`/`aac_path` — those
    columns are set to exactly this route at write time (see
    app/workers/tasks.py generate_lecture_audio), so the frontend just
    plays `<audio src={opus_path}>` with no separate fetch-then-blob step.
    """
    cache_key, media_type = _split_cache_filename(filename)
    row = (
        await db.execute(select(LectureAudio).where(LectureAudio.cache_key == cache_key))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not found")
    data = row.opus_data if media_type.startswith("audio/ogg") else row.aac_data
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not yet generated")
    return Response(content=data, media_type=media_type, headers={"Cache-Control": _CACHE_CONTROL})


@corpus_router.post("/{corpus_item_id}/audio", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_corpus_audio(
    corpus_item_id: str,
    body: CorpusAudioRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Same content-addressed cache-hit-shortcut as lecture audio, but for
    one /listening sentence. No login required — hearing a sentence read
    aloud is a public read like GET /corpus/items; only progress (đã nhớ/
    chưa nhớ) needs an account, so `Job.owner_id` stays null here.
    """
    cache_key = f"{corpus_item_id}:{body.voice}:{body.prompt_version}"

    cached = await db.execute(select(CorpusItemAudio).where(CorpusItemAudio.cache_key == cache_key))
    hit = cached.scalar_one_or_none()

    idempotency_key = request.headers.get("Idempotency-Key", cache_key)

    existing_job = await db.execute(
        select(Job).where(
            Job.type == "generate_corpus_audio",
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running", "succeeded"]),
        )
    )
    job = existing_job.scalar_one_or_none()

    if job is None:
        job = Job(
            type="generate_corpus_audio",
            owner_id=None,
            idempotency_key=idempotency_key,
            status="succeeded" if hit else "queued",
            progress=1.0 if hit else 0.0,
            result={
                "cache_key": hit.cache_key,
                "opus_path": f"/api/v1/corpus/audio/{hit.cache_key}.opus",
                "aac_path": f"/api/v1/corpus/audio/{hit.cache_key}.aac",
                "duration_sec": hit.duration_sec,
            }
            if hit
            else None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        if not hit:
            generate_corpus_audio.delay(
                str(job.id), corpus_item_id, body.text_ko, body.voice, body.prompt_version
            )

    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@corpus_router.get("/audio/{filename}")
async def stream_corpus_audio(filename: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """Serves audio.corpus_item_audio bytes — same shape as
    stream_lecture_audio, keyed by the corpus item's own cache_key."""
    cache_key, media_type = _split_cache_filename(filename)
    row = (
        await db.execute(select(CorpusItemAudio).where(CorpusItemAudio.cache_key == cache_key))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not found")
    data = row.opus_data if media_type.startswith("audio/ogg") else row.aac_data
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not yet generated")
    return Response(content=data, media_type=media_type, headers={"Cache-Control": _CACHE_CONTROL})


@vocab_router.post("/{vocab_item_id}/audio", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def request_vocab_audio(
    vocab_item_id: int,
    body: VocabAudioRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Same content-addressed cache-hit-shortcut as corpus-item audio, but
    for one vocab flashcard's hangul. No login required — same rationale
    as corpus: hearing a word read aloud is a public read, only the "đã
    nhớ/chưa nhớ" progress button needs an account. Covers both the
    lesson-flashcard screen and an editorial article's vocab list, since
    both read from the exact same content.vocab_item rows."""
    cache_key = f"{vocab_item_id}:{body.voice}:{body.prompt_version}"

    cached = await db.execute(select(VocabItemAudio).where(VocabItemAudio.cache_key == cache_key))
    hit = cached.scalar_one_or_none()

    idempotency_key = request.headers.get("Idempotency-Key", cache_key)

    existing_job = await db.execute(
        select(Job).where(
            Job.type == "generate_vocab_audio",
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running", "succeeded"]),
        )
    )
    job = existing_job.scalar_one_or_none()

    if job is None:
        job = Job(
            type="generate_vocab_audio",
            owner_id=None,
            idempotency_key=idempotency_key,
            status="succeeded" if hit else "queued",
            progress=1.0 if hit else 0.0,
            result={
                "cache_key": hit.cache_key,
                "opus_path": f"/api/v1/vocab-items/audio/{hit.cache_key}.opus",
                "aac_path": f"/api/v1/vocab-items/audio/{hit.cache_key}.aac",
                "duration_sec": hit.duration_sec,
            }
            if hit
            else None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        if not hit:
            generate_vocab_audio.delay(
                str(job.id), str(vocab_item_id), body.text_ko, body.voice, body.prompt_version
            )

    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@vocab_router.get("/audio/{filename}")
async def stream_vocab_audio(filename: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """Serves audio.vocab_item_audio bytes — same shape as
    stream_corpus_audio, keyed by the vocab item's own cache_key."""
    cache_key, media_type = _split_cache_filename(filename)
    row = (
        await db.execute(select(VocabItemAudio).where(VocabItemAudio.cache_key == cache_key))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not found")
    data = row.opus_data if media_type.startswith("audio/ogg") else row.aac_data
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio not yet generated")
    return Response(content=data, media_type=media_type, headers={"Cache-Control": _CACHE_CONTROL})
