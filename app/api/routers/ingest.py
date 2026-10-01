"""Import / review-queue module — data NEVER auto-writes to main
tables (SDD principle: "dữ liệu vào hệ thống chỉ qua khu chờ duyệt").
This router covers the generic import_batch/import_item flow that backs
the Studio "lô nhập" screen: lesson image/PDF ingestion (FR-42..FR-51),
film-subtitle/corpus ingestion (FR-17..FR-21, Gate G6), and exam-paper
ingestion (content.exam_paper/exam_passage/exam_item), all staging
Gemini's proposals into import_item for a human editor/admin to review,
edit and confirm before anything reaches content.*/corpus.*.

File handling: uploads are NOT persisted to disk/object storage (none is
provisioned yet) — they're base64-encoded and passed straight through
Celery/Redis to the extraction task, capped at MAX_INGEST_FILE_MB. Fine
for admin-tool volumes (occasional lesson pages / one subtitle file per
film); swap for real object storage if ingestion volume or file sizes
grow past what's comfortable in a Celery message.
"""
import base64
import hashlib
import mimetypes
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, require_role
from app.core.config import settings
from app.db import get_db
from app.models import CorpusItem, Film, ImportBatch, ImportItem, Job, Profile
from app.schemas import (
    CorpusEnrichmentStatusOut,
    HiddenCorpusItemOut,
    ImportBatchAccepted,
    ImportBatchOut,
    ImportItemOut,
    ImportItemPatch,
    JobAccepted,
)
from app.services import corpus_enrich, ingestion, lesson_extract
from app.workers.tasks import (
    apply_import_batch_task,
    enrich_corpus_items,
    extract_corpus_import,
    extract_editorial_import,
    extract_exam_paper_import,
    extract_lesson_import,
)

router = APIRouter(prefix="/imports", tags=["ingest"])

_editor_or_admin = require_role("editor", "admin")

_LESSON_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf", "text/plain"}
_MAX_BYTES = settings.MAX_INGEST_FILE_MB * 1024 * 1024


async def start_editorial_import(
    db: AsyncSession,
    profile: Profile,
    source_url: str,
    source_name: str,
    title_ko: str | None,
    published_date: datetime | None = None,
    *,
    force: bool = False,
) -> ImportBatchAccepted:
    """Shared by POST /imports?kind=editorial_article (admin types a URL
    directly) and POST /editorial-candidates/{id}/ingest (admin picks one
    of the RSS-discovered candidates instead — see app/api/routers/
    editorial.py). No file to hash here, so `source_url` itself is the
    dedup identity, same slot `file_hash` fills for the other three kinds:
    resubmitting the same URL reuses the existing batch/job instead of
    firing a second Gemini extraction.

    `force=True` skips that reuse and always stages a FRESH batch (still
    against the SAME editorial_article row — find_or_create_editorial_article
    dedupes by source_url regardless) — the one legitimate reason to want a
    second extraction of a URL already ingested: re-scraping/re-analyzing to
    pick up a pipeline fix (e.g. the improved fetch_article_text boilerplate
    stripping + og:site_name outlet-name suggestion), not a duplicate."""
    source_url = source_url.strip()
    file_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    idempotency_key = f"editorial_article:{file_hash}"
    if force:
        idempotency_key = f"{idempotency_key}:refresh:{uuid.uuid4()}"

    existing_batch = (
        None
        if force
        else (
            await db.execute(
                select(ImportBatch).where(ImportBatch.kind == "editorial_article", ImportBatch.file_hash == file_hash)
            )
        ).scalar_one_or_none()
    )
    batch = existing_batch
    if batch is None:
        batch = ImportBatch(
            kind="editorial_article",
            owner_id=profile.id,
            status="queued",
            source_file=source_url,
            file_hash=file_hash,
        )
        db.add(batch)
        await db.commit()
        await db.refresh(batch)

    existing_job = await db.execute(
        select(Job).where(
            Job.type == "extract_import_editorial_article", Job.idempotency_key == idempotency_key
        )
    )
    job = existing_job.scalar_one_or_none()
    if job is not None and job.status not in ("queued", "running", "succeeded"):
        # A failed extraction (bad URL, Gemini error, ...) must not
        # permanently block retrying the same URL — `idempotency_key` has a
        # unique constraint per job type, so leaving the old row in place
        # would 500 (IntegrityError) on every future attempt instead of
        # actually retrying.
        await db.delete(job)
        await db.flush()
        job = None

    if job is None:
        job = Job(
            type="extract_import_editorial_article",
            owner_id=profile.id,
            idempotency_key=idempotency_key,
            status="queued",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        extract_editorial_import.delay(
            str(job.id),
            str(batch.id),
            source_url,
            source_name,
            title_ko,
            published_date.isoformat() if published_date else None,
        )

    return ImportBatchAccepted(
        import_batch_id=batch.id,
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@router.post("", response_model=ImportBatchAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_import_batch(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    kind: Annotated[Literal["lesson", "corpus", "exam_paper", "editorial_article"], Form()],
    file: UploadFile | None = None,
    files: list[UploadFile] | None = None,
    film_title: Annotated[str | None, Form()] = None,
    exam_kind: Annotated[str | None, Form()] = None,
    session_label: Annotated[str | None, Form()] = None,
    source_url: Annotated[str | None, Form()] = None,
    source_name: Annotated[str | None, Form()] = None,
    title_ko: Annotated[str | None, Form()] = None,
    force: Annotated[bool, Form()] = False,
):
    """kind="corpus" requires film_title (which film this subtitle file is
    filed under; created on first use, per Film.title being the SRS's
    only field for it). kind="exam_paper" requires exam_kind + session_label
    (content.exam_paper's admin-typed fields — e.g. "TOPIK II" / "64회 읽기").
    kind="editorial_article" takes a URL instead of a file upload — no
    `file` at all, so it's handled separately before any of the upload
    validation below runs.

    "lesson"/"exam_paper" accept text/plain as well as image/PDF — Gemini's
    Part.from_bytes is mime-agnostic (see gemini_client.part_from_bytes),
    so a plain-text lesson (typed/pasted, or a real .txt file) goes through
    the exact same multimodal extraction call as a photographed page,
    letting an admin skip the photo step entirely when they already have
    the lesson as text.

    "corpus" (phụ đề phim) has no mime whitelist — a real .srt/.vtt file
    often reports an unregistered/empty content_type in the browser, so
    enforcing one here would reject legitimate subtitle uploads. It DOES
    now compute mime_type (same as lesson/exam_paper) and pass it through
    to extraction: an image/PDF upload (a photographed/screenshotted
    subtitle list) gets OCR'd via Gemini vision first instead of being
    silently base64-decoded as garbage text (see
    ingestion.extract_corpus_source_text).

    "exam_paper" is the one kind that takes MULTIPLE files under the same
    `files` field — a real TOPIK paper is often split into a reading-passage
    file, a listening/writing file, and a separate answer-key file, and an
    admin shouldn't have to run three separate uploads/reviews for what is
    really one đề thi. `file` (singular) still works for a single-file exam
    upload too, for backward compat."""
    if kind == "editorial_article":
        if not (source_url and source_name):
            raise _problem(
                status.HTTP_400_BAD_REQUEST,
                "source_url and source_name are required for kind=editorial_article",
                "validation_error",
            )
        return await start_editorial_import(db, profile, source_url, source_name, title_ko, force=force)

    if kind == "exam_paper":
        return await _create_exam_paper_batch(db, profile, files or ([file] if file else []), exam_kind, session_label)

    if file is None:
        raise _problem(status.HTTP_400_BAD_REQUEST, "file is required for this kind", "validation_error")

    if kind == "corpus" and not film_title:
        raise _problem(status.HTTP_400_BAD_REQUEST, "film_title is required for kind=corpus", "validation_error")

    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise _problem(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "File too large",
            "file_too_large",
            f"limit is {settings.MAX_INGEST_FILE_MB} MB",
        )
    mime_type: str | None = file.content_type or mimetypes.guess_type(file.filename or "")[0]
    if kind == "lesson" and mime_type not in _LESSON_MIME_TYPES:
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Unsupported file type",
            "unsupported_media_type",
            f"expected one of {sorted(_LESSON_MIME_TYPES)}, got {mime_type!r}",
        )

    file_hash = hashlib.sha256(raw).hexdigest()

    # Dedup: re-uploading the exact same file returns the existing batch's
    # job instead of re-running Gemini (SDD: "AI vừa đắt vừa chậm... không
    # tạo job trùng"). Only useful once that batch already has a job to
    # point back to, which is why we look up by (kind, file_hash) AND a
    # still-live job row for the same idempotency key.
    existing_batch = await db.execute(
        select(ImportBatch).where(ImportBatch.kind == kind, ImportBatch.file_hash == file_hash)
    )
    batch = existing_batch.scalar_one_or_none()

    if batch is None:
        batch = ImportBatch(
            kind=kind, owner_id=profile.id, status="queued", source_file=file.filename, file_hash=file_hash
        )
        db.add(batch)
        await db.commit()
        await db.refresh(batch)

    lookup_keys, idempotency_key = upload_job_keys(kind, file_hash, batch.status)
    existing_job = await db.execute(
        select(Job).where(Job.type == f"extract_import_{kind}", Job.idempotency_key.in_(lookup_keys))
    )
    job = existing_job.scalars().first()
    if job is not None and job.status not in ("queued", "running", "succeeded"):
        # See the editorial_article branch above: a failed row must not
        # permanently block retrying the same file upload.
        await db.delete(job)
        await db.flush()
        job = None

    if job is None:
        job = Job(
            type=f"extract_import_{kind}", owner_id=profile.id, idempotency_key=idempotency_key, status="queued"
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        file_b64 = base64.b64encode(raw).decode("ascii")
        if kind == "lesson":
            extract_lesson_import.delay(str(job.id), str(batch.id), file_b64, mime_type)
        else:
            extract_corpus_import.delay(str(job.id), str(batch.id), file_b64, film_title, mime_type)

    return ImportBatchAccepted(
        import_batch_id=batch.id,
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


def upload_job_keys(kind: str, file_hash: str, batch_status: str) -> tuple[list[str], str]:
    """(idempotency keys that count as "this upload already ran", key for a new job).

    A lesson's key carries the extraction version, so a file that was read by an
    older (less complete) extraction is read again when it is re-uploaded — onto
    the same batch, which still holds only unreviewed proposals. A confirmed
    batch is the exception: its lesson is already in the library, so re-reading
    would only propose duplicates, and any earlier job (old key included) still
    answers the upload."""
    plain = f"{kind}:{file_hash}"
    if kind != "lesson":
        return [plain], plain
    versioned = f"{plain}:{lesson_extract.PROMPT_VERSION}"
    if batch_status == "confirmed":
        return [versioned, plain], versioned
    return [versioned], versioned


async def _create_exam_paper_batch(
    db: AsyncSession,
    profile: Profile,
    upload_files: list[UploadFile],
    exam_kind: str | None,
    session_label: str | None,
) -> ImportBatchAccepted:
    """Reading passage / listening+writing / answer-key files (1-3 of
    them) for ONE đề thi, all staged as a single import_batch and sent to
    Gemini as separate multimodal parts of ONE call (see
    ingestion.build_exam_prompt_parts) so it can cross-reference an answer
    key against the actual questions. `file_hash` covers the whole set
    (sorted by filename first, so upload ORDER never changes the dedup
    identity), which is what both ImportBatch's and ExamPaper's
    (owner_id, file_hash) uniqueness keys off."""
    if not upload_files:
        raise _problem(status.HTTP_400_BAD_REQUEST, "file is required for this kind", "validation_error")
    if len(upload_files) > 3:
        raise _problem(
            status.HTTP_400_BAD_REQUEST, "At most 3 files per exam paper", "validation_error"
        )
    if not (exam_kind and session_label):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "exam_kind and session_label are required for kind=exam_paper",
            "validation_error",
        )

    read_files: list[tuple[bytes, str, str]] = []  # (raw, mime_type, filename)
    for f in upload_files:
        raw = await f.read()
        if len(raw) > _MAX_BYTES:
            raise _problem(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "File too large",
                "file_too_large",
                f"{f.filename}: limit is {settings.MAX_INGEST_FILE_MB} MB",
            )
        mime_type = f.content_type or mimetypes.guess_type(f.filename or "")[0]
        if mime_type not in _LESSON_MIME_TYPES:
            raise _problem(
                status.HTTP_400_BAD_REQUEST,
                "Unsupported file type",
                "unsupported_media_type",
                f"{f.filename}: expected one of {sorted(_LESSON_MIME_TYPES)}, got {mime_type!r}",
            )
        read_files.append((raw, mime_type, f.filename or "file"))

    read_files.sort(key=lambda t: t[2])
    file_hash = hashlib.sha256(b"".join(r for r, _, _ in read_files)).hexdigest()
    source_label = ", ".join(fn for _, _, fn in read_files)

    existing_batch = await db.execute(
        select(ImportBatch).where(ImportBatch.kind == "exam_paper", ImportBatch.file_hash == file_hash)
    )
    batch = existing_batch.scalar_one_or_none()
    idempotency_key = f"exam_paper:{file_hash}"

    if batch is None:
        batch = ImportBatch(
            kind="exam_paper",
            owner_id=profile.id,
            status="queued",
            source_file=source_label,
            file_hash=file_hash,
        )
        db.add(batch)
        await db.commit()
        await db.refresh(batch)

    existing_job = await db.execute(
        select(Job).where(Job.type == "extract_import_exam_paper", Job.idempotency_key == idempotency_key)
    )
    job = existing_job.scalar_one_or_none()
    if job is not None and job.status not in ("queued", "running", "succeeded"):
        # See the editorial_article branch above: a failed row must not
        # permanently block retrying the same upload.
        await db.delete(job)
        await db.flush()
        job = None

    if job is None:
        job = Job(
            type="extract_import_exam_paper", owner_id=profile.id, idempotency_key=idempotency_key, status="queued"
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        files_b64 = [
            {"data": base64.b64encode(raw).decode("ascii"), "mime_type": mime_type}
            for raw, mime_type, _ in read_files
        ]
        extract_exam_paper_import.delay(str(job.id), str(batch.id), files_b64, exam_kind, session_label)

    return ImportBatchAccepted(
        import_batch_id=batch.id,
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@router.get("", response_model=list[ImportBatchOut])
async def list_import_batches(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    kind: Annotated[Literal["lesson", "corpus", "exam_paper", "editorial_article"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
):
    """Backs the Studio screen's batch list — newest first."""
    query = select(ImportBatch).order_by(ImportBatch.created_at.desc()).limit(limit)
    if kind:
        query = query.where(ImportBatch.kind == kind)
    result = await db.execute(query)
    return result.scalars().all()


# ------------------------------------------------- corpus enrichment (Studio) --
# Declared BEFORE the "/{import_id}" routes below: a literal path must come
# first or "/imports/corpus-enrichment" would be read as an import id.
_ENRICH_JOB_TYPE = "enrich_corpus_items"
# A run that has not reported progress for this long is treated as dead (its
# worker was killed), so it cannot block a new run forever.
_ENRICH_STALE_AFTER = timedelta(minutes=15)


async def _count(db: AsyncSession, *conditions) -> int:
    query = select(func.count()).select_from(CorpusItem)
    for condition in conditions:
        query = query.where(condition)
    return int((await db.execute(query)).scalar_one())


async def _active_enrichment_job(db: AsyncSession) -> Job | None:
    fresh = datetime.now(timezone.utc) - _ENRICH_STALE_AFTER
    return (
        await db.execute(
            select(Job)
            .where(Job.type == _ENRICH_JOB_TYPE, Job.status.in_(["queued", "running"]), Job.updated_at > fresh)
            .order_by(Job.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


@router.get("/corpus-enrichment", response_model=CorpusEnrichmentStatusOut)
async def corpus_enrichment_status(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """How much of the corpus has a Vietnamese meaning, usage note and
    naturalness verdict — what the Studio "Làm giàu kho câu" card shows."""
    total = await _count(db)
    pending = await _count(db, ingestion.corpus_enrichment_pending_filter())
    active = await _active_enrichment_job(db)
    return CorpusEnrichmentStatusOut(
        version=corpus_enrich.ENRICH_VERSION,
        total=total,
        pending=pending,
        enriched=total - pending,
        hidden=await _count(db, CorpusItem.naturalness == corpus_enrich.UNNATURAL),
        awkward=await _count(db, CorpusItem.naturalness == corpus_enrich.AWKWARD),
        active_job_id=active.id if active else None,
    )


@router.post("/corpus-enrichment", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def start_corpus_enrichment(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """Starts the background run that adds a meaning, usage note, naturalness
    verdict and topics to every sentence that lacks them. One run at a time:
    pressing the button while a run is going returns that run. Resumable — a
    later run only does what is still missing."""
    job = await _active_enrichment_job(db)
    if job is None:
        if await _count(db, ingestion.corpus_enrichment_pending_filter()) == 0:
            raise _problem(
                status.HTTP_409_CONFLICT,
                "Nothing to enrich",
                "nothing_pending",
                "Mọi câu trong kho đã có nghĩa và ghi chú cách dùng.",
            )
        job = Job(type=_ENRICH_JOB_TYPE, owner_id=profile.id, status="queued")
        db.add(job)
        await db.commit()
        await db.refresh(job)
        enrich_corpus_items.delay(str(job.id))
    return JobAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@router.get("/corpus-enrichment/hidden", response_model=list[HiddenCorpusItemOut])
async def list_hidden_corpus_items(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    limit: Annotated[int, Query(ge=1, le=300)] = 100,
):
    """Sentences the model judged unnatural and learners therefore never see —
    listed so an editor can check the filter and restore a good line."""
    rows = (
        await db.execute(
            select(CorpusItem.id, CorpusItem.film_id, CorpusItem.text_ko)
            .where(CorpusItem.naturalness == corpus_enrich.UNNATURAL)
            .order_by(CorpusItem.film_id, CorpusItem.id)
            .limit(limit)
        )
    ).all()
    film_ids = {r[1] for r in rows}
    films = dict((await db.execute(select(Film.id, Film.title).where(Film.id.in_(film_ids)))).all()) if film_ids else {}
    return [HiddenCorpusItemOut(id=r[0], film_title=films.get(r[1], ""), text_ko=r[2]) for r in rows]


@router.post("/corpus-enrichment/hidden/{item_id}/restore", status_code=status.HTTP_204_NO_CONTENT)
async def restore_hidden_corpus_item(
    item_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """An editor says this line is fine: learners see it again, the model's
    verdict can no longer hide it, and the next enrichment run adds its meaning."""
    item = await db.get(CorpusItem, item_id)
    if item is None or item.naturalness != corpus_enrich.UNNATURAL:
        raise _problem(status.HTTP_404_NOT_FOUND, "Hidden corpus item not found", "not_found")
    item.naturalness = corpus_enrich.APPROVED
    item.enriched_version = None  # so the next run gives it a meaning
    await db.commit()


@router.get("/{import_id}", response_model=ImportBatchOut)
async def get_import_batch(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    batch = await db.get(ImportBatch, import_id)
    if batch is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import batch not found", "not_found")
    return batch


@router.get("/{import_id}/items", response_model=list[ImportItemOut])
async def list_import_items(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
):
    query = select(ImportItem).where(ImportItem.import_batch_id == import_id)
    if status_filter:
        query = query.where(ImportItem.status == status_filter)
    result = await db.execute(query)
    return result.scalars().all()


@router.patch("/{import_id}/items/{item_id}", response_model=ImportItemOut)
async def patch_import_item(
    import_id: uuid.UUID,
    item_id: uuid.UUID,
    body: ImportItemPatch,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    item = await db.get(ImportItem, item_id)
    if item is None or item.import_batch_id != import_id:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import item not found", "not_found")
    if body.payload is not None:
        item.payload = body.payload
    if body.status is not None:
        item.status = body.status  # human confirm — always required, even for green/high-confidence items
    await db.commit()
    await db.refresh(item)
    return item


@router.post("/{import_id}/confirm", response_model=ImportBatchAccepted, status_code=status.HTTP_202_ACCEPTED)
async def confirm_import_batch(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """Writes confirmed import_items into the real content/corpus tables.
    Runs as a Celery job (not inline) because the corpus path's last step
    is a Gemini embedding call per item — every AI call goes through
    Celery, confirm included. Safe to call again after a reviewer
    confirms more items later: already-applied rows are skipped (see
    app/services/ingestion.py apply_lesson_batch/apply_corpus_batch)."""
    batch = await db.get(ImportBatch, import_id)
    if batch is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import batch not found", "not_found")

    job = Job(type="apply_import_batch", owner_id=profile.id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    apply_import_batch_task.delay(str(job.id), str(batch.id))

    return ImportBatchAccepted(
        import_batch_id=batch.id,
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


@router.post("/{import_id}/rollback", response_model=ImportBatchOut)
async def rollback_import_batch(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """See app/services/ingestion.py rollback_import_batch for what "undo"
    means per kind — in particular, a confirmed `corpus` batch can't be
    rolled back (corpus_item has no import_item_id lineage per the SRS)."""
    batch = await db.get(ImportBatch, import_id)
    if batch is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import batch not found", "not_found")

    try:
        await ingestion.rollback_import_batch(db, batch)
    except ValueError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "Cannot roll back", "rollback_conflict", str(exc)) from exc

    return batch
