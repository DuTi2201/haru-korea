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
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Form, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, require_role
from app.core.config import settings
from app.db import get_db
from app.models import ImportBatch, ImportItem, Job, Profile
from app.schemas import ImportBatchAccepted, ImportBatchOut, ImportItemOut, ImportItemPatch
from app.services import ingestion
from app.workers.tasks import (
    apply_import_batch_task,
    extract_corpus_import,
    extract_exam_paper_import,
    extract_lesson_import,
)

router = APIRouter(prefix="/imports", tags=["ingest"])

_editor_or_admin = require_role("editor", "admin")

_LESSON_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
_MAX_BYTES = settings.MAX_INGEST_FILE_MB * 1024 * 1024


@router.post("", response_model=ImportBatchAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_import_batch(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
    file: UploadFile,
    kind: Annotated[Literal["lesson", "corpus", "exam_paper"], Form()],
    film_title: Annotated[str | None, Form()] = None,
    exam_kind: Annotated[str | None, Form()] = None,
    session_label: Annotated[str | None, Form()] = None,
):
    """kind="corpus" requires film_title (which film this subtitle file is
    filed under; created on first use, per Film.title being the SRS's
    only field for it). kind="exam_paper" requires exam_kind + session_label
    (content.exam_paper's admin-typed fields — e.g. "TOPIK II" / "64회 읽기").
    """
    if kind == "corpus" and not film_title:
        raise _problem(status.HTTP_400_BAD_REQUEST, "film_title is required for kind=corpus", "validation_error")
    if kind == "exam_paper" and not (exam_kind and session_label):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "exam_kind and session_label are required for kind=exam_paper",
            "validation_error",
        )

    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise _problem(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "File too large",
            "file_too_large",
            f"limit is {settings.MAX_INGEST_FILE_MB} MB",
        )
    if kind in ("lesson", "exam_paper"):
        mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
        if mime_type not in _LESSON_MIME_TYPES:
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
    idempotency_key = f"{kind}:{file_hash}"

    if batch is None:
        batch = ImportBatch(
            kind=kind, owner_id=profile.id, status="queued", source_file=file.filename, file_hash=file_hash
        )
        db.add(batch)
        await db.commit()
        await db.refresh(batch)

    existing_job = await db.execute(
        select(Job).where(
            Job.type == f"extract_import_{kind}",
            Job.idempotency_key == idempotency_key,
            Job.status.in_(["queued", "running", "succeeded"]),
        )
    )
    job = existing_job.scalar_one_or_none()

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
        elif kind == "corpus":
            extract_corpus_import.delay(str(job.id), str(batch.id), file_b64, film_title)
        else:
            extract_exam_paper_import.delay(str(job.id), str(batch.id), file_b64, mime_type, exam_kind, session_label)

    return ImportBatchAccepted(
        import_batch_id=batch.id,
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
        events_url=f"/api/v1/jobs/{job.id}/events",
    )


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
