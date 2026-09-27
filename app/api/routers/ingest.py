"""Import / review-queue module — data NEVER auto-writes to main
tables (SDD principle: "dữ liệu vào hệ thống chỉ qua khu chờ duyệt").
This router covers the generic import_batch/import_item flow that
backs both the Studio "lô nhập" screen and the exam-ingestion pipeline.
"""
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, require_role
from app.db import get_db
from app.models import ImportBatch, ImportItem, Profile
from app.schemas import ImportBatchOut, ImportItemOut, ImportItemPatch

router = APIRouter(prefix="/imports", tags=["ingest"])

_editor_or_admin = require_role("editor", "admin")


@router.post("", response_model=ImportBatchOut, status_code=status.HTTP_201_CREATED)
async def create_import_batch(
    kind: Literal["lesson", "corpus", "exam_paper"],
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    batch = ImportBatch(kind=kind, owner_id=profile.id, status="queued")
    db.add(batch)
    await db.commit()
    await db.refresh(batch)
    # TODO: dispatch the pass-1/2/3 Gemini ingestion Celery chain here,
    # flipping status queued -> extracting -> validating -> awaiting_review.
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


@router.post("/{import_id}/confirm", response_model=ImportBatchOut)
async def confirm_import_batch(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    """Writes confirmed import_items into the real content/corpus/vocab
    tables in one transaction. Left as a TODO per-kind implementation —
    the important invariant already enforced here is that nothing reaches
    main tables except through this explicit, human-triggered call."""
    batch = await db.get(ImportBatch, import_id)
    if batch is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import batch not found", "not_found")
    batch.status = "confirmed"
    await db.commit()
    await db.refresh(batch)
    return batch


@router.post("/{import_id}/rollback", response_model=ImportBatchOut)
async def rollback_import_batch(
    import_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_editor_or_admin)],
):
    batch = await db.get(ImportBatch, import_id)
    if batch is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Import batch not found", "not_found")
    batch.status = "rolled_back"
    await db.commit()
    await db.refresh(batch)
    return batch
