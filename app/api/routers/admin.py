from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_role
from app.db import get_db
from app.models import AppConfig, Profile

router = APIRouter(prefix="/admin", tags=["admin"])

_admin_only = require_role("admin")


@router.get("/config/{name}")
async def get_config(
    name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(_admin_only)],
):
    result = await db.execute(
        select(AppConfig).where(AppConfig.name == name, AppConfig.active.is_(True))
    )
    return result.scalar_one_or_none()


@router.get("/ai-usage")
async def ai_usage(profile: Annotated[Profile, Depends(_admin_only)]):
    # TODO: aggregate analytics.learning_event / job cost logs once the
    # AI Gateway logs cost per call (see services/gemini_client.py TODO).
    return {"note": "not yet implemented — wire up cost aggregation here"}
