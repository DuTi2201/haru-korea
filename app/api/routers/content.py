"""Learner-facing read endpoints: content (topics/lessons), curriculum
(goal/placement/plan) and practice (quick-checks). Grouped in one router
for the scaffold; split into content.py/curriculum.py/practice.py once
each grows past a handful of routes.
"""
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_profile
from app.db import get_db
from app.models import Profile
from app.schemas import ProfileOut

router = APIRouter(tags=["content"])


@router.get("/topics")
async def list_topics(db: Annotated[AsyncSession, Depends(get_db)]):
    # TODO: query content.topic once that table/module is built.
    return []


@router.get("/me/plan")
async def get_my_plan(profile: Annotated[Profile, Depends(get_current_profile)]):
    # TODO: derive today's plan from curriculum module once implemented;
    # for now the Lovable /today screen can render its own mock plan.
    return {"learner_id": str(profile.id), "tasks": []}


@router.patch("/me/goal", response_model=ProfileOut)
async def set_my_goal(
    goal: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    profile.goal = goal
    await db.commit()
    await db.refresh(profile)
    return profile
