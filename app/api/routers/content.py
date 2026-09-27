"""Learner-facing read endpoints: content (topics/lessons/vocab/grammar),
corpus (listening sentences), progress (item-state check-ins), curriculum
(goal/placement/plan) and practice (quick-checks). Grouped in one router
for the scaffold; split into content.py/curriculum.py/practice.py once
each grows past a handful of routes.
"""
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile
from app.db import get_db
from app.models import CorpusItem, Film, GrammarPoint, ItemState, Lesson, LessonTopic, Profile, Topic, VocabItem
from app.schemas import (
    CorpusItemOut,
    GrammarPointOut,
    ItemStateOut,
    ItemStateReviewRequest,
    LessonOut,
    ProfileOut,
    TopicOut,
    VocabItemOut,
)

router = APIRouter(tags=["content"])


@router.get("/topics")
async def list_topics(db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(Topic).order_by(Topic.name))
    topics = result.scalars().all()
    return [{"id": t.id, "name": t.name, "quizlet_url": t.quizlet_url} for t in topics]


@router.get("/lessons/{lesson_id}", response_model=LessonOut)
async def get_lesson(lesson_id: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """`lesson_id="today"` is a placeholder "next lesson" heuristic (picks
    the earliest confirmed lesson) — real adaptive next-lesson selection
    is SRS scheduling logic, still stubbed elsewhere in this scaffold; a
    numeric id fetches that exact lesson."""
    if lesson_id == "today":
        lesson = (await db.execute(select(Lesson).order_by(Lesson.id).limit(1))).scalars().first()
    else:
        try:
            lid = int(lesson_id)
        except ValueError as exc:
            raise _problem(status.HTTP_404_NOT_FOUND, "Lesson not found", "not_found") from exc
        lesson = await db.get(Lesson, lid)

    if lesson is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Lesson not found", "not_found")

    topics = (
        (
            await db.execute(
                select(Topic).join(LessonTopic, LessonTopic.topic_id == Topic.id).where(LessonTopic.lesson_id == lesson.id)
            )
        )
        .scalars()
        .all()
    )
    vocab = (
        (await db.execute(select(VocabItem).where(VocabItem.lesson_id == lesson.id).order_by(VocabItem.id)))
        .scalars()
        .all()
    )
    grammar = (
        (await db.execute(select(GrammarPoint).where(GrammarPoint.lesson_id == lesson.id).order_by(GrammarPoint.id)))
        .scalars()
        .all()
    )
    return LessonOut(
        id=lesson.id,
        title=lesson.title,
        level=lesson.level,
        content=lesson.content,
        created_at=lesson.created_at,
        topics=[TopicOut.model_validate(t) for t in topics],
        vocab=[VocabItemOut.model_validate(v) for v in vocab],
        grammar=[GrammarPointOut.model_validate(g) for g in grammar],
    )


@router.get("/corpus/items", response_model=list[CorpusItemOut])
async def list_corpus_items(
    db: Annotated[AsyncSession, Depends(get_db)],
    level: Annotated[int | None, Query(ge=1, le=6)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """Random sample of corpus.corpus_item for listening practice. No SQL
    JOIN across corpus/content (SDD module-boundary principle) — film/
    topic/grammar names are resolved via separate bulk queries and zipped
    together in application code instead."""
    query = select(CorpusItem).where(CorpusItem.kind == "câu")
    if level is not None:
        query = query.where(CorpusItem.level == level)
    items = (await db.execute(query.order_by(func.random()).limit(limit))).scalars().all()

    film_ids = {i.film_id for i in items}
    films = dict((await db.execute(select(Film.id, Film.title).where(Film.id.in_(film_ids)))).all()) if film_ids else {}

    topic_ids = {tid for i in items for tid in i.topic_ids}
    topics = dict((await db.execute(select(Topic.id, Topic.name).where(Topic.id.in_(topic_ids)))).all()) if topic_ids else {}

    grammar_ids = {gid for i in items for gid in i.grammar_point_ids}
    grammar = (
        dict((await db.execute(select(GrammarPoint.id, GrammarPoint.pattern).where(GrammarPoint.id.in_(grammar_ids)))).all())
        if grammar_ids
        else {}
    )

    return [
        CorpusItemOut(
            id=i.id,
            film_id=i.film_id,
            film_title=films.get(i.film_id, ""),
            text_ko=i.text_ko,
            kind=i.kind,
            level=i.level,
            register=i.register,
            topics=[topics[t] for t in i.topic_ids if t in topics],
            grammar_patterns=[grammar[g] for g in i.grammar_point_ids if g in grammar],
        )
        for i in items
    ]


@router.post("/progress/reviews", response_model=ItemStateOut)
async def record_item_review(
    body: ItemStateReviewRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """A learner's quick in-app check nudges ItemState.strength — SRS §5
    ITEM_STATE's decay-style nudge, not a full SM-2 scheduler (see
    ItemState's docstring in app/models.py)."""
    existing = await db.execute(
        select(ItemState).where(
            ItemState.learner_id == profile.id,
            ItemState.item_type == body.item_type,
            ItemState.item_id == body.item_id,
        )
    )
    state = existing.scalar_one_or_none()
    delta = 0.2 if body.correct else -0.2
    new_strength = max(0.0, min(1.0, (state.strength if state else 0.0) + delta))
    now = datetime.now(timezone.utc)
    if state is None:
        state = ItemState(
            learner_id=profile.id, item_type=body.item_type, item_id=body.item_id, strength=new_strength, last_seen=now
        )
        db.add(state)
    else:
        state.strength = new_strength
        state.last_seen = now
    await db.commit()
    await db.refresh(state)
    return state


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
