"""Learner-facing read endpoints: content (topics/lessons/vocab/grammar),
corpus (listening sentences), progress (item-state check-ins), curriculum
(goal/placement/plan) and practice (quick-checks). Grouped in one router
for the scaffold; split into content.py/curriculum.py/practice.py once
each grows past a handful of routes.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile
from app.db import get_db
from app.models import (
    CorpusItem,
    EditorialArticle,
    EditorialOutlineSubmission,
    Film,
    GrammarPoint,
    ItemState,
    Lesson,
    LessonTopic,
    Profile,
    Topic,
    VocabItem,
)
from app.schemas import (
    CorpusItemOut,
    GrammarPointOut,
    ItemStateOut,
    ItemStateReviewRequest,
    LessonOut,
    ProfileOut,
    TodayPlanOut,
    TodayPlanTask,
    TopicOut,
    VocabItemOut,
)
from app.services.progress import ItemKey, pick_next_lesson

router = APIRouter(tags=["content"])


@router.get("/topics")
async def list_topics(db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(Topic).order_by(Topic.name))
    topics = result.scalars().all()
    return [{"id": t.id, "name": t.name, "quizlet_url": t.quizlet_url} for t in topics]


@router.get("/lessons/{lesson_id}", response_model=LessonOut)
async def get_lesson(lesson_id: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """`lesson_id="today"` is the anonymous fallback (the earliest lesson) —
    this route is public, so it cannot know who is asking. The learner-aware
    "next lesson" is chosen by /me/plan (see app.services.progress), which
    hands the frontend a concrete numeric `lesson_id`; a numeric id fetches
    that exact lesson."""
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
    ItemState's docstring in app/models.py). The item must exist: item_id
    has no FK (content lives in another schema), so without this check a
    typo'd or made-up id would quietly become a phantom ItemState row that
    skews the learner's streak and readiness."""
    content_model = VocabItem if body.item_type == "vocab_item" else GrammarPoint
    if (await db.execute(select(content_model.id).where(content_model.id == body.item_id))).first() is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Item not found", "not_found")

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


@router.get("/me/plan", response_model=TodayPlanOut)
async def get_my_plan(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """Real /today data — every field here comes from a query, never a
    hardcoded placeholder. `streak_days`/`readiness_pct` are derived from
    ItemState (the only per-learner history table that exists so far —
    see its docstring), so a brand-new learner honestly gets 0/None
    rather than fake numbers; the frontend renders that as a first-run
    empty state instead of a lie. Each task is only included when there
    is real content behind it (a lesson/corpus item/confirmed article
    actually exists) and `status` is only ever "done"/"in_progress" when
    that is something this endpoint can actually verify — no fabricated
    checkmarks like the old Lovable mock had."""
    states = (
        (await db.execute(select(ItemState).where(ItemState.learner_id == profile.id))).scalars().all()
    )
    today = datetime.now(timezone.utc).date()

    seen_dates = {s.last_seen.astimezone(timezone.utc).date() for s in states}
    streak_days = 0
    cursor = today if today in seen_dates else today - timedelta(days=1)
    while cursor in seen_dates:
        streak_days += 1
        cursor -= timedelta(days=1)

    readiness_pct = round(sum(s.strength for s in states) / len(states) * 100, 1) if states else None

    tasks: list[TodayPlanTask] = []

    # -- vocab/grammar review: the first lesson this learner has not yet
    # mastered (see app.services.progress.pick_next_lesson), so finishing
    # lesson 1 moves them on to lesson 2 instead of showing lesson 1 forever.
    # Two narrow bulk queries (ids only) — vocab/grammar that belong to no
    # lesson (editorial-article imports) are left out, they are not lesson
    # content.
    lesson_items: dict[int, list[ItemKey]] = defaultdict(list)
    for lesson_id_, v_id in (await db.execute(select(VocabItem.lesson_id, VocabItem.id))).all():
        if lesson_id_ is not None:
            lesson_items[lesson_id_].append(("vocab_item", v_id))
    for lesson_id_, g_id in (await db.execute(select(GrammarPoint.lesson_id, GrammarPoint.id))).all():
        if lesson_id_ is not None:
            lesson_items[lesson_id_].append(("grammar_point", g_id))

    picked = pick_next_lesson(lesson_items, {(s.item_type, s.item_id): (s.strength, s.last_seen) for s in states})
    lesson = await db.get(Lesson, picked.lesson_id) if picked is not None else None
    if picked is not None and lesson is not None:
        total = picked.total
        lesson_keys = set(lesson_items[picked.lesson_id])
        reviewed_today = sum(
            1
            for s in states
            if s.last_seen.astimezone(timezone.utc).date() == today and (s.item_type, s.item_id) in lesson_keys
        )
        status_ = "done" if reviewed_today >= total else ("in_progress" if reviewed_today > 0 else "todo")
        subtitle = f"{total} từ/ngữ pháp"
        if picked.mastered:
            subtitle += f" · đã thuộc {picked.mastered}/{total}"
        if reviewed_today:
            subtitle += f" · đã ôn {reviewed_today}/{total} hôm nay"
        tasks.append(
            TodayPlanTask(
                kind="vocab_review",
                title=lesson.title,
                subtitle=subtitle,
                status=status_,
                lesson_id=lesson.id,
            )
        )

    # -- listening: no per-item completion is tracked yet for corpus
    # items (only lesson vocab/grammar go through /progress/reviews), so
    # this is honestly always "todo" rather than a faked checkmark.
    corpus_count = (
        await db.execute(select(func.count()).select_from(CorpusItem).where(CorpusItem.kind == "câu"))
    ).scalar_one()
    if corpus_count > 0:
        tasks.append(
            TodayPlanTask(
                kind="listening",
                title="Luyện nghe câu mẫu từ phim",
                subtitle=f"{corpus_count} câu có sẵn, giọng đọc Gemini chuẩn Seoul",
                status="todo",
            )
        )

    # -- reading: most recently created *confirmed* article (body_ko set
    # by apply_editorial_batch, so this only ever surfaces something a
    # human has actually reviewed and confirmed).
    article = (
        (
            await db.execute(
                select(EditorialArticle)
                .where(EditorialArticle.body_ko.isnot(None))
                .order_by(EditorialArticle.created_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if article is not None:
        submission = (
            await db.execute(
                select(EditorialOutlineSubmission).where(
                    EditorialOutlineSubmission.learner_id == profile.id,
                    EditorialOutlineSubmission.editorial_article_id == article.id,
                )
            )
        ).scalar_one_or_none()
        tasks.append(
            TodayPlanTask(
                kind="reading",
                title=article.title_ko or article.source_name,
                subtitle="Đọc xã luận · luyện dàn ý câu 54",
                status="done" if submission and submission.revision_count > 0 else "todo",
                article_id=article.id,
            )
        )

    return TodayPlanOut(
        streak_days=streak_days,
        readiness_pct=readiness_pct,
        goal=profile.goal,
        exam_date=profile.exam_date,
        tasks=tasks,
    )


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
