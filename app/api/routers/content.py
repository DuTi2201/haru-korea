"""Learner-facing read endpoints: content (topics/lessons/vocab/grammar),
corpus (listening sentences), progress (item-state check-ins), curriculum
(goal/placement/plan) and practice (quick-checks). Grouped in one router
for the scaffold; split into content.py/curriculum.py/practice.py once
each grows past a handful of routes.
"""
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile, get_optional_profile
from app.db import get_db
from app.models import (
    CorpusItem,
    EditorialArticle,
    EditorialOutlineSubmission,
    ErrorLog,
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
    ClozeOut,
    CorpusFacetsOut,
    CorpusFilmCount,
    CorpusGrammarCount,
    CorpusItemOut,
    CorpusLevelCount,
    CorpusPageOut,
    CorpusRegisterCount,
    CorpusSimilarOut,
    CorpusTopicCount,
    GrammarPointOut,
    ItemStateOut,
    ItemStateReviewRequest,
    LessonOut,
    LessonSummaryOut,
    ProfileOut,
    ReviewQueueItem,
    ReviewQueueOut,
    TodayPlanOut,
    TodayPlanTask,
    TopicOut,
    VocabItemOut,
)
from app.services import corpus_browse, srs, weakness
from app.services.corpus_browse import Sentence
from app.services.exercises import build_cloze
from app.services.progress import ItemKey, count_mastered, pick_next_lesson

router = APIRouter(tags=["content"])


@router.get("/topics")
async def list_topics(db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(Topic).order_by(Topic.name))
    topics = result.scalars().all()
    return [{"id": t.id, "name": t.name, "quizlet_url": t.quizlet_url} for t in topics]


async def _lesson_item_keys(db: AsyncSession) -> dict[int, list[ItemKey]]:
    """lesson id -> the (type, id) of every vocab/grammar item in it. Two
    narrow bulk queries (ids only); items that belong to no lesson
    (editorial-article imports) are left out — they are not lesson content."""
    lesson_items: dict[int, list[ItemKey]] = defaultdict(list)
    for lesson_id, item_id in (await db.execute(select(VocabItem.lesson_id, VocabItem.id))).all():
        if lesson_id is not None:
            lesson_items[lesson_id].append(("vocab_item", item_id))
    for lesson_id, item_id in (await db.execute(select(GrammarPoint.lesson_id, GrammarPoint.id))).all():
        if lesson_id is not None:
            lesson_items[lesson_id].append(("grammar_point", item_id))
    return lesson_items


@dataclass
class ArticleWords:
    """The vocabulary of the confirmed news articles, newest article first.
    `keys` is every word that exists; `hold` the ones above the learner's level
    (not started as new cards), `title_of` / `article_of` say where a word came from."""

    keys: dict[str, list[ItemKey]] = field(default_factory=dict)
    hold: set[ItemKey] = field(default_factory=set)
    title_of: dict[str, str] = field(default_factory=dict)
    article_of: dict[int, str] = field(default_factory=dict)


async def _article_words(db: AsyncSession, level_cap: int) -> ArticleWords:
    """Words of the articles a human has confirmed. A word that was rolled back is
    left out (it no longer exists); a word shared by two articles belongs to the
    newer one. Words above `level_cap` stay in `keys` but are listed in `hold`."""
    out = ArticleWords()
    rows = (
        await db.execute(
            select(
                EditorialArticle.id,
                EditorialArticle.title_ko,
                EditorialArticle.source_name,
                EditorialArticle.vocab_ids,
            )
            .where(EditorialArticle.body_ko.isnot(None), func.cardinality(EditorialArticle.vocab_ids) > 0)
            .order_by(EditorialArticle.created_at.desc())
        )
    ).all()
    if not rows:
        return out
    wanted = {vid for _, _, _, vocab_ids in rows for vid in vocab_ids}
    level_of = dict(
        (await db.execute(select(VocabItem.id, VocabItem.level).where(VocabItem.id.in_(wanted)))).all()
    )
    for article_id, title, source_name, vocab_ids in rows:
        key = str(article_id)
        words = [vid for vid in vocab_ids if vid in level_of and vid not in out.article_of]
        if not words:
            continue
        out.keys[key] = [("vocab_item", vid) for vid in words]
        out.title_of[key] = title or source_name
        for vid in words:
            out.article_of[vid] = key
            if level_of[vid] > level_cap:
                out.hold.add(("vocab_item", vid))
    return out


@router.get("/lessons", response_model=list[LessonSummaryOut])
async def list_lessons(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile | None, Depends(get_optional_profile)],
):
    """The lesson picker: every lesson that has something to study, by level
    then id. Public; when the caller is signed in each row also carries how
    many of its items they have mastered, and `is_next` marks the lesson
    /me/plan would hand them (same rule — app.services.progress)."""
    lessons = (await db.execute(select(Lesson.id, Lesson.title, Lesson.level).order_by(Lesson.level, Lesson.id))).all()
    lesson_items = await _lesson_item_keys(db)

    topics: dict[int, list[str]] = defaultdict(list)
    topic_rows = (
        await db.execute(
            select(LessonTopic.lesson_id, Topic.name)
            .join(Topic, Topic.id == LessonTopic.topic_id)
            .order_by(Topic.name)
        )
    ).all()
    for lesson_id, name in topic_rows:
        topics[lesson_id].append(name)

    states: dict[ItemKey, tuple[float, datetime]] | None = None
    next_id: int | None = None
    if profile is not None:
        rows = (await db.execute(select(ItemState).where(ItemState.learner_id == profile.id))).scalars().all()
        states = {(s.item_type, s.item_id): (s.strength, s.last_seen) for s in rows}
        picked = pick_next_lesson(lesson_items, states)
        next_id = picked.lesson_id if picked is not None else None

    out: list[LessonSummaryOut] = []
    for lesson_id, title, level in lessons:
        keys = lesson_items.get(lesson_id, [])
        if not keys:  # nothing to study yet: not a choice worth offering
            continue
        vocab_count = sum(1 for kind, _ in keys if kind == "vocab_item")
        out.append(
            LessonSummaryOut(
                id=lesson_id,
                title=title,
                level=level,
                topics=topics.get(lesson_id, []),
                vocab_count=vocab_count,
                grammar_count=len(keys) - vocab_count,
                mastered=count_mastered(keys, states) if states is not None else None,
                is_next=lesson_id == next_id,
            )
        )
    return out


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
    query = select(CorpusItem).where(CorpusItem.kind == "câu", _visible_to_learners())
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
            meaning_vi=i.meaning_vi,
            usage_note_vi=i.usage_note_vi,
        )
        for i in items
    ]


# --------------------------------------------------------- corpus browsing --
# A light copy of corpus.corpus_item (no embedding, de-duplicated, register
# corrected — see app/services/corpus_browse.py). Three endpoints and every
# page of a listening session read from it, so it is kept for a short while
# per process instead of being rebuilt on each request; a freshly confirmed
# import shows up within _SNAPSHOT_TTL_SECONDS.
_SNAPSHOT_TTL_SECONDS = 30.0
_snapshot: tuple[float, list[Sentence]] | None = None


def reset_corpus_snapshot() -> None:
    global _snapshot
    _snapshot = None


def _visible_to_learners():
    """Lines the enrichment step judged unnatural (machine-translated, garbled)
    are kept in the table but never offered; not-yet-judged lines (NULL) are."""
    return CorpusItem.naturalness.is_distinct_from("unnatural")


async def _corpus_sentences(db: AsyncSession) -> list[Sentence]:
    global _snapshot
    now = time.monotonic()
    if _snapshot is not None and now - _snapshot[0] < _SNAPSHOT_TTL_SECONDS:
        return _snapshot[1]
    rows = (
        await db.execute(
            select(
                CorpusItem.id,
                CorpusItem.film_id,
                CorpusItem.text_ko,
                CorpusItem.kind,
                CorpusItem.level,
                CorpusItem.register,
                CorpusItem.topic_ids,
                CorpusItem.grammar_point_ids,
                CorpusItem.meaning_vi,
                CorpusItem.usage_note_vi,
            ).where(CorpusItem.kind == "câu", _visible_to_learners())
        )
    ).all()
    sentences = corpus_browse.dedupe(
        [
            corpus_browse.with_effective_register(
                Sentence(
                    id=r[0],
                    film_id=r[1],
                    text_ko=r[2],
                    kind=r[3],
                    level=r[4],
                    register=r[5],
                    topic_ids=tuple(r[6] or ()),
                    grammar_ids=tuple(r[7] or ()),
                    meaning_vi=r[8],
                    usage_note_vi=r[9],
                )
            )
            for r in rows
        ]
    )
    _snapshot = (now, sentences)
    return sentences


async def _id_names(db: AsyncSession, id_col, name_col, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    return dict((await db.execute(select(id_col, name_col).where(id_col.in_(ids)))).all())


async def _corpus_out(db: AsyncSession, sentences: list[Sentence]) -> list[CorpusItemOut]:
    """Names for the ids a page of sentences refers to — separate bulk queries
    zipped in code, never a JOIN across corpus/content (module boundary)."""
    films = await _id_names(db, Film.id, Film.title, {s.film_id for s in sentences})
    topics = await _id_names(db, Topic.id, Topic.name, {t for s in sentences for t in s.topic_ids})
    grammar = await _id_names(db, GrammarPoint.id, GrammarPoint.pattern, {g for s in sentences for g in s.grammar_ids})
    return [
        CorpusItemOut(
            id=s.id,
            film_id=s.film_id,
            film_title=films.get(s.film_id, ""),
            text_ko=s.text_ko,
            kind=s.kind,
            level=s.level,
            register=s.register,
            topics=[topics[t] for t in s.topic_ids if t in topics],
            grammar_patterns=[grammar[g] for g in s.grammar_ids if g in grammar],
            meaning_vi=s.meaning_vi,
            usage_note_vi=s.usage_note_vi,
        )
        for s in sentences
    ]


@router.get("/corpus/facets", response_model=CorpusFacetsOut)
async def corpus_facets(db: Annotated[AsyncSession, Depends(get_db)]):
    """What the listening picker can filter by (level, speech level, topic,
    grammar pattern, source film) and how many distinct sentences each holds.
    Public, like the sentences themselves."""
    counts = corpus_browse.count_facets(await _corpus_sentences(db))
    films = await _id_names(db, Film.id, Film.title, {i for i, _ in counts.films})
    topics = await _id_names(db, Topic.id, Topic.name, {i for i, _ in counts.topics})
    grammar = await _id_names(db, GrammarPoint.id, GrammarPoint.pattern, {i for i, _ in counts.grammar})
    return CorpusFacetsOut(
        total=counts.total,
        levels=[CorpusLevelCount(level=lv, count=n) for lv, n in counts.levels],
        registers=[CorpusRegisterCount(register=r, count=n) for r, n in counts.registers],
        topics=[CorpusTopicCount(id=i, name=topics[i], count=n) for i, n in counts.topics if i in topics],
        grammar=[CorpusGrammarCount(id=i, pattern=grammar[i], count=n) for i, n in counts.grammar if i in grammar],
        films=[CorpusFilmCount(id=i, title=films[i], count=n) for i, n in counts.films if i in films],
    )


@router.get("/corpus/browse", response_model=CorpusPageOut)
async def browse_corpus(
    db: Annotated[AsyncSession, Depends(get_db)],
    level: Annotated[int | None, Query(ge=1, le=6)] = None,
    topic_id: Annotated[int | None, Query(ge=1)] = None,
    register: Annotated[Literal["존댓말", "반말", "hỗn hợp"] | None, Query()] = None,
    grammar_id: Annotated[int | None, Query(ge=1)] = None,
    film_id: Annotated[int | None, Query(ge=1)] = None,
    seed: Annotated[str, Query(max_length=40)] = "",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """A page of distinct sentences matching the filters, in an order fixed by
    `seed`: ask for offset 0, 10, 20… with the same seed and every sentence
    comes up exactly once (a fresh random sample per request cannot promise
    that). The client picks a new random seed for each listening session."""
    matching = [
        s
        for s in await _corpus_sentences(db)
        if corpus_browse.matches(
            s, level=level, topic_id=topic_id, register=register, grammar_id=grammar_id, film_id=film_id
        )
    ]
    page = corpus_browse.seeded_order(matching, seed)[offset : offset + limit]
    return CorpusPageOut(total=len(matching), offset=offset, items=await _corpus_out(db, page))


@router.get("/corpus/items/{item_id}/similar", response_model=list[CorpusSimilarOut])
async def similar_corpus_items(
    item_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=10)] = 3,
    max_distance: Annotated[float, Query(ge=0, le=2)] = 0.4,
):
    """Sentences close in meaning to this one — the same idea said another way
    (a politer or plainer ending, other word forms) — nearest first. Uses the
    embedding stored when the sentence was imported; word-for-word repeats are
    never offered as "similar"."""
    target = await db.get(CorpusItem, item_id)
    if target is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Corpus item not found", "not_found")
    if target.embedding is None:
        return []

    distance = CorpusItem.embedding.cosine_distance(target.embedding)
    rows = (
        await db.execute(
            select(CorpusItem.id, distance.label("distance"))
            .where(
                CorpusItem.id != item_id,
                CorpusItem.kind == "câu",
                CorpusItem.embedding.is_not(None),
                distance <= max_distance,
            )
            .order_by(distance)
            .limit(limit * 6)  # headroom: copies and repeats are filtered out below
        )
    ).all()

    by_id = {s.id: s for s in await _corpus_sentences(db)}
    own_key = corpus_browse.normalize_key(target.text_ko)
    picked: list[tuple[Sentence, float]] = []
    seen_keys = {own_key}
    for near_id, dist in rows:
        sentence = by_id.get(near_id)  # absent = a repeat of a sentence kept elsewhere
        if sentence is None:
            continue
        key = corpus_browse.normalize_key(sentence.text_ko)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picked.append((sentence, float(dist)))
        if len(picked) == limit:
            break

    outs = await _corpus_out(db, [s for s, _ in picked])
    return [CorpusSimilarOut(**o.model_dump(), distance=round(d, 3)) for o, (_, d) in zip(outs, picked)]


@router.post("/progress/reviews", response_model=ItemStateOut)
async def record_item_review(
    body: ItemStateReviewRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
):
    """A learner's quick in-app check. It nudges ItemState.strength (SRS §5
    ITEM_STATE's decay-style nudge, the readiness signal) and moves the item's
    review schedule (app.services.srs: when it is due again, how often it was
    forgotten). A wrong answer is also written to the error log, so the same
    mistake can be found again (app.services.weakness). The item must exist:
    item_id has no FK (content lives in another schema), so without this check a
    typo'd or made-up id would quietly become a phantom ItemState row that skews
    the learner's streak and readiness."""
    content_model = VocabItem if body.item_type == "vocab_item" else GrammarPoint
    row = (await db.execute(select(content_model).where(content_model.id == body.item_id))).scalar_one_or_none()
    if row is None:
        raise _problem(status.HTTP_404_NOT_FOUND, "Item not found", "not_found")

    existing = await db.execute(
        select(ItemState).where(
            ItemState.learner_id == profile.id,
            ItemState.item_type == body.item_type,
            ItemState.item_id == body.item_id,
        )
    )
    state = existing.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    after = srs.review(_schedule_of(state) if state is not None else None, body.correct, now)
    if state is None:
        state = ItemState(learner_id=profile.id, item_type=body.item_type, item_id=body.item_id, last_seen=now)
        db.add(state)
    state.strength = after.strength
    state.reps = after.reps
    state.lapses = after.lapses
    state.ease = after.ease
    state.interval_days = after.interval_days
    state.due_at = after.due_at
    state.introduced_at = after.introduced_at
    state.last_seen = now
    if not body.correct:
        db.add(_error_for(profile.id, body, row))
    await db.commit()
    await db.refresh(state)
    return state


def _error_for(learner_id: uuid.UUID, body: ItemStateReviewRequest, row: VocabItem | GrammarPoint) -> ErrorLog:
    """The error-log row for a wrong review answer."""
    mode = body.mode or "recognize"
    is_vocab = isinstance(row, VocabItem)
    detail: dict[str, str] = {}
    if body.chosen:
        detail["chosen"] = body.chosen
    family = getattr(row, "family", None)
    if family:
        detail["family"] = family
    return ErrorLog(
        learner_id=learner_id,
        skill="vocab" if is_vocab else "grammar",
        error_type=weakness.error_type_for_review(body.item_type, mode, bool(getattr(row, "node_word", None))),
        example_ko=row.hangul if is_vocab else row.pattern,
        item_type=body.item_type,
        item_id=body.item_id,
        mode=mode,
        detail=detail or None,
    )


def _schedule_of(state: ItemState) -> srs.Schedule:
    return srs.Schedule(
        strength=state.strength,
        reps=state.reps or 0,
        lapses=state.lapses or 0,
        ease=state.ease or srs.START_EASE,
        interval_days=state.interval_days or 0.0,
        due_at=state.due_at,
        introduced_at=state.introduced_at,
    )


@router.get("/me/review-queue", response_model=ReviewQueueOut)
async def get_review_queue(
    db: Annotated[AsyncSession, Depends(get_db)],
    profile: Annotated[Profile, Depends(get_current_profile)],
    limit: Annotated[int, Query(ge=1, le=60)] = srs.DEFAULT_REVIEW_LIMIT,
    new: Annotated[int, Query(ge=0, le=30)] = srs.DEFAULT_NEW_PER_DAY,
    focus: Annotated[Literal["due", "weak"], Query()] = "due",
):
    """Today's sitting: the cards that are due (cards the learner keeps forgetting
    first, then the most overdue), then new ones — from the earliest lessons that
    still have unseen cards, plus words from the confirmed news articles that fit
    the learner's goal level — at most `new` per rolling 24 hours (about one
    grammar point and one article word in four). A due vocabulary card that has
    been answered right before is asked as a fill-in-the-blank when one can be
    built from it (app.services.exercises); a new, forgotten or grammar card is
    shown for the ordinary flip-and-judge review.

    `focus=weak` is practice on demand instead: the cards the learner keeps
    forgetting, due or not (an answer before the due time still counts for
    strength and the error log, but does not push the schedule out)."""
    now = datetime.now(timezone.utc)
    states = (await db.execute(select(ItemState).where(ItemState.learner_id == profile.id))).scalars().all()
    lesson_items = await _lesson_item_keys(db)
    articles = await _article_words(db, srs.article_level_cap(profile.goal))
    schedules = {(s.item_type, s.item_id): _schedule_of(s) for s in states}
    if focus == "weak":
        existing = {key for keys in lesson_items.values() for key in keys}
        existing |= {key for keys in articles.keys.values() for key in keys}
        weak = srs.weak_keys(schedules, existing, limit=limit)
        queue = srs.Queue(due=weak, due_total=len(srs.weak_keys(schedules, existing, limit=10_000)))
    else:
        queue = srs.build_queue(
            schedules, lesson_items, now, limit=limit, new_limit=new, articles=articles.keys, hold=articles.hold
        )

    keys = [*queue.due, *queue.new]
    vocab_ids = [i for t, i in keys if t == "vocab_item"]
    grammar_ids = [i for t, i in keys if t == "grammar_point"]
    vocab = {}
    if vocab_ids:
        vocab = {v.id: v for v in (await db.execute(select(VocabItem).where(VocabItem.id.in_(vocab_ids)))).scalars().all()}
    grammar = {}
    if grammar_ids:
        grammar = {
            g.id: g for g in (await db.execute(select(GrammarPoint).where(GrammarPoint.id.in_(grammar_ids)))).scalars().all()
        }
    lesson_ids = {v.lesson_id for v in vocab.values() if v.lesson_id} | {g.lesson_id for g in grammar.values() if g.lesson_id}
    titles: dict[int, str] = {}
    mates: dict[int, list[VocabItem]] = defaultdict(list)
    if lesson_ids:
        titles = dict((await db.execute(select(Lesson.id, Lesson.title).where(Lesson.id.in_(lesson_ids)))).all())
        for v in (await db.execute(select(VocabItem).where(VocabItem.lesson_id.in_(lesson_ids)))).scalars().all():
            mates[v.lesson_id].append(v)
    # an article word's mates, for the fill-in-the-blank, are the other words of its article
    article_mates: dict[str, list[VocabItem]] = defaultdict(list)
    article_ids = {articles.article_of[v.id] for v in vocab.values() if v.id in articles.article_of}
    if article_ids:
        wanted = [vid for aid in article_ids for _, vid in articles.keys.get(aid, [])]
        for v in (await db.execute(select(VocabItem).where(VocabItem.id.in_(wanted)))).scalars().all():
            article_mates[articles.article_of[v.id]].append(v)

    items: list[ReviewQueueItem] = []
    due_keys = set(queue.due)
    for item_type, item_id in keys:
        row = vocab.get(item_id) if item_type == "vocab_item" else grammar.get(item_id)
        if row is None:  # deleted between the two queries
            continue
        schedule = schedules.get((item_type, item_id))
        is_new = (item_type, item_id) not in due_keys
        article_id = articles.article_of.get(item_id) if item_type == "vocab_item" and row.lesson_id is None else None
        cloze = None
        if item_type == "vocab_item" and not is_new and schedule is not None and schedule.reps >= 1:
            pool = article_mates.get(article_id, []) if article_id else mates.get(row.lesson_id, [])
            built = build_cloze(row, pool, seed=f"{profile.id}:{item_id}:{schedule.reps}")
            if built is not None:
                cloze = ClozeOut(prompt_ko=built.prompt_ko, answer=built.answer, choices=list(built.choices))
        items.append(
            ReviewQueueItem(
                item_type=item_type,  # type: ignore[arg-type]
                item_id=item_id,
                is_new=is_new,
                mode="cloze" if cloze is not None else "recognize",
                reps=schedule.reps if schedule else 0,
                lapses=schedule.lapses if schedule else 0,
                leech=bool(schedule and srs.is_leech(schedule)),
                due_at=schedule.due_at if schedule else None,
                lesson_title=titles.get(row.lesson_id) if row.lesson_id else None,
                article_id=uuid.UUID(article_id) if article_id else None,
                article_title=articles.title_of.get(article_id) if article_id else None,
                vocab=VocabItemOut.model_validate(row) if item_type == "vocab_item" else None,
                grammar=GrammarPointOut.model_validate(row) if item_type == "grammar_point" else None,
                cloze=cloze,
            )
        )
    return ReviewQueueOut(
        due_total=queue.due_total,
        new_available=queue.new_budget,
        new_today=queue.new_today,
        items=items,
        focus=focus,
        article_waiting=queue.article_waiting,
        article_above_level=sum(
            1 for key in articles.hold if key not in schedules
        ),
    )


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
    lesson_items = await _lesson_item_keys(db)

    # -- today's review sitting (the /review screen): cards that are due plus the
    # day's new ones, counted by the same rule that builds the queue itself.
    now = datetime.now(timezone.utc)
    articles = await _article_words(db, srs.article_level_cap(profile.goal))
    queue = srs.build_queue(
        {(s.item_type, s.item_id): _schedule_of(s) for s in states},
        lesson_items,
        now,
        articles=articles.keys,
        hold=articles.hold,
    )
    lesson_keys_all = {key for keys in lesson_items.values() for key in keys}
    lesson_keys_all |= {key for keys in articles.keys.values() for key in keys}
    studied_today = sum(
        1
        for s in states
        if (s.item_type, s.item_id) in lesson_keys_all and s.last_seen.astimezone(timezone.utc).date() == today
    )
    waiting = queue.due_total + len(queue.new)
    if waiting > 0:
        parts = [f"{queue.due_total} thẻ đến hạn"] if queue.due_total else []
        if queue.new:
            parts.append(f"{len(queue.new)} thẻ mới")
        tasks.append(
            TodayPlanTask(
                kind="review",
                title="Ôn tập hôm nay",
                subtitle=" · ".join(parts),
                status="in_progress" if studied_today else "todo",
            )
        )
    elif studied_today:
        tasks.append(
            TodayPlanTask(
                kind="review",
                title="Ôn tập hôm nay",
                subtitle="Đã ôn hết thẻ đến hạn, hẹn lần sau",
                status="done",
            )
        )

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
                subtitle=f"{corpus_count} câu có sẵn, giọng đọc chuẩn Seoul",
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
