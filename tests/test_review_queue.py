"""GET /me/review-queue, the schedule written by POST /progress/reviews, and the
"Ôn tập hôm nay" task in /me/plan. The database is a small in-memory fake that
answers the router's queries from lists of real model objects."""
import re
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from sqlalchemy.dialects import postgresql

from app.api.routers import content
from app.models import GrammarPoint, ItemState, VocabItem
from app.schemas import ItemStateReviewRequest

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
FAMILY = "Động từ đi với thời tiết"


def vocab(id, hangul, lesson_id=1, **kw):
    return VocabItem(id=id, lesson_id=lesson_id, hangul=hangul, meaning_vi=f"nghĩa {hangul}", level=1, **kw)


def grammar(id, pattern, lesson_id=1):
    return GrammarPoint(id=id, lesson_id=lesson_id, pattern=pattern, meaning_vi="m", level=2)


CONTENT_V = [
    vocab(1, "비가 오다", node_word="오다", family=FAMILY, distractors=["내리다", "떨어지다"]),
    vocab(2, "눈이 오다", node_word="오다", family=FAMILY),
    vocab(3, "바람이 불다", node_word="불다", family=FAMILY),
    vocab(4, "날씨", pos="명사", example_ko="오늘 날씨가 좋아요."),
    vocab(5, "기온", pos="명사"),
    vocab(6, "하늘", pos="명사"),
    vocab(7, "바람", pos="명사"),
    vocab(8, "봄", lesson_id=2),
]
CONTENT_G = [grammar(1, "V + -(으)ㄹ 것 같다"), grammar(2, "V + -(으)면서"), grammar(3, "V + -군요", lesson_id=2)]
LESSONS = [(1, "날씨와 계절"), (2, "Bài hai")]


def sql(stmt):
    return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def ids_in(text, column):
    match = re.search(rf"{column} IN \(([^)]*)\)", text)
    return {int(x) for x in match.group(1).split(",")} if match else None


class FakeDB:
    def __init__(self, states=()):
        self.states = list(states)
        self.queries = []

    async def execute(self, stmt):
        text = sql(stmt)
        self.queries.append(text)
        res = mock.MagicMock()
        if "FROM item_state" in text:
            res.scalars.return_value.all.return_value = self.states
        elif "FROM content.vocab_item" in text or "FROM content.grammar_point" in text:
            table, rows = ("vocab_item", CONTENT_V) if "FROM content.vocab_item" in text else ("grammar_point", CONTENT_G)
            full = "meaning_vi" in text
            if not full:  # the (lesson_id, id) listing
                res.all.return_value = [(r.lesson_id, r.id) for r in rows]
            else:
                by_id = ids_in(text, f"content.{table}.id")
                by_lesson = ids_in(text, f"content.{table}.lesson_id")
                picked = [r for r in rows if (by_id is None or r.id in by_id) and (by_lesson is None or r.lesson_id in by_lesson)]
                res.scalars.return_value.all.return_value = picked
        elif "FROM content.lesson" in text:
            wanted = ids_in(text, "content.lesson.id")
            res.all.return_value = [l for l in LESSONS if wanted is None or l[0] in wanted]
        else:
            raise AssertionError(f"unexpected query: {text[:200]}")
        return res


def state(item_type, item_id, **kw):
    base = dict(learner_id=uuid.uuid4(), item_type=item_type, item_id=item_id, strength=0.2, last_seen=NOW - DAY, reps=1,
                lapses=0, ease=2.2, interval_days=1.0, due_at=NOW - timedelta(hours=2), introduced_at=NOW - 3 * DAY)
    return ItemState(**{**base, **kw})


async def queue(db, **params):
    profile = SimpleNamespace(id=uuid.uuid4())
    kwargs = dict(limit=20, new=8)
    kwargs.update(params)
    with mock.patch.object(content, "datetime", wraps=datetime) as dt:
        dt.now.return_value = NOW
        return await content.get_review_queue(db, profile, **kwargs)


class ReviewQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_new_learner_gets_new_cards_in_lesson_order_with_grammar_spread_in(self):
        out = await queue(FakeDB())
        self.assertEqual((out.due_total, out.new_today, out.new_available), (0, 0, 8))
        self.assertEqual(len(out.items), 8)
        self.assertTrue(all(i.is_new and i.mode == "recognize" and i.cloze is None for i in out.items))
        kinds = [i.item_type for i in out.items]
        self.assertEqual(kinds.count("grammar_point"), 2)
        self.assertEqual([i.vocab.hangul for i in out.items if i.vocab][:3], ["비가 오다", "눈이 오다", "바람이 불다"])
        self.assertTrue(all(i.lesson_title == "날씨와 계절" for i in out.items[:6]))

    async def test_new_vocab_cards_carry_their_chunk_layers_to_the_app(self):
        out = await queue(FakeDB())
        first = out.items[0].vocab
        self.assertEqual((first.family, first.node_word, first.distractors), (FAMILY, "오다", ["내리다", "떨어지다"]))

    async def test_a_due_card_that_was_answered_right_before_is_asked_as_a_blank(self):
        out = await queue(FakeDB([state("vocab_item", 1, reps=2)]))
        (due,) = [i for i in out.items if not i.is_new]
        self.assertEqual((due.item_id, due.mode, due.reps), (1, "cloze", 2))
        self.assertEqual((due.cloze.prompt_ko, due.cloze.answer), ("비가 ____", "오다"))
        self.assertIn("오다", due.cloze.choices)
        self.assertGreaterEqual(len(due.cloze.choices), 3)

    async def test_a_forgotten_card_goes_back_to_the_plain_review(self):
        out = await queue(FakeDB([state("vocab_item", 1, reps=0, lapses=1)]))
        (due,) = [i for i in out.items if not i.is_new]
        self.assertEqual((due.mode, due.cloze, due.lapses), ("recognize", None, 1))

    async def test_a_due_card_that_cannot_make_a_blank_is_reviewed_the_plain_way(self):
        out = await queue(FakeDB([state("vocab_item", 8, reps=3)]))  # 봄: no node word, no example
        (due,) = [i for i in out.items if not i.is_new]
        self.assertEqual((due.item_id, due.mode), (8, "recognize"))

    async def test_a_noun_with_an_example_is_asked_with_same_part_of_speech_words(self):
        out = await queue(FakeDB([state("vocab_item", 4, reps=1)]))
        (due,) = [i for i in out.items if not i.is_new]
        self.assertEqual((due.mode, due.cloze.answer), ("cloze", "날씨"))
        self.assertTrue(set(due.cloze.choices) <= {"날씨", "기온", "하늘", "바람"})

    async def test_grammar_is_always_the_plain_review(self):
        out = await queue(FakeDB([state("grammar_point", 1, reps=4)]))
        (due,) = [i for i in out.items if not i.is_new]
        self.assertEqual((due.item_type, due.mode, due.cloze), ("grammar_point", "recognize", None))
        self.assertEqual(due.grammar.pattern, "V + -(으)ㄹ 것 같다")

    async def test_due_cards_come_before_new_ones_most_overdue_first(self):
        states = [
            state("vocab_item", 2, due_at=NOW - timedelta(hours=1)),
            state("vocab_item", 1, due_at=NOW - 2 * DAY),
        ]
        out = await queue(FakeDB(states))
        self.assertEqual([(i.item_id, i.is_new) for i in out.items[:2]], [(1, False), (2, False)])
        self.assertTrue(all(i.is_new for i in out.items[2:]))

    async def test_cards_already_started_are_not_offered_as_new_and_use_up_the_days_cap(self):
        states = [state("vocab_item", i, due_at=NOW + 3 * DAY, introduced_at=NOW - timedelta(hours=2)) for i in (1, 2, 3, 4, 5)]
        out = await queue(FakeDB(states))
        self.assertEqual((out.new_today, out.new_available), (5, 3))
        self.assertEqual(len(out.items), 3)
        self.assertTrue({i.item_id for i in out.items if i.item_type == "vocab_item"}.isdisjoint({1, 2, 3, 4, 5}))

    async def test_a_state_whose_lesson_was_rolled_back_is_ignored(self):
        out = await queue(FakeDB([state("vocab_item", 999)]))
        self.assertEqual(out.due_total, 0)
        self.assertTrue(all(i.is_new for i in out.items))

    async def test_the_limits_are_respected_and_the_total_still_reports_what_was_cut(self):
        states = [state("vocab_item", i, due_at=NOW - timedelta(hours=i)) for i in (1, 2, 3, 4)]
        out = await queue(FakeDB(states), limit=2, new=0)
        self.assertEqual((out.due_total, len(out.items), out.new_available), (4, 2, 0))
        self.assertEqual([i.item_id for i in out.items], [4, 3])  # the most overdue

    async def test_an_empty_library_gives_an_empty_queue(self):
        db = FakeDB()
        with mock.patch.object(content, "_lesson_item_keys", mock.AsyncMock(return_value={})):
            out = await queue(db)
        self.assertEqual((out.due_total, out.items), (0, []))

    async def test_the_choices_are_stable_for_the_same_learner_and_card(self):
        profile = SimpleNamespace(id=uuid.uuid4())
        with mock.patch.object(content, "datetime", wraps=datetime) as dt:
            dt.now.return_value = NOW
            x = await content.get_review_queue(FakeDB([state("vocab_item", 1, reps=2)]), profile, limit=20, new=8)
            y = await content.get_review_queue(FakeDB([state("vocab_item", 1, reps=2)]), profile, limit=20, new=8)
        self.assertEqual(x.items[0].cloze.choices, y.items[0].cloze.choices)


class RecordReviewScheduleTests(unittest.IsolatedAsyncioTestCase):
    def _db(self, existing=None):
        db = mock.MagicMock()
        db.add = mock.MagicMock()
        db.commit = mock.AsyncMock()
        db.refresh = mock.AsyncMock()

        async def execute(stmt):
            res = mock.MagicMock()
            if "item_state" in str(stmt):
                res.scalar_one_or_none.return_value = existing
            else:
                res.first.return_value = (1,)
            return res

        db.execute = execute
        return db

    async def record(self, db, correct, item_type="vocab_item", item_id=1):
        body = ItemStateReviewRequest(item_type=item_type, item_id=item_id, correct=correct)
        with mock.patch.object(content, "datetime", wraps=datetime) as dt:
            dt.now.return_value = NOW
            await content.record_item_review(body, db, SimpleNamespace(id=uuid.uuid4()))

    async def test_a_first_right_answer_creates_the_state_due_tomorrow(self):
        db = self._db()
        await self.record(db, True)
        (created,) = db.add.call_args.args
        self.assertEqual((created.reps, created.lapses, created.interval_days), (1, 0, 1.0))
        self.assertEqual((created.due_at, created.introduced_at, created.last_seen), (NOW + DAY, NOW, NOW))
        self.assertAlmostEqual(created.strength, 0.2)

    async def test_a_first_wrong_answer_is_due_again_in_ten_minutes(self):
        db = self._db()
        await self.record(db, False)
        (created,) = db.add.call_args.args
        self.assertEqual((created.reps, created.lapses, created.strength), (0, 1, 0.0))
        self.assertEqual(created.due_at, NOW + timedelta(minutes=10))

    async def test_an_existing_state_moves_up_the_ladder(self):
        existing = state("vocab_item", 1, reps=1, interval_days=1.0, strength=0.2, due_at=NOW - timedelta(hours=1))
        db = self._db(existing)
        await self.record(db, True)
        db.add.assert_not_called()
        self.assertEqual((existing.reps, existing.interval_days, existing.due_at), (2, 3.0, NOW + 3 * DAY))
        self.assertAlmostEqual(existing.strength, 0.4)
        db.commit.assert_awaited_once()

    async def test_a_wrong_answer_on_an_existing_state_is_a_lapse(self):
        existing = state("vocab_item", 1, reps=4, interval_days=15.0, strength=0.8, due_at=NOW - DAY)
        await self.record(self._db(existing), False)
        self.assertEqual((existing.reps, existing.lapses), (0, 1))
        self.assertAlmostEqual(existing.strength, 0.6)

    async def test_an_early_right_answer_leaves_the_schedule_alone(self):
        due = NOW + 5 * DAY
        existing = state("vocab_item", 1, reps=3, interval_days=7.0, strength=0.6, due_at=due)
        await self.record(self._db(existing), True)
        self.assertEqual((existing.reps, existing.interval_days, existing.due_at), (3, 7.0, due))
        self.assertAlmostEqual(existing.strength, 0.8)

    async def test_a_row_from_before_the_schedule_existed_still_works(self):
        legacy = SimpleNamespace(strength=0.4, last_seen=NOW - DAY, reps=None, lapses=None, ease=None, interval_days=None,
                                 due_at=None, introduced_at=None)
        await self.record(self._db(legacy), True)
        self.assertEqual((legacy.reps, legacy.interval_days), (1, 1.0))
        self.assertEqual(legacy.introduced_at, NOW)


class PlanReviewTaskTests(unittest.IsolatedAsyncioTestCase):
    async def plan(self, states):
        db = FakeDB(states)
        # the plan also looks up the lesson, the corpus and the editorial article
        real = db.execute

        async def execute(stmt):
            text = sql(stmt)
            if "count(" in text:
                res = mock.MagicMock()
                res.scalar_one.return_value = 0
                return res
            if "editorial" in text:
                res = mock.MagicMock()
                res.scalars.return_value.first.return_value = None
                res.scalar_one_or_none.return_value = None
                return res
            return await real(stmt)

        db.execute = execute

        async def get(model, pk):
            return SimpleNamespace(id=pk, title=dict(LESSONS).get(pk, "?"))

        db.get = get
        profile = SimpleNamespace(id=uuid.uuid4(), goal="topik4", exam_date=None)
        with mock.patch.object(content, "datetime", wraps=datetime) as dt:
            dt.now.return_value = NOW
            return await content.get_my_plan(db, profile)

    def review_task(self, plan):
        tasks = [t for t in plan.tasks if t.kind == "review"]
        return tasks[0] if tasks else None

    async def test_a_new_learner_is_offered_todays_new_cards(self):
        task = self.review_task(await self.plan([]))
        self.assertEqual((task.title, task.subtitle, task.status), ("Ôn tập hôm nay", "8 thẻ mới", "todo"))

    async def test_due_and_new_are_both_counted_and_the_review_comes_first(self):
        plan = await self.plan([state("vocab_item", 1), state("vocab_item", 2, due_at=NOW + 3 * DAY)])
        self.assertEqual(plan.tasks[0].kind, "review")
        self.assertEqual(plan.tasks[0].subtitle, "1 thẻ đến hạn · 8 thẻ mới")

    async def test_a_sitting_in_progress_says_so(self):
        states = [state("vocab_item", 1, last_seen=NOW - timedelta(hours=1))]
        self.assertEqual(self.review_task(await self.plan(states)).status, "in_progress")

    async def test_when_nothing_is_left_after_studying_today_the_task_is_done(self):
        every = [("vocab_item", v.id) for v in CONTENT_V] + [("grammar_point", g.id) for g in CONTENT_G]
        states = [state(t, i, last_seen=NOW - timedelta(hours=1), due_at=NOW + 3 * DAY) for t, i in every]
        task = self.review_task(await self.plan(states))
        self.assertEqual(task.status, "done")

    async def test_nothing_due_and_nothing_studied_today_shows_no_review_task(self):
        every = [("vocab_item", v.id) for v in CONTENT_V] + [("grammar_point", g.id) for g in CONTENT_G]
        states = [state(t, i, last_seen=NOW - 2 * DAY, due_at=NOW + 3 * DAY) for t, i in every]
        self.assertIsNone(self.review_task(await self.plan(states)))


if __name__ == "__main__":
    unittest.main()
