"""Progress logic: which lesson a learner sees next (pure function), and the
two routes that use it — /progress/reviews rejects unknown items, /me/plan
follows the learner's own progress. DB session is faked (no Postgres)."""
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from app.api.routers import content
from app.schemas import ItemStateReviewRequest
from app.services.progress import MASTERY_THRESHOLD, is_mastered, pick_next_lesson

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def lessons():
    return {
        1: [("vocab_item", 1), ("vocab_item", 2), ("grammar_point", 1)],
        2: [("vocab_item", 3), ("vocab_item", 4)],
        3: [("vocab_item", 5)],
    }


class MasteryTests(unittest.TestCase):
    def test_threshold_tolerates_float_accumulation(self):
        strength = 0.0
        for _ in range(4):  # four correct nudges of +0.2
            strength += 0.2
        self.assertTrue(is_mastered(strength))
        self.assertTrue(is_mastered(MASTERY_THRESHOLD))
        self.assertFalse(is_mastered(0.6))
        self.assertFalse(is_mastered(None))


class PickNextLessonTests(unittest.TestCase):
    def test_new_learner_starts_at_first_lesson(self):
        got = pick_next_lesson(lessons(), {})
        self.assertEqual((got.lesson_id, got.total, got.mastered), (1, 3, 0))

    def test_finished_lesson_moves_learner_on(self):
        states = {k: (0.8, NOW) for k in lessons()[1]}
        got = pick_next_lesson(lessons(), states)
        self.assertEqual(got.lesson_id, 2)

    def test_resumes_a_half_done_lesson_and_counts_mastered(self):
        states = {("vocab_item", 1): (1.0, NOW), ("vocab_item", 2): (0.4, NOW), ("grammar_point", 1): (0.8, NOW)}
        got = pick_next_lesson(lessons(), states)
        self.assertEqual((got.lesson_id, got.mastered, got.total), (1, 2, 3))

    def test_forgotten_item_pulls_learner_back(self):
        # lesson 1 was mastered, then a wrong answer dropped one item below 0.8
        states = {k: (0.8, NOW) for k in lessons()[1]}
        states[("vocab_item", 2)] = (0.6, NOW)
        self.assertEqual(pick_next_lesson(lessons(), states).lesson_id, 1)

    def test_lessons_without_items_are_skipped(self):
        got = pick_next_lesson({1: [], 2: [("vocab_item", 3)]}, {})
        self.assertEqual(got.lesson_id, 2)

    def test_no_content_returns_none(self):
        self.assertIsNone(pick_next_lesson({}, {}))
        self.assertIsNone(pick_next_lesson({1: []}, {}))

    def test_states_for_unrelated_items_are_ignored(self):
        # e.g. article vocab (no lesson) that the learner reviewed
        got = pick_next_lesson(lessons(), {("vocab_item", 999): (1.0, NOW)})
        self.assertEqual((got.lesson_id, got.mastered), (1, 0))

    def test_everything_mastered_refreshes_the_stalest_lesson(self):
        states = {}
        for lid, days_ago in ((1, 1), (2, 9), (3, 4)):
            for key in lessons()[lid]:
                states[key] = (1.0, NOW - timedelta(days=days_ago))
        got = pick_next_lesson(lessons(), states)
        self.assertEqual(got.lesson_id, 2)
        self.assertTrue(got.complete)


def _rows(rows):
    res = mock.MagicMock()
    res.all.return_value = rows
    return res


class RecordReviewTests(unittest.IsolatedAsyncioTestCase):
    def _db(self, *, item_exists):
        db = mock.MagicMock()
        db.add = mock.MagicMock()
        db.commit = mock.AsyncMock()
        db.refresh = mock.AsyncMock()

        async def execute(stmt):
            text = str(stmt)
            res = mock.MagicMock()
            if "item_state" in text:
                res.scalar_one_or_none.return_value = None
            else:  # existence probe on the vocab/grammar table
                res.first.return_value = (1,) if item_exists else None
            return res

        db.execute = execute
        return db

    async def test_unknown_item_is_404_and_writes_nothing(self):
        db = self._db(item_exists=False)
        body = ItemStateReviewRequest(item_type="vocab_item", item_id=999999, correct=True)
        with self.assertRaises(HTTPException) as ctx:
            await content.record_item_review(body, db, SimpleNamespace(id=uuid.uuid4()))
        self.assertEqual(ctx.exception.status_code, 404)
        db.add.assert_not_called()
        db.commit.assert_not_awaited()

    async def test_known_item_records_a_nudge(self):
        db = self._db(item_exists=True)
        body = ItemStateReviewRequest(item_type="grammar_point", item_id=7, correct=True)
        await content.record_item_review(body, db, SimpleNamespace(id=uuid.uuid4()))
        (created,) = db.add.call_args.args
        self.assertEqual((created.item_type, created.item_id), ("grammar_point", 7))
        self.assertAlmostEqual(created.strength, 0.2)
        db.commit.assert_awaited_once()


class PlanTests(unittest.IsolatedAsyncioTestCase):
    def _db(self, states, *, vocab, grammar, lessons_by_id):
        db = mock.MagicMock()

        async def get(model, pk):
            return lessons_by_id.get(pk)

        db.get = get

        async def execute(stmt):
            text = str(stmt)
            res = mock.MagicMock()
            if "FROM item_state" in text:
                res.scalars.return_value.all.return_value = states
            elif "FROM content.vocab_item" in text:
                res.all.return_value = vocab
            elif "FROM content.grammar_point" in text:
                res.all.return_value = grammar
            elif "count(" in text:
                res.scalar_one.return_value = 0
            else:  # editorial article / submission: none
                res.scalars.return_value.first.return_value = None
                res.scalar_one_or_none.return_value = None
            return res

        db.execute = execute
        return db

    def _state(self, item_type, item_id, strength, when=NOW):
        return SimpleNamespace(item_type=item_type, item_id=item_id, strength=strength, last_seen=when)

    async def _plan(self, states):
        db = self._db(
            states,
            vocab=[(1, 1), (1, 2), (2, 3), (None, 50)],  # 50 = article vocab, no lesson
            grammar=[(1, 1)],
            lessons_by_id={
                1: SimpleNamespace(id=1, title="Bài 1"),
                2: SimpleNamespace(id=2, title="Bài 2"),
            },
        )
        profile = SimpleNamespace(id=uuid.uuid4(), goal="topik4", exam_date=None)
        with mock.patch.object(content, "datetime", wraps=datetime) as dt:
            dt.now.return_value = NOW
            return await content.get_my_plan(db, profile)

    async def test_new_learner_gets_first_lesson(self):
        plan = await self._plan([])
        (task,) = plan.tasks
        self.assertEqual((task.kind, task.lesson_id, task.status), ("vocab_review", 1, "todo"))
        self.assertEqual(task.subtitle, "3 từ/ngữ pháp")

    async def test_mastering_lesson_one_moves_plan_to_lesson_two(self):
        old = NOW - timedelta(days=2)
        states = [self._state(t, i, 0.8, old) for t, i in (("vocab_item", 1), ("vocab_item", 2), ("grammar_point", 1))]
        # article vocab reviewed today must not count toward any lesson
        states.append(self._state("vocab_item", 50, 1.0))
        plan = await self._plan(states)
        (task,) = plan.tasks
        self.assertEqual((task.lesson_id, task.title, task.status), (2, "Bài 2", "todo"))

    async def test_half_done_lesson_reports_mastered_and_today_counts(self):
        states = [self._state("vocab_item", 1, 1.0), self._state("vocab_item", 2, 0.2)]
        plan = await self._plan(states)
        (task,) = plan.tasks
        self.assertEqual((task.lesson_id, task.status), (1, "in_progress"))
        self.assertEqual(task.subtitle, "3 từ/ngữ pháp · đã thuộc 1/3 · đã ôn 2/3 hôm nay")


if __name__ == "__main__":
    unittest.main()
