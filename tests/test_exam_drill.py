"""Choosing and grading real exam questions for the mini-drill: pure functions."""
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from app.services import exam_drill
from app.services.exam_drill import Attempt, Candidate

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


def cand(n, qtype="read_blank", **kw):
    base = dict(
        item_id=uuid.UUID(int=n), number=n, qtype_code=qtype, stem_ko=f"( ) 문제 {n}", options=["가", "나", "다", "라"],
        answer=2, answer_source="editor", skill="đọc", passage_kind=None, passage_ko=None,
    )
    return Candidate(**{**base, **kw})


def attempt(n, correct, ago):
    return Attempt(item_id=uuid.UUID(int=n), correct=correct, at=NOW - ago)


class UsableTests(unittest.TestCase):
    def test_a_confirmed_reading_question_is_usable(self):
        self.assertTrue(exam_drill.usable(cand(1)))

    def test_an_answer_guessed_by_the_model_is_never_used(self):
        self.assertFalse(exam_drill.usable(cand(1, answer_source="ai_guess")))
        self.assertFalse(exam_drill.usable(cand(1, answer_source=None)))

    def test_a_question_without_an_answer_is_not_usable(self):
        self.assertFalse(exam_drill.usable(cand(1, answer=None)))

    def test_listening_questions_are_left_out(self):
        self.assertFalse(exam_drill.usable(cand(1, skill="nghe")))
        self.assertFalse(exam_drill.usable(cand(1, passage_kind="nghe")))

    def test_the_answer_must_be_one_of_the_options(self):
        self.assertFalse(exam_drill.usable(cand(1, answer=5)))
        self.assertFalse(exam_drill.usable(cand(1, answer=0)))
        self.assertTrue(exam_drill.usable(cand(1, answer=4)))

    def test_options_must_be_a_list_of_two_to_five_non_empty_strings(self):
        for bad in (None, {}, "가나다", ["가"], ["가", "나", "다", "라", "마", "바"], ["가", "", "다"], ["가", 2, "다"]):
            self.assertFalse(exam_drill.usable(cand(1, options=bad, answer=1)), bad)

    def test_a_blank_stem_is_not_usable(self):
        self.assertFalse(exam_drill.usable(cand(1, stem_ko="  ")))

    def test_a_passage_question_needs_its_passage_text(self):
        self.assertFalse(exam_drill.usable(cand(1, qtype="read_short_passage")))
        self.assertFalse(exam_drill.usable(cand(1, qtype="read_chart_info", passage_ko=" ")))
        self.assertTrue(exam_drill.usable(cand(1, qtype="read_short_passage", passage_ko="글")))

    def test_clean_options_trims(self):
        self.assertEqual(exam_drill.clean_options([" 가 ", "나"]), ["가", "나"])


class PickTests(unittest.TestCase):
    def ids(self, picked):
        return [c.number for c in picked]

    def test_it_asks_the_requested_number_of_usable_questions(self):
        cs = [cand(i) for i in range(1, 9)] + [cand(20, skill="nghe")]
        picked = exam_drill.pick(cs, [], {}, NOW, n=5)
        self.assertEqual(len(picked), 5)
        self.assertNotIn(20, self.ids(picked))

    def test_the_size_is_capped(self):
        cs = [cand(i) for i in range(1, 30)]
        self.assertEqual(len(exam_drill.pick(cs, [], {}, NOW, n=99)), exam_drill.MAX_DRILL_SIZE)

    def test_a_question_answered_wrong_comes_before_new_ones_after_half_a_day(self):
        cs = [cand(1), cand(2), cand(3)]
        picked = exam_drill.pick(cs, [attempt(2, False, DAY)], {}, NOW, n=3)
        self.assertEqual(self.ids(picked)[0], 2)

    def test_a_question_answered_wrong_a_moment_ago_is_not_asked_again_yet(self):
        picked = exam_drill.pick([cand(1), cand(2)], [attempt(1, False, 10 * HOUR)], {}, NOW, n=5)
        self.assertEqual(self.ids(picked), [2])

    def test_nothing_is_asked_twice_in_one_sitting(self):
        picked = exam_drill.pick([cand(1), cand(2)], [attempt(1, False, timedelta(minutes=5))], {}, NOW, n=5)
        self.assertEqual(self.ids(picked), [2])

    def test_a_question_answered_right_this_week_is_left_alone_then_returns_last(self):
        cs = [cand(1), cand(2), cand(3)]
        picked = exam_drill.pick(cs, [attempt(1, True, 2 * DAY)], {}, NOW, n=5)
        self.assertEqual(sorted(self.ids(picked)), [2, 3])
        later = exam_drill.pick(cs, [attempt(1, True, 9 * DAY)], {}, NOW, n=5)
        self.assertEqual(self.ids(later)[-1], 1)

    def test_only_the_latest_attempt_counts(self):
        history = [attempt(1, False, 3 * DAY), attempt(1, True, 2 * DAY)]
        self.assertEqual(self.ids(exam_drill.pick([cand(1)], history, {}, NOW)), [])

    def test_the_weakest_question_type_comes_first_within_a_group(self):
        cs = [cand(1, "read_order"), cand(2, "read_blank"), cand(3, "read_order"), cand(4, "read_blank")]
        picked = exam_drill.pick(cs, [], {"read_blank": 20, "read_order": 90}, NOW, n=4)
        self.assertEqual({c.qtype_code for c in picked[:2]}, {"read_blank"})

    def test_the_same_seed_gives_the_same_set_and_a_new_seed_can_differ(self):
        cs = [cand(i) for i in range(1, 25)]
        a = self.ids(exam_drill.pick(cs, [], {}, NOW, n=5, seed="x"))
        b = self.ids(exam_drill.pick(cs, [], {}, NOW, n=5, seed="x"))
        c = self.ids(exam_drill.pick(cs, [], {}, NOW, n=5, seed="y"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_without_questions_the_drill_is_empty(self):
        self.assertEqual(exam_drill.pick([], [], {}, NOW), [])
        self.assertEqual(exam_drill.pick([cand(1)], [], {}, NOW, n=0), [])


class GradeTests(unittest.TestCase):
    def test_the_key_is_one_based(self):
        c = cand(1, answer=2)
        self.assertTrue(exam_drill.grade(c, 2))
        self.assertFalse(exam_drill.grade(c, 1))

    def test_accuracy_per_type_is_a_whole_percent(self):
        out = exam_drill.accuracy_by_type([("a", True), ("a", False), ("a", False), ("b", True)])
        self.assertEqual(out, {"a": 33, "b": 100})


if __name__ == "__main__":
    unittest.main()
