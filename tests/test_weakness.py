"""The weakness report's classification and counting: pure functions."""
import unittest

from app.services import weakness
from app.services.weakness import ErrorRow


def row(error_type, item=None, skill="vocab"):
    return ErrorRow(skill=skill, error_type=error_type, item_type=item[0] if item else None, item_id=item[1] if item else None)


V = lambda i: ("vocab_item", i)  # noqa: E731


class ErrorTypeTests(unittest.TestCase):
    def test_a_wrong_blank_on_a_chunk_is_a_collocation_mistake(self):
        self.assertEqual(weakness.error_type_for_review("vocab_item", "cloze", True), "collocation")

    def test_a_wrong_blank_on_a_plain_word_is_a_word_in_context_mistake(self):
        self.assertEqual(weakness.error_type_for_review("vocab_item", "cloze", False), "word_in_context")

    def test_a_card_judged_not_remembered_is_a_recall_failure(self):
        self.assertEqual(weakness.error_type_for_review("vocab_item", "recognize", True), "vocab_recall")
        self.assertEqual(weakness.error_type_for_review("grammar_point", "recognize", False), "grammar_recall")

    def test_every_kind_has_a_label_a_learner_can_read(self):
        for kind in ("collocation", "word_in_context", "vocab_recall", "grammar_recall", "writing_grammar",
                     "writing_spelling", "writing_register", "writing_word_choice", "writing_content"):
            self.assertNotEqual(weakness.label_for(kind), kind)

    def test_an_exam_question_type_uses_its_own_name(self):
        self.assertEqual(weakness.label_for("read_blank", {"read_blank": "Điền chỗ trống"}), "Điền chỗ trống")

    def test_an_unknown_kind_falls_back_to_its_code(self):
        self.assertEqual(weakness.label_for("mystery"), "mystery")


class CountTests(unittest.TestCase):
    def test_kinds_are_listed_most_frequent_first(self):
        rows = [row("vocab_recall")] * 2 + [row("collocation")] * 5 + [row("grammar_recall", skill="grammar")]
        out = weakness.count_by_type(rows)
        self.assertEqual([(c.error_type, c.count) for c in out], [("collocation", 5), ("vocab_recall", 2), ("grammar_recall", 1)])
        self.assertEqual(out[0].label, weakness.ERROR_LABELS["collocation"])
        self.assertEqual(out[2].skill, "grammar")

    def test_ties_are_ordered_by_name_so_the_list_does_not_jump_around(self):
        out = weakness.count_by_type([row("b"), row("a")])
        self.assertEqual([c.error_type for c in out], ["a", "b"])

    def test_no_errors_means_an_empty_report(self):
        self.assertEqual(weakness.count_by_type([]), [])
        self.assertEqual(weakness.repeat_items([]), [])

    def test_a_card_is_hay_sai_only_after_going_wrong_twice(self):
        rows = [row("vocab_recall", V(1)), row("collocation", V(2)), row("collocation", V(2)), row("collocation", V(2))]
        self.assertEqual(weakness.repeat_items(rows), [weakness.RepeatItem(V(2), 3)])

    def test_rows_without_a_card_are_not_counted_as_repeats(self):
        rows = [row("writing_grammar", skill="viết")] * 3
        self.assertEqual(weakness.repeat_items(rows), [])

    def test_the_list_of_repeat_items_is_cut(self):
        rows = [row("collocation", V(i)) for i in range(1, 11) for _ in range(2)]
        self.assertEqual(len(weakness.repeat_items(rows, limit=4)), 4)

    def test_mistakes_gather_in_families_but_one_slip_is_not_a_pattern(self):
        family_of = {V(1): "Động từ đi với thời tiết", V(2): "Động từ đi với thời tiết", V(3): "Thang nhiệt độ", V(4): None}
        rows = [row("collocation", V(1)), row("collocation", V(2)), row("collocation", V(1)),
                row("collocation", V(3)), row("collocation", V(4)), row("collocation", V(4))]
        out = weakness.count_by_family(rows, family_of)
        self.assertEqual(out, [weakness.FamilyCount("Động từ đi với thời tiết", 3)])


class ExamAccuracyTests(unittest.TestCase):
    NAMES = {"read_blank": "Điền chỗ trống", "read_order": "Sắp xếp câu"}

    def test_the_weakest_type_comes_first_and_counts_are_kept(self):
        attempts = [("read_blank", True), ("read_blank", False), ("read_blank", False),
                    ("read_order", True), ("read_order", True)]
        out = weakness.exam_accuracy(attempts, self.NAMES)
        self.assertEqual([(a.qtype_code, a.attempts, a.correct, a.accuracy_pct) for a in out],
                         [("read_blank", 3, 1, 33), ("read_order", 2, 2, 100)])
        self.assertEqual(out[0].name, "Điền chỗ trống")

    def test_a_type_without_a_name_shows_its_code(self):
        self.assertEqual(weakness.exam_accuracy([("x", True)], {})[0].name, "x")

    def test_without_attempts_there_is_nothing_to_report(self):
        self.assertEqual(weakness.exam_accuracy([], self.NAMES), [])


if __name__ == "__main__":
    unittest.main()
