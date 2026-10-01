"""Fill-in-the-blank questions built from cards, without a model call."""
import unittest
from types import SimpleNamespace

from app.services.exercises import BLANK, MAX_CHOICES, MIN_CHOICES, build_cloze


def card(id, hangul, **kw):
    base = dict(id=id, hangul=hangul, pos=None, example_ko=None, family=None, node_word=None, distractors=None, lesson_id=1)
    return SimpleNamespace(**{**base, **kw})


FAMILY = "Động từ đi với thời tiết"
RAIN = card(1, "비가 오다", node_word="오다", family=FAMILY, distractors=["내리다", "떨어지다"])
SNOW = card(2, "눈이 오다", node_word="오다", family=FAMILY)
WIND = card(3, "바람이 불다", node_word="불다", family=FAMILY)
FLOWER = card(4, "꽃이 피다", node_word="피다", family=FAMILY)
LEAVES = card(5, "단풍이 들다", node_word="들다", family=FAMILY)
ALL = [RAIN, SNOW, WIND, FLOWER, LEAVES]


class ChunkClozeTests(unittest.TestCase):
    def test_the_node_word_is_blanked_and_is_the_answer(self):
        q = build_cloze(RAIN, ALL, "s")
        self.assertEqual((q.prompt_ko, q.answer), (f"비가 {BLANK}", "오다"))
        self.assertIn("오다", q.choices)

    def test_the_cards_own_distractors_come_first_then_the_familys_other_verbs(self):
        q = build_cloze(RAIN, ALL, "s")
        self.assertEqual(len(q.choices), MAX_CHOICES)
        self.assertEqual(sorted(q.choices), sorted(["오다", "내리다", "떨어지다", "불다"]))  # 2 own, then 불다 from 바람이 불다

    def test_the_answer_is_never_among_the_wrong_choices_twice(self):
        # 눈이 오다 has the same node word as the answer: it must not appear as a second 오다
        q = build_cloze(RAIN, ALL, "s")
        self.assertEqual(q.choices.count("오다"), 1)

    def test_a_chunk_without_distractors_borrows_from_the_family(self):
        q = build_cloze(WIND, ALL, "s")
        self.assertEqual(sorted(q.choices), sorted(["불다", "오다", "피다", "들다"]))

    def test_the_family_only_lends_its_own_words(self):
        other = card(9, "우산을 쓰다", node_word="쓰다", family="Đồ dùng")
        q = build_cloze(WIND, [WIND, SNOW, FLOWER, other], "s")
        self.assertNotIn("쓰다", q.choices)

    def test_too_few_choices_means_no_question(self):
        self.assertIsNone(build_cloze(WIND, [WIND, SNOW], "s"))  # one wrong choice would be a coin flip

    def test_a_node_word_that_is_not_part_of_the_phrase_is_not_used(self):
        broken = card(7, "비가 오다", node_word="내리다", distractors=["불다", "들다"], family=FAMILY)
        self.assertIsNone(build_cloze(broken, [broken], "s"))  # falls through to the example route, which has none

    def test_a_node_word_equal_to_the_whole_card_is_not_a_chunk(self):
        single = card(8, "날씨", node_word="날씨", distractors=["기온", "하늘"])
        self.assertIsNone(build_cloze(single, [single], "s"))

    def test_the_order_is_stable_for_one_seed_and_varies_between_seeds(self):
        self.assertEqual(build_cloze(RAIN, ALL, "a").choices, build_cloze(RAIN, ALL, "a").choices)
        orders = {build_cloze(RAIN, ALL, f"seed{i}").choices for i in range(20)}
        self.assertGreater(len(orders), 1)

    def test_the_prompt_keeps_the_rest_of_the_phrase(self):
        card_ = card(1, "우산을 가지고 오다", node_word="가지고 오다", distractors=["쓰다", "들다"])
        q = build_cloze(card_, [card_], "s")
        self.assertEqual(q.prompt_ko, f"우산을 {BLANK}")


class ExampleClozeTests(unittest.TestCase):
    NOUN = card(10, "날씨", pos="명사", example_ko="오늘 날씨가 좋아요.")
    OTHERS = [
        card(11, "기온", pos="명사", example_ko="기온이 높아요."),
        card(12, "하늘", pos="명사"),
        card(13, "바람", pos="명사"),
        card(14, "맑다", pos="형용사"),
    ]

    def test_a_word_found_verbatim_in_its_example_is_blanked(self):
        q = build_cloze(self.NOUN, [self.NOUN, *self.OTHERS], "s")
        self.assertEqual((q.prompt_ko, q.answer), (f"오늘 {BLANK}가 좋아요.", "날씨"))
        self.assertEqual(sorted(q.choices), sorted(["날씨", "기온", "하늘", "바람"]))  # same part of speech only

    def test_a_word_of_another_part_of_speech_is_never_offered(self):
        q = build_cloze(self.NOUN, [self.NOUN, *self.OTHERS], "s")
        self.assertNotIn("맑다", q.choices)

    def test_without_a_part_of_speech_there_is_no_safe_pool(self):
        bare = card(10, "날씨", example_ko="오늘 날씨가 좋아요.")
        self.assertIsNone(build_cloze(bare, [bare, *self.OTHERS], "s"))

    def test_a_conjugated_verb_is_matched_by_its_stem_when_the_stem_is_long_enough(self):
        verb = card(20, "따뜻하다", pos="형용사", example_ko="날씨가 따뜻하고 좋아요.")
        mates = [verb, card(21, "선선하다", pos="형용사"), card(22, "쌀쌀하다", pos="형용사"), card(23, "시원하다", pos="형용사")]
        q = build_cloze(verb, mates, "s")
        self.assertEqual((q.prompt_ko, q.answer), (f"날씨가 {BLANK}고 좋아요.", "따뜻하"))
        self.assertEqual(sorted(q.choices), sorted(["따뜻하", "선선하", "쌀쌀하", "시원하"]))

    def test_a_short_stem_is_not_matched_by_accident(self):
        verb = card(30, "오다", pos="동사", example_ko="비가 와요, 오늘은 오후에 와요.")
        mates = [verb, card(31, "가다", pos="동사"), card(32, "먹다", pos="동사"), card(33, "보다", pos="동사")]
        self.assertIsNone(build_cloze(verb, mates, "s"))

    def test_a_card_without_an_example_or_a_node_has_no_question(self):
        bare = card(40, "봄", pos="명사")
        self.assertIsNone(build_cloze(bare, [bare, *self.OTHERS], "s"))

    def test_a_mate_already_in_the_sentence_is_not_a_wrong_choice(self):
        noun = card(50, "날씨", pos="명사", example_ko="날씨가 좋고 기온도 높아요.")
        mates = [noun, card(51, "기온", pos="명사"), card(52, "하늘", pos="명사"), card(53, "바람", pos="명사")]
        q = build_cloze(noun, mates, "s")
        self.assertNotIn("기온", q.choices)  # it is right there in the sentence

    def test_there_are_always_at_least_the_minimum_number_of_choices(self):
        q = build_cloze(self.NOUN, [self.NOUN, *self.OTHERS], "s")
        self.assertGreaterEqual(len(q.choices), MIN_CHOICES)


if __name__ == "__main__":
    unittest.main()
