"""Writing practice (TOPIK 51–52 style): choosing cards, checking the generated
exercise, checking the learner's sentences. The model is a fake."""
import json
import unittest

from app.services import writing_drill as wd
from app.services.writing_drill import SourceCard

FAMILY = "Động từ đi với thời tiết"
RAIN = SourceCard(1, "비가 오다", "trời mưa", node_word="오다", family=FAMILY, errors=3, shaky=True)
SNOW = SourceCard(2, "눈이 오다", "tuyết rơi", node_word="오다", family=FAMILY)
UMBRELLA = SourceCard(3, "우산을 쓰다", "che ô", node_word="쓰다", family="Động từ đi với danh từ", errors=1)
WORD = SourceCard(4, "날씨", "thời tiết")


def good_exercise(**over):
    data = {
        "text_type": "공지",
        "title_ko": "야외 행사 안내",
        "body_ko": "다음 주 토요일에 야외 행사가 있습니다. ㉠ 행사는 오전 10시에 시작합니다. ㉡",
        "register": "합쇼체",
        "blanks": [
            {"label": "㉠", "intent_vi": "Nói rằng nếu trời mưa thì cần mang ô", "model_answer": "비가 오면 우산을 쓰고 오십시오.",
             "alt_answers": ["비가 오면 우산을 준비해 오십시오."], "uses": ["비가 오다", "우산을 쓰다", "không có trong danh sách"]},
            {"label": "㉡", "intent_vi": "Mời mọi người tham dự", "model_answer": "많은 참석 바랍니다.", "alt_answers": [], "uses": []},
        ],
    }
    data.update(over)
    return data


class FakeModel:
    """Answers each kind of call from a queue, and remembers the prompts."""

    def __init__(self, generated=(), verified=(), graded=()):
        self.generated = list(generated)
        self.verified = list(verified)
        self.graded = list(graded)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        schema = kw["response_schema"]
        queue = {id(wd.GENERATE_SCHEMA): self.generated, id(wd.VERIFY_SCHEMA): self.verified, id(wd.GRADE_SCHEMA): self.graded}[id(schema)]
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"text": item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}


OK_CHECK = {"text_ok": True, "uses_target": True, "blanks": [{"label": "㉠", "ok": True}, {"label": "㉡", "ok": True}]}


class PickSourcesTests(unittest.TestCase):
    def test_the_cards_the_learner_gets_wrong_most_come_first(self):
        picked = wd.pick_sources([WORD, SNOW, RAIN, UMBRELLA])
        self.assertEqual(picked[0].item_id, 1)

    def test_the_second_card_is_from_the_same_set_when_there_is_one(self):
        picked = wd.pick_sources([UMBRELLA, SNOW, RAIN, WORD])
        self.assertEqual([c.item_id for c in picked], [1, 2])

    def test_without_a_mate_the_next_best_card_fills_in(self):
        picked = wd.pick_sources([RAIN, UMBRELLA, WORD])
        self.assertEqual([c.item_id for c in picked], [1, 3])

    def test_cards_used_recently_are_skipped_unless_nothing_else_is_left(self):
        self.assertEqual(wd.pick_sources([RAIN, SNOW, UMBRELLA], recently_used=[1])[0].item_id, 3)
        self.assertEqual([c.item_id for c in wd.pick_sources([RAIN], recently_used=[1])], [1])

    def test_a_chunk_beats_a_single_word_when_nothing_else_differs(self):
        self.assertEqual(wd.pick_sources([WORD, SNOW])[0].item_id, 2)

    def test_no_cards_means_no_exercise(self):
        self.assertEqual(wd.pick_sources([]), [])

    def test_the_count_is_respected(self):
        self.assertEqual(len(wd.pick_sources([RAIN, SNOW, UMBRELLA, WORD], count=1)), 1)


class SentenceToolsTests(unittest.TestCase):
    def test_one_sentence_is_counted_as_one(self):
        self.assertEqual(wd.sentence_count("비가 오면 우산을 쓰고 오십시오."), 1)
        self.assertEqual(wd.sentence_count("비가 와요"), 1)

    def test_two_sentences_are_two(self):
        self.assertEqual(wd.sentence_count("비가 와요. 우산을 쓰세요."), 2)

    def test_a_decimal_point_is_not_a_sentence_end(self):
        self.assertEqual(wd.sentence_count("기온이 3.5도 올랐어요."), 1)

    def test_register_is_read_from_the_ending(self):
        self.assertEqual(wd.register_of("참석해 주십시오."), "합쇼체")
        self.assertEqual(wd.register_of("행사가 있습니다."), "합쇼체")
        self.assertEqual(wd.register_of("우산을 가져오세요."), "해요체")
        self.assertEqual(wd.register_of("같이 가시죠?"), "해요체")
        self.assertEqual(wd.register_of("비가 온다."), "plain")
        self.assertIsNone(wd.register_of("참석 바람"))
        self.assertIsNone(wd.register_of(""))

    def test_han_characters_are_found(self):
        self.assertTrue(wd.has_han("學校에 갑니다"))
        self.assertFalse(wd.has_han("학교에 갑니다"))


class ParseGeneratedTests(unittest.TestCase):
    TARGETS = [RAIN, UMBRELLA]

    def test_a_good_exercise_is_kept_and_only_real_targets_are_listed_as_used(self):
        out = wd.parse_generated(good_exercise(), self.TARGETS)
        self.assertEqual([b["label"] for b in out["blanks"]], ["㉠", "㉡"])
        self.assertEqual(out["blanks"][0]["uses"], ["비가 오다", "우산을 쓰다"])
        self.assertEqual(out["targets"], [{"item_id": 1, "hangul": "비가 오다", "meaning_vi": "trời mưa"},
                                          {"item_id": 3, "hangul": "우산을 쓰다", "meaning_vi": "che ô"}])
        self.assertEqual((out["text_type"], out["register"]), ("공지", "합쇼체"))

    def test_each_blank_marker_must_appear_exactly_once(self):
        self.assertIsNone(wd.parse_generated(good_exercise(body_ko="㉠ 안내입니다. ㉠ 감사합니다."), self.TARGETS))
        self.assertIsNone(wd.parse_generated(good_exercise(body_ko="안내입니다. ㉠ 감사합니다."), self.TARGETS))

    def test_there_must_be_two_distinct_blanks(self):
        one = good_exercise()
        one["blanks"] = one["blanks"][:1]
        self.assertIsNone(wd.parse_generated(one, self.TARGETS))
        dup = good_exercise()
        dup["blanks"][1]["label"] = "㉠"
        self.assertIsNone(wd.parse_generated(dup, self.TARGETS))

    def test_a_model_answer_must_be_one_clean_korean_sentence(self):
        for bad in ("", "비가 옵니다. 우산을 쓰십시오.", "雨が降ります", "x" * 90 + "입니다.", "rain"):
            data = good_exercise()
            data["blanks"][0]["model_answer"] = bad
            self.assertIsNone(wd.parse_generated(data, self.TARGETS), bad)

    def test_han_characters_in_the_text_are_rejected(self):
        self.assertIsNone(wd.parse_generated(good_exercise(body_ko="學校 행사입니다. ㉠ ㉡"), self.TARGETS))

    def test_a_blank_without_an_intent_is_rejected(self):
        data = good_exercise()
        data["blanks"][1]["intent_vi"] = " "
        self.assertIsNone(wd.parse_generated(data, self.TARGETS))

    def test_alternatives_are_cleaned_and_capped(self):
        data = good_exercise()
        data["blanks"][0]["alt_answers"] = ["비가 오면 우산을 쓰고 오십시오.", "A 입니다. B 입니다.", "우산을 가져오십시오.", "우산을 챙기십시오.", "우산을 준비하십시오."]
        alts = wd.parse_generated(data, self.TARGETS)["blanks"][0]["alt_answers"]
        self.assertEqual(alts, ["우산을 가져오십시오.", "우산을 챙기십시오."])

    def test_an_unknown_text_type_and_register_get_safe_defaults(self):
        out = wd.parse_generated(good_exercise(text_type="편지", register="반말"), self.TARGETS)
        self.assertEqual((out["text_type"], out["register"]), ("안내문", "합쇼체"))  # register read from the text

    def test_garbage_is_rejected_not_raised(self):
        for data in ({}, {"body_ko": 5}, {"body_ko": "㉠㉡", "blanks": "x"}, {"body_ko": "㉠㉡", "blanks": [1, 2]}):
            self.assertIsNone(wd.parse_generated(data, self.TARGETS))


class VerifiedTests(unittest.TestCase):
    def test_everything_ok_passes(self):
        self.assertEqual(wd.verified(OK_CHECK), (True, None))

    def test_one_doubtful_answer_fails_with_its_reason(self):
        check = {**OK_CHECK, "blanks": [{"label": "㉠", "ok": True}, {"label": "㉡", "ok": False, "issue_vi": "không tự nhiên"}]}
        self.assertEqual(wd.verified(check), (False, "không tự nhiên"))

    def test_a_doubtful_text_fails(self):
        self.assertFalse(wd.verified({**OK_CHECK, "text_ok": False, "issue_vi": "văn bản lạ"})[0])

    def test_answers_that_do_not_use_the_target_fail(self):
        self.assertFalse(wd.verified({**OK_CHECK, "uses_target": False})[0])

    def test_a_missing_blank_fails(self):
        self.assertFalse(wd.verified({**OK_CHECK, "blanks": [{"label": "㉠", "ok": True}]})[0])


class BuildExerciseTests(unittest.TestCase):
    def build(self, model):
        return wd.build_exercise([RAIN, UMBRELLA], 4, model, "m")

    def test_a_good_exercise_passes_both_calls(self):
        model = FakeModel(generated=[good_exercise()], verified=[OK_CHECK])
        out = self.build(model)
        self.assertEqual(out["blanks"][0]["model_answer"], "비가 오면 우산을 쓰고 오십시오.")
        self.assertEqual(len(model.calls), 2)
        self.assertIn("비가 오다", model.calls[0]["prompt"])
        self.assertIn("비가 오면 우산을 쓰고 오십시오.", model.calls[1]["prompt"])

    def test_an_exercise_the_checker_doubts_is_generated_again_with_the_reason(self):
        bad = {**OK_CHECK, "blanks": [{"label": "㉠", "ok": False, "issue_vi": "câu này không tự nhiên"}, {"label": "㉡", "ok": True}]}
        model = FakeModel(generated=[good_exercise(), good_exercise()], verified=[bad, OK_CHECK])
        self.build(model)
        self.assertEqual(len(model.calls), 4)
        self.assertIn("câu này không tự nhiên", model.calls[2]["prompt"])

    def test_two_doubted_exercises_give_up_rather_than_serve_a_doubtful_one(self):
        bad = {**OK_CHECK, "text_ok": False}
        model = FakeModel(generated=[good_exercise()] * 2, verified=[bad, bad])
        with self.assertRaises(ValueError):
            self.build(model)

    def test_a_rule_breaking_exercise_is_not_even_sent_to_the_checker(self):
        model = FakeModel(generated=[good_exercise(body_ko="없음"), good_exercise()], verified=[OK_CHECK])
        self.build(model)
        self.assertEqual([c["response_schema"] is wd.VERIFY_SCHEMA for c in model.calls], [False, False, True])

    def test_a_reply_that_is_not_json_is_asked_again(self):
        model = FakeModel(generated=["not json", good_exercise()], verified=[OK_CHECK])
        self.build(model)
        self.assertEqual(len(model.calls), 3)

    def test_a_fenced_json_reply_is_read(self):
        model = FakeModel(generated=["```json\n" + json.dumps(good_exercise(), ensure_ascii=False) + "\n```"], verified=[OK_CHECK])
        self.assertEqual(self.build(model)["text_type"], "공지")

    def test_a_model_outage_is_not_swallowed(self):
        model = FakeModel(generated=[RuntimeError("503")])
        with self.assertRaises(RuntimeError):
            self.build(model)


EXERCISE = wd.parse_generated(good_exercise(), [RAIN, UMBRELLA])


def ai(label, verdict="good", fixes=(), comment="Tốt."):
    return {"label": label, "verdict": verdict, "comment_vi": comment, "fixes": list(fixes)}


class LocalCheckTests(unittest.TestCase):
    def test_nothing_written_blocks_grading(self):
        self.assertIsNotNone(wd.local_check("  ", "합쇼체").blocking)

    def test_an_answer_without_korean_blocks_grading(self):
        self.assertIsNotNone(wd.local_check("I will come", "합쇼체").blocking)

    def test_two_sentences_in_one_blank_are_flagged(self):
        notes = wd.local_check("비가 옵니다. 우산을 쓰십시오.", "합쇼체").notes
        self.assertEqual([n["category"] for n in notes], ["content"])
        self.assertIn("MỘT câu", notes[0]["reason_vi"])

    def test_a_different_politeness_from_the_text_is_flagged_on_the_ending(self):
        notes = wd.local_check("비가 오면 우산을 쓰세요.", "합쇼체").notes
        (note,) = notes
        self.assertEqual(note["category"], "register")
        self.assertIn(note["original"], "비가 오면 우산을 쓰세요.")
        self.assertIn("합쇼체", note["reason_vi"])

    def test_the_same_politeness_is_not_flagged(self):
        self.assertEqual(wd.local_check("우산을 쓰십시오.", "합쇼체").notes, ())
        self.assertEqual(wd.local_check("우산을 쓰세요.", "해요체").notes, ())

    def test_an_ending_that_is_not_clear_is_not_flagged(self):
        self.assertEqual(wd.local_check("우산 지참 바람", "합쇼체").notes, ())


class AssembleResultTests(unittest.TestCase):
    ANSWERS = {"㉠": "비가 오면 우산을 쓰고 오십시오.", "㉡": "많은 참석 바랍니다."}

    def test_a_clean_pass(self):
        out = wd.assemble_result(EXERCISE, self.ANSWERS, {"blanks": [ai("㉠"), ai("㉡")]})
        self.assertEqual([b["verdict"] for b in out["blanks"]], ["good", "good"])
        self.assertTrue(out["ai_checked"])
        self.assertEqual(out["blanks"][0]["model_answer"], "비가 오면 우산을 쓰고 오십시오.")
        self.assertEqual(wd.error_rows(out), [])

    def test_a_correction_must_quote_the_learners_own_words(self):
        fixes = [
            {"original": "우산을 쓰고", "corrected": "우산을 쓰고", "category": "grammar", "reason_vi": "x"},
            {"original": "ĐIỀU BỊA RA", "corrected": "y", "category": "grammar", "reason_vi": "bịa"},
        ]
        out = wd.assemble_result(EXERCISE, self.ANSWERS, {"blanks": [ai("㉠", "minor", fixes), ai("㉡")]})
        kept = out["blanks"][0]["fixes"]
        self.assertEqual([f["original"] for f in kept], ["우산을 쓰고"])
        self.assertEqual(kept[0]["source"], "ai")

    def test_a_pass_with_a_correction_found_is_not_a_clean_pass(self):
        fixes = [{"original": "많은", "corrected": "많이", "category": "word_choice", "reason_vi": "r"}]
        out = wd.assemble_result(EXERCISE, self.ANSWERS, {"blanks": [ai("㉠"), ai("㉡", "good", fixes)]})
        self.assertEqual(out["blanks"][1]["verdict"], "minor")

    def test_an_automatic_note_lowers_a_pass_the_model_gave(self):
        answers = {"㉠": "비가 오면 우산을 쓰세요.", "㉡": "많은 참석 바랍니다."}
        out = wd.assemble_result(EXERCISE, answers, {"blanks": [ai("㉠"), ai("㉡")]})
        self.assertEqual(out["blanks"][0]["verdict"], "minor")
        self.assertEqual(out["blanks"][0]["fixes"][0]["source"], "auto")

    def test_an_empty_answer_is_off_without_asking_the_model_about_it(self):
        out = wd.assemble_result(EXERCISE, {"㉠": "", "㉡": "많은 참석 바랍니다."}, {"blanks": [ai("㉡")]})
        self.assertEqual(out["blanks"][0]["verdict"], "off")
        self.assertIn("chưa viết", out["blanks"][0]["comment_vi"])

    def test_an_unknown_verdict_is_treated_as_minor(self):
        out = wd.assemble_result(EXERCISE, self.ANSWERS, {"blanks": [ai("㉠", "excellent!!"), ai("㉡")]})
        self.assertEqual(out["blanks"][0]["verdict"], "minor")

    def test_an_unknown_fix_category_becomes_grammar_and_han_in_a_fix_is_dropped(self):
        fixes = [{"original": "많은", "corrected": "多い", "category": "vibes", "reason_vi": "r"}]
        out = wd.assemble_result(EXERCISE, self.ANSWERS, {"blanks": [ai("㉠"), ai("㉡", "minor", fixes)]})
        (fix,) = out["blanks"][1]["fixes"]
        self.assertEqual((fix["category"], fix["corrected"]), ("grammar", ""))

    def test_without_the_model_only_the_automatic_checks_remain_and_it_says_so(self):
        out = wd.assemble_result(EXERCISE, self.ANSWERS, None)
        self.assertFalse(out["ai_checked"])
        self.assertEqual([b["verdict"] for b in out["blanks"]], ["unchecked", "unchecked"])
        self.assertIn("AI chưa nhận xét", out["blanks"][0]["comment_vi"])
        self.assertEqual(wd.error_rows(out), [])

    def test_the_answer_is_capped_in_length(self):
        out = wd.assemble_result(EXERCISE, {"㉠": "가" * 1000, "㉡": "x"}, None)
        self.assertLessEqual(len(out["blanks"][0]["answer"]), wd.MAX_ANSWER_CHARS * 2)


class ErrorRowsTests(unittest.TestCase):
    def test_each_kind_of_slip_in_a_blank_is_one_row(self):
        fixes = [
            {"original": "쓰고", "corrected": "쓰고", "category": "grammar", "reason_vi": "a"},
            {"original": "오십시오", "corrected": "와 주십시오", "category": "grammar", "reason_vi": "b"},
            {"original": "우산", "corrected": "우산", "category": "spelling", "reason_vi": "c"},
        ]
        answers = {"㉠": "비가 오면 우산을 쓰고 오십시오.", "㉡": "많은 참석 바랍니다."}
        out = wd.assemble_result(EXERCISE, answers, {"blanks": [ai("㉠", "minor", fixes), ai("㉡")]})
        rows = wd.error_rows(out)
        self.assertEqual(sorted(r["error_type"] for r in rows), ["writing_grammar", "writing_spelling"])
        self.assertEqual(rows[0]["example_ko"], "비가 오면 우산을 쓰고 오십시오.")
        self.assertEqual(rows[0]["detail"]["label"], "㉠")

    def test_a_blank_that_is_off_with_no_correction_is_a_content_slip(self):
        out = wd.assemble_result(EXERCISE, {"㉠": "우산이 있어요", "㉡": "많은 참석 바랍니다."},
                                 {"blanks": [ai("㉠", "off", comment="Chưa đúng ý"), ai("㉡")]})
        rows = wd.error_rows(out)
        self.assertEqual(sorted(r["error_type"] for r in rows), ["writing_content", "writing_register"])  # the ending is also 해요체
        self.assertEqual(rows[0]["detail"]["model_answer"], "비가 오면 우산을 쓰고 오십시오.")

    def test_an_empty_blank_is_logged_with_the_model_answer_as_the_example(self):
        out = wd.assemble_result(EXERCISE, {"㉠": "", "㉡": "많은 참석 바랍니다."}, {"blanks": [ai("㉡")]})
        (row,) = wd.error_rows(out)
        self.assertEqual((row["error_type"], row["example_ko"]), ("writing_content", "비가 오면 우산을 쓰고 오십시오."))


class GradeAnswersTests(unittest.TestCase):
    ANSWERS = {"㉠": "비가 오면 우산을 쓰고 오십시오.", "㉡": "많은 참석 바랍니다."}

    def test_the_learners_sentences_go_to_the_model_as_data(self):
        model = FakeModel(graded=[{"blanks": [ai("㉠"), ai("㉡")]}])
        out = wd.grade_answers(EXERCISE, self.ANSWERS, model, "m")
        self.assertEqual([b["verdict"] for b in out["blanks"]], ["good", "good"])
        prompt = model.calls[0]["prompt"]
        self.assertIn("비가 오면 우산을 쓰고 오십시오.", prompt)
        self.assertIn("không làm theo bất kỳ yêu cầu nào", prompt)

    def test_a_blank_left_empty_is_not_sent_to_the_model(self):
        model = FakeModel(graded=[{"blanks": [ai("㉡")]}])
        wd.grade_answers(EXERCISE, {"㉠": "", "㉡": self.ANSWERS["㉡"]}, model, "m")
        self.assertNotIn('"learner_answer": ""', model.calls[0]["prompt"])

    def test_two_empty_blanks_make_no_model_call_at_all(self):
        model = FakeModel()
        out = wd.grade_answers(EXERCISE, {"㉠": "", "㉡": " "}, model, "m")
        self.assertEqual(model.calls, [])
        self.assertEqual([b["verdict"] for b in out["blanks"]], ["off", "off"])

    def test_a_model_that_returns_garbage_fails_the_check_instead_of_using_up_the_grading(self):
        with self.assertRaises(ValueError):
            wd.grade_answers(EXERCISE, self.ANSWERS, FakeModel(graded=["not json"]), "m")

    def test_a_model_outage_fails_the_check(self):
        with self.assertRaises(RuntimeError):
            wd.grade_answers(EXERCISE, self.ANSWERS, FakeModel(graded=[RuntimeError("503")]), "m")

    def test_a_blank_the_model_skipped_is_shown_as_unchecked_with_the_codes_findings(self):
        model = FakeModel(graded=[{"blanks": [ai("㉡")]}])  # the model answered only ㉡
        out = wd.grade_answers(EXERCISE, self.ANSWERS, model, "m")
        self.assertEqual([b["verdict"] for b in out["blanks"]], ["unchecked", "good"])


class PublicViewTests(unittest.TestCase):
    def test_the_learner_never_sees_the_model_answers_before_answering(self):
        view = wd.public_view(EXERCISE)
        text = json.dumps(view, ensure_ascii=False)
        self.assertNotIn("model_answer", text)
        self.assertNotIn("비가 오면 우산을 쓰고 오십시오.", text)
        self.assertEqual([b["label"] for b in view["blanks"]], ["㉠", "㉡"])
        self.assertIn("intent_vi", view["blanks"][0])

    def test_no_exercise_no_view(self):
        self.assertIsNone(wd.public_view(None))


if __name__ == "__main__":
    unittest.main()
