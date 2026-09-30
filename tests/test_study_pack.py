"""Study pack: sentence splitting, batching, validation and assembly, all
against a fake `generate` (no Gemini)."""
import unittest

from app.services import study_pack as sp

P1 = "정부가 야심 차게 추진하고 있는 3대 메가프로젝트는 반도체, 인공지능(AI) 데이터센터, 피지컬 AI(로봇) 등 3대 분야에 약 1500조원이 투자되는 단군 이래 최대 사업이다. 성장 둔화를 극복하고 지역균형발전을 도모해 대한민국의 대도약을 이루겠다는 목표는 기대감을 갖게 한다."
P2 = '"제발 아이들을 품어주세요." 지난 17일 서울 강서구 여명학교 교사 건립 주민설명회에서 조명숙 교장이 눈물로 호소했다.'
P3 = "정부 추진 18.4GW 규모 데이터센터, 전기요금 등 갈등 유발 가능성"


class SplitTests(unittest.TestCase):
    def test_two_sentences(self):
        self.assertEqual(len(sp.split_sentences(P1)), 2)

    def test_split_after_closing_quote(self):
        s = sp.split_sentences(P2)
        self.assertEqual(len(s), 2)
        self.assertTrue(s[0].endswith('주세요."'), s[0])
        self.assertTrue(s[1].startswith("지난 17일"))

    def test_decimals_and_percent_do_not_split(self):
        s = sp.split_sentences("전력 수요가 18.4GW로 늘고 금리는 5.3%를 넘었다. 다음 문장이다.")
        self.assertEqual(len(s), 2)
        self.assertIn("18.4GW", s[0])

    def test_heading_without_terminator_is_one_sentence(self):
        self.assertEqual(sp.split_sentences(P3), [P3])

    def test_tiny_numbering_fragment_merged(self):
        s = sp.split_sentences("1. 첫째로 전력 수요를 제대로 점검해야 한다. 둘째로 규제가 필요하다.")
        self.assertEqual(len(s), 2)
        self.assertTrue(s[0].startswith("1. 첫째로"))

    def test_empty_and_whitespace(self):
        self.assertEqual(sp.split_sentences(""), [])
        self.assertEqual(sp.split_sentences("  \n "), [])

    def test_deterministic_and_lossless(self):
        joined = " ".join(sp.split_sentences(P1))
        self.assertEqual(joined, P1)


class BatchPlanTests(unittest.TestCase):
    def test_groups_under_budget(self):
        lists = [["가" * 400], ["나" * 400], ["다" * 400], ["라" * 400]]
        self.assertEqual(sp.plan_batches(lists), [[0, 1, 2], [3]])

    def test_huge_paragraph_gets_its_own_batch(self):
        lists = [["가" * 100], ["나" * 3000], ["다" * 100]]
        self.assertEqual(sp.plan_batches(lists), [[0], [1], [2]])

    def test_sentence_cap(self):
        lists = [["짧다."] * 10, ["짧다."] * 10]
        self.assertEqual(sp.plan_batches(lists), [[0], [1]])

    def test_empty(self):
        self.assertEqual(sp.plan_batches([]), [])


def fake_generate_factory(*, drop_translation_for=None, fail_first_batch=False, garbage_words=False):
    calls = {"overview": 0, "batch": 0}

    def generate(prompt, schema):
        if schema is sp.OVERVIEW_SCHEMA:
            calls["overview"] += 1
            return {
                "summary_vi": "Chính phủ thúc đẩy dự án lớn.",
                "key_points_vi": ["Ý 1", "Ý 2", ""],
                "key_terms": [{"ko": "메가프로젝트", "vi": "dự án quy mô cực lớn"}, {"ko": "", "vi": "bỏ"}],
            }
        calls["batch"] += 1
        if fail_first_batch and calls["batch"] == 1:
            raise ValueError("boom")
        # parse the numbered paragraphs/sentences back out of the prompt
        import re

        paragraphs = []
        for m in re.finditer(r"Đoạn (\d+):\n((?:  \[\d+\] .*\n?)+)", prompt):
            idx = int(m.group(1))
            sents = re.findall(r"  \[(\d+)\] (.*)", m.group(2))
            paragraphs.append(
                {
                    "index": idx,
                    "easy_ko": f"쉬운 문장 {idx}.",
                    "sentences": [
                        {
                            "i": int(i),
                            "vi": "" if (drop_translation_for == idx) else f"dịch {idx}-{i}",
                            "words": (
                                [{"surface": "없는단어", "base": "없다", "pos": "danh từ", "meaning_vi": "x"}]
                                if garbage_words
                                else [
                                    {"surface": ko.split()[0], "base": None, "pos": "danh từ", "meaning_vi": "nghĩa"},
                                    {"surface": ko.split()[0], "base": None, "pos": None, "meaning_vi": "trùng"},
                                ]
                            ),
                            "grammar_notes_vi": ["-다: kết câu", "b", "c"],
                        }
                        for i, ko in sents
                    ],
                }
            )
        return {"paragraphs": paragraphs}

    return generate, calls


class BuildPackTests(unittest.TestCase):
    def build(self, generate, paragraphs=None):
        return sp.build_study_pack(
            "[사설] 제목", paragraphs or [P1, P2, P3], generate, text_sig="sig1", sleep=lambda s: None
        )

    def test_happy_path_shape_and_alignment(self):
        gen, calls = fake_generate_factory()
        pack = self.build(gen)
        self.assertEqual(pack["version"], sp.STUDY_VERSION)
        self.assertEqual(pack["text_sig"], "sig1")
        self.assertEqual(pack["summary_vi"], "Chính phủ thúc đẩy dự án lớn.")
        self.assertEqual(pack["key_points_vi"], ["Ý 1", "Ý 2"])  # blank dropped
        self.assertEqual(pack["key_terms"], [{"ko": "메가프로젝트", "vi": "dự án quy mô cực lớn"}])
        self.assertEqual(len(pack["paragraphs"]), 3)
        self.assertEqual([len(p["sentences"]) for p in pack["paragraphs"]], [2, 2, 1])
        self.assertEqual(pack["paragraphs"][0]["sentences"][1]["vi"], "dịch 0-1")
        self.assertEqual(pack["paragraphs"][1]["easy_ko"], "쉬운 문장 1.")
        # sentence ko is the article's own sentence, in order
        self.assertEqual(" ".join(s["ko"] for s in pack["paragraphs"][0]["sentences"]), P1)
        self.assertEqual(calls["overview"], 1)

    def test_words_deduped_and_notes_capped(self):
        gen, _ = fake_generate_factory()
        s = self.build(gen)["paragraphs"][0]["sentences"][0]
        self.assertEqual(len(s["words"]), 1)  # duplicate surface dropped
        self.assertEqual(s["words"][0]["meaning_vi"], "nghĩa")
        self.assertEqual(len(s["grammar_notes_vi"]), 2)

    def test_hallucinated_words_not_in_sentence_are_dropped(self):
        gen, _ = fake_generate_factory(garbage_words=True)
        pack = self.build(gen)
        self.assertTrue(all(s["words"] == [] for p in pack["paragraphs"] for s in p["sentences"]))

    def test_missing_translation_tolerated_when_most_present(self):
        gen, _ = fake_generate_factory(drop_translation_for=2)
        pack = self.build(gen)
        self.assertIsNone(pack["paragraphs"][2]["sentences"][0]["vi"])
        self.assertIsNotNone(pack["paragraphs"][0]["sentences"][0]["vi"])

    def test_failed_first_attempt_is_retried(self):
        gen, calls = fake_generate_factory(fail_first_batch=True)
        pack = self.build(gen)
        self.assertGreaterEqual(calls["batch"], 2)
        self.assertEqual(pack["paragraphs"][0]["sentences"][0]["vi"], "dịch 0-0")

    def test_unusable_when_nothing_translated(self):
        def gen(prompt, schema):
            if schema is sp.OVERVIEW_SCHEMA:
                return {"summary_vi": "x", "key_points_vi": [], "key_terms": []}
            raise ValueError("always fails")

        with self.assertRaises(RuntimeError):
            self.build(gen)

    def test_overview_failure_does_not_sink_translations(self):
        inner, _ = fake_generate_factory()

        def gen(prompt, schema):
            if schema is sp.OVERVIEW_SCHEMA:
                raise ValueError("no overview")
            return inner(prompt, schema)

        pack = self.build(gen)
        self.assertEqual(pack["summary_vi"], "")
        self.assertEqual(pack["paragraphs"][0]["sentences"][0]["vi"], "dịch 0-0")

    def test_progress_reported(self):
        gen, _ = fake_generate_factory()
        seen = []
        sp.build_study_pack("t", [P1, P2], gen, text_sig="s", on_progress=lambda d, t: seen.append((d, t)), sleep=lambda s: None)
        self.assertEqual(seen[0][0], 0)
        self.assertEqual(seen[-1][0], seen[-1][1])


class SigAndEasyTests(unittest.TestCase):
    def test_sig_changes_with_text(self):
        self.assertNotEqual(sp.study_text_sig("가"), sp.study_text_sig("나"))
        self.assertEqual(sp.study_text_sig("가"), sp.study_text_sig("가"))

    def test_easy_tts_text_uses_simple_paragraphs_and_title(self):
        pack = {
            "paragraphs": [
                {"easy_ko": "정부가 큰 일을 해요.", "sentences": [{"ko": "원문."}]},
                {"easy_ko": None, "sentences": [{"ko": "원문 하나입니다."}, {"ko": "원문 둘입니다."}]},
            ]
        }
        text = sp.easy_tts_text("[사설] 여명학교", pack)
        lines = text.split("\n")
        self.assertEqual(lines[0], "여명학교.")
        self.assertEqual(lines[1], "정부가 큰 일을 해요.")
        self.assertEqual(lines[2], "원문 하나입니다. 원문 둘입니다.")


if __name__ == "__main__":
    unittest.main()
