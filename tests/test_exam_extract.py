"""Exam extraction v2 (app/services/exam_extract.py).

The regression behind it, read off a real paper (TOPIK II 102회 읽기): one call over
the whole PDF, one text field per question. The model wrote the group instruction,
a passage's first sentence or a banner into `stem_ko`, paraphrased ("고르십시오" →
"고르시오", "우표" → "우체"), guessed answers (which turned EVERY card red) and
lost the underlining. Gemini is a fake here: it "sees" only the pages of the window
it is asked about and reads a question completely only when every page the question
needs is in view, so what is asserted is the logic around it — windows that
overlap, the best reading kept, two reads compared, flags with reasons, answers
only from a key."""
import json
import re
import unittest
from unittest import mock

from app.services import exam_drill, exam_extract as ex, gemini_client

QTYPES = [
    ("read_blank", "Điền chỗ trống", "đọc"),
    ("read_grammar_choice", "Ngữ pháp", "đọc"),
    ("read_short_passage", "Đoạn ngắn", "đọc"),
    ("read_long_passage", "Đoạn dài", "đọc"),
    ("read_order", "Sắp xếp", "đọc"),
    ("read_title_topic", "Chủ đề", "đọc"),
    ("read_chart_info", "Biểu đồ", "đọc"),
    ("listen_picture", "Nghe tranh", "nghe"),
]

# ---------------------------------------------------------------- a tiny paper --
# Each question: the pages it is printed on, and the extra pages it needs to be read
# whole (its group instruction, its passage). Page 1 is the cover.
INSTR_1 = "[1~2] ( )에 들어갈 말로 가장 알맞은 것을 고르십시오."
INSTR_3 = "[3~4] 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오."
INSTR_5 = "[5~6] 다음 글을 읽고 물음에 답하십시오."
P1_BODY = "공원에는 사람이 많았다. 우표 박물관은 문을 닫았다.\n그래서 우리는 집에 갔다."
TRUTH = [
    dict(n=1, pages={2}, needs={2}, instr=INSTR_1, stem="이 동네로 이사를 ( ) 일 년이 됐다.", opts=["온 지", "올 때", "오거나", "오다가"], qtype="read_blank"),
    dict(n=2, pages={2}, needs={2}, instr=INSTR_1, stem="가을이 되면서 나뭇잎 색이 점점 붉게 ( ).", opts=["변해 간다", "변할 뻔했다", "변한 척했다", "변하면 된다"], qtype="read_blank"),
    dict(n=3, pages={3}, needs={3}, instr=INSTR_3, stem="바람이 <u>시원하다</u>.", opts=["덥다", "선선하다", "춥다", "쌀쌀하다"], qtype="read_grammar_choice"),
    dict(n=5, pages={5}, needs={4, 5}, instr=INSTR_5, stem="윗글의 내용과 같은 것을 고르십시오.", opts=["가", "나", "다", "라"], qtype="read_short_passage", passage="P1"),
    dict(n=6, pages={5}, needs={4, 5}, instr=INSTR_5, stem="윗글을 쓴 이유로 알맞은 것을 고르십시오.", opts=["가", "나", "다", "라"], qtype="read_short_passage", passage="P1"),
    dict(n=7, pages={6}, needs={6}, instr="[7] 다음 글을 읽고 물음에 답하십시오.", stem="밑줄 친 부분에 나타난 '그'의 심정으로 알맞은 것을 고르십시오.", opts=["섭섭하다", "기쁘다", "놀랍다", "그립다"], qtype="read_short_passage", passage="W"),
    dict(n=8, pages={7}, needs={7}, instr="[8] ( )에 들어갈 말로 가장 알맞은 것을 고르십시오.", stem="비가 ( ) 우산을 챙겼다.", opts=["와서", "오면", "올까", "오자"], qtype="read_blank"),
    dict(n=9, pages={9}, needs={8, 9}, instr="[9] 다음 글 또는 도표의 내용과 같은 것을 고르십시오.", stem="", opts=["하나", "둘", "셋", "넷"], qtype="read_chart_info", passage="P9"),
]
PASSAGES = {
    "P1": dict(kind="đọc hiểu", body=P1_BODY, pages={4, 5}, start=4, withheld=False),
    "W": dict(kind="đọc hiểu", body=None, pages={6}, start=6, withheld=True),
    "P9": dict(kind="biểu đồ", body="여행지 선택: 가격 48% | 거리 20%", pages={8}, start=8, withheld=False),
}
PAGE_COUNT = 9
KEY_TABLE = {"읽기": {1: 1, 2: 1, 3: 4, 5: 2, 6: 3, 7: 1, 8: 2, 9: 4}, "듣기": {1: 2, 2: 1}}


def page_bytes(n):
    return f"p{n}".encode()


def files(count=PAGE_COUNT):
    return [(page_bytes(i), "image/png") for i in range(1, count + 1)]


class Paper:
    """The fake model: reads what is on the pages it is given, as a real one would,
    and records every call."""

    def __init__(self, *, mutate=None, fail_windows=None, truth=TRUTH, passages=PASSAGES, key=KEY_TABLE, key_reads=None, refine_data=None, fail_refine=False):
        self.mutate = mutate or {}  # (second read?, number) -> {"stem"/"option1"/…: text}
        self.fail = fail_windows or set()  # page tuples that raise
        self.truth, self.passages, self.key = truth, passages, key
        self.key_reads = list(key_reads or [])
        self.refine_data = refine_data or {}  # number -> what a narrow re-read answers for it
        self.refine_calls = []
        self.fail_refine = fail_refine
        self.models = set()
        self.prompts = []
        self.calls = []
        self.first_pass_windows = set()

    def pages_of(self, prompt):
        return tuple(int(p[1:]) for p in prompt[:-1] if isinstance(p, bytes))

    def labels_of(self, prompt):
        return [p for p in prompt[:-1] if isinstance(p, str)]

    def __call__(self, *, model, prompt, response_schema, prompt_version):
        visible = self.pages_of(prompt)
        self.calls.append((response_schema is ex.KEY_SCHEMA, visible))
        self.models.add(model)
        self.prompts.append((self.labels_of(prompt), prompt[-1]))
        if response_schema is ex.REFINE_SCHEMA:
            self.refine_calls.append(visible)
            if self.fail_refine:
                raise RuntimeError("model unavailable")
            return {"text": json.dumps({"items": [self.refine_data[n] for n in self.refine_for(prompt[-1]) if n in self.refine_data]})}
        if response_schema is ex.KEY_SCHEMA:
            return {"text": json.dumps(self.key_reads.pop(0) if self.key_reads else self.key_json())}
        if visible in self.fail:
            raise RuntimeError("model unavailable")
        if visible in getattr(self, "bad_json", set()):
            return {"text": "not json at all"}
        second = visible not in self.first_pass_windows
        return {"text": json.dumps(self.read(set(visible), second, visible[0]))}

    @staticmethod
    def refine_for(prompt):
        return [int(n) for n in re.findall(r"- Câu (\d+):", prompt)]

    def key_json(self):
        return {"sections": [{"section": s, "answers": [{"number": n, "answer": a} for n, a in t.items()]} for s, t in self.key.items()]}

    def read(self, visible, second, first_page):
        passages, items = {}, []
        for q in self.truth:
            if not (q["pages"] & visible):
                continue
            whole = q["needs"] <= visible
            ref = q.get("passage")
            if ref and ref not in passages and (self.passages[ref]["pages"] & visible):
                p = self.passages[ref]
                seen_whole = p["pages"] <= visible
                passages[ref] = {
                    "local_ref": ref,
                    "kind": "đọc hiểu" if p["kind"] not in ("biểu đồ", "nghe") else p["kind"],
                    "body_ko": None if p["withheld"] else (p["body"] if seen_whole else (p["body"] or "")[:12]),
                    "withheld": p["withheld"],
                    "page": p["start"],
                    "confidence": 0.9 if seen_whole else 0.4,
                }
            stem, opts = q["stem"], list(q["opts"])
            for field, text in self.mutate.get((second, q["n"]), {}).items():
                if field == "stem":
                    stem = text
                elif field.startswith("option"):
                    opts[int(field[6:]) - 1] = text
            items.append({
                "number": q["n"],
                "passage_ref": ref if ref in passages else None,
                "qtype_code": q["qtype"],
                "instruction_ko": ("※ " + q["instr"] + " (각 2점)") if q["instr"] and (whole or min(q["needs"]) in visible) else None,
                "group_from": None,
                "group_to": None,
                "stem_ko": stem,
                "options": [f"① {o}" for o in opts] if whole else [f"① {opts[0]}", f"② {opts[1]}"],
                "answer_guess": None,
                "page": min(q["pages"]),
                "confidence": 0.9 if whole else 0.4,
            })
        return {"paper_section": "읽기" if 1 in visible or 2 in visible else None, "passages": list(passages.values()), "items": items, "answer_key": []}

    def mark_first_pass(self):
        self.first_pass_windows = {tuple(range(a + 1, b + 1)) for a, b in ex.windows(PAGE_COUNT)}
        return self


def run(paper=None, **kw):
    paper = (paper or Paper()).mark_first_pass()
    with mock.patch.object(gemini_client, "part_from_bytes", side_effect=lambda data, mime: data):
        draft = ex.extract(files(), QTYPES, generate=paper, session_label=kw.pop("session_label", "102회 읽기"), **kw)
    return draft, paper


def item(draft, n):
    return next(i for i in draft.items if i["payload"]["number"] == n)


class WindowTests(unittest.TestCase):
    def test_every_page_is_in_some_window_and_windows_overlap(self):
        for count in range(1, 60):
            for phase in (0, ex.VERIFY_PHASE):
                spans = ex.windows(count, phase=phase)
                covered = set()
                for a, b in spans:
                    self.assertLessEqual(b - a, ex.WINDOW_PAGES)
                    covered |= set(range(a, b))
                self.assertEqual(covered, set(range(count)), (count, phase))
                for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
                    self.assertLess(a2, b1, "consecutive windows must overlap")

    def test_a_short_paper_is_one_window_and_nothing_is_none(self):
        self.assertEqual(ex.windows(3), [(0, 3)])
        self.assertEqual(ex.windows(0), [])

    def test_the_two_reads_cut_the_paper_in_different_places(self):
        first, second = ex.windows(25), ex.windows(25, phase=ex.VERIFY_PHASE)
        self.assertNotEqual({a for a, _ in first[1:]}, {a for a, _ in second[1:]})

    def test_a_file_is_one_page_per_image_and_a_pdf_is_cut_into_pages(self):
        self.assertEqual(len(ex.load_pages(files(4))), 4)
        with self.assertRaises(ValueError):
            ex.load_pages([(b"x", "image/png")] * (ex.MAX_PAGES + 1))

    def test_a_pdf_that_cannot_be_cut_is_kept_whole(self):
        self.assertEqual(ex.load_pages([(b"%PDF-broken", "application/pdf")]), [(b"%PDF-broken", "application/pdf")])


class TextTests(unittest.TestCase):
    def test_instruction_loses_the_mark_and_the_score_but_keeps_the_range(self):
        self.assertEqual(
            ex.clean_instruction("※ [9~12] 다음 글 또는 도표의 내용과 같은 것을 고르십시오. (각 2점)"),
            "[9~12] 다음 글 또는 도표의 내용과 같은 것을 고르십시오.",
        )
        self.assertEqual(ex.group_of("[9~12] 다음"), (9, 12))
        self.assertIsNone(ex.group_of("[12~9] x"))
        self.assertIsNone(ex.clean_instruction("  "))

    def test_wording_is_never_touched_only_spacing_and_markup(self):
        self.assertEqual(ex.tidy_text("고르십시오.  <U>밑줄</U>"), "고르십시오. <u>밑줄</u>")
        self.assertEqual(ex.tidy_text("a<u></u>b"), "ab")
        self.assertEqual(ex.tidy_text("줄1 \n\n\n\n 줄2", multiline=True), "줄1\n\n줄2")

    def test_option_numbers_are_dropped(self):
        self.assertEqual(ex.clean_option("① 온 지"), "온 지")
        self.assertEqual(ex.clean_option("3번 방"), "3번 방")

    def test_diffs_show_where_two_reads_differ(self):
        diffs = ex.span_diffs("우표 박물관에 갔다", "우체 박물관에 갔다")
        self.assertEqual(diffs, [("우표 박물관에", "우체 박물관에")])  # readable: the words as printed, with spaces
        self.assertEqual(ex.span_diffs("같은  글", "같은\n글"), [])  # layout is not content
        self.assertEqual(ex.span_diffs("고르십시오", "고르시오"), [("고르십시오", "고르시오")])

    def test_spacing_and_punctuation_are_not_differences(self):
        self.assertEqual(ex.span_diffs("온 지.", "온 지"), [])
        self.assertEqual(ex.span_diffs("“가을”이 되면서, 나뭇잎이", "'가을'이 되면서 나뭇잎이"), [])
        self.assertEqual(ex.span_diffs("우표박물관", "우표 박물관"), [])
        self.assertEqual(ex.span_diffs("( ) 안에", "()안에"), [])
        self.assertEqual(ex.span_diffs("① 가나", "가나"), [("① 가나", "가나")])  # a number is not punctuation

    def test_a_dropped_decimal_point_or_comma_inside_a_number_is_a_difference(self):
        self.assertEqual(len(ex.span_diffs("평균 은퇴 연령은 23.6세로", "평균 은퇴 연령은 236세로")), 1)
        self.assertEqual(len(ex.span_diffs("성인 남녀 1,600명", "성인 남녀 1600명")), 1)
        self.assertEqual(ex.span_diffs("23.6세", "23.6세."), [])

    def test_an_underline_one_read_missed_is_a_difference_with_the_words_around_it(self):
        diffs = ex.span_diffs("지금 출발하지 않으면 <u>늦을지도 모른다</u>.", "지금 출발하지 않으면 늦을지도 모른다.")
        self.assertEqual(len(diffs), 1)  # the opening and the closing mark are one difference
        self.assertEqual(diffs[0][0], "않으면 <u>늦을지도 모른다</u>.")
        self.assertEqual(diffs[0][1], "않으면 늦을지도 모른다.")

    def test_a_long_text_shows_only_the_places_that_differ(self):
        a = "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 열하나 열둘 열셋 열넷 열다섯"
        b = a.replace("일곱", "일급").replace("열넷", "열녯")
        diffs = ex.span_diffs(a, b)
        self.assertEqual(diffs, [("여섯 일곱 여덟", "여섯 일급 여덟"), ("열셋 열넷 열다섯", "열셋 열녯 열다섯")])

    def test_near_differences_are_one_difference(self):
        diffs = ex.span_diffs("가나다라마바사", "가난다랑마바사")
        self.assertEqual(len(diffs), 1)

    def test_underline_must_be_balanced(self):
        self.assertTrue(ex.underline_balanced("<u>가</u>나"))
        self.assertFalse(ex.underline_balanced("<u>가나"))


class ParseTests(unittest.TestCase):
    def test_a_bad_piece_costs_itself_not_the_window(self):
        data = {
            "paper_section": "읽기",
            "passages": ["junk", {"local_ref": ""}, {"local_ref": "P1", "kind": "?", "body_ko": " 글 ", "page": 12, "confidence": "x"}],
            "items": [
                7,
                {"number": "x"},
                {"number": 3, "stem_ko": "a", "options": "not a list", "page": 1, "confidence": 2},
                {"number": 4, "stem_ko": "b", "options": ["① 가", "나"], "instruction_ko": "※ [3~4] 가 (각 2점)", "page": 14},
            ],
            "answer_key": [{"number": 1, "answer": 9}, {"number": 2, "answer": 3, "section": "읽기"}, "x"],
        }
        read = ex.parse_read(data, 0, (10, 15))
        self.assertEqual(read.section, "đọc")
        p = read.passages[(0, "P1")]
        self.assertEqual((p.kind, p.body, p.page, p.confidence), ("đọc hiểu", "글", 12, 0.5))  # the page is the [Trang N] the model named
        self.assertEqual([q.number for q in read.questions], [3, 4])
        self.assertEqual(read.questions[0].options, [])
        self.assertEqual([q.page for q in read.questions], [0, 14])  # page 1 is not a page of this window: unknown
        self.assertEqual(read.questions[0].confidence, 1.0)
        self.assertEqual((read.questions[1].options, read.questions[1].group), (["가", "나"], (3, 4)))
        self.assertEqual(read.key_rows, [("đọc", 2, 3)])

    def test_a_withheld_passage_never_keeps_text(self):
        read = ex.parse_read({"passages": [{"local_ref": "P1", "kind": "đọc hiểu", "body_ko": "bịa", "withheld": True, "page": 1}]}, 0, (0, 5))
        self.assertIsNone(read.passages[(0, "P1")].body)


class MergeTests(unittest.TestCase):
    def reads(self, paper, spans):
        paper.mark_first_pass()
        with mock.patch.object(gemini_client, "part_from_bytes", side_effect=lambda d, m: d):
            reads, failed = ex._read_windows(paper, ex.load_pages(files()), spans, QTYPES, lambda: None)
        self.assertEqual(failed, [])
        return reads

    def test_the_whole_window_reading_wins_over_the_cut_one(self):
        # question 5 needs pages 4-5: window p1-5 sees it whole, p4-8 too, but a window
        # starting at page 5 sees it without its passage or instruction
        merged = ex.merge(self.reads(Paper(), [(4, 9), (0, 5)]))
        q = merged.questions[5]
        self.assertEqual(len(q.options), 4)
        self.assertTrue(q.instruction)
        self.assertEqual(merged.passage_of(q).body, P1_BODY.replace("\n", "\n"))

    def test_a_passage_two_questions_share_is_kept_once(self):
        merged = ex.merge(self.reads(Paper(), ex.windows(PAGE_COUNT)))
        self.assertIs(merged.passage_of(merged.questions[5]), merged.passage_of(merged.questions[6]))
        refs = {p.body for p in merged.passages.values() if p.body}
        self.assertEqual(refs, {P1_BODY, "여행지 선택: 가격 48% | 거리 20%"})

    def test_the_fuller_reading_of_a_passage_is_kept(self):
        merged = ex.merge(self.reads(Paper(), [(3, 4), (0, 5)]))  # first sees page 4 only: a stub of the passage
        self.assertEqual(merged.passage_of(merged.questions[5]).body, P1_BODY)

    def test_the_same_number_in_windows_that_do_not_touch_is_flagged(self):
        r1 = ex._Read(index=0, span=(0, 5), questions=[ex._Question(5, "하나", ["a", "b", "c", "d"], None, None, None, None, None, 1, 0.9, 0)])
        r2 = ex._Read(index=1, span=(20, 25), questions=[ex._Question(5, "다른 문제", ["a", "b", "c", "d"], None, None, None, None, None, 21, 0.9, 1)])
        self.assertEqual(ex.merge([r1, r2]).duplicates, {5})
        r3 = ex._Read(index=2, span=(3, 8), questions=[ex._Question(5, "하나", ["a", "b", "c", "d"], None, None, None, None, None, 4, 0.9, 2)])
        self.assertEqual(ex.merge([r1, r3]).duplicates, set())  # touching windows: the same question

    def test_withheld_passages_are_told_apart_by_their_page(self):
        w1 = ex._Passage((0, "P1"), "đọc hiểu", None, True, 9, 0.9)
        w2 = ex._Passage((1, "P1"), "đọc hiểu", None, True, 9, 0.9)
        w3 = ex._Passage((1, "P2"), "đọc hiểu", None, True, 18, 0.9)
        self.assertTrue(ex.same_passage(w1, w2))
        self.assertFalse(ex.same_passage(w1, w3))


class CompareTests(unittest.TestCase):
    def merged(self, mutate):
        paper = Paper(mutate={(False, k): v for k, v in mutate.items()}).mark_first_pass()
        with mock.patch.object(gemini_client, "part_from_bytes", side_effect=lambda d, m: d):
            reads, _ = ex._read_windows(paper, ex.load_pages(files()), ex.windows(PAGE_COUNT), QTYPES, lambda: None)
        return ex.merge(reads)

    def test_identical_reads_have_no_differences(self):
        a, b = self.merged({}), self.merged({})
        self.assertEqual(ex.compare(a, b), ({}, {}))

    def test_a_changed_word_in_an_option_is_shown_with_both_readings(self):
        a, b = self.merged({}), self.merged({1: {"option1": "온 적"}})
        question_diffs, _ = ex.compare(a, b)
        # the whole second reading of the field is kept, to be taken in one tap
        self.assertEqual(question_diffs[1], [{"field": "option 1", "a": "온 지", "b": "온 적", "b_full": "온 적"}])

    def test_spacing_and_punctuation_alone_are_not_a_mismatch(self):
        a, b = self.merged({}), self.merged({1: {"option1": "온  지."}, 2: {"stem": "가을이 되면서 나뭇잎 색이 점점 붉게( )."}})
        self.assertEqual(ex.compare(a, b), ({}, {}))

    def test_only_the_first_difference_of_a_field_carries_the_whole_second_reading(self):
        a, b = self.merged({}), self.merged({2: {"stem": "가을이 되면서 나뭇닢 색이 점점 붉개 ( )."}})
        diffs = [d for d in ex.compare(a, b)[0][2] if d["field"] == "stem"]
        self.assertEqual(len(diffs), 2)
        self.assertEqual(diffs[0]["b_full"], "가을이 되면서 나뭇닢 색이 점점 붉개 ( ).")
        self.assertNotIn("b_full", diffs[1])

    def test_underline_missing_in_one_read_is_a_difference(self):
        a, b = self.merged({}), self.merged({3: {"stem": "바람이 시원하다."}})
        self.assertIn("stem", {d["field"] for d in ex.compare(a, b)[0][3]})


class FlagTests(unittest.TestCase):
    def q(self, **kw):
        base = dict(number=1, stem="( ) 일 년이 됐다.", options=["a", "b", "c", "d"], instruction="[1~2] ( )에 들어갈 말로", group=None,
                    passage_key=None, qtype="read_blank", guess=None, page=1, confidence=0.9, window=0)
        return ex._Question(**{**base, **kw})

    def passage(self, **kw):
        base = dict(key=(0, "P1"), kind="đọc hiểu", body="본문", withheld=False, page=1, confidence=0.9)
        return ex._Passage(**{**base, **kw})

    def flags(self, q, passage=None, qtype="read_blank", diffs=None, single=False):
        return ex.question_flags(q, passage, qtype, diffs=diffs, single=single)

    def test_a_clean_question_has_no_flags_and_is_green(self):
        flags = self.flags(self.q())
        self.assertEqual(flags, [])
        self.assertEqual(ex.status_for(flags, 0.9), "pending")

    def test_no_answer_key_is_not_a_reason_to_flag_anything(self):
        # the old version flagged every question because it had guessed an answer
        self.assertEqual(ex.status_for(self.flags(self.q()), 0.95), "pending")

    def test_serious_problems_are_red_and_milder_ones_yellow(self):
        self.assertEqual(ex.status_for(["withheld"], 0.9), "flagged_red")
        self.assertEqual(ex.status_for(["text_mismatch"], 0.9), "flagged_yellow")
        self.assertEqual(ex.status_for([], 0.7), "flagged_yellow")
        self.assertEqual(ex.status_for([], 0.4), "flagged_red")

    def test_a_question_that_needs_a_passage_but_has_none(self):
        self.assertIn("missing_passage", self.flags(self.q(stem="윗글의 내용과 같은 것", instruction=None), None, "read_blank"))
        self.assertIn("missing_passage", self.flags(self.q(), None, "read_short_passage"))
        self.assertNotIn("missing_passage", self.flags(self.q(), self.passage(), "read_short_passage"))
        self.assertIn("missing_passage", self.flags(self.q(), self.passage(body=None), "read_short_passage"))

    def test_a_withheld_passage_is_flagged_as_withheld(self):
        flags = self.flags(self.q(), self.passage(body=None, withheld=True), "read_short_passage")
        self.assertEqual(flags, ["withheld"])

    def test_options_other_than_four_unreadable_marks_and_broken_underline(self):
        self.assertIn("options_count", self.flags(self.q(options=["a", "b"])))
        self.assertIn("unclear", self.flags(self.q(options=["a", "b[?]", "c", "d"])))
        self.assertIn("unclear", self.flags(self.q(stem="<u>깨진")))
        self.assertIn("missing_qtype", self.flags(self.q(), None, None))

    def test_the_underline_must_be_marked_when_the_instruction_speaks_of_it(self):
        q = self.q(instruction="[3~4] 밑줄 친 부분과 의미가 가장 비슷한 것", stem="바람이 시원하다.")
        self.assertIn("no_underline", self.flags(q, None, "read_grammar_choice"))
        ok = self.q(instruction=q.instruction, stem="바람이 <u>시원하다</u>.")
        self.assertNotIn("no_underline", self.flags(ok, None, "read_grammar_choice"))
        on_passage = self.flags(self.q(instruction="밑줄 친 부분", stem=""), self.passage(body="그는 <u>웃었다</u>"), "read_short_passage")
        self.assertNotIn("no_underline", on_passage)

    def test_disagreement_single_read_and_low_confidence(self):
        flags = self.flags(self.q(confidence=0.6), diffs=[{"field": "stem"}], single=True)
        self.assertEqual(flags, ["text_mismatch", "single_read", "low_confidence"])


class QtypeTests(unittest.TestCase):
    valid = {c for c, _, _ in QTYPES}

    def test_the_printed_instruction_decides_when_the_model_did_not(self):
        g = lambda instr, stem="", body=None: ex.guess_qtype(instr, stem, body, self.valid)  # noqa: E731
        self.assertEqual(g("[13~15] 다음을 순서에 맞게 배열한 것을 고르십시오."), "read_order")
        self.assertEqual(g("[3~4] 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오."), "read_grammar_choice")
        self.assertEqual(g("[5~8] 다음은 무엇에 대한 글인지 고르십시오."), "read_title_topic")
        self.assertEqual(g("[9~12] 다음 글 또는 도표의 내용과 같은 것을 고르십시오."), "read_chart_info")
        self.assertEqual(g("[16~18] 다음을 읽고 ( )에 들어갈 내용으로 가장 알맞은 것을 고르십시오."), "read_blank")
        self.assertEqual(g(None, "윗글의 내용과 같은 것", "짧은 글"), "read_short_passage")
        self.assertEqual(g(None, "윗글의 내용과 같은 것", "글" * 400), "read_long_passage")
        self.assertEqual(g(None, "가을이 되면서"), "read_grammar_choice")

    def test_a_code_that_does_not_exist_is_never_returned(self):
        self.assertIsNone(ex.guess_qtype("순서에 맞게", "", None, {"read_blank"}))


class ExtractTests(unittest.TestCase):
    def test_every_question_comes_out_split_into_its_parts(self):
        draft, paper = run()
        self.assertEqual([i["payload"]["number"] for i in draft.items], [1, 2, 3, 5, 6, 7, 8, 9])
        q1 = item(draft, 1)["payload"]
        self.assertEqual(q1["instruction_ko"], INSTR_1)  # the mark and the score are gone
        self.assertEqual(q1["group_range"], [1, 2])
        self.assertEqual(q1["stem_ko"], "이 동네로 이사를 ( ) 일 년이 됐다.")
        self.assertEqual(q1["options"], ["온 지", "올 때", "오거나", "오다가"])
        self.assertEqual(q1["qtype_code"], "read_blank")
        self.assertIsNone(q1["answer"])
        self.assertIn("<u>시원하다</u>", item(draft, 3)["payload"]["stem_ko"])
        self.assertEqual(item(draft, 9)["payload"]["stem_ko"], "")  # an ad/chart question has no line of its own

    def test_a_passage_is_shared_and_complete(self):
        draft, _ = run()
        by_ref = {p["payload"]["local_ref"]: p["payload"] for p in draft.passages}
        self.assertEqual(set(by_ref), {"P1", "P2", "P3"})
        q5, q6 = item(draft, 5)["payload"], item(draft, 6)["payload"]
        self.assertEqual(q5["passage_ref"], q6["passage_ref"])
        self.assertEqual(by_ref[q5["passage_ref"]]["body_ko"], P1_BODY)

    def test_a_withheld_passage_is_marked_and_its_question_is_red(self):
        draft, _ = run()
        q7 = item(draft, 7)
        self.assertIn("withheld", q7["flags"])
        self.assertEqual(ex.status_for(q7["flags"], q7["confidence"]), "flagged_red")
        withheld = [p for p in draft.passages if p["payload"]["withheld"]]
        self.assertEqual(len(withheld), 1)
        self.assertIsNone(withheld[0]["payload"]["body_ko"])
        self.assertEqual(withheld[0]["payload"]["source_page"], 6)

    def test_a_paper_with_no_answer_key_is_not_red(self):
        draft, _ = run()
        clean = [i for i in draft.items if i["payload"]["number"] in (1, 2, 5, 6, 8, 9)]
        for i in clean:
            self.assertEqual(ex.status_for(i["flags"], i["confidence"]), "pending", i["payload"]["number"])
        self.assertTrue(all(i["payload"]["answer"] is None for i in draft.items))

    def test_a_misread_character_shows_up_as_a_difference_between_the_reads(self):
        draft, _ = run(Paper(mutate={(True, 2): {"option2": "변할 뻔했나"}}))
        flags = item(draft, 2)["flags"]
        self.assertIn("text_mismatch", flags)
        alt = item(draft, 2)["payload"]["alt"]
        self.assertEqual(alt, [{"field": "option 2", "a": "변할 뻔했다", "b": "변할 뻔했나", "b_full": "변할 뻔했나"}])
        self.assertEqual(item(draft, 2)["payload"]["options"][1], "변할 뻔했다")  # the first read stays; the editor decides
        self.assertEqual(ex.status_for(flags, 0.9), "flagged_yellow")

    def test_no_second_read_when_verification_is_off(self):
        paper = Paper().mark_first_pass()
        draft, paper = run(paper, verify=False)
        self.assertEqual(len(paper.calls), len(ex.windows(PAGE_COUNT)))
        self.assertFalse(draft.summary["verified"])

    def test_each_page_window_stays_small(self):
        draft, paper = run()
        self.assertTrue(all(len(pages) <= ex.WINDOW_PAGES for _, pages in paper.calls))
        self.assertEqual(draft.summary["windows"], 3)
        self.assertEqual(draft.summary["verify_windows"], 3)

    def test_a_window_that_fails_costs_only_what_no_other_window_saw(self):
        paper = Paper(fail_windows={(7, 8, 9)}).mark_first_pass()  # question 9 is also in the second read's last window
        draft, _ = run(paper)
        self.assertEqual(draft.summary["failed_windows"], [2])
        numbers = [i["payload"]["number"] for i in draft.items]
        self.assertIn(9, numbers)  # filled in from the second read ...
        self.assertIn("single_read", item(draft, 9)["flags"])  # ... and said so

    def test_a_missing_question_is_reported_as_a_gap(self):
        truth = [q for q in TRUTH if q["n"] != 6]
        draft, _ = run(Paper(truth=truth))
        self.assertEqual(draft.summary["gaps"], [4, 6])
        self.assertEqual(draft.summary["numbers"], [1, 9])

    def test_every_window_failing_is_an_error_not_an_empty_paper(self):
        paper = Paper(fail_windows={tuple(range(a + 1, b + 1)) for a, b in ex.windows(PAGE_COUNT)}).mark_first_pass()
        with mock.patch.object(gemini_client, "part_from_bytes", side_effect=lambda d, m: d), self.assertRaises(RuntimeError) as caught:
            ex.extract(files(), QTYPES, generate=paper)
        # the editor is told which model was asked and why it failed (a model name that does not exist shows here)
        self.assertIn(ex._model(), str(caught.exception))
        self.assertIn("model unavailable", str(caught.exception))

    def test_unusable_json_is_asked_again_once_then_the_window_fails(self):
        paper = Paper().mark_first_pass()
        paper.bad_json = {(1, 2, 3, 4, 5)}
        draft, paper = run(paper)
        self.assertGreaterEqual(sum(1 for key, pages in paper.calls if pages == (1, 2, 3, 4, 5)), 2)
        self.assertEqual(draft.summary["failed_windows"], [0])

    def test_the_summary_records_how_the_paper_was_read(self):
        draft, _ = run()
        s = draft.summary
        self.assertEqual((s["prompt_version"], s["section"], s["pages"], s["questions"], s["verified"]), ("exam-v3", "đọc", 9, 8, True))
        self.assertEqual(s["duplicates"], [])

    def test_progress_is_reported_per_window(self):
        seen = []
        run(on_progress=lambda done, total, step: seen.append((done, total)))
        self.assertEqual(seen, [(i, 6) for i in range(1, 7)])

    def test_an_answer_key_printed_on_the_paper_is_returned_for_the_right_part(self):
        paper = Paper().mark_first_pass()
        original = paper.read
        paper.read = lambda visible, second, first: {**original(visible, second, first), "answer_key": [{"section": "읽기", "number": 1, "answer": 3}]}
        draft, _ = run(paper)
        self.assertEqual(draft.key, {"đọc": {1: 3}})


# ------------------------------------------------------------ exam-v3 ----------
def hard_truth():
    """The paper again, but question 9 (a poster question) has no passage in the first reading."""
    truth = [dict(q) for q in TRUTH]
    for q in truth:
        if q["n"] == 9:
            q.pop("passage")
    return truth


NO_UNDERLINE_BOTH = {(False, 3): {"stem": "바람이 시원하다."}, (True, 3): {"stem": "바람이 시원하다."}}
UNDERLINED_STEM = dict(number=3, stem_ko="바람이 <u>시원하다</u>.", underline_in="stem", confidence=0.8)
POSTER = dict(number=9, passage_ko="여행지 선택: 가격 48% | 거리 20%", confidence=0.7)


class PageLabelTests(unittest.TestCase):
    def test_every_page_goes_to_the_model_under_its_own_number(self):
        _, paper = run()
        windows = [(labels, text) for labels, text in paper.prompts if labels]
        self.assertTrue(windows)
        for (labels, _text), (_key, pages) in zip(paper.prompts, paper.calls):
            self.assertEqual(labels, [f"[Trang {n}]" for n in pages])  # N counts pages of the file, not printed numbers

    def test_the_prompt_names_the_first_and_last_page_of_the_window(self):
        _, paper = run()
        labels, text = paper.prompts[0]
        self.assertIn("[Trang 1] đến\n[Trang 5]", text)
        self.assertIn("KHÔNG phải số in", text)

    def test_the_prompt_does_not_teach_the_model_a_wording_the_paper_does_not_print(self):
        example = [line for line in ex.build_prompt(5, QTYPES).splitlines() if "[9~12]" in line][0]
        self.assertIn("그래프", example)  # 102회 prints 그래프, not 도표, in the 9~12 instruction
        self.assertNotIn("도표", example)

    def test_the_source_page_is_the_page_the_model_named(self):
        draft, _ = run()
        self.assertEqual({i["payload"]["number"]: i["payload"]["source_page"] for i in draft.items}, {1: 2, 2: 2, 3: 3, 5: 5, 6: 5, 7: 6, 8: 7, 9: 9})
        by_ref = {p["payload"]["body_ko"] or "withheld": p["payload"]["source_page"] for p in draft.passages}
        self.assertEqual(by_ref, {P1_BODY: 4, "withheld": 6, "여행지 선택: 가격 48% | 거리 20%": 8})

    def test_a_page_outside_the_window_is_unknown_not_trusted(self):
        self.assertEqual(ex.page_in(3, (0, 5)), 3)
        self.assertEqual(ex.page_in(3, (4, 9)), 0)  # a position inside the window, or the printed number
        self.assertEqual(ex.page_in(10, (4, 9)), 0)
        self.assertEqual(ex.page_in("x", (0, 5)), 0)

    def test_a_question_with_no_page_takes_its_neighbours_and_a_passage_is_never_pageless(self):
        paper = Paper().mark_first_pass()
        original = paper.read

        def read(visible, second, first_page):
            data = original(visible, second, first_page)
            for q in data["items"]:
                if q["number"] == 6:
                    q["page"] = 99
            for p in data["passages"]:
                p["page"] = 99
            return data

        paper.read = read
        draft, _ = run(paper)
        self.assertEqual(item(draft, 6)["payload"]["source_page"], 5)  # the page of question 5, the nearest numbered one
        self.assertTrue(all(isinstance(p["payload"]["source_page"], int) and p["payload"]["source_page"] >= 1 for p in draft.passages))

    def test_the_model_that_read_the_paper_is_recorded(self):
        draft, paper = run()
        self.assertEqual(draft.summary["model"], ex._model())
        self.assertEqual(paper.models, {ex._model()})

    def test_the_exam_models_default_to_the_stronger_one_and_are_not_the_lesson_model(self):
        from app.core.config import Settings

        fields = Settings.model_fields
        self.assertEqual(fields["GEMINI_MODEL_EXAM_INGEST"].default, "gemini-3.8-flash")
        self.assertEqual(fields["GEMINI_MODEL_EXAM_ANALYSIS"].default, "gemini-3.8-flash")
        self.assertNotEqual(fields["GEMINI_MODEL_EXAM_INGEST"].default, fields["GEMINI_MODEL_LESSON_INGEST"].default)

    def test_an_empty_setting_falls_back_instead_of_calling_a_model_with_no_name(self):
        with mock.patch.object(ex.settings, "GEMINI_MODEL_EXAM_INGEST", ""), mock.patch.object(ex.settings, "GEMINI_MODEL_LESSON_INGEST", "lesson-model"):
            self.assertEqual(ex._model(), "lesson-model")
        with mock.patch.object(ex.settings, "GEMINI_MODEL_EXAM_ANALYSIS", ""), mock.patch.object(ex.settings, "GEMINI_MODEL_STUDY", "study-model"):
            self.assertEqual(ex.analysis_model(), "study-model")


class StandardWordingTests(unittest.TestCase):
    def test_a_dropped_word_is_flagged_with_the_standard_wording_and_the_group_mark_kept(self):
        got = ex.standard_suggestion("[3~4] 밑줄 친 부분과 의미가 비슷한 것을 고르십시오.", "")
        self.assertEqual(got, {"instruction_ko": "[3~4] 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오."})

    def test_an_ending_cut_short_is_a_near_miss_too(self):
        got = ex.standard_suggestion("[9~12] 다음 글 또는 그래프의 내용과 같은 것을 고르시오.", "")
        self.assertEqual(got, {"instruction_ko": "[9~12] 다음 글 또는 그래프의 내용과 같은 것을 고르십시오."})

    def test_without_a_group_mark_none_is_invented(self):
        got = ex.standard_suggestion("밑줄 친 부분과 의미가 비슷한 것을 고르십시오.", "")
        self.assertEqual(got, {"instruction_ko": "밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오."})

    def test_every_standard_instruction_passes_and_so_does_the_older_wording(self):
        for text in ex.STANDARD_INSTRUCTIONS + ex.ALSO_ACCEPTED_INSTRUCTIONS:
            self.assertEqual(ex.standard_suggestion(f"[1~2] {text}", ""), {}, text)
        for text in ex.STANDARD_STEMS:
            self.assertEqual(ex.standard_suggestion(None, text), {}, text)

    def test_spacing_and_quotes_do_not_make_a_standard_line_a_near_miss(self):
        self.assertEqual(ex.standard_suggestion(None, "밑줄 친 부분에 나타난 ‘그’의 심정으로  가장 알맞은 것을 고르십시오."), {})
        self.assertEqual(ex.standard_suggestion("[1~2] ( ) 에 들어갈 말로 가장 알맞은 것을 고르십시오", ""), {})

    def test_a_replaced_or_added_word_is_a_different_question_not_a_misreading(self):
        self.assertEqual(ex.standard_suggestion(None, "윗글의 내용과 다른 것을 고르십시오."), {})
        self.assertEqual(ex.standard_suggestion("[1~2] ( )에 들어갈 말로 가장 알맞지 않은 것을 고르십시오.", ""), {})

    def test_wording_that_is_not_in_the_lists_is_left_alone(self):
        self.assertEqual(ex.standard_suggestion("[1~3] 다음을 듣고 알맞은 그림을 고르십시오.", "남자는 무엇을 합니까?"), {})
        self.assertEqual(ex.standard_suggestion("[1~2] 다음을 읽고 ( )에 들어갈 말을 고르십시오.", ""), {})

    def test_the_stem_is_checked_too(self):
        self.assertEqual(ex.standard_suggestion(None, "윗글의 내용과 같은 것 고르십시오."), {"stem_ko": "윗글의 내용과 같은 것을 고르십시오."})

    def test_the_paper_is_flagged_but_never_corrected(self):
        truth = [dict(q) for q in TRUTH]
        for q in truth:
            if q["n"] == 3:
                q["instr"] = "[3~4] 밑줄 친 부분과 의미가 비슷한 것을 고르십시오."
        draft, _ = run(Paper(truth=truth))
        q3 = item(draft, 3)
        self.assertIn("instruction_variant", q3["flags"])
        self.assertEqual(ex.status_for(q3["flags"], q3["confidence"]), "flagged_yellow")
        self.assertEqual(q3["payload"]["instruction_ko"], "[3~4] 밑줄 친 부분과 의미가 비슷한 것을 고르십시오.")  # as read
        self.assertEqual(q3["payload"]["instruction_suggest"], {"instruction_ko": INSTR_3})
        self.assertNotIn("instruction_variant", item(draft, 1)["flags"])
        self.assertNotIn("instruction_suggest", item(draft, 1)["payload"])

    def test_a_stem_that_repeats_the_group_instruction_is_dropped(self):
        data = {"items": [{"number": 1, "stem_ko": "( )에 들어갈 말로 가장 알맞은 것을 고르십시오.", "options": ["a", "b", "c", "d"], "instruction_ko": "※ [1~2] ( )에 들어갈 말로 가장 알맞은 것을 고르십시오. (각 2점)", "page": 1}]}
        read = ex.parse_read(data, 0, (0, 5))
        self.assertEqual(read.questions[0].stem, "")
        self.assertEqual(read.questions[0].instruction, "[1~2] ( )에 들어갈 말로 가장 알맞은 것을 고르십시오.")

    def test_an_instruction_the_model_put_in_the_stem_is_moved_to_where_it_belongs(self):
        data = {"items": [{"number": 3, "stem_ko": "※ [3~4] 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오. (각 2점)", "options": ["a", "b", "c", "d"], "page": 1}]}
        q = ex.parse_read(data, 0, (0, 5)).questions[0]
        self.assertEqual((q.stem, q.instruction, q.group), ("", "[3~4] 밑줄 친 부분과 의미가 가장 비슷한 것을 고르십시오.", (3, 4)))

    def test_a_real_stem_beside_an_instruction_stays(self):
        data = {"items": [{"number": 1, "stem_ko": "이 동네로 이사를 ( ) 일 년이 됐다.", "options": ["a", "b", "c", "d"], "instruction_ko": INSTR_1, "page": 1}]}
        self.assertEqual(ex.parse_read(data, 0, (0, 5)).questions[0].stem, "이 동네로 이사를 ( ) 일 년이 됐다.")


class RefineTests(unittest.TestCase):
    def test_a_clean_paper_is_not_read_again(self):
        draft, paper = run()
        self.assertEqual(paper.refine_calls, [])
        self.assertEqual((draft.summary["refine_pages"], draft.summary["refined"]), ([], []))

    def test_an_underline_nobody_saw_is_looked_for_on_its_own_page(self):
        draft, paper = run(Paper(mutate=NO_UNDERLINE_BOTH, refine_data={3: UNDERLINED_STEM}))
        self.assertEqual(paper.refine_calls, [(3,)])  # one page, not the window
        labels, text = paper.prompts[-1]
        self.assertEqual(labels, ["[Trang 3]"])
        self.assertIn("Câu 3", text)
        self.assertNotIn("Câu 1:", text)
        q3 = item(draft, 3)
        self.assertEqual(q3["payload"]["stem_ko"], "바람이 <u>시원하다</u>.")
        self.assertIn("refined", q3["flags"])
        self.assertNotIn("no_underline", q3["flags"])
        self.assertEqual(ex.status_for(q3["flags"], q3["confidence"]), "flagged_yellow")  # a human compares it with the page
        self.assertEqual((draft.summary["refine_pages"], draft.summary["refined"]), ([3], [3]))

    def test_a_re_read_that_rewrites_the_words_is_not_trusted(self):
        rewritten = dict(UNDERLINED_STEM, stem_ko="바람이 <u>선선하다</u>.")
        draft, _ = run(Paper(mutate=NO_UNDERLINE_BOTH, refine_data={3: rewritten}))
        q3 = item(draft, 3)
        self.assertEqual(q3["payload"]["stem_ko"], "바람이 시원하다.")
        self.assertIn("no_underline", q3["flags"])  # still red: nothing new was found
        self.assertNotIn("refined", q3["flags"])
        self.assertEqual(ex.status_for(q3["flags"], q3["confidence"]), "flagged_red")

    def test_a_re_read_that_finds_no_underline_changes_nothing(self):
        none = dict(number=3, stem_ko="바람이 시원하다.", underline_in="none", confidence=0.8)
        draft, _ = run(Paper(mutate=NO_UNDERLINE_BOTH, refine_data={3: none}))
        self.assertIn("no_underline", item(draft, 3)["flags"])

    def test_a_poster_question_with_no_passage_gets_its_text_from_its_page(self):
        draft, paper = run(Paper(truth=hard_truth(), refine_data={9: POSTER}))
        self.assertEqual(paper.refine_calls, [(9,)])
        q9 = item(draft, 9)
        self.assertIn("refined", q9["flags"])
        self.assertNotIn("missing_passage", q9["flags"])
        ref = q9["payload"]["passage_ref"]
        passage = next(p for p in draft.passages if p["payload"]["local_ref"] == ref)["payload"]
        self.assertEqual(passage["body_ko"], "여행지 선택: 가격 48% | 거리 20%")
        self.assertEqual((passage["kind"], passage["source_page"]), ("biểu đồ", 9))
        self.assertIn("refined", passage["flags"])
        self.assertEqual(ex.status_for(passage["flags"], 0.7), "flagged_yellow")

    def test_a_poster_that_cannot_be_read_stays_red(self):
        draft, _ = run(Paper(truth=hard_truth(), refine_data={}))
        q9 = item(draft, 9)
        self.assertIn("missing_passage", q9["flags"])
        self.assertEqual(ex.status_for(q9["flags"], q9["confidence"]), "flagged_red")

    def test_a_re_read_with_unreadable_marks_is_not_taken(self):
        draft, _ = run(Paper(truth=hard_truth(), refine_data={9: dict(POSTER, passage_ko="여행지 [?]")}))
        self.assertIn("missing_passage", item(draft, 9)["flags"])

    def test_a_page_that_fails_costs_only_itself(self):
        draft, paper = run(Paper(truth=hard_truth(), mutate=NO_UNDERLINE_BOTH, fail_refine=True))
        self.assertEqual(draft.summary["refine_failed"], [3, 9])
        self.assertEqual(draft.summary["refine_pages"], [])
        self.assertEqual(len(draft.items), 8)  # the paper still comes out

    def test_two_hard_spots_on_one_page_are_one_call(self):
        paper = Paper(truth=hard_truth(), mutate={**NO_UNDERLINE_BOTH}, refine_data={3: UNDERLINED_STEM, 9: POSTER})
        _, paper = run(paper)
        self.assertEqual(sorted(paper.refine_calls), [(3,), (9,)])  # different pages: one each

    def test_re_reading_can_be_switched_off(self):
        draft, paper = run(Paper(mutate=NO_UNDERLINE_BOTH, refine_data={3: UNDERLINED_STEM}), refine_pass=False)
        self.assertEqual(paper.refine_calls, [])
        self.assertIn("no_underline", item(draft, 3)["flags"])

    def test_what_was_refined_is_compared_with_the_second_read_again(self):
        # the first read missed the underline, the second saw it: after the narrow look the two agree
        paper = Paper(mutate={(False, 3): {"stem": "바람이 시원하다."}}, refine_data={3: UNDERLINED_STEM})
        draft, _ = run(paper)
        q3 = item(draft, 3)
        self.assertNotIn("text_mismatch", q3["flags"])
        self.assertNotIn("alt", q3["payload"])
        self.assertIn("refined", q3["flags"])

    def test_a_withheld_passage_is_never_asked_for_again(self):
        _, paper = run(Paper(refine_data={7: dict(number=7, passage_ko="bịa", confidence=0.9)}))
        self.assertEqual(paper.refine_calls, [])  # question 7 says 밑줄 but its passage is withheld

    def test_progress_counts_the_re_read_pages_too(self):
        seen = []
        run(Paper(truth=hard_truth(), mutate=NO_UNDERLINE_BOTH, refine_data={3: UNDERLINED_STEM, 9: POSTER}), on_progress=lambda d, t, s: seen.append((d, t, s)))
        self.assertEqual([d for d, _t, _s in seen], list(range(1, 9)))
        self.assertEqual(seen[-1][1], 8)
        self.assertIn("đọc lại", seen[-1][2])

    def test_the_summary_says_what_was_read_again(self):
        draft, _ = run(Paper(truth=hard_truth(), mutate=NO_UNDERLINE_BOTH, refine_data={3: UNDERLINED_STEM, 9: POSTER}))
        self.assertEqual((draft.summary["refine_pages"], draft.summary["refined"]), ([3, 9], [3, 9]))


class KeyTests(unittest.TestCase):
    def read(self, paper):
        with mock.patch.object(gemini_client, "part_from_bytes", side_effect=lambda d, m: d):
            return ex.extract_key([(b"p1", "image/png"), (b"p2", "image/png")], generate=paper)

    def test_a_key_with_two_parts_is_read_per_part_and_writing_is_ignored(self):
        key = dict(KEY_TABLE)
        key["쓰기"] = {51: 1}
        read = self.read(Paper(key=key))
        self.assertEqual(read.sections, {"đọc": KEY_TABLE["읽기"], "nghe": KEY_TABLE["듣기"]})
        self.assertEqual(read.conflicts, {})

    def test_a_number_the_two_reads_disagree_on_is_not_used(self):
        agree = {"sections": [{"section": "읽기", "answers": [{"number": 1, "answer": 1}, {"number": 2, "answer": 2}]}]}
        differ = {"sections": [{"section": "읽기", "answers": [{"number": 1, "answer": 1}, {"number": 2, "answer": 4}]}]}
        read = self.read(Paper(key_reads=[agree, differ]))
        self.assertEqual(read.sections, {"đọc": {1: 1}})
        self.assertEqual(read.conflicts, {"đọc": [2]})

    def test_a_number_only_one_read_saw_is_not_used_either(self):
        a = {"sections": [{"section": "읽기", "answers": [{"number": 1, "answer": 1}, {"number": 2, "answer": 2}]}]}
        b = {"sections": [{"section": "읽기", "answers": [{"number": 1, "answer": 1}]}]}
        read = self.read(Paper(key_reads=[a, b]))
        self.assertEqual((read.sections, read.conflicts), ({"đọc": {1: 1}}, {"đọc": [2]}))

    def test_one_read_giving_a_number_two_answers_is_a_conflict(self):
        a = {"sections": [{"section": "읽기", "answers": [{"number": 1, "answer": 1}, {"number": 1, "answer": 2}]}]}
        read = self.read(Paper(key_reads=[a, a]))
        self.assertEqual((read.sections, read.conflicts), ({}, {"đọc": [1]}))

    def test_the_part_of_the_key_that_belongs_to_the_paper(self):
        key = {"đọc": {1: 1}, "nghe": {1: 2}}
        self.assertEqual(ex.choose_key_section(key, "đọc"), "đọc")
        self.assertEqual(ex.choose_key_section(key, "nghe"), "nghe")
        self.assertIsNone(ex.choose_key_section(key, None))  # two parts and no idea: do not guess
        self.assertEqual(ex.choose_key_section({"đọc": {1: 1}}, None), "đọc")
        self.assertIsNone(ex.choose_key_section({"đọc": {1: 1}, "nghe": {}}, "viết"))
        self.assertIsNone(ex.choose_key_section({"nghe": {1: 2}}, "đọc"))  # never the listening answers for a reading paper

    def test_keys_printed_in_a_paper_drop_a_number_given_two_answers(self):
        rows = [("đọc", 1, 2), ("đọc", 1, 2), ("đọc", 2, 1), ("đọc", 2, 3), (None, 3, 4)]
        self.assertEqual(ex.collect_key(rows, "đọc"), {"đọc": {1: 2, 3: 4}})

    def test_the_part_a_label_names(self):
        self.assertEqual(ex.section_from_label("102회 읽기"), "đọc")
        self.assertEqual(ex.section_from_label("TOPIK II 102 - đề đọc"), "đọc")
        self.assertEqual(ex.section_from_label("102회 듣기"), "nghe")
        self.assertIsNone(ex.section_from_label("102회 듣기·쓰기"))
        self.assertIsNone(ex.section_from_label("102회"))


class AnswerTextTests(unittest.TestCase):
    def test_pairs_in_the_forms_people_paste(self):
        want = {1: 2, 2: 1, 3: 4}
        for text in ("1-2, 2-1, 3-4", "1:2 2:1 3:4", "1. ② 2. ① 3. ④", "1②2①3④", "1 ② 2 ① 3 ④", "1=2\\n2=1\\n3=4".replace("\\n", "\n"), "１-２, ２-１, ３-４"):
            self.assertEqual(ex.parse_answer_text(text), want, text)

    def test_two_digit_numbers(self):
        self.assertEqual(ex.parse_answer_text("10-3, 11-4"), {10: 3, 11: 4})

    def test_an_unbroken_run_counts_from_the_start_number(self):
        self.assertEqual(ex.parse_answer_text("2134"), {1: 2, 2: 1, 3: 3, 4: 4})
        self.assertEqual(ex.parse_answer_text("2 1, 3 4", start=26), {26: 2, 27: 1, 28: 3, 29: 4})

    def test_anything_unclear_is_refused_not_guessed(self):
        for bad in ("", "abc", "1-2, 2-9", "1-2, 3", "1-2, 1-3", "2169", "1-2 and then 7"):
            with self.assertRaises(ex.AnswerTextError, msg=bad):
                ex.parse_answer_text(bad)


class DrillRuleTests(unittest.TestCase):
    """What the drill refuses to offer: questions that cannot be answered fairly."""

    def cand(self, **kw):
        base = dict(item_id=None, number=5, qtype_code="read_short_passage", stem_ko="윗글의 내용과 같은 것을 고르십시오.",
                    options=["가", "나", "다", "라"], answer=2, answer_source="editor", skill="đọc", passage_kind="đọc hiểu",
                    passage_ko="본문", instruction_ko="[5~6] 다음 글을 읽고 물음에 답하십시오.", passage_linked=True)
        return exam_drill.Candidate(**{**base, **kw})

    def test_a_question_with_its_passage_is_usable(self):
        self.assertTrue(exam_drill.usable(self.cand()))

    def test_a_withheld_passage_is_not_usable(self):
        self.assertFalse(exam_drill.usable(self.cand(passage_ko=None)))
        self.assertFalse(exam_drill.usable(self.cand(passage_ko="  ")))

    def test_a_question_that_points_at_a_passage_it_does_not_have(self):
        for stem in ("윗글의 내용과 같은 것", "㉠에 들어갈 말", "<보기>의 문장이 들어갈 곳"):
            self.assertFalse(exam_drill.usable(self.cand(qtype_code="read_blank", stem_ko=stem, passage_ko=None, passage_linked=False)), stem)

    def test_an_ordering_question_needs_its_sentences(self):
        self.assertFalse(exam_drill.usable(self.cand(qtype_code="read_order", passage_ko=None, passage_linked=False)))

    def test_an_underlined_phrase_must_be_marked(self):
        plain = self.cand(qtype_code="read_grammar_choice", instruction_ko="[3~4] 밑줄 친 부분과 의미가 비슷한 것", stem_ko="바람이 시원하다.", passage_ko=None, passage_linked=False)
        self.assertFalse(exam_drill.usable(plain))
        marked = self.cand(qtype_code="read_grammar_choice", instruction_ko=plain.instruction_ko, stem_ko="바람이 <u>시원하다</u>.", passage_ko=None, passage_linked=False)
        self.assertTrue(exam_drill.usable(marked))

    def test_an_ad_question_has_no_stem_of_its_own_but_an_instruction(self):
        ad = self.cand(qtype_code="read_chart_info", stem_ko="", instruction_ko="[9~12] 다음 글 또는 도표의 내용과 같은 것을 고르십시오.")
        self.assertTrue(exam_drill.usable(ad))
        self.assertFalse(exam_drill.usable(self.cand(stem_ko="", instruction_ko="")))

    def test_a_question_saved_by_the_old_extraction_is_still_judged_by_what_it_has(self):
        old = self.cand(instruction_ko="", passage_linked=False, qtype_code="read_blank", stem_ko="( ) 일 년이 됐다.", passage_ko=None)
        self.assertTrue(exam_drill.usable(old))


if __name__ == "__main__":
    unittest.main()
