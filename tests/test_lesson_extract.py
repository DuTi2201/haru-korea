"""Lesson extraction: an inventory first, then the cards in small groups.

The regression behind it: a real lesson (4.6 kB, ~40 words in bold, 6 grammar
points) came back as 14 words and 4 patterns because one structured answer was
asked to hold everything and the model stopped early. Gemini is a fake here; the
fake answers each group from the refs in the prompt, so what is asserted is the
coverage logic (nothing teachable is lost, a gap is shown, not hidden)."""
import json
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from app.services import ingestion, lesson_extract
from app.services.lesson_extract import (
    GRAMMAR_CARDS_SCHEMA,
    INVENTORY_PROMPT,
    INVENTORY_SCHEMA,
    MAX_FAMILY,
    ORGANISE_SCHEMA,
    OVERVIEW_SCHEMA,
    VOCAB_CARDS_SCHEMA,
    VOCAB_GROUP,
    bold_terms,
    clean_terms,
    cluster_order,
    extract,
    merge_terms,
    parse_groups,
)

# the lesson that lost most of its content (pasted note, with its citation marks)
LESSON = """### 날씨와 계절 (Thời tiết và Mùa)
### 1. Tổng hợp Từ vựng chủ đề 날씨와 계절 (어휘)

#### **a. Thời tiết & Khí hậu (날씨 관련 어휘)**
* **날씨**: Thời tiết [1, 3, 5, 15]
* **일기예보**: Dự báo thời tiết [3, 5, 6]
* **맑다**: Trong xanh, quang đãng [3, 5, 15, 16]
* **흐리다**: Âm u, u ám [3, 5, 15, 16]
* **비가 오다**: Trời mưa [3, 5, 15, 16]
* **눈이 오다**: Tuyết rơi [3, 5, 15, 16]
* **바람이 불다**: Gió thổi [3, 5, 15, 16]
* **덥다**: Nóng [3, 5, 16-18]
* **따뜻하다**: Ấm áp [3, 5, 6, 8, 16, 17]
* **선선하다**: Mát mẻ [3, 5, 16-18]
* **쌀쌀하다**: Se lạnh [3, 5, 16]
* **춥다**: Lạnh [3, 5, 15-18]

#### **b. Bốn mùa & Hoạt động đặc trưng (계절 및 관련 활동)**
* **사계절**: Bốn mùa [3, 5, 18, 19]
  * **봄 (Mùa xuân - Tháng 3~5)**: Thời tiết ấm áp (**따뜻하다**), hoa anh đào nở (**벚꽃이 피다 / 꽃이 피다**), đi dã ngoại (**소풍을 가다**) [3, 5, 8, 18, 19].
  * **여름 (Mùa hè - Tháng 6~8)**: Thời tiết nóng (**덥다**), đi tắm biển/bãi biển (**해수욕장에 가다 / 바다**), bơi lội (**수영하다**) [3, 5, 18, 19].
  * **가을 (Mùa thu - Tháng 9~11)**: Thời tiết mát mẻ (**선선하다**), lá đổi màu sang đỏ/vàng (**단풍이 들다**), ngắm lá thu (**단풍 구경을 하다**), đi leo núi (**등산하다**) [3, 5, 18, 19].
  * **겨울 (Mùa đông - Tháng 12~2)**: Thời tiết lạnh (**춥다**), làm người tuyết (**눈사람을 만들다 / 눈사람**), chơi ném tuyết (**눈싸움을 하다**), trượt tuyết (**스키를 타다**) [3, 5, 15, 18-20].

#### **c. Vật dụng & Từ vựng liên quan**
* **우산 / 우산을 쓰다 / 우산을 가지고 오다**: Ô (dù), che ô, mang ô theo [3, 5, 6].
* **두꺼운 옷 / 따뜻한 옷 / 얇은 옷 / 두꺼운 양말**: Quần áo dày / quần áo ấm / quần áo mỏng / tất dày [14, 17, 21].
* **Địa danh xuất hiện trong bài**: 서울 (Seoul), 제주도 (đảo Jeju), 모스크바 (Moskva), 시드니 (Sydney), 도쿄 (Tokyo), 상파울루 (São Paulo) [12, 16, 22].

---

### 2. Tổng hợp Ngữ pháp trọng tâm (문법)

#### **1. -는군요 / -군요 (Đuôi câu cảm thán khi trực tiếp nhận ra thực tế)**
* **Ý nghĩa**: Diễn tả sự ngạc nhiên hoặc cảm thán khi vừa mới trực tiếp chứng kiến, phát hiện một sự việc/trạng thái [1, 2, 5, 23].
* **Cấu trúc & Quy tắc kết hợp**:
  * **Động từ (hiện tại)** + **-는군요**: *비가 오는군요* (Ôi, trời đang mưa kìa!) [5, 6, 23]; *눈이 오는군요* (Tuyết rơi rồi kìa!); *사람들이 많이 내리는군요* [8].
  * **Tính từ** + **-군요**: *날씨가 좋군요* (Thời tiết đẹp thật đấy!) [1, 2, 5, 23]; *날씨가 흐리군요* (Trời u u ám quá!) [24]; *길이 복잡하군요* [23].
  * **Danh từ** + **-(이)군요**: *명동이군요* [23].

#### **2. -(으)ㄹ 것 같다 (Phỏng đoán sự việc / thời tiết tương lai hoặc chưa chắc chắn)**
* **Ý nghĩa**: Diễn tả sự phỏng đoán, nhận định chưa chắc chắn về một sự việc hoặc khả năng xảy ra của thời tiết ("có vẻ như sẽ...", "chắc là...") [1, 2, 5, 24, 25].
* **Cách chia**: Thân động từ/tính từ + **-(으)ㄹ 것 같다** [5, 25].
* **Ví dụ trong bài**: 
  * *비가 올 것 같다* (Có vẻ trời sắp mưa) [3, 5, 24].
  * *날씨가 추울 것 같다* (Chắc thời tiết sẽ lạnh) [21].
  * *친구들이 이 사진을 보면 좋아할 것 같다* (Các bạn nếu xem ảnh này chắc sẽ thích lắm) [15].

#### **3. -는/은/ㄴ 것 같다 (Phỏng đoán trạng thái / sự việc hiện tại)**
* **Ý nghĩa**: Thể hiện sự phỏng đoán, nhận định nhẹ nhàng về trạng thái hoặc hành động đang diễn ra ở hiện tại ("hình như...", "có vẻ...") [1, 2, 5, 26].
* **Cách chia**:
  * **Tính từ** + **-은/ㄴ 것 같다**: *날씨가 따뜻한 것 같다* (Có vẻ thời tiết ấm áp) [1-3, 5, 27]; *선선한 것 같다* [28, 29].
  * **Động từ** + **-는 것 같다**: *비가 오는 것 같다* [26]; *소풍을 가는 사람이 많은 것 같다* [8].

#### **4. -(으)면서 (Diễn tả hai hành động diễn ra song song)**
* **Ý nghĩa**: Liên kết hai động từ do cùng một chủ thể thực hiện đồng thời ("vừa... vừa...") [1-3, 5].
* **Cách chia**: Thân động từ kết thúc bằng nguyên âm hoặc **ㄹ** + **-면서**; có patchim khác + **-으면서** [1-3, 5].
* **Ví dụ trong bài**:
  * *벚꽃을 구경하면서 천천히 걸읍시다* (Chúng ta vừa ngắm hoa anh đào vừa đi dạo chậm rãi nhé) [1-3, 5, 8].
  * *등산을 하면서 단풍 구경을 합니다* (Vừa đi leo núi vừa ngắm lá thu) [19].
  * *바다를 보면서 걷고 싶어요* (Tôi muốn vừa ngắm biển vừa đi dạo) [27].

#### **5. Ngữ pháp bổ trợ liên quan**
* **-(으)니까**: Dùng lý do thời tiết để rủ rê/đề nghị (*날씨가 좋으니까 공원에 갑시다* [9, 10], *비가 오니까 옷을 두껍게 입으세요*) [6, 14, 30].
* **보다**: So sánh thời tiết giữa các địa điểm (*서울이 제주도보다 추워요*) [11, 12].

---

### 3. Ghi chú Văn hóa & Diễn đạt thực tế (문화 및 활용)
* **Văn hóa ngắm hoa xuân & ngắm lá thu (꽃구경과 단풍놀이)** [1, 31]:
  * **Mùa xuân (tháng 4)**: Thời tiết ấm áp (**따뜻하다**), người dân Hàn Quốc thường đến **여의도 (Yeouido)** ở Seoul để ngắm hoa anh đào nở (**벚꽃구경**) [31].
  * **Mùa thu (tháng 9~11)**: Khi lá cây chuyển sang màu đỏ/vàng (**단풍이 들다**), mọi người thường đi leo núi ngắm cảnh lá thu tại các ngọn núi nổi tiếng như **설악산 (Seoraksan)** hay **속리산 (Songnisan)** [31].
* **Thói quen sinh hoạt theo thời tiết**:
  * Nghe dự báo thời tiết (**일기예보**) vào buổi sáng để chuẩn bị ô (**우산을 가지고 오다/쓰다**) [6].
  * Khi trời lạnh (**추운 날씨**), rủ nhau đi uống trà ấm (**따뜻한 차를 마시다**) [6] hoặc mặc trang phục/tất dày (**두꺼운 옷/양말**) [14, 17]."""

# what the model listed on its own: only the first block of the lesson
LAZY_VOCAB = ["날씨", "일기예보", "맑다", "흐리다", "비가 오다", "눈이 오다", "바람이 불다", "덥다", "따뜻하다", "선선하다", "쌀쌀하다", "춥다", "사계절", "우산"]
LAZY_GRAMMAR = ["V/A + -는군요 / -군요", "V + -(으)ㄹ 것 같다", "V/A + -는/은/ㄴ 것 같다", "V + -(으)면서"]


def reply(obj):
    return {"text": json.dumps(obj, ensure_ascii=False), "prompt_version": "test"}


def refs_in(prompt):
    return re.findall(r"^([vg]\d+)\. (.+)$", prompt, flags=re.MULTILINE)


class FakeGemini:
    """Inventory -> `inventory`; organise -> `organise` (no sets by default); card
    groups -> one card per ref in the prompt, minus anything in `skip` (omitted)
    or `omit_first_time`, plus whatever `extra` holds for that term; overview -> text."""

    def __init__(
        self,
        inventory,
        skip=(),
        omit_first_time=(),
        fail_overview=False,
        break_all_cards=False,
        organise=None,
        fail_organise=False,
        extra=None,
    ):
        self.inventory = inventory
        self.skip = set(skip)
        self.omit_first_time = set(omit_first_time)
        self.fail_overview = fail_overview
        self.break_all_cards = break_all_cards
        self.organise = organise or {"vocab_families": [], "grammar_groups": []}
        self.fail_organise = fail_organise
        self.extra = extra or {}
        self.calls = []
        self.seen = set()

    def __call__(self, *, model, prompt, response_schema, prompt_version):
        document, text = prompt
        self.calls.append((response_schema, text, document))
        if response_schema is INVENTORY_SCHEMA:
            return reply(self.inventory)
        if response_schema is ORGANISE_SCHEMA:
            if self.fail_organise:
                raise RuntimeError("organise down")
            return reply(self.organise)
        if response_schema is OVERVIEW_SCHEMA:
            if self.fail_overview:
                raise RuntimeError("overview down")
            return reply({"content": "Bài học về thời tiết và bốn mùa."})
        if self.break_all_cards:
            raise RuntimeError("quota")
        items = []
        for ref, term in refs_in(text):
            if term in self.skip or (term in self.omit_first_time and term not in self.seen):
                self.seen.add(term)
                continue
            self.seen.add(term)
            if response_schema is VOCAB_CARDS_SCHEMA:
                items.append({"ref": ref, "hangul": term, "meaning_vi": f"nghĩa của {term}", "level": 1, "confidence": 0.9,
                              **self.extra.get(term, {})})
            else:
                items.append({"ref": ref, "pattern": term, "meaning_vi": f"nghĩa của {term}", "level": 2,
                              "usage_context_vi": "dùng khi...", "confidence": 0.9, **self.extra.get(term, {})})
        return reply({"items": items})

    def card_calls(self):
        return [c for c in self.calls if c[0] in (VOCAB_CARDS_SCHEMA, GRAMMAR_CARDS_SCHEMA)]


def lazy_inventory(**extra):
    return {"title": "Thời tiết và Mùa", "level": 2, "topics": ["Thời tiết"], "confidence": 0.9,
            "vocab_terms": LAZY_VOCAB, "grammar_patterns": LAZY_GRAMMAR, **extra}


class BoldTermsTests(unittest.TestCase):
    def setUp(self):
        self.terms = bold_terms(LESSON)

    def test_the_words_nested_in_other_bullets_are_found(self):
        for term in ("봄", "벚꽃이 피다", "소풍을 가다", "여름", "해수욕장에 가다", "단풍이 들다", "등산하다", "겨울",
                     "눈사람을 만들다", "눈싸움을 하다", "스키를 타다", "우산을 쓰다", "우산을 가지고 오다",
                     "두꺼운 옷", "두꺼운 양말", "여의도", "설악산", "속리산"):
            self.assertIn(term, self.terms)

    def test_grammar_formulas_and_glosses_are_not_taken_as_words(self):
        joined = " | ".join(self.terms)
        self.assertNotIn("것 같다", joined)  # the -(으)ㄹ / -는 것 같다 headings
        self.assertNotIn("Mùa", joined)  # the gloss in parentheses is dropped
        self.assertNotIn("1.", joined)  # heading numbers are stripped
        self.assertNotIn("은", self.terms)  # "-는/은/ㄴ" must not be split into particles
        self.assertTrue(all(any("\uAC00" <= c <= "\uD7A3" for c in t) for t in self.terms))

    def test_there_are_many_more_than_the_14_the_model_listed(self):
        self.assertGreaterEqual(len(self.terms), 40)

    def test_slash_variants_become_separate_terms_and_duplicates_collapse(self):
        got = bold_terms("**우산 / 우산을 쓰다**: ô. Lại **우산** nữa. **우 산**")
        self.assertEqual(got, ["우산", "우산을 쓰다"])


class MergeTests(unittest.TestCase):
    def test_what_the_model_missed_is_added_after_its_own_list(self):
        merged = merge_terms(["날씨", "맑다"], ["맑 다", "봄", "날씨"])
        self.assertEqual(merged, ["날씨", "맑다", "봄"])

    def test_non_korean_and_blank_entries_are_dropped(self):
        self.assertEqual(clean_terms(["", "  ", "weather", "비가 오다", "비가오다"]), ["비가 오다"])


class ExtractTests(unittest.TestCase):
    def run_extract(self, gemini, text=LESSON, **kw):
        return extract(text.encode("utf-8"), "text/plain", generate=gemini, **kw)

    def test_every_term_in_the_lesson_gets_a_card_even_when_the_model_listed_only_14(self):
        gem = FakeGemini(lazy_inventory())
        draft = self.run_extract(gem)
        hangul = {c.hangul for c in draft.vocab}
        for term in ("봄", "벚꽃이 피다", "스키를 타다", "여의도", "두꺼운 양말", "날씨", "사계절"):
            self.assertIn(term, hangul)
        self.assertGreaterEqual(len(draft.vocab), 40)
        self.assertEqual(draft.missing, [])
        self.assertEqual([g.pattern for g in draft.grammar], LAZY_GRAMMAR)
        self.assertEqual((draft.title, draft.level, draft.confidence), ("Thời tiết và Mùa", 2, 0.9))
        self.assertEqual(draft.content, "Bài học về thời tiết và bốn mùa.")

    def test_cards_are_asked_for_in_small_groups_each_carrying_the_document(self):
        gem = FakeGemini(lazy_inventory())
        draft = self.run_extract(gem)
        vocab_calls = [c for c in gem.card_calls() if c[0] is VOCAB_CARDS_SCHEMA]
        self.assertEqual(len(vocab_calls), -(-len(draft.vocab) // VOCAB_GROUP))
        for _schema, text, document in gem.card_calls():
            self.assertLessEqual(len(refs_in(text)), VOCAB_GROUP)
            self.assertIsNotNone(document.inline_data)  # the lesson travels with every call

    def test_a_card_the_model_leaves_out_is_asked_for_again(self):
        gem = FakeGemini(lazy_inventory(), omit_first_time={"봄", "우산"})
        draft = self.run_extract(gem)
        self.assertEqual(draft.missing, [])
        self.assertTrue({"봄", "우산"} <= {c.hangul for c in draft.vocab})
        retries = [c for c in gem.card_calls() if "bị bỏ sót" in c[1]]
        self.assertEqual(len(retries), 1)

    def test_a_card_that_never_comes_is_shown_as_an_empty_red_one_not_dropped(self):
        gem = FakeGemini(lazy_inventory(), skip={"봄"})
        draft = self.run_extract(gem)
        self.assertEqual(draft.missing, ["봄"])
        card = next(c for c in draft.vocab if c.hangul == "봄")
        self.assertEqual((card.meaning_vi, card.confidence), ("", 0.2))
        self.assertEqual(ingestion.confidence_status(card.confidence), "flagged_red")

    def test_the_cards_line_up_with_the_inventory_order(self):
        draft = self.run_extract(FakeGemini(lazy_inventory()))
        self.assertEqual([c.hangul for c in draft.vocab[:3]], ["날씨", "일기예보", "맑다"])

    def test_a_total_outage_fails_the_job_instead_of_staging_empty_cards(self):
        with self.assertRaises(RuntimeError):
            self.run_extract(FakeGemini(lazy_inventory(), break_all_cards=True))

    def test_one_bad_number_does_not_cost_the_group(self):
        base = FakeGemini(lazy_inventory())

        def sloppy(**kw):
            out = base(**kw)
            data = json.loads(out["text"])
            for item in data.get("items", []):
                item["level"] = 9
                item["confidence"] = "high"
            return reply(data)

        draft = self.run_extract(sloppy)
        self.assertTrue(all(1 <= c.level <= 6 for c in draft.vocab))
        self.assertEqual(draft.missing, [])

    def test_unknown_and_duplicate_refs_are_ignored(self):
        base = FakeGemini(lazy_inventory())

        def noisy(**kw):
            out = base(**kw)
            data = json.loads(out["text"])
            if data.get("items") and kw["response_schema"] is VOCAB_CARDS_SCHEMA:
                first = data["items"][0]
                data["items"] += [dict(first, ref="v999", hangul="가짜"), dict(first, meaning_vi="khác")]
            return reply(data)

        draft = self.run_extract(noisy)
        self.assertNotIn("가짜", {c.hangul for c in draft.vocab})
        self.assertEqual(draft.vocab[0].meaning_vi, "nghĩa của 날씨")  # the first answer for a ref wins

    def test_the_overview_failing_leaves_the_cards_intact(self):
        draft = self.run_extract(FakeGemini(lazy_inventory(), fail_overview=True))
        self.assertEqual(draft.content, "")
        self.assertGreaterEqual(len(draft.vocab), 40)

    def test_a_lesson_that_is_not_plain_text_relies_on_the_models_inventory(self):
        gem = FakeGemini(lazy_inventory())
        draft = extract(b"\x89PNG fake image bytes", "image/png", generate=gem)
        self.assertEqual(len(draft.vocab), len(LAZY_VOCAB))

    def test_an_empty_inventory_uses_the_one_call_fallback_and_without_one_it_fails(self):
        empty = {"title": "x", "level": 1, "confidence": 0.5, "vocab_terms": [], "grammar_patterns": []}
        sentinel = lesson_extract.LessonDraft("Cũ", 1, "", [], 0.6)
        self.assertIs(extract(b"hello", "text/plain", generate=FakeGemini(empty), fallback=lambda: sentinel), sentinel)
        with self.assertRaises(RuntimeError):
            extract(b"hello", "text/plain", generate=FakeGemini(empty))

    def test_progress_runs_from_the_inventory_to_the_last_group_without_overshooting(self):
        seen = []
        self.run_extract(FakeGemini(lazy_inventory()), on_progress=lambda d, t, step: seen.append((d, t, step)))
        done = [d for d, _t, _s in seen]
        self.assertEqual(done, sorted(done))
        self.assertTrue(all(d <= t for d, t, _s in seen))
        self.assertIn("danh mục", seen[0][2])
        self.assertEqual(len({t for _d, t, _s in seen}), 1)

    def test_grammar_cards_keep_the_usage_fields(self):
        draft = self.run_extract(FakeGemini(lazy_inventory()))
        self.assertTrue(all(g.usage_context_vi == "dùng khi..." for g in draft.grammar))


class PromptContractTests(unittest.TestCase):
    def test_the_inventory_defines_what_a_card_is(self):
        for needle in ("lồng trong", "gạch chéo", "bổ trợ", "địa danh", "Danh mục thiếu mục là lỗi nặng nhất"):
            self.assertIn(needle.lower(), " ".join(INVENTORY_PROMPT.lower().split()))

    def test_a_card_group_names_its_refs_and_demands_one_card_each(self):
        prompt = lesson_extract.cards_prompt("vocab", "Thời tiết", [("v1", "봄"), ("v2", "여름")])
        self.assertIn("v1. 봄", prompt)
        self.assertIn("ĐÚNG MỘT thẻ", prompt)
        self.assertNotIn("bị bỏ sót", prompt)
        self.assertIn("bị bỏ sót", lesson_extract.cards_prompt("vocab", "x", [("v1", "봄")], retry=True))

    def test_the_grammar_rules_keep_tips_as_observations_not_orders(self):
        prompt = lesson_extract.cards_prompt("grammar", "x", [("g1", "V + -(으)면서")])
        self.assertIn("NHẬN XÉT VỀ NGÔN NGỮ", " ".join(prompt.split()))
        self.assertIn("usage_context_vi", prompt)


def inventory_terms():
    """The vocabulary list the pipeline builds for LESSON with the lazy inventory
    (what the organise call's v1, v2... refer to)."""
    return merge_terms(clean_terms(LAZY_VOCAB), bold_terms(LESSON))


def ref_of(term):
    return f"v{inventory_terms().index(term) + 1}"


WEATHER_VERBS = ["비가 오다", "눈이 오다", "바람이 불다", "꽃이 피다", "단풍이 들다"]
TEMPERATURE = ["덥다", "따뜻하다", "선선하다", "쌀쌀하다", "춥다"]


def weather_organise():
    return {
        "vocab_families": [
            {"label": "Động từ đi với thời tiết", "refs": [ref_of(t) for t in WEATHER_VERBS]},
            {"label": "Thang nhiệt độ", "refs": [ref_of(t) for t in TEMPERATURE]},
        ],
        "grammar_groups": [{"label": "Phỏng đoán với 것 같다", "refs": ["g2", "g3"]}],
    }


class FamilyTests(unittest.TestCase):
    def run_weather(self, **kw):
        gem = FakeGemini(lazy_inventory(), organise=weather_organise(), **kw)
        return gem, extract(LESSON.encode("utf-8"), "text/plain", generate=gem)

    def test_the_members_of_a_family_sit_side_by_side_and_carry_its_label(self):
        _gem, draft = self.run_weather()
        names = [c.hangul for c in draft.vocab]
        first = names.index("비가 오다")
        self.assertEqual(names[first : first + 5], WEATHER_VERBS)
        family = {c.hangul: c.family for c in draft.vocab}
        for term in WEATHER_VERBS:
            self.assertEqual(family[term], "Động từ đi với thời tiết")
        for term in TEMPERATURE:
            self.assertEqual(family[term], "Thang nhiệt độ")
        self.assertIsNone(family["날씨"])  # not every word has a family

    def test_the_rest_of_the_lesson_keeps_its_own_order(self):
        _gem, draft = self.run_weather()
        plain = [c.hangul for c in draft.vocab if c.family is None]
        self.assertEqual(plain, [t for t in inventory_terms() if t in plain])
        self.assertEqual(len(draft.vocab), len(inventory_terms()))  # nothing lost by regrouping

    def test_a_card_call_sees_the_whole_family_even_when_only_part_of_it_is_in_the_call(self):
        gem, _draft = self.run_weather()
        weather_calls = [c for c in gem.card_calls() if c[0] is VOCAB_CARDS_SCHEMA and "비가 오다" in c[1]]
        self.assertTrue(weather_calls)
        text = weather_calls[0][1]
        self.assertIn("Động từ đi với thời tiết: 비가 오다 | 눈이 오다 | 바람이 불다 | 꽃이 피다 | 단풍이 들다", text)
        # the lines the model must answer stay a bare "v1. term" list
        for _ref, term in refs_in(text):
            self.assertNotIn("Động từ", term)

    def test_organising_costs_one_short_call_and_the_progress_total_includes_it(self):
        seen = []
        gem = FakeGemini(lazy_inventory(), organise=weather_organise())
        extract(LESSON.encode("utf-8"), "text/plain", generate=gem, on_progress=lambda d, t, s: seen.append((d, t, s)))
        self.assertEqual(sum(1 for c in gem.calls if c[0] is ORGANISE_SCHEMA), 1)
        done = [d for d, _t, _s in seen]
        self.assertEqual(done, sorted(done))
        self.assertTrue(all(d <= t for d, t, _s in seen))
        self.assertEqual(len({t for _d, t, _s in seen}), 1)
        self.assertTrue(any("họ cụm" in step for _d, _t, step in seen))

    def test_a_failed_organise_call_only_means_no_sets(self):
        gem = FakeGemini(lazy_inventory(), fail_organise=True)
        draft = extract(LESSON.encode("utf-8"), "text/plain", generate=gem)
        self.assertTrue(all(c.family is None for c in draft.vocab))
        self.assertTrue(all(g.contrast_group is None for g in draft.grammar))
        self.assertEqual(draft.missing, [])
        self.assertEqual([c.hangul for c in draft.vocab[:3]], ["날씨", "일기예보", "맑다"])  # lesson order kept

    def test_a_card_cannot_name_its_own_family(self):
        gem = FakeGemini(lazy_inventory(), extra={"비가 오다": {"family": "tự đặt", "contrast_group": "tự đặt"}})
        draft = extract(LESSON.encode("utf-8"), "text/plain", generate=gem)
        self.assertIsNone(next(c for c in draft.vocab if c.hangul == "비가 오다").family)


class ParseGroupsTests(unittest.TestCase):
    def test_only_sets_that_hold_up_are_kept(self):
        data = {
            "vocab_families": [
                {"label": "Họ A", "refs": ["v1", "v2", "v2", "v99", "x3"]},
                {"label": "Họ B", "refs": ["v2", "v3"]},  # v2 already belongs to A: B has one member left
                {"label": "  ", "refs": ["v4", "v5"]},  # no label
                {"label": "Họ C", "refs": ["v4", "v5"]},
                "not a dict",
                {"label": "Họ D", "refs": "v6"},
            ]
        }
        got = parse_groups(data, "vocab_families", "v", 10)
        self.assertEqual(got, [("Họ A", [0, 1]), ("Họ C", [3, 4])])

    def test_a_set_is_cut_at_the_cap_and_members_come_back_in_lesson_order(self):
        refs = [f"v{i}" for i in range(12, 0, -1)]
        (label, members), = parse_groups({"vocab_families": [{"label": "Lớn", "refs": refs}]}, "vocab_families", "v", 12)
        self.assertEqual(members, list(range(MAX_FAMILY)))

    def test_the_wrong_kind_of_ref_is_ignored(self):
        data = {"grammar_groups": [{"label": "G", "refs": ["v1", "v2", "g1", "g2"]}]}
        self.assertEqual(parse_groups(data, "grammar_groups", "g", 4), [("G", [0, 1])])

    def test_garbage_gives_no_sets(self):
        for data in ({}, {"vocab_families": None}, {"vocab_families": "x"}, {"vocab_families": [None]}):
            self.assertEqual(parse_groups(data, "vocab_families", "v", 5), [])


class ClusterOrderTests(unittest.TestCase):
    def test_members_gather_at_the_first_members_place(self):
        self.assertEqual(cluster_order(7, [("A", [1, 5]), ("B", [2, 3])]), [0, 1, 5, 2, 3, 4, 6])

    def test_without_sets_nothing_moves(self):
        self.assertEqual(cluster_order(4, []), [0, 1, 2, 3])

    def test_every_index_appears_once(self):
        order = cluster_order(10, [("A", [0, 9]), ("B", [4, 5, 6])])
        self.assertEqual(sorted(order), list(range(10)))


class ChunkLayerTests(unittest.TestCase):
    def card(self, term, **fields):
        gem = FakeGemini(lazy_inventory(), extra={term: fields})
        draft = extract(LESSON.encode("utf-8"), "text/plain", generate=gem)
        return next(c for c in draft.vocab if c.hangul == term)

    def test_a_good_chunk_card_keeps_every_layer(self):
        card = self.card(
            "비가 오다",
            node_word="오다",
            distractors=["내리다", "불다", "들다"],
            collocations=[{"ko": "눈이 오다", "vi": "tuyết rơi"}],
            register="neutral",
            usage_note_vi="Hàn nói 'mưa đến', không nói 'mưa rơi'.",
        )
        self.assertEqual(card.node_word, "오다")
        self.assertEqual(card.distractors, ["내리다", "불다", "들다"])
        self.assertEqual([(c.ko, c.vi) for c in card.collocations], [("눈이 오다", "tuyết rơi")])
        self.assertEqual((card.register, card.usage_note_vi), ("neutral", "Hàn nói 'mưa đến', không nói 'mưa rơi'."))

    def test_a_node_word_that_is_not_part_of_the_phrase_is_dropped_with_its_distractors(self):
        card = self.card("비가 오다", node_word="내리다", distractors=["불다"])
        self.assertIsNone(card.node_word)
        self.assertIsNone(card.distractors)

    def test_a_node_word_equal_to_the_whole_word_is_no_chunk(self):
        card = self.card("날씨", node_word="날씨", distractors=["기온"])
        self.assertIsNone(card.node_word)
        self.assertIsNone(card.distractors)

    def test_distractors_are_cleaned(self):
        card = self.card("비가 오다", node_word="오다", distractors=["오다", "불다", "불 다", "rain", "", None, 7, "들다", "피다", "끼다"])
        self.assertEqual(card.distractors, ["불다", "들다", "피다"])  # no answer, no repeat, no non-Korean, at most 3

    def test_collocations_are_cleaned(self):
        card = self.card(
            "비가 오다",
            collocations=[
                {"ko": "비가 오다", "vi": "chính nó"},  # the headword itself
                {"ko": "눈이 오다"},  # no meaning
                {"ko": "snow", "vi": "tuyết"},  # not Korean
                "눈이 오다",
                {"ko": "눈이 오다", "vi": "tuyết rơi"},
                {"ko": "눈이오다", "vi": "lặp lại"},
                {"ko": "바람이 불다", "vi": "gió thổi"},
                {"ko": "꽃이 피다", "vi": "hoa nở"},
                {"ko": "단풍이 들다", "vi": "lá đổi màu"},
            ],
        )
        self.assertEqual([c.ko for c in card.collocations], ["눈이 오다", "바람이 불다", "꽃이 피다"])

    def test_an_unknown_register_is_dropped(self):
        self.assertIsNone(self.card("비가 오다", register="colloquial").register)
        self.assertEqual(self.card("비가 오다", register="Written").register, "written")

    def test_blank_or_wrongly_typed_layers_become_none(self):
        card = self.card("비가 오다", usage_note_vi="   ", collocations="눈이 오다", distractors={"a": 1}, node_word=3)
        self.assertEqual((card.usage_note_vi, card.collocations, card.distractors, card.node_word), (None, None, None, None))

    def test_a_card_without_any_layer_is_still_a_card(self):
        card = self.card("날씨")
        self.assertEqual((card.meaning_vi, card.node_word, card.collocations, card.register), ("nghĩa của 날씨", None, None, None))

    def test_the_layers_survive_into_the_staged_payload(self):
        card = self.card("비가 오다", node_word="오다", distractors=["불다"], register="spoken")
        payload = card.model_dump(exclude={"confidence"})
        for key in ("family", "node_word", "register", "usage_note_vi", "collocations", "distractors"):
            self.assertIn(key, payload)
        json.dumps(payload, ensure_ascii=False)  # JSON-safe: it goes into a JSONB column


class ContrastTests(unittest.TestCase):
    PAIR = {"vocab_families": [], "grammar_groups": [{"label": "Phỏng đoán với 것 같다", "refs": ["g2", "g3"]}]}

    def run_pair(self, contrasts_for):
        extra = {term: {"contrasts": c} for term, c in contrasts_for.items()}
        gem = FakeGemini(lazy_inventory(), organise=self.PAIR, extra=extra)
        draft = extract(LESSON.encode("utf-8"), "text/plain", generate=gem)
        return gem, {g.pattern: g for g in draft.grammar}

    def test_the_group_is_labelled_and_its_members_sit_together(self):
        _gem, by = self.run_pair({})
        self.assertEqual(by["V + -(으)ㄹ 것 같다"].contrast_group, "Phỏng đoán với 것 같다")
        self.assertEqual(by["V/A + -는/은/ㄴ 것 같다"].contrast_group, "Phỏng đoán với 것 같다")
        self.assertIsNone(by["V + -(으)면서"].contrast_group)

    def test_a_contrast_is_linked_to_the_mate_it_names(self):
        _gem, by = self.run_pair(
            {
                "V + -(으)ㄹ 것 같다": [{"pattern": "V/A + -는/은/ㄴ 것 같다", "diff_vi": "Tương lai, chưa xảy ra."}],
                "V/A + -는/은/ㄴ 것 같다": [{"pattern": "V + -(으)ㄹ 것 같다", "diff_vi": "Hiện tại, đang xảy ra."}],
            }
        )
        a = by["V + -(으)ㄹ 것 같다"].contrasts
        self.assertEqual([(c.pattern, c.diff_vi) for c in a], [("V/A + -는/은/ㄴ 것 같다", "Tương lai, chưa xảy ra.")])
        b = by["V/A + -는/은/ㄴ 것 같다"].contrasts
        self.assertEqual([c.pattern for c in b], ["V + -(으)ㄹ 것 같다"])

    def test_in_a_pair_a_reworded_name_can_only_mean_the_other_one(self):
        _gem, by = self.run_pair({"V + -(으)ㄹ 것 같다": [{"pattern": "-는 것 같다", "diff_vi": "Hiện tại."}]})
        self.assertEqual([c.pattern for c in by["V + -(으)ㄹ 것 같다"].contrasts], ["V/A + -는/은/ㄴ 것 같다"])

    def test_a_contrast_naming_nothing_in_the_group_is_dropped(self):
        trio = {"vocab_families": [], "grammar_groups": [{"label": "Nhóm ba", "refs": ["g1", "g2", "g3"]}]}
        extra = {"V + -(으)ㄹ 것 같다": {"contrasts": [{"pattern": "V + -(으)면서", "diff_vi": "không cùng nhóm"}]}}
        gem = FakeGemini(lazy_inventory(), organise=trio, extra=extra)
        draft = extract(LESSON.encode("utf-8"), "text/plain", generate=gem)
        card = next(g for g in draft.grammar if g.pattern == "V + -(으)ㄹ 것 같다")
        self.assertIsNone(card.contrasts)

    def test_a_pattern_never_contrasts_with_itself(self):
        _gem, by = self.run_pair({"V + -(으)ㄹ 것 같다": [{"pattern": "V + -(으)ㄹ 것 같다", "diff_vi": "chính nó"}]})
        self.assertIsNone(by["V + -(으)ㄹ 것 같다"].contrasts)

    def test_the_card_call_is_told_to_contrast_and_sees_its_group_mates(self):
        gem, _by = self.run_pair({})
        grammar_calls = [c for c in gem.card_calls() if c[0] is GRAMMAR_CARDS_SCHEMA]
        text = " ".join(grammar_calls[0][1].split())
        self.assertIn("Phỏng đoán với 것 같다: V + -(으)ㄹ 것 같다 | V/A + -는/은/ㄴ 것 같다", text)
        self.assertIn("contrasts", text)


class OrganisePromptTests(unittest.TestCase):
    def test_the_organise_prompt_numbers_both_lists_with_the_refs_the_answer_must_use(self):
        prompt = lesson_extract.organise_prompt(["비가 오다", "눈이 오다"], ["V + -(으)면서"])
        self.assertIn("v1. 비가 오다", prompt)
        self.assertIn("v2. 눈이 오다", prompt)
        self.assertIn("g1. V + -(으)면서", prompt)
        self.assertIn("không thêm ref", " ".join(prompt.split()))

    def test_the_new_schemas_are_valid_for_the_gemini_sdk(self):
        from google.genai import types

        for schema in (VOCAB_CARDS_SCHEMA, GRAMMAR_CARDS_SCHEMA, ORGANISE_SCHEMA):
            types.Schema.model_validate(schema)

    def test_the_vocab_rules_ask_for_chunk_layers_without_inviting_invention(self):
        text = " ".join(lesson_extract.cards_prompt("vocab", "x", [("v1", "비가 오다")]).split())
        for needle in ("node_word", "distractors", "collocations", "register", "usage_note_vi", "không bịa"):
            self.assertIn(needle, text)

    def test_a_distractor_that_is_also_good_korean_is_ruled_out_in_the_prompt(self):
        # a learner who picks a valid alternative must not be marked wrong
        text = " ".join(lesson_extract.cards_prompt("vocab", "x", [("v1", "비가 오다")]).split())
        self.assertIn("người Hàn vẫn nói được", text)
        self.assertIn("bị chấm sai oan", text)

    def test_the_lesson_version_moved_so_a_reupload_reads_the_file_again(self):
        self.assertEqual(lesson_extract.PROMPT_VERSION, "lesson-v4")


class StagingTests(unittest.TestCase):
    def test_the_inventory_and_gap_cards_are_all_staged_with_traffic_light_status(self):
        draft = lesson_extract.LessonDraft(
            "Thời tiết", 2, "nội dung", ["Thời tiết"], 0.9,
            vocab=[lesson_extract.VocabCard(hangul="봄", meaning_vi="mùa xuân", confidence=0.9),
                   lesson_extract.VocabCard(hangul="여름", meaning_vi="", confidence=0.2)],
            grammar=[lesson_extract.GrammarCard(pattern="V + -(으)면서", meaning_vi="vừa... vừa", confidence=0.7)],
            missing=["여름"],
        )
        added = []
        db = SimpleNamespace(add=added.append, flush=lambda: None)
        batch = SimpleNamespace(id="b1")
        with mock.patch.object(ingestion, "extract_lesson", return_value=draft):
            staged, flagged = ingestion.run_lesson_extraction(db, batch, b"x", "text/plain")
        self.assertEqual(staged, 4)  # the lesson, 2 words, 1 pattern
        self.assertEqual(flagged, 2)  # the empty card (red) and the 0.7 pattern (yellow)
        kinds = [(i.kind, i.status) for i in added]
        self.assertEqual(kinds, [("lesson", "pending"), ("vocab_item", "pending"), ("vocab_item", "flagged_red"),
                                 ("grammar_point", "flagged_yellow")])
        self.assertEqual(added[1].payload["hangul"], "봄")
        self.assertNotIn("confidence", added[1].payload)

    def test_chunk_layers_and_contrasts_are_staged_in_the_payload(self):
        draft = lesson_extract.LessonDraft(
            "Thời tiết", 2, "", [], 0.9,
            vocab=[lesson_extract.VocabCard(hangul="비가 오다", meaning_vi="trời mưa", family="Thời tiết", node_word="오다",
                                            distractors=["불다"], register="neutral", confidence=0.9)],
            grammar=[lesson_extract.GrammarCard(pattern="V + -(으)ㄹ 것 같다", meaning_vi="có vẻ sẽ", contrast_group="것 같다",
                                                contrasts=[lesson_extract.Contrast(pattern="-는 것 같다", diff_vi="hiện tại")],
                                                confidence=0.9)],
        )
        added = []
        db = SimpleNamespace(add=added.append, flush=lambda: None)
        with mock.patch.object(ingestion, "extract_lesson", return_value=draft):
            ingestion.run_lesson_extraction(db, SimpleNamespace(id="b1"), b"x", "text/plain")
        vocab, grammar = added[1].payload, added[2].payload
        self.assertEqual((vocab["family"], vocab["node_word"], vocab["distractors"]), ("Thời tiết", "오다", ["불다"]))
        self.assertEqual(grammar["contrast_group"], "것 같다")
        self.assertEqual(grammar["contrasts"], [{"pattern": "-는 것 같다", "diff_vi": "hiện tại"}])


if __name__ == "__main__":
    unittest.main()
