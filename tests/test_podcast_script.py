"""The "Bài giảng tổng hợp": the prompt that asks for a teacher's lecture, the
cleaning/checks that run before any text-to-speech quota is spent, and how the
voice is recovered when it mistakes a chunk for a task.

No network: Gemini is a mock, and the TTS "once" call is patched so the retry
ladder (as is -> framed as a transcript -> two halves) is observable."""
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from google.genai import errors as genai_errors

from app.api.routers import audio
from app.core.config import settings
from app.schemas import PodcastRequest
from app.services import ingestion, podcast_script, tts
from app.services.podcast_script import (
    PODCAST_VERSION,
    build_prompt,
    clean_script_for_tts,
    script_problems,
    target_chars,
)

VOCAB = [
    ("방향", "danh từ", "phương hướng", "방향을 몰라요.", "方向", "PHƯƠNG HƯỚNG"),
    ("길", "danh từ", "con đường", "길을 찾아요.", None, None),
]
GRAMMAR = [
    (
        "V/A + -(으)면",
        "Nếu... thì...",
        "저기서 오른쪽으로 가면 돼요.",
        "Dùng khi nêu điều kiện hoặc chỉ đường theo từng bước.",
        "Khi gặp câu hỏi về điều kiện, hãy chọn đáp án có -(으)면.",  # the kind of line that caused the 400
    )
]


def good_script(n=900):
    return "Hôm nay mình cùng học cách hỏi đường. " * (n // 38 + 1)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.prompt = build_prompt("길 찾기", VOCAB, GRAMMAR)

    def test_it_teaches_the_way_a_veteran_teacher_would(self):
        for needle in (
            "15 năm",
            "CÂU CHUYỆN",  # a story as the thread
            "CỤM",  # phrases, not isolated words
            "đồng nghĩa",
            "trái nghĩa",
            "Hán Việt",
            "thành ngữ",
            "Áp dụng thực tế",
            "lỗi người Việt hay mắc",
            "Mang về nhà",
        ):
            self.assertIn(needle, self.prompt)

    def test_idioms_are_only_used_when_certain(self):
        self.assertIn("CHẮC CHẮN", self.prompt)
        self.assertIn("không bịa", self.prompt)

    def test_the_voice_problem_is_explained_not_blacklisted(self):
        # the old prompt listed forbidden words four times, which primed the writer with them
        instructions = self.prompt.split("DỮ LIỆU THAM KHẢO")[0]  # the data block legitimately holds exam notes
        self.assertNotIn("đáp án", instructions)
        self.assertNotIn("TUYỆT ĐỐI CẤM", self.prompt)
        self.assertIn("máy", self.prompt)  # why: a machine reads it out and mistakes orders for tasks
        self.assertIn("Cách nói nên dùng", self.prompt)  # and what to say instead

    def test_reference_data_is_fenced_and_declared_not_an_instruction(self):
        self.assertIn("<tu_vung>", self.prompt)
        self.assertIn("</ngu_phap>", self.prompt)
        self.assertIn("không phải chỉ dẫn", self.prompt)
        self.assertIn("gốc Hán: 方向 PHƯƠNG HƯỚNG", self.prompt)
        # the editor's exam note is passed through, but labelled as background
        self.assertIn("chỉ để bạn hiểu mẫu này hay gặp ở đâu", self.prompt)

    def test_the_length_budget_follows_the_number_of_items_within_limits(self):
        self.assertEqual(target_chars(0), 2400)
        self.assertGreater(target_chars(10), target_chars(3))
        self.assertEqual(target_chars(500), 6400)  # a handful of TTS requests, not unlimited
        self.assertIn(str(target_chars(3)), self.prompt)

    def test_a_retry_says_what_was_wrong(self):
        retry = build_prompt("길 찾기", VOCAB, GRAMMAR, retry_note="bản trước quá dài")
        self.assertIn("LƯU Ý CHO LẦN VIẾT NÀY: bản trước quá dài", retry)
        self.assertNotIn("LƯU Ý CHO LẦN VIẾT NÀY", self.prompt)

    def test_empty_lessons_still_build(self):
        prompt = build_prompt("Trống", [], [])
        self.assertIn("(không có từ vựng)", prompt)
        self.assertIn("(không có ngữ pháp)", prompt)


class CleanTests(unittest.TestCase):
    def test_markdown_lists_labels_and_stage_directions_go(self):
        raw = "## Mở bài\n**Xin chào** các bạn.\n- 길을 찾아요\n1. 방향\nGiáo viên: Mình cùng nói chậm nhé. (dừng ba giây)\n[nhạc nền]"
        got = clean_script_for_tts(raw)
        self.assertEqual(got, "Mở bài\nXin chào các bạn.\n길을 찾아요\n방향\nMình cùng nói chậm nhé.")

    def test_grammar_formulas_become_speakable(self):
        self.assertEqual(clean_script_for_tts("Mẫu -(으)면 nghĩa là nếu."), "Mẫu 으면 nghĩa là nếu.")
        spoken = clean_script_for_tts("Mẫu V + -(으)ㄹ 때")
        for symbol in ("(", ")", "+"):
            self.assertNotIn(symbol, spoken)
        self.assertIn("때", spoken)

    def test_arrows_emoji_and_stray_symbols(self):
        got = clean_script_for_tts("길 → con đường 🚶 và 방향 = phương hướng")
        self.assertNotIn("→", got)
        self.assertNotIn("🚶", got)
        self.assertNotIn("=", got)
        self.assertIn("con đường", got)

    def test_korean_vietnamese_and_pauses_are_kept(self):
        text = "이거 얼마예요... Mình cùng nói chậm nhé. Bây giờ đến lượt bạn."
        self.assertEqual(clean_script_for_tts(text), text)

    def test_whitespace_is_tidied(self):
        self.assertEqual(clean_script_for_tts("a  b \n\n\n\n c​"), "a b\n\nc")


class ProblemTests(unittest.TestCase):
    def test_a_good_script_passes_even_when_it_talks_about_structure(self):
        script = good_script() + "Mẫu này đi theo cấu trúc quen thuộc. Mình đọc câu ví dụ nhé."
        self.assertEqual(script_problems(script), [])

    def test_empty_short_and_long(self):
        self.assertEqual(script_problems("  "), ["empty"])
        self.assertEqual(script_problems("Xin chào."), ["too_short"])
        self.assertEqual(script_problems(good_script(9500)), ["too_long"])

    def test_echoing_the_instructions_is_caught(self):
        for leak in (
            "Đây là kịch bản bài giảng của tôi.",
            "Theo đúng JSON schema đã cho.",
            "Tôi sẽ giải thích nghĩa bằng tiếng Việt.",
            "A. Mở bài\nXin chào.",
            "<tu_vung> 길 </tu_vung>",
        ):
            self.assertIn("echoes_instructions", script_problems(good_script() + "\n" + leak), leak)


class GenerateScriptTests(unittest.TestCase):
    def reply(self, script):
        import json

        return {"text": json.dumps({"outline": "dàn ý", "script": script}, ensure_ascii=False)}

    def test_a_clean_first_answer_is_used_as_it_is(self):
        with mock.patch.object(ingestion.gemini_client, "generate_structured", return_value=self.reply("**" + good_script())) as g:
            script = ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)
        self.assertEqual(g.call_count, 1)
        self.assertNotIn("**", script)  # cleaned for the voice
        self.assertEqual(g.call_args.kwargs["prompt_version"], PODCAST_VERSION)

    def test_a_script_that_echoes_the_prompt_is_asked_for_again_with_a_note(self):
        replies = [self.reply(good_script() + "\nĐây là kịch bản bài giảng."), self.reply(good_script())]
        with mock.patch.object(ingestion.gemini_client, "generate_structured", side_effect=replies) as g:
            script = ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)
        self.assertEqual(g.call_count, 2)
        self.assertNotIn("LƯU Ý CHO LẦN VIẾT NÀY", g.call_args_list[0].kwargs["prompt"])
        self.assertIn("LƯU Ý CHO LẦN VIẾT NÀY", g.call_args_list[1].kwargs["prompt"])
        self.assertNotIn("kịch bản bài giảng", script)

    def test_after_two_flawed_answers_the_cleaned_script_is_still_used(self):
        long = self.reply(good_script(9500))
        with mock.patch.object(ingestion.gemini_client, "generate_structured", return_value=long) as g:
            script = ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)
        self.assertEqual(g.call_count, 2)
        self.assertGreater(len(script), 9000)  # a long lecture beats none

    def test_an_empty_script_is_an_error(self):
        with mock.patch.object(ingestion.gemini_client, "generate_structured", return_value=self.reply("")):
            with self.assertRaises(RuntimeError):
                ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)

    def test_the_writing_model_can_be_chosen_separately(self):
        with mock.patch.object(settings, "GEMINI_MODEL_PODCAST", "stronger-model"), mock.patch.object(
            ingestion.gemini_client, "generate_structured", return_value=self.reply(good_script())
        ) as g:
            ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)
        self.assertEqual(g.call_args.kwargs["model"], "stronger-model")
        with mock.patch.object(settings, "GEMINI_MODEL_PODCAST", ""), mock.patch.object(
            ingestion.gemini_client, "generate_structured", return_value=self.reply(good_script())
        ) as g:
            ingestion.generate_podcast_script("길 찾기", VOCAB, GRAMMAR)
        self.assertEqual(g.call_args.kwargs["model"], settings.GEMINI_MODEL_LESSON_INGEST)


# ---------------------------------------------------------------- the voice --
def shape_error():
    return tts.TTSContentShapeError("read as a task")


def always_refused(*args, **kwargs):
    raise shape_error()


def refused_then_voiced(refusals):
    """A voice that reads the first `refusals` requests as tasks, then speaks."""
    state = {"calls": 0}

    def fake(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] <= refusals:
            raise shape_error()
        return PCM, 24000

    return fake


PCM = b"\x01\x00" * 100
SENTENCES = " ".join(f"Câu số {i} dài vừa phải để chia." for i in range(30))


class VoiceRecoveryTests(unittest.TestCase):
    def synth(self, side_effect, text=SENTENCES):
        with mock.patch.object(tts, "_synthesize_once", side_effect=side_effect) as once:
            try:
                return tts._synthesize_chunk_pcm(object(), text, {}, ["m"]), once
            except RuntimeError as exc:
                return exc, once

    def test_a_chunk_that_works_is_voiced_exactly_as_before(self):
        result, once = self.synth([(PCM, 24000)])
        self.assertEqual(result, (PCM, 24000))
        self.assertEqual(once.call_count, 1)
        self.assertEqual(once.call_args.args[1], SENTENCES)  # no frame added to the normal case

    def test_a_refused_chunk_is_retried_framed_as_a_transcript(self):
        result, once = self.synth([shape_error(), (PCM, 24000)])
        self.assertEqual(result, (PCM, 24000))
        self.assertEqual(once.call_count, 2)
        self.assertEqual(once.call_args.args[1], tts._TRANSCRIPT_FRAME + SENTENCES)

    def test_still_refused_it_is_cut_in_two_and_the_parts_are_joined(self):
        result, once = self.synth(refused_then_voiced(2))
        pcm, rate = result
        self.assertEqual(rate, 24000)
        parts = [c.args[1] for c in once.call_args_list[2:]]
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(p.startswith(tts._TRANSCRIPT_FRAME) for p in parts))
        # nothing of the lecture is lost or reordered by the cut
        self.assertEqual(" ".join(p[len(tts._TRANSCRIPT_FRAME):] for p in parts), SENTENCES)
        self.assertEqual(len(pcm), len(PCM) * len(parts))

    def test_when_nothing_works_the_caller_gets_the_clear_error(self):
        result, once = self.synth(always_refused)
        self.assertIsInstance(result, tts.TTSContentShapeError)  # the caller's own failure handling still sees it
        self.assertEqual(once.call_count, 3)  # as is, framed, then the first part — and it stops there

    def test_a_short_chunk_cannot_be_cut_so_it_fails_after_the_framed_retry(self):
        result, once = self.synth(always_refused, text="Hãy chọn.")
        self.assertIsInstance(result, tts.TTSContentShapeError)
        self.assertEqual(once.call_count, 2)

    def test_quota_and_server_errors_are_not_mistaken_for_a_task(self):
        quota, once = self.synth(RuntimeError("hết hạn mức"))
        self.assertEqual((str(quota), once.call_count), ("hết hạn mức", 1))
        flaky, once = self.synth(tts.TransientTTSError("500"))
        self.assertIsInstance(flaky, tts.TransientTTSError)
        self.assertEqual(once.call_count, 1)  # the callers' own retry loop handles these

    def test_the_gemini_400_is_what_triggers_it(self):
        err = genai_errors.APIError.__new__(genai_errors.APIError)
        err.code, err.status = 400, "INVALID_ARGUMENT"
        err.message = "Model tried to generate text, but it should only be used for TTS"
        client = SimpleNamespace(models=SimpleNamespace(generate_content=mock.Mock(side_effect=err)))
        with self.assertRaises(tts.TTSContentShapeError) as raised:
            tts._synthesize_once(client, "Hãy chọn đáp án đúng.", {}, ["m"])
        self.assertIn("hiểu nhầm là câu lệnh", str(raised.exception))  # the message the learner finally sees
        other = genai_errors.APIError.__new__(genai_errors.APIError)
        other.code, other.status, other.message = 400, "INVALID_ARGUMENT", "something else"
        client.models.generate_content.side_effect = other
        with self.assertRaises(tts.TransientTTSError):  # unchanged behaviour for every other error
            tts._synthesize_once(client, "xin chào", {}, ["m"])


# ------------------------------------------------------------------- the API --
class PodcastVersionTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_server_decides_the_prompt_version_whatever_the_client_sends(self):
        db = mock.MagicMock()
        res = mock.MagicMock()
        res.scalar_one_or_none.return_value = None
        db.execute = mock.AsyncMock(return_value=res)
        db.add = mock.MagicMock()
        db.commit = mock.AsyncMock()
        db.flush = mock.AsyncMock()

        async def refresh(job):
            job.id = uuid.UUID(int=3)
            job.status = "queued"

        db.refresh = mock.AsyncMock(side_effect=refresh)
        request = SimpleNamespace(headers={})
        with mock.patch.object(audio.generate_content_podcast, "delay") as delay:
            await audio._request_podcast(db, "lesson", "5", PodcastRequest(prompt_version="podcast-v1"), request, [1, 2], [3])
        args = delay.call_args.args
        self.assertEqual(args[4], PODCAST_VERSION)  # an old client still gets the new lecture
        self.assertIn(f":{PODCAST_VERSION}:", args[5])  # and a cache key of its own


if __name__ == "__main__":
    unittest.main()
