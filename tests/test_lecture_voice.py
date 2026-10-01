"""The lecture voiced by Google Chirp 3 HD: one voice per language.

No network and no real credential: `google_tts.synthesize_pcm` is patched, so
what is asserted is which voice/pace each run goes to, the silence between
runs, the retry/replace behaviour and the Gemini fallback bookkeeping."""
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from app.api.routers import audio
from app.core.config import settings
from app.schemas import PodcastRequest
from app.services import google_tts, lecture_voice, tts
from app.services.lecture_voice import KO, VI, split_runs
from app.workers import tasks

SCRIPT = (
    'Hôm qua Lan ra chợ. Cô ấy hỏi bác bán hàng: "이거 얼마예요?"\n'
    "Mình dừng ở cụm này. 이거 얼마예요... Mình cùng nói chậm nhé. 이거... 얼마예요...\n"
    "얼마 nghĩa là bao nhiêu, và 이거 là cái này."
)
RATE = 24000
PCM_KO = b"\x02\x00" * 240  # 240 samples
PCM_VI = b"\x01\x00" * 120  # 120 samples


class Chirp:
    """Stands in for synthesize_pcm: remembers (voice, rate, text) per call."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, client=None, voice=None, rate=None):
        self.calls.append((voice, rate, text))
        return (PCM_KO if voice.startswith("ko-") else PCM_VI), RATE


class SplitTests(unittest.TestCase):
    def test_runs_follow_the_language_in_reading_order(self):
        runs = split_runs(SCRIPT)
        self.assertEqual([r.lang for r in runs], [VI, KO, VI, KO, VI, KO, KO, VI, KO, VI])
        self.assertEqual(runs[3].text, "이거 얼마예요...")
        self.assertEqual(runs[7].text, "nghĩa là bao nhiêu, và")
        # nothing is lost: every Hangul and every Vietnamese word is in some run
        joined = " ".join(r.text for r in runs)
        for word in ("Hôm qua Lan", "이거", "얼마예요", "bác bán hàng", "cái này."):
            self.assertIn(word, joined)

    def test_a_korean_phrase_to_repeat_gets_a_real_gap(self):
        runs = split_runs("Mình nói chậm nhé. 이거... 얼마예요... Bây giờ đến lượt bạn.")
        ko = next(r for r in runs if r.lang == KO)
        vi_between = runs[0]
        self.assertEqual(ko.pause_after_ms, lecture_voice.GAP_AFTER_ELLIPSIS_KO_MS)
        self.assertEqual(vi_between.pause_after_ms, lecture_voice.GAP_LANGUAGE_SWITCH_MS)
        self.assertEqual(runs[-1].pause_after_ms, 0)  # nothing after the last run

    def test_a_paragraph_ends_with_a_pause(self):
        runs = split_runs("Câu một.\nCâu hai.")
        self.assertEqual([r.pause_after_ms for r in runs], [lecture_voice.GAP_AFTER_PARAGRAPH_MS, 0])

    def test_runs_with_nothing_to_say_are_dropped(self):
        self.assertEqual(split_runs("...\n  \n— \n"), [])
        self.assertEqual([r.text for r in split_runs("Xin chào ... 안녕하세요 !!!")], ["Xin chào ...", "안녕하세요 !!!"])

    def test_pure_vietnamese_and_pure_korean_are_one_run_each(self):
        self.assertEqual([r.lang for r in split_runs("Chào các bạn. Hôm nay học bài mới.")], [VI])
        self.assertEqual([r.lang for r in split_runs("길을 찾아요. 저기서 오른쪽으로 가요.")], [KO])


class VoiceSettingsTests(unittest.TestCase):
    def test_the_vietnamese_voice_is_the_same_persona_unless_configured(self):
        with mock.patch.object(settings, "TTS_CHIRP_VOICE", "ko-KR-Chirp3-HD-Iapetus"), mock.patch.object(
            settings, "TTS_CHIRP_VOICE_VI", ""
        ):
            self.assertEqual(google_tts.vi_voice_name(), "vi-VN-Chirp3-HD-Iapetus")
        with mock.patch.object(settings, "TTS_CHIRP_VOICE_VI", "vi-VN-Chirp3-HD-Charon"):
            self.assertEqual(google_tts.vi_voice_name(), "vi-VN-Chirp3-HD-Charon")

    def test_either_voice_or_pace_changes_the_lecture_cache_spec(self):
        base = google_tts.lecture_spec()
        with mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE_VI", 1.1):
            self.assertNotEqual(google_tts.lecture_spec(), base)
        with mock.patch.object(settings, "TTS_CHIRP_VOICE_VI", "vi-VN-Chirp3-HD-Charon"):
            self.assertNotEqual(google_tts.lecture_spec(), base)
        with mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE", 0.7):
            self.assertNotEqual(google_tts.lecture_spec(), base)
        self.assertRegex(base, r"^gc3bi-[a-z0-9-]+$")

    def test_a_request_uses_the_voice_it_is_given_and_its_own_language_code(self):
        resp = mock.MagicMock(status_code=200)
        wav = b"RIFF" + (36 + 4).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
        wav += (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + RATE.to_bytes(4, "little")
        wav += (RATE * 2).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        wav += b"data" + (4).to_bytes(4, "little") + b"\x01\x00\x02\x00"
        import base64

        resp.json.return_value = {"audioContent": base64.b64encode(wav).decode()}
        with mock.patch.object(settings, "TTS_GG_CHIRP", "fake-test-credential"), mock.patch.object(
            google_tts.httpx, "post", return_value=resp
        ) as post:
            google_tts.synthesize_pcm("Xin chào", voice="vi-VN-Chirp3-HD-Iapetus", rate=0.95)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["voice"], {"languageCode": "vi-VN", "name": "vi-VN-Chirp3-HD-Iapetus"})
        self.assertEqual(body["audioConfig"]["speakingRate"], 0.95)
        self.assertNotIn("fake-test-credential", str(body))  # the credential travels in a header only

    def test_a_replacement_vietnamese_voice_prefers_the_persona_then_the_gender(self):
        vi = [
            {"name": "vi-VN-Chirp3-HD-Aoede", "gender": "FEMALE"},
            {"name": "vi-VN-Chirp3-HD-Charon", "gender": "MALE"},
        ]
        ko = [{"name": "ko-KR-Chirp3-HD-Iapetus", "gender": "MALE"}]
        lists = {"vi-VN": vi, "ko-KR": ko}
        with mock.patch.object(settings, "TTS_CHIRP_VOICE", "ko-KR-Chirp3-HD-Iapetus"), mock.patch.object(
            google_tts, "list_chirp_voices", side_effect=lambda lang: lists[lang]
        ):
            self.assertEqual(google_tts.pick_vi_voice("vi-VN-Chirp3-HD-Iapetus"), "vi-VN-Chirp3-HD-Charon")
            lists["vi-VN"] = vi + [{"name": "vi-VN-Chirp3-HD-Iapetus", "gender": "MALE"}]
            self.assertEqual(google_tts.pick_vi_voice("vi-VN-Chirp3-HD-Charon"), "vi-VN-Chirp3-HD-Iapetus")
            lists["vi-VN"] = []
            self.assertIsNone(google_tts.pick_vi_voice("vi-VN-Chirp3-HD-Iapetus"))


class PlanTests(unittest.TestCase):
    def test_every_request_fits_the_byte_limit(self):
        long_vi = " ".join(f"Đây là câu số {i} của một đoạn giải thích rất dài." for i in range(120))
        plan = tts.plan_lecture(long_vi + "\n이거 얼마예요...\n" + long_vi)
        self.assertGreater(len(plan), 3)
        for _lang, text, _pause in plan:
            self.assertLessEqual(len(text.encode("utf-8")), google_tts.MAX_REQUEST_BYTES)
        langs = [lang for lang, _t, _p in plan]
        self.assertEqual(langs.count(KO), 1)

    def test_pieces_of_one_long_run_are_joined_with_a_short_gap_and_the_run_end_pause_stays(self):
        long_vi = " ".join(f"Đây là câu số {i} của một đoạn giải thích rất dài." for i in range(60)) + "\nHết."
        plan = tts.plan_lecture(long_vi)
        pauses = [p for _l, _t, p in plan[:-1]]
        self.assertEqual(pauses[-1], lecture_voice.GAP_AFTER_PARAGRAPH_MS)
        self.assertTrue(all(p == tts._LECTURE_SPLIT_GAP_MS for p in pauses[:-1]))


@mock.patch.object(tts.time, "sleep", lambda *_: None)
@mock.patch.object(tts, "transcode_pcm", lambda pcm, rate: (pcm, b"", len(pcm)))
class SynthesizeLectureTests(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(settings, "TTS_GG_CHIRP", "fake-test-credential"),
            mock.patch.object(settings, "TTS_CHIRP_VOICE", "ko-KR-Chirp3-HD-Iapetus"),
            mock.patch.object(settings, "TTS_CHIRP_VOICE_VI", ""),
            mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE", 0.85),
            mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE_VI", 0.95),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_each_run_goes_to_the_voice_and_pace_of_its_language(self):
        chirp = Chirp()
        with mock.patch.object(google_tts, "synthesize_pcm", chirp):
            tts.synthesize_lecture_chirp("Chào các bạn. 이거 얼마예요... Nhớ nhé.")
        self.assertEqual(
            [(v, r) for v, r, _t in chirp.calls],
            [("vi-VN-Chirp3-HD-Iapetus", 0.95), ("ko-KR-Chirp3-HD-Iapetus", 0.85), ("vi-VN-Chirp3-HD-Iapetus", 0.95)],
        )

    def test_the_audio_is_joined_in_order_with_the_silences(self):
        chirp = Chirp()
        with mock.patch.object(google_tts, "synthesize_pcm", chirp):
            pcm, _aac, _dur = tts.synthesize_lecture_chirp("Chào các bạn. 이거 얼마예요... Nhớ nhé.")
        gap = lambda ms: b"\x00" * (int(RATE * ms / 1000) * 2)  # noqa: E731
        expected = (
            PCM_VI + gap(lecture_voice.GAP_LANGUAGE_SWITCH_MS)
            + PCM_KO + gap(lecture_voice.GAP_AFTER_ELLIPSIS_KO_MS)
            + PCM_VI
        )
        self.assertEqual(pcm, expected)

    def test_a_phrase_said_twice_is_voiced_once(self):
        chirp = Chirp()
        with mock.patch.object(google_tts, "synthesize_pcm", chirp):
            tts.synthesize_lecture_chirp("Nghe nhé. 이거 얼마예요... Nghe lại nhé. 이거 얼마예요...")
        korean = [t for v, _r, t in chirp.calls if v.startswith("ko-")]
        self.assertEqual(korean, ["이거 얼마예요..."])

    def test_progress_is_reported_per_request(self):
        seen = []
        with mock.patch.object(google_tts, "synthesize_pcm", Chirp()):
            tts.synthesize_lecture_chirp("Chào. 안녕하세요. Tạm biệt.", lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_a_throttled_request_is_retried(self):
        calls = {"n": 0}

        def flaky(text, *, client=None, voice=None, rate=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise google_tts.GoogleTTSTransient("Google TTS HTTP 429")
            return PCM_VI, RATE

        with mock.patch.object(google_tts, "synthesize_pcm", flaky):
            tts.synthesize_lecture_chirp("Chào các bạn.")
        self.assertEqual(calls["n"], 2)

    def test_a_vietnamese_voice_google_does_not_have_is_replaced_once_for_the_whole_lecture(self):
        calls = []

        def synth(text, *, client=None, voice=None, rate=None):
            calls.append(voice)
            if voice == "vi-VN-Chirp3-HD-Iapetus":
                raise google_tts.GoogleTTSError("Google TTS HTTP 400 — voice does not exist")
            return (PCM_KO if voice.startswith("ko-") else PCM_VI), RATE

        with mock.patch.object(google_tts, "synthesize_pcm", synth), mock.patch.object(
            google_tts, "pick_vi_voice", return_value="vi-VN-Chirp3-HD-Charon"
        ) as pick:
            tts.synthesize_lecture_chirp("Chào các bạn. 이거 얼마예요... Nhớ nhé. 또 봐요... Hết rồi.")
        self.assertEqual(pick.call_count, 1)
        self.assertEqual(calls.count("vi-VN-Chirp3-HD-Iapetus"), 1)  # tried once, never again
        self.assertEqual(calls.count("vi-VN-Chirp3-HD-Charon"), 3)

    def test_when_there_is_no_vietnamese_chirp_voice_at_all_the_error_reaches_the_caller(self):
        def synth(text, *, client=None, voice=None, rate=None):
            raise google_tts.GoogleTTSError("Google TTS HTTP 400")

        with mock.patch.object(google_tts, "synthesize_pcm", synth), mock.patch.object(
            google_tts, "pick_vi_voice", return_value=None
        ):
            with self.assertRaises(RuntimeError):
                tts.synthesize_lecture_chirp("Chào các bạn.")

    def test_without_the_credential_it_refuses_instead_of_guessing(self):
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""):
            with self.assertRaises(RuntimeError):
                tts.synthesize_lecture_chirp("Chào các bạn.")

    def test_an_empty_script_is_an_error(self):
        with self.assertRaises(RuntimeError):
            tts.synthesize_lecture_chirp("  \n ... \n")


class VoicePodcastTests(unittest.TestCase):
    """tasks._voice_podcast: which engine, and where the recording is filed."""

    def db_with(self, existing=None):
        db = mock.MagicMock()
        db.execute.return_value.scalar_one_or_none.return_value = existing
        return db

    def test_without_a_fallback_key_it_is_gemini_only_as_before(self):
        with mock.patch.object(tts, "synthesize_korean_tts", return_value=(b"o", b"a", 9)) as gem, mock.patch.object(
            tts, "synthesize_lecture_chirp"
        ) as chirp:
            out = tasks._voice_podcast(self.db_with(), "script", "ko-female-1", "key", None, None)
        self.assertEqual(out, (None, b"o", b"a", 9, "key"))
        gem.assert_called_once()
        chirp.assert_not_called()

    def test_chirp_is_used_and_filed_under_the_spec_key(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=True), mock.patch.object(
            tts, "synthesize_lecture_chirp", return_value=(b"o", b"a", 9)
        ), mock.patch.object(tts, "synthesize_korean_tts") as gem:
            out = tasks._voice_podcast(self.db_with(), "script", "ko-female-1", "key:spec", "key", None)
        self.assertEqual(out, (None, b"o", b"a", 9, "key:spec"))
        gem.assert_not_called()

    def test_chirp_failure_falls_back_to_gemini_under_the_plain_key(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=True), mock.patch.object(
            tts, "synthesize_lecture_chirp", side_effect=RuntimeError("boom")
        ), mock.patch.object(tts, "synthesize_korean_tts", return_value=(b"g", b"g", 7)):
            out = tasks._voice_podcast(self.db_with(), "script", "ko-female-1", "key:spec", "key", None)
        self.assertEqual(out, (None, b"g", b"g", 7, "key"))

    def test_an_existing_fallback_recording_is_reused_not_regenerated(self):
        existing = SimpleNamespace(duration_sec=11)
        with mock.patch.object(tts, "chirp_enabled", return_value=True), mock.patch.object(
            tts, "synthesize_lecture_chirp", side_effect=RuntimeError("boom")
        ), mock.patch.object(tts, "synthesize_korean_tts") as gem:
            out = tasks._voice_podcast(self.db_with(existing), "script", "ko-female-1", "key:spec", "key", None)
        self.assertIs(out[0], existing)
        self.assertEqual(out[4], "key")
        gem.assert_not_called()

    def test_when_both_engines_fail_the_message_names_both(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=True), mock.patch.object(
            tts, "synthesize_lecture_chirp", side_effect=RuntimeError("chirp down")
        ), mock.patch.object(tts, "synthesize_korean_tts", side_effect=RuntimeError("quota")):
            with self.assertRaises(RuntimeError) as raised:
                tasks._voice_podcast(self.db_with(), "script", "ko-female-1", "key:spec", "key", None)
        self.assertIn("chirp down", str(raised.exception))
        self.assertIn("quota", str(raised.exception))

    def test_a_missing_credential_on_the_worker_goes_straight_to_the_fallback(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=False), mock.patch.object(
            tts, "synthesize_korean_tts", return_value=(b"g", b"g", 7)
        ):
            out = tasks._voice_podcast(self.db_with(), "script", "ko-female-1", "key:spec", "key", None)
        self.assertEqual(out[4], "key")


class PodcastRequestKeyTests(unittest.IsolatedAsyncioTestCase):
    def make_db(self, hit=None, job=None):
        db = mock.MagicMock()
        results = [mock.MagicMock(), mock.MagicMock()]
        results[0].scalar_one_or_none.return_value = hit
        results[1].scalar_one_or_none.return_value = job
        db.execute = mock.AsyncMock(side_effect=results)
        db.add = mock.MagicMock()
        db.commit = mock.AsyncMock()
        db.flush = mock.AsyncMock()
        db.delete = mock.AsyncMock()

        async def refresh(j):
            j.id = uuid.UUID(int=4)
            j.status = "queued"

        db.refresh = mock.AsyncMock(side_effect=refresh)
        return db

    async def request(self, db):
        with mock.patch.object(audio.generate_content_podcast, "delay") as delay:
            await audio._request_podcast(db, "lesson", "5", PodcastRequest(), SimpleNamespace(headers={}), [1], [2])
        return delay

    async def test_with_chirp_the_key_carries_the_voice_spec_and_the_plain_key_is_the_fallback(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=True):
            delay = await self.request(self.make_db())
        key, fallback = delay.call_args.args[5], delay.call_args.args[6]
        self.assertEqual(key, f"{fallback}:{tts.lecture_spec()}")
        self.assertNotIn("gc3bi", fallback)

    async def test_without_chirp_the_key_and_flow_are_unchanged(self):
        with mock.patch.object(tts, "chirp_enabled", return_value=False):
            delay = await self.request(self.make_db())
        self.assertIsNone(delay.call_args.args[6])
        self.assertNotIn("gc3bi", delay.call_args.args[5])

    async def test_a_succeeded_job_without_a_recording_under_the_wanted_key_runs_again(self):
        # Chirp failed last time: the job succeeded with the Gemini recording filed under the plain key
        stale = SimpleNamespace(status="succeeded")
        db = self.make_db(hit=None, job=stale)
        with mock.patch.object(tts, "chirp_enabled", return_value=True):
            delay = await self.request(db)
        db.delete.assert_awaited_once_with(stale)
        delay.assert_called_once()


if __name__ == "__main__":
    unittest.main()
