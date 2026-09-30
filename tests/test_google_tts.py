"""Google Chirp 3 HD: REST transport, credential handling (never in a URL or
an error message), WAV unwrapping, Chirp-first / Gemini-fallback orchestration,
the worker's cache-key bookkeeping and the route's key/stale-job rules — all
with fakes (no network, no Gemini, no Postgres)."""
import base64
import os
import struct
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import httpx

from app.api.routers import editorial
from app.core.config import Settings, settings
from app.schemas import ArticleAudioRequest
from app.services import google_tts, tts
from app.workers import tasks

# A deliberately distinctive fake credential: the tests assert it never shows
# up in a URL, an exception message or a log line.
SECRET = "unit-test-fake-credential-not-a-real-key"


def wav(pcm: bytes, rate: int = 24000, channels: int = 1, bits: int = 16, extra_chunk: bool = False) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
    if extra_chunk:
        body += b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\x00"  # odd size -> pad byte
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def ok_response(pcm: bytes = b"\x01\x00" * 2400, rate: int = 24000) -> httpx.Response:
    return httpx.Response(200, json={"audioContent": base64.b64encode(wav(pcm, rate)).decode()})


def err_response(status: int, message: str = "boom", code: str = "PERMISSION_DENIED") -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message, "status": code}})


class ChirpOn(unittest.TestCase):
    def setUp(self):
        for name, value in [
            ("TTS_GG_CHIRP", SECRET),
            ("TTS_CHIRP_VOICE", "ko-KR-Chirp3-HD-Iapetus"),
            ("TTS_CHIRP_SPEAKING_RATE", 0.85),
        ]:
            mock.patch.object(settings, name, value).start()
        mock.patch.object(tts.time, "sleep").start()
        google_tts._sa_credentials = None

    def tearDown(self):
        mock.patch.stopall()


class SettingsTests(unittest.TestCase):
    def test_env_name_is_case_insensitive(self):
        # Railway variable is spelled `TTS_GG_Chirp`
        with mock.patch.dict(os.environ, {"TTS_GG_Chirp": "abc123"}, clear=False):
            self.assertEqual(Settings(_env_file=None).TTS_GG_CHIRP, "abc123")

    def test_defaults_are_iapetus_at_085_and_key_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_GG_Chirp", None)
            os.environ.pop("TTS_GG_CHIRP", None)
            s = Settings(_env_file=None)
        self.assertEqual(s.TTS_GG_CHIRP, "")
        self.assertEqual(s.TTS_CHIRP_VOICE, "ko-KR-Chirp3-HD-Iapetus")
        self.assertEqual(s.TTS_CHIRP_SPEAKING_RATE, 0.85)


class WavTests(unittest.TestCase):
    def test_strips_header(self):
        pcm = b"\x01\x02" * 100
        self.assertEqual(google_tts.wav_to_pcm(wav(pcm)), (pcm, 24000))

    def test_skips_other_chunks_and_reads_rate(self):
        pcm = b"\x03\x04" * 50
        self.assertEqual(google_tts.wav_to_pcm(wav(pcm, rate=16000, extra_chunk=True)), (pcm, 16000))

    def test_data_size_zero_takes_the_rest(self):
        pcm = b"\x05\x06" * 10
        data = bytearray(wav(pcm))
        data[40:44] = struct.pack("<I", 0)
        self.assertEqual(google_tts.wav_to_pcm(bytes(data))[0], pcm)

    def test_raw_pcm_passthrough(self):
        self.assertEqual(google_tts.wav_to_pcm(b"\x01\x00" * 4), (b"\x01\x00" * 4, 24000))

    def test_stereo_rejected(self):
        with self.assertRaises(google_tts.GoogleTTSError):
            google_tts.wav_to_pcm(wav(b"\x00\x00" * 4, channels=2))


class SpecTests(ChirpOn):
    def test_spec_reflects_voice_and_rate(self):
        self.assertEqual(google_tts.spec(), "gc3-iapetus-r85")
        with mock.patch.object(settings, "TTS_CHIRP_VOICE", "ko-KR-Chirp3-HD-Erinome"), mock.patch.object(
            settings, "TTS_CHIRP_SPEAKING_RATE", 0.75
        ):
            self.assertEqual(google_tts.spec(), "gc3-erinome-r75")

    def test_rate_clamped_to_api_range(self):
        with mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE", 9):
            self.assertEqual(google_tts.speaking_rate(), 2.0)
        with mock.patch.object(settings, "TTS_CHIRP_SPEAKING_RATE", 0):
            self.assertEqual(google_tts.speaking_rate(), 0.25)

    def test_configured_flag_and_quote_stripping(self):
        self.assertTrue(google_tts.is_configured())
        with mock.patch.object(settings, "TTS_GG_CHIRP", '  "abc"  '):
            self.assertEqual(google_tts._credential(), "abc")
        with mock.patch.object(settings, "TTS_GG_CHIRP", "   "):
            self.assertFalse(google_tts.is_configured())


class RequestTests(ChirpOn):
    def test_request_shape_and_key_only_in_header(self):
        with mock.patch.object(google_tts.httpx, "post", return_value=ok_response()) as post:
            pcm, rate = google_tts.synthesize_pcm("안녕하세요.")
        self.assertEqual(rate, 24000)
        self.assertTrue(pcm)
        (url,), kw = post.call_args
        self.assertEqual(url, "https://texttospeech.googleapis.com/v1/text:synthesize")
        self.assertNotIn(SECRET, url)
        self.assertEqual(kw["headers"], {"X-Goog-Api-Key": SECRET})
        self.assertEqual(
            kw["json"],
            {
                "input": {"text": "안녕하세요."},
                "voice": {"languageCode": "ko-KR", "name": "ko-KR-Chirp3-HD-Iapetus"},
                "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 0.85},
            },
        )
        self.assertNotIn("params", kw)  # no ?key= query string

    def test_permission_error_is_not_retryable_and_hides_key(self):
        resp = err_response(403, "Cloud Text-to-Speech API has not been used in project 123", "PERMISSION_DENIED")
        with mock.patch.object(google_tts.httpx, "post", return_value=resp):
            with self.assertRaises(google_tts.GoogleTTSError) as cm:
                google_tts.synthesize_pcm("x")
        self.assertNotIsInstance(cm.exception, google_tts.GoogleTTSTransient)
        self.assertIn("403", str(cm.exception))
        self.assertIn("has not been used", str(cm.exception))
        self.assertNotIn(SECRET, str(cm.exception))

    def test_rate_limit_and_5xx_are_transient(self):
        for status in (429, 500, 503):
            with mock.patch.object(google_tts.httpx, "post", return_value=err_response(status, "busy", "UNAVAILABLE")):
                with self.assertRaises(google_tts.GoogleTTSTransient):
                    google_tts.synthesize_pcm("x")

    def test_timeout_and_network_errors_are_transient_and_hide_key(self):
        for exc in (httpx.ReadTimeout("t"), httpx.ConnectError("c")):
            with mock.patch.object(google_tts.httpx, "post", side_effect=exc):
                with self.assertRaises(google_tts.GoogleTTSTransient) as cm:
                    google_tts.synthesize_pcm("x")
            self.assertNotIn(SECRET, str(cm.exception))

    def test_bad_voice_name_never_hits_the_network(self):
        with mock.patch.object(settings, "TTS_CHIRP_VOICE", "Iapetus"), mock.patch.object(
            google_tts.httpx, "post"
        ) as post:
            with self.assertRaises(google_tts.GoogleTTSError):
                google_tts.synthesize_pcm("x")
        post.assert_not_called()

    def test_oversized_text_rejected_locally(self):
        with mock.patch.object(google_tts.httpx, "post") as post:
            with self.assertRaises(google_tts.GoogleTTSError):
                google_tts.synthesize_pcm("가" * 1700)  # 5,100 bytes
        post.assert_not_called()

    def test_empty_audio_is_transient(self):
        with mock.patch.object(google_tts.httpx, "post", return_value=httpx.Response(200, json={"audioContent": ""})):
            with self.assertRaises(google_tts.GoogleTTSTransient):
                google_tts.synthesize_pcm("x")

    def test_service_account_json_uses_bearer_token(self):
        creds = mock.MagicMock(valid=True, token="tok123")
        sa_json = '{"type": "service_account", "client_email": "a@b.c", "private_key": "k", "token_uri": "t"}'
        with mock.patch.object(settings, "TTS_GG_CHIRP", sa_json), mock.patch(
            "google.oauth2.service_account.Credentials.from_service_account_info", return_value=creds
        ), mock.patch.object(google_tts.httpx, "post", return_value=ok_response()) as post:
            google_tts.synthesize_pcm("x")
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer tok123"})


class ChunkSizeTests(unittest.TestCase):
    def test_worst_case_chunk_fits_the_5000_byte_limit(self):
        self.assertLessEqual(google_tts.CHUNK_CHARS * 4, google_tts.MAX_REQUEST_BYTES)


def para(n: int) -> str:
    return ("가" * (n - 1)) + "."


ONE_SEC = b"\x01\x00" * 24000


class ArticleChirpTests(ChirpOn):
    def setUp(self):
        super().setUp()
        mock.patch.object(tts, "transcode_pcm", side_effect=lambda pcm, rate: (pcm, pcm, len(pcm) // (rate * 2))).start()

    def test_chunks_fit_request_limit_pauses_and_progress(self):
        seen: list[str] = []

        def fake(text, **_):
            seen.append(text)
            return ONE_SEC, 24000

        events = []
        text = "\n".join([para(800), para(800), para(800)])
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=fake):
            opus, _aac, dur = tts.synthesize_article_chirp(text, on_progress=lambda d, t: events.append((d, t)))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(len(c) <= google_tts.CHUNK_CHARS for c in seen))
        self.assertTrue(all(len(c.encode()) <= google_tts.MAX_REQUEST_BYTES for c in seen))
        self.assertEqual(events, [(0, 3), (1, 3), (2, 3), (3, 3)])
        # 3 x 1s speech + 2 x 0.7s paragraph pauses, none after the last chunk
        self.assertEqual(len(opus), (3 * 24000 + 2 * int(24000 * 0.7)) * 2)
        self.assertEqual(dur, 4)

    def test_transient_retried_then_succeeds(self):
        calls = {"n": 0}

        def flaky(text, **_):
            calls["n"] += 1
            if calls["n"] < 3:
                raise google_tts.GoogleTTSTransient("429")
            return ONE_SEC, 24000

        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=flaky):
            tts.synthesize_article_chirp(para(100))
        self.assertEqual(calls["n"], 3)

    def test_hard_error_not_retried(self):
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=google_tts.GoogleTTSError("403")) as syn:
            with self.assertRaises(RuntimeError):
                tts.synthesize_article_chirp(para(100))
        self.assertEqual(syn.call_count, 1)

    def test_mixed_sample_rates_refused(self):
        rates = iter([24000, 16000])
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=lambda *_a, **_k: (ONE_SEC, next(rates))):
            with self.assertRaises(RuntimeError):
                tts.synthesize_article_chirp("\n".join([para(800), para(800)]))

    def test_long_article_not_cut_at_geminis_8000_chars(self):
        text = "\n".join(para(900) for _ in range(12))  # 10.8k chars
        with mock.patch.object(google_tts, "synthesize_pcm", return_value=(ONE_SEC, 24000)) as syn:
            tts.synthesize_article_chirp(text)
        voiced = sum(len(c.args[0]) for c in syn.call_args_list)
        self.assertGreater(voiced, tts._ARTICLE_MAX_CHARS)


class KoreanChirpFirstTests(ChirpOn):
    def setUp(self):
        super().setUp()
        mock.patch.object(tts, "transcode_pcm", side_effect=lambda pcm, rate: (b"opus", b"aac", 1)).start()

    def test_off_means_gemini_only(self):
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""), mock.patch.object(
            google_tts, "synthesize_pcm"
        ) as chirp, mock.patch.object(tts, "synthesize_korean_tts", return_value=("o", "a", 2)) as gem:
            self.assertEqual(tts.synthesize_korean_tts_chirp_first("안녕"), ("o", "a", 2))
        chirp.assert_not_called()
        gem.assert_called_once()

    def test_on_uses_chirp_and_skips_gemini(self):
        with mock.patch.object(google_tts, "synthesize_pcm", return_value=(ONE_SEC, 24000)), mock.patch.object(
            tts, "synthesize_korean_tts"
        ) as gem:
            self.assertEqual(tts.synthesize_korean_tts_chirp_first("안녕하세요."), (b"opus", b"aac", 1))
        gem.assert_not_called()

    def test_chirp_failure_falls_back_to_gemini(self):
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=google_tts.GoogleTTSError("403")), mock.patch.object(
            tts, "synthesize_korean_tts", return_value=("o", "a", 2)
        ) as gem:
            self.assertEqual(tts.synthesize_korean_tts_chirp_first("안녕"), ("o", "a", 2))
        gem.assert_called_once()

    def test_unexpected_client_error_also_falls_back(self):
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=ValueError("bug")), mock.patch.object(
            tts, "synthesize_korean_tts", return_value=("o", "a", 2)
        ):
            self.assertEqual(tts.synthesize_korean_tts_chirp_first("안녕"), ("o", "a", 2))

    def test_both_fail_reports_both_reasons(self):
        with mock.patch.object(google_tts, "synthesize_pcm", side_effect=google_tts.GoogleTTSError("HTTP 403")), mock.patch.object(
            tts, "synthesize_korean_tts", side_effect=RuntimeError("het quota")
        ):
            with self.assertRaises(RuntimeError) as cm:
                tts.synthesize_korean_tts_chirp_first("안녕")
        self.assertIn("HTTP 403", str(cm.exception))
        self.assertIn("het quota", str(cm.exception))
        self.assertNotIn(SECRET, str(cm.exception))


class WorkerBookkeepingTests(ChirpOn):
    """tasks._voice_article: which recording is stored under which key."""

    def db_with(self, fallback_row=None, job_result=None):
        db = mock.MagicMock()

        def execute(stmt):
            res = mock.MagicMock()
            is_job_lookup = "SELECT jobs.result" in str(stmt)
            res.scalar_one_or_none.return_value = job_result if is_job_lookup else fallback_row
            return res

        db.execute.side_effect = execute
        return db

    def test_legacy_gemini_only_when_api_had_chirp_off(self):
        with mock.patch.object(tts, "synthesize_article_tts", return_value=("o", "a", 5)) as gem, mock.patch.object(
            tts, "synthesize_article_chirp"
        ) as chirp:
            out = tasks._voice_article(self.db_with(), "t", "v", "KEY", None, None)
        self.assertEqual(out, (None, "o", "a", 5, "KEY", None))
        chirp.assert_not_called()
        gem.assert_called_once()

    def test_chirp_success_stored_under_primary_key(self):
        with mock.patch.object(tts, "synthesize_article_chirp", return_value=("o", "a", 5)), mock.patch.object(
            tts, "synthesize_article_tts"
        ) as gem:
            out = tasks._voice_article(self.db_with(), "t", "v", "KEY:gc3", "KEY", None)
        self.assertEqual(out, (None, "o", "a", 5, "KEY:gc3", None))
        gem.assert_not_called()

    def test_chirp_failure_redoes_whole_article_with_gemini_under_fallback_key(self):
        with mock.patch.object(tts, "synthesize_article_chirp", side_effect=google_tts.GoogleTTSError("403")), mock.patch.object(
            tts, "synthesize_article_tts", return_value=("go", "ga", 7)
        ) as gem:
            out = tasks._voice_article(self.db_with(), "t", "v", "KEY:gc3", "KEY", None)
        self.assertEqual(out, (None, "go", "ga", 7, "KEY", None))  # NOT the chirp key
        gem.assert_called_once()

    def test_existing_fallback_recording_reused_without_spending_gemini(self):
        row = SimpleNamespace(duration_sec=99)
        with mock.patch.object(tts, "synthesize_article_chirp", side_effect=google_tts.GoogleTTSError("403")), mock.patch.object(
            tts, "synthesize_article_tts"
        ) as gem:
            reuse, _o, _a, dur, key, _t = tasks._voice_article(self.db_with(row), "t", "v", "KEY:gc3", "KEY", None)
        self.assertIs(reuse, row)
        self.assertEqual((dur, key), (99, "KEY"))
        gem.assert_not_called()

    def test_reused_fallback_recording_keeps_its_read_along_timeline(self):
        row = SimpleNamespace(duration_sec=99)
        timeline = {"v": 1, "items": [[0, 0, 0], [0, 1, 4200]]}
        db = self.db_with(row, job_result={"cache_key": "KEY", "timings": timeline})
        with mock.patch.object(tts, "synthesize_article_chirp", side_effect=google_tts.GoogleTTSError("403")), mock.patch.object(
            tts, "synthesize_article_tts"
        ) as gem:
            out = tasks._voice_article(db, "t", "v", "KEY:gc3", "KEY", None)
        self.assertEqual(out[5], timeline)
        gem.assert_not_called()

    def test_units_switch_to_the_timed_voices(self):
        units = [[0, 0, "안녕하세요."], [0, 1, "반갑습니다."]]
        timeline = {"v": 1, "items": [[0, 0, 0], [0, 1, 1500]]}
        with mock.patch.object(
            tts, "synthesize_article_chirp_timed", return_value=("o", "a", 5, timeline)
        ) as chirp, mock.patch.object(tts, "synthesize_article_chirp") as plain:
            out = tasks._voice_article(self.db_with(), "t", "v", "KEY:gc3", "KEY", None, units)
        self.assertEqual(out, (None, "o", "a", 5, "KEY:gc3", timeline))
        self.assertEqual([tuple(u) for u in chirp.call_args.args[0]], [(0, 0, "안녕하세요."), (0, 1, "반갑습니다.")])
        plain.assert_not_called()

    def test_worker_without_the_variable_falls_back_and_says_why(self):
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""), mock.patch.object(
            tts, "synthesize_article_chirp"
        ) as chirp, mock.patch.object(tts, "synthesize_article_tts", return_value=("go", "ga", 7)):
            out = tasks._voice_article(self.db_with(), "t", "v", "KEY:gc3", "KEY", None)
        chirp.assert_not_called()
        self.assertEqual(out[4], "KEY")

    def test_both_fail_message_names_both(self):
        with mock.patch.object(tts, "synthesize_article_chirp", side_effect=google_tts.GoogleTTSError("HTTP 403")), mock.patch.object(
            tts, "synthesize_article_tts", side_effect=RuntimeError("het quota")
        ):
            with self.assertRaises(RuntimeError) as cm:
                tasks._voice_article(self.db_with(), "t", "v", "KEY:gc3", "KEY", None)
        self.assertIn("HTTP 403", str(cm.exception))
        self.assertIn("het quota", str(cm.exception))


# ---- route-level: cache key + stale succeeded job --------------------------
P1 = "미 국채시장이 불안하다. 10년물 금리가 5%를 넘나들고 있다."


def article():
    return SimpleNamespace(
        id=uuid.uuid4(), source_name="경향신문", source_url="https://x/1", title_ko="[정동칼럼]미 국채시장",
        level_estimate=5, topic_tags=[], body_ko=P1, vocab_ids=[], grammar_ids=[], thinking_guide_text=None,
        created_at=datetime.now(timezone.utc), images=[], images_fetched_at=None, study_pack=None,
        study_status="none", study_updated_at=None,
    )


def fake_db(art, *, existing_job=None, cached_audio=None):
    db = mock.MagicMock()
    db.get = mock.AsyncMock(return_value=art)
    db.add = mock.MagicMock(side_effect=lambda o: setattr(o, "id", uuid.uuid4()) if getattr(o, "id", "x") is None else None)
    db.commit = mock.AsyncMock()
    db.refresh = mock.AsyncMock()
    db.delete = mock.AsyncMock()
    db.flush = mock.AsyncMock()

    async def execute(stmt):
        text = str(stmt)
        res = mock.MagicMock()
        if "lecture_audio" in text:
            res.scalar_one_or_none.return_value = cached_audio
        elif "jobs" in text:
            res.scalar_one_or_none.return_value = existing_job
        return res

    db.execute = execute
    return db


class RouteKeyTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(headers={})

    async def test_chirp_off_keeps_the_old_key_and_no_fallback(self):
        art = article()
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), fake_db(art))
        args = delay.call_args.args
        self.assertNotIn("gc3-", args[4])
        self.assertIsNone(args[5])

    async def test_chirp_on_key_carries_the_spec_and_fallback_is_the_plain_key(self):
        art = article()
        with mock.patch.object(settings, "TTS_GG_CHIRP", SECRET), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), fake_db(art))
        args = delay.call_args.args
        self.assertTrue(args[4].endswith(":gc3-iapetus-r85"), args[4])
        self.assertEqual(args[5], args[4][: -len(":gc3-iapetus-r85")])
        self.assertNotIn(SECRET, "".join(str(a) for a in args))  # the credential never travels in task args

    async def test_easy_variant_spec_goes_after_easy_marker(self):
        art = article()
        pack = SimpleNamespace()
        with mock.patch.object(settings, "TTS_GG_CHIRP", SECRET), mock.patch.object(
            editorial, "_valid_pack", return_value=pack
        ), mock.patch.object(
            editorial.read_along, "easy_units", return_value=[editorial.read_along.Unit(0, 0, "쉬운 글이다.")]
        ), mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            await editorial.request_article_audio(
                art.id, ArticleAudioRequest(variant="easy"), self.request(), fake_db(art)
            )
        args = delay.call_args.args
        self.assertTrue(args[4].endswith(":easy:gc3-iapetus-r85"), args[4])
        self.assertTrue(args[5].endswith(":easy"), args[5])

    async def test_succeeded_job_without_a_recording_is_rerun(self):
        # the previous run fell back to Gemini (filed under the plain key), so
        # nothing sits under the Chirp key: run again to give Chirp another go
        art = article()
        done = SimpleNamespace(id=uuid.uuid4(), status="succeeded", updated_at=datetime.now(timezone.utc))
        db = fake_db(art, existing_job=done, cached_audio=None)
        with mock.patch.object(settings, "TTS_GG_CHIRP", SECRET), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        db.delete.assert_awaited_once_with(done)
        delay.assert_called_once()

    async def test_succeeded_job_with_recording_is_reused(self):
        art = article()
        hit = SimpleNamespace(cache_key="k", opus_path="/o", aac_path="/a", duration_sec=42)
        done = SimpleNamespace(id=uuid.uuid4(), status="succeeded", updated_at=datetime.now(timezone.utc), progress=1.0)
        db = fake_db(art, existing_job=done, cached_audio=hit)
        with mock.patch.object(settings, "TTS_GG_CHIRP", SECRET), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            out = await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        delay.assert_not_called()
        db.delete.assert_not_awaited()
        self.assertEqual(out.job_id, done.id)


if __name__ == "__main__":
    unittest.main()
