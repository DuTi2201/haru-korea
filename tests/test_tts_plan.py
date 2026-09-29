"""No network / no Gemini: chunk planning, silence insertion, retry and
progress of the article read-aloud path (app.services.tts)."""
import unittest
from unittest import mock

from app.core.config import settings
from app.services import tts


def para(n: int, ch: str = "가") -> str:
    return (ch * (n - 1)) + "."


class PlanTests(unittest.TestCase):
    def test_paragraph_aligned_and_pauses(self):
        text = "\n".join([para(500), para(500), para(500), para(200)])
        plan = tts.plan_article_chunks(text, max_chars=1200)
        # 500+500 fit (1002 incl. blank line); the third starts a new chunk
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0][0].count("\n\n"), 1)
        self.assertEqual(plan[0][1], tts._PAUSE_AFTER_PARAGRAPH_MS)
        self.assertEqual(plan[-1][1], 0)  # no trailing silence
        # nothing lost, nothing reordered
        joined = "\n\n".join(c for c, _ in plan).replace("\n\n", "\n")
        self.assertEqual(joined, text)

    def test_oversized_paragraph_split_at_sentences_with_short_pause(self):
        sentence = "이것은 문장이다."
        big = " ".join([sentence] * 40)  # ~ 400 chars
        plan = tts.plan_article_chunks("\n".join([big, para(50)]), max_chars=150)
        self.assertGreater(len(plan), 2)
        self.assertTrue(all(len(c) <= 150 for c, _ in plan))
        self.assertIn(tts._PAUSE_AFTER_SENTENCE_MS, [p for _, p in plan])

    def test_length_cap(self):
        text = "\n".join(para(900) for _ in range(20))
        plan = tts.plan_article_chunks(text)
        self.assertLessEqual(sum(len(c) for c, _ in plan), tts._ARTICLE_MAX_CHARS + 900)

    def test_empty(self):
        self.assertEqual(tts.plan_article_chunks("  \n \n"), [])


class SynthTests(unittest.TestCase):
    def setUp(self):
        self.key = mock.patch.object(settings, "GEMINI_API_KEY", "test-key")
        self.key.start()
        self.client = mock.patch.object(tts, "_get_client", return_value=object())
        self.client.start()
        self.sleep = mock.patch.object(tts.time, "sleep")
        self.sleep.start()
        self.enc = mock.patch.object(tts, "transcode_pcm", side_effect=lambda pcm, rate: (pcm, pcm, len(pcm) // (rate * 2)))
        self.enc.start()

    def tearDown(self):
        mock.patch.stopall()

    def test_silence_between_chunks_progress_and_duration(self):
        one_sec = b"\x01\x00" * 24000
        with mock.patch.object(tts, "_synthesize_chunk_pcm", return_value=(one_sec, 24000)) as syn:
            events = []
            text = "\n".join([para(900), para(900), para(900)])
            opus, _aac, dur = tts.synthesize_article_tts(text, on_progress=lambda d, t: events.append((d, t)))
        self.assertEqual(syn.call_count, 3)
        self.assertEqual(events, [(0, 3), (1, 3), (2, 3), (3, 3)])
        # 3 x 1s speech + 2 x 0.7s paragraph pauses, none after the last chunk
        self.assertEqual(len(opus), (3 * 24000 + 2 * int(24000 * 0.7)) * 2)
        self.assertEqual(dur, 4)

    def test_transient_error_retried_then_succeeds(self):
        good = (b"\x01\x00" * 2400, 24000)
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise tts.TransientTTSError("500")
            return good

        with mock.patch.object(tts, "_synthesize_chunk_pcm", side_effect=flaky):
            tts.synthesize_article_tts(para(100))
        self.assertEqual(calls["n"], 3)

    def test_non_transient_error_not_retried(self):
        with mock.patch.object(tts, "_synthesize_chunk_pcm", side_effect=RuntimeError("quota")) as syn:
            with self.assertRaises(RuntimeError):
                tts.synthesize_article_tts(para(100))
        self.assertEqual(syn.call_count, 1)

    def test_transient_gives_up_after_retries(self):
        with mock.patch.object(tts, "_synthesize_chunk_pcm", side_effect=tts.TransientTTSError("500")) as syn:
            with self.assertRaises(tts.TransientTTSError):
                tts.synthesize_article_tts(para(100))
        self.assertEqual(syn.call_count, 1 + len(tts._CHUNK_RETRY_DELAYS_SEC))


if __name__ == "__main__":
    unittest.main()
