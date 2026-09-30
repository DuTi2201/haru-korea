"""Read-along: sentence units line up with what the reader shows, chunks are
planned without losing or reordering anything, and the timeline is anchored on
exact clip lengths. A fake voice (length proportional to the text) stands in for
Chirp/Gemini — no network, no ffmpeg, no Postgres."""
import re
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from app.api.routers import editorial
from app.core.config import settings
from app.schemas import ArticleAudioRequest
from app.services import read_along as ra
from app.services import tts
from app.services.article_extract import article_tts_text, clean_article_text
from app.services.study_pack import split_sentences

P1 = "미 국채시장이 불안하다. 10년물 금리가 5.3%를 넘나들고 있다. 이란전쟁과 인플레 우려가 직접적 계기지만, 문제의 뿌리는 깊다."
P2 = "이 상황에서 떠올리게 되는 것이 스티븐 마이런의 보고서다. 관세 정책을 이해하는 이론적 배경이 되어왔다."
P3 = "자세한 내용은 https://example.com/report 에서 볼 수 있다."
TITLE = "[정동칼럼]미 국채시장"
FAKE_CREDENTIAL = "unit-test-not-a-real-credential"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def units_from(*paragraphs: str) -> list[ra.Unit]:
    return ra.original_units(None, "\n".join(paragraphs))


class UnitTests(unittest.TestCase):
    def test_indices_follow_the_study_pack_sentence_split(self):
        body = "\n".join([P1, P2])
        units = ra.original_units(TITLE, body)
        self.assertEqual(units[0], ra.Unit(ra.TITLE_PARAGRAPH, 0, "미 국채시장."))
        expected = [(p, s, sent) for p, para in enumerate([P1, P2]) for s, sent in enumerate(split_sentences(para))]
        self.assertEqual([(u.p, u.s, u.text) for u in units[1:]], expected)

    def test_voiced_text_is_unchanged_from_article_tts_text(self):
        body = clean_article_text("\n".join([P1, P2, P3]))
        units = ra.original_units(TITLE, body)
        self.assertEqual(norm(ra.units_to_text(units)), norm(article_tts_text(TITLE, body)))

    def test_url_only_sentence_is_skipped_but_keeps_its_number(self):
        units = ra.original_units(None, "첫 문장이다. https://example.com/a. 마지막 문장이다.")
        self.assertEqual([(u.p, u.s) for u in units], [(0, 0), (0, 2)])
        self.assertNotIn("https", " ".join(u.text for u in units))

    def test_paragraph_made_only_of_a_url_leaves_a_gap_in_numbering(self):
        units = ra.original_units(None, "\n".join([P1, "https://example.com/only", P2]))
        self.assertEqual(sorted({u.p for u in units}), [0, 2])

    def test_easy_units_use_pack_paragraph_numbers_and_fall_back_to_sentences(self):
        pack = {
            "paragraphs": [
                {"easy_ko": "쉬운 글이다.", "sentences": [{"ko": "어려운 글이다."}]},
                {"easy_ko": "", "sentences": [{"ko": "첫째 문장이다."}, {"ko": "둘째 문장이다."}]},
                {"easy_ko": "", "sentences": []},
                {"easy_ko": "마지막 글이다.", "sentences": []},
            ]
        }
        units = ra.easy_units(TITLE, pack)
        self.assertEqual(
            [(u.p, u.s, u.text) for u in units],
            [
                (ra.TITLE_PARAGRAPH, 0, "미 국채시장."),
                (0, 0, "쉬운 글이다."),
                (1, 0, "첫째 문장이다. 둘째 문장이다."),
                (3, 0, "마지막 글이다."),
            ],
        )

    def test_units_to_text_one_paragraph_per_line(self):
        units = [ra.Unit(-1, 0, "제목."), ra.Unit(0, 0, "가."), ra.Unit(0, 1, "나."), ra.Unit(1, 0, "다.")]
        self.assertEqual(ra.units_to_text(units), "제목.\n가. 나.\n다.")


def plan(units, **kw):
    params = dict(
        pack_to=240, split_over=900, max_total=12000, sentence_gap_ms=150, paragraph_gap_ms=700, cross_paragraph=False
    )
    params.update(kw)
    return ra.plan_chunks(units, **params)


def covered(chunks, units):
    """unit indices in the order the chunks voice them (repeats collapsed)"""
    seen: list[int] = []
    for c in chunks:
        for pc in c.pieces:
            if not seen or seen[-1] != pc.unit:
                seen.append(pc.unit)
    return seen


def long_sentence(n: int) -> str:
    return "가" * (n - 1) + "."


class PlanTests(unittest.TestCase):
    def test_small_chunks_never_cross_a_paragraph_and_cover_every_unit_in_order(self):
        units = units_from(P1, P2, P3)
        chunks = plan(units)
        self.assertEqual(covered(chunks, units), list(range(len(units))))
        for c in chunks:
            self.assertEqual(len({pc.para for pc in c.pieces}), 1)
            self.assertLessEqual(len(c.text), max(240, max(len(pc.text) for pc in c.pieces)))

    def test_pauses_sentence_gap_inside_paragraph_paragraph_gap_at_its_end_none_at_the_end(self):
        body = "\n".join([" ".join([long_sentence(200)] * 3), long_sentence(50)])
        chunks = plan(ra.original_units(None, body))  # 200+200 > 240: one sentence per chunk
        self.assertEqual(len(chunks), 4)
        self.assertEqual([c.pause_after_ms for c in chunks], [150, 150, 700, 0])

        body = "\n".join([long_sentence(100) + " " + long_sentence(100), long_sentence(50)])
        self.assertEqual([c.pause_after_ms for c in plan(ra.original_units(None, body))], [700, 0])

    def test_a_sentence_above_the_pack_size_stays_whole(self):
        units = units_from(long_sentence(500))
        chunks = plan(units)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].text), 500)

    def test_a_sentence_above_the_request_cap_is_cut_but_still_one_unit(self):
        words = " ".join(["단어"] * 700) + "."  # ~2100 chars, no sentence end until the last char
        units = units_from(words)
        chunks = plan(units)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 900 for c in chunks))
        self.assertEqual({pc.unit for c in chunks for pc in c.pieces}, {0})

    def test_length_cap_drops_the_tail(self):
        paras = [long_sentence(200) for _ in range(20)]
        units = units_from(*paras)
        chunks = plan(units, max_total=1000)
        voiced = covered(chunks, units)
        self.assertEqual(voiced, list(range(len(voiced))))
        self.assertLess(len(voiced), len(units))
        self.assertGreaterEqual(len(voiced), 1)

    def test_cross_paragraph_mode_packs_whole_paragraphs_like_the_old_gemini_plan(self):
        paras = [long_sentence(500), long_sentence(500), long_sentence(500), long_sentence(200)]
        units = units_from(*paras)
        chunks = plan(units, pack_to=1200, split_over=1200, cross_paragraph=True)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text.count("\n\n"), 1)
        self.assertEqual([c.pause_after_ms for c in chunks], [700, 0])
        self.assertEqual(covered(chunks, units), [0, 1, 2, 3])

    def test_empty(self):
        self.assertEqual(plan([]), [])


def fake_pcm(text: str, rate: int = 24000, ms_per_weight: float = 200.0) -> tuple[bytes, int]:
    """Speech whose length is exactly proportional to the planner's own weight
    of the text — the best case for the within-chunk estimate."""
    samples = int(rate * ra._weight(text.replace("\n", " ")) * ms_per_weight / 1000)
    return b"\x01\x00" * samples, rate


class TimelineTests(unittest.TestCase):
    def run_timeline(self, units, **plan_kw):
        chunks = plan(units, **plan_kw)
        pcm, rate, starts, speech = tts._synthesize_timed(chunks, fake_pcm, None)
        return chunks, pcm, rate, starts, speech, ra.build_timeline(units, chunks, starts, speech)

    def test_chunk_starts_are_exact_and_include_the_pauses(self):
        units = units_from(P1, P2)
        chunks, pcm, rate, starts, speech, _ = self.run_timeline(units)
        self.assertEqual(starts[0], 0)
        for i in range(1, len(chunks)):
            expected = starts[i - 1] + speech[i - 1] + chunks[i - 1].pause_after_ms
            self.assertAlmostEqual(starts[i], expected, places=3)
        total_ms = len(pcm) / 2 / rate * 1000
        self.assertAlmostEqual(total_ms, starts[-1] + speech[-1], places=3)  # no trailing silence

    def test_every_unit_gets_a_start_in_order_inside_the_audio(self):
        units = ra.original_units(TITLE, "\n".join([P1, P2, P3]))
        timeline = self.run_timeline(units)[5]
        items = timeline["items"]
        self.assertEqual(timeline["v"], 1)
        self.assertEqual([(p, s) for p, s, _ in items], [(u.p, u.s) for u in units])
        starts = [t for _, _, t in items]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0)

    def test_sentences_in_one_chunk_split_it_in_proportion_to_their_weight(self):
        a, b = long_sentence(50), long_sentence(150)  # 1 : 3 by weight
        units = ra.original_units(None, a + " " + b)
        chunks, _, _, starts, speech, timeline = self.run_timeline(units)
        self.assertEqual(len(chunks), 1)
        (_, _, t0), (_, _, t1) = timeline["items"]
        self.assertEqual(t0, 0)
        self.assertAlmostEqual(t1 / speech[0], 50 / 200, delta=0.01)

    def test_estimate_error_is_bounded_by_the_chunk_even_when_the_voice_is_uneven(self):
        # The voice reads the second sentence 30% slower than the estimate.
        a, b = long_sentence(100), long_sentence(100)
        units = ra.original_units(None, a + " " + b)
        chunks = plan(units)

        def uneven(text: str):
            pcm, rate = fake_pcm(text)
            return pcm + b"\x01\x00" * int(len(pcm) / 2 * 0.15), rate  # +15% overall

        _, _, starts, speech = tts._synthesize_timed(chunks, uneven, None)
        timeline = ra.build_timeline(units, chunks, starts, speech)
        true_second_start = speech[0] * 0.5  # if both took the same time
        est = timeline["items"][1][2]
        self.assertLess(abs(est - true_second_start), 0.2 * speech[0])  # well inside the chunk

    def test_a_unit_cut_over_several_requests_starts_at_its_first_piece(self):
        units = units_from(" ".join(["단어"] * 700) + ".")
        chunks, _, _, starts, speech, timeline = self.run_timeline(units)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(timeline["items"], [[0, 0, 0]])

    def test_unvoiced_tail_units_have_no_entry(self):
        units = units_from(*[long_sentence(200) for _ in range(20)])
        chunks = plan(units, max_total=1000)
        _pcm, _rate, starts, speech = tts._synthesize_timed(chunks, fake_pcm, None)
        timeline = ra.build_timeline(units, chunks, starts, speech)
        self.assertLess(len(timeline["items"]), len(units))

    def test_mixed_sample_rates_are_an_error(self):
        chunks = plan(units_from(long_sentence(200), long_sentence(200)))
        rates = iter([24000, 16000])
        with self.assertRaises(RuntimeError):
            tts._synthesize_timed(chunks, lambda t: (b"\x00\x00" * 100, next(rates)), None)


class TimedVoiceTests(unittest.TestCase):
    def test_chirp_timed_reports_progress_and_returns_the_timeline(self):
        units = ra.original_units(TITLE, "\n".join([P1, P2]))
        progress: list[tuple[int, int]] = []
        with mock.patch.object(settings, "TTS_GG_CHIRP", FAKE_CREDENTIAL), mock.patch.object(
            tts, "_chirp_chunk_pcm", side_effect=fake_pcm
        ), mock.patch.object(tts, "transcode_pcm", return_value=(b"opus", b"aac", 42)):
            opus, aac, dur, timings = tts.synthesize_article_chirp_timed(units, lambda d, t: progress.append((d, t)))
        self.assertEqual((opus, aac, dur), (b"opus", b"aac", 42))
        self.assertEqual(progress[0][0], 0)
        self.assertEqual(progress[-1][0], progress[-1][1])
        self.assertEqual([(p, s) for p, s, _ in timings["items"]], [(u.p, u.s) for u in units])

    def test_chirp_timed_needs_the_credential(self):
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""):
            with self.assertRaises(RuntimeError):
                tts.synthesize_article_chirp_timed(units_from(P1))

    def test_a_failed_request_propagates_so_the_caller_can_fall_back(self):
        units = units_from(P1, P2)
        with mock.patch.object(settings, "TTS_GG_CHIRP", FAKE_CREDENTIAL), mock.patch.object(
            tts, "_chirp_chunk_pcm", side_effect=RuntimeError("HTTP 403")
        ):
            with self.assertRaises(RuntimeError):
                tts.synthesize_article_chirp_timed(units)


# ---- route: the unit list goes to the worker and cached timings come back ----
def article(body: str = "\n".join([P1, P2])):
    return SimpleNamespace(
        id=uuid.uuid4(), source_name="경향신문", source_url="https://x/1", title_ko=TITLE,
        level_estimate=5, topic_tags=[], body_ko=body, vocab_ids=[], grammar_ids=[], thinking_guide_text=None,
        created_at=datetime.now(timezone.utc), images=[], images_fetched_at=None, study_pack=None,
        study_status="none", study_updated_at=None,
    )


def fake_db(art, *, cached_audio=None, earlier_job_result=None):
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
        elif "SELECT jobs.result" in text:
            res.scalar_one_or_none.return_value = earlier_job_result
        elif "jobs" in text:
            res.scalar_one_or_none.return_value = None
        return res

    db.execute = execute
    return db


class RouteTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(headers={})

    async def test_worker_gets_the_sentence_units(self):
        art = article()
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), fake_db(art))
        units = delay.call_args.kwargs["units"]
        self.assertEqual(units[0], [-1, 0, "미 국채시장."])
        self.assertEqual([(p, s) for p, s, _ in units[1:]], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
        self.assertEqual(delay.call_args.args[1], ra.units_to_text([ra.Unit(*u) for u in units]))

    async def test_the_key_changes_with_the_read_along_version(self):
        art = article()
        keys = []
        for version in ("ra1", "ra2"):
            with mock.patch.object(ra, "READ_ALONG_VERSION", version), mock.patch.object(
                editorial.generate_article_audio, "delay"
            ) as delay:
                await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), fake_db(art))
            keys.append(delay.call_args.args[4])
        self.assertNotEqual(keys[0], keys[1])

    async def test_cache_hit_returns_the_timeline_of_the_job_that_made_it(self):
        art = article()
        timeline = {"v": 1, "items": [[-1, 0, 0], [0, 0, 1800]]}
        hit = SimpleNamespace(cache_key="K", opus_path="/o", aac_path="/a", duration_sec=30)
        db = fake_db(art, cached_audio=hit, earlier_job_result={"cache_key": "K", "timings": timeline})
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""), mock.patch.object(
            editorial.generate_article_audio, "delay"
        ) as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        delay.assert_not_called()
        job = db.add.call_args.args[0]
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.result["timings"], timeline)

    async def test_cache_hit_without_a_timeline_just_has_none(self):
        art = article()
        hit = SimpleNamespace(cache_key="K", opus_path="/o", aac_path="/a", duration_sec=30)
        db = fake_db(art, cached_audio=hit, earlier_job_result=None)
        with mock.patch.object(settings, "TTS_GG_CHIRP", ""):
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        self.assertIsNone(db.add.call_args.args[0].result["timings"])


if __name__ == "__main__":
    unittest.main()
