"""Route-level smoke tests with a faked DB session (no Postgres): the read
path cleans legacy body text + queues the one-off photo backfill, and the
audio path voices the CLEANED title+body and reuses cached audio."""
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from app.api.routers import editorial
from app.schemas import ArticleAudioRequest
from app.services import study_pack as sp
from app.services.article_extract import clean_article_text, split_paragraphs

P1 = "미 국채시장이 불안하다. 10년물 금리가 5%, 30년물은 5.3%를 넘나들고 있다. 이란전쟁과 인플레 우려가 직접적 계기지만, 문제의 뿌리는 깊다."
P2 = "이 상황에서 떠올리게 되는 것이 스티븐 마이런의 2024년 보고서다. 세계 무역체제 재편을 위한 안내서인데, 트럼프 2기의 관세 정책을 이해하는 이론적 배경이 되어왔다."
DIRTY = "\n".join(["보기 설정", "닫기", "글자 크기", "보통", "크게", P1, P2, "지금 많이 보는 기사", "오마주", "닫기"])


def article(**kw):
    base = dict(
        id=uuid.uuid4(), source_name="경향신문", source_url="https://x/1", title_ko="[정동칼럼]미 국채시장",
        level_estimate=5, topic_tags=["경제"], body_ko=DIRTY, vocab_ids=[], grammar_ids=[],
        thinking_guide_text=None, created_at=datetime.now(timezone.utc), images=[], images_fetched_at=None,
        study_pack=None, study_status="none", study_updated_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def fake_db(art, *, existing_job=None, cached_audio=None):
    db = mock.MagicMock()
    db.get = mock.AsyncMock(return_value=art)
    def add(obj):
        # a real DB assigns the primary key on flush/commit
        if getattr(obj, "id", "x") is None:
            obj.id = uuid.uuid4()

    db.add = mock.MagicMock(side_effect=add)
    db.commit = mock.AsyncMock()
    db.refresh = mock.AsyncMock()
    db.rollback = mock.AsyncMock()
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


class ReadPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_body_cleaned_and_photo_backfill_queued_once(self):
        art = article()
        db = fake_db(art)
        with mock.patch.object(editorial.refresh_editorial_images, "delay") as delay, mock.patch.object(
            editorial.generate_study_pack, "delay"
        ):
            out = await editorial.get_editorial(art.id, db)
        self.assertEqual(out.body_ko.split("\n"), [P1, P2])
        self.assertTrue(out.images_pending)
        delay.assert_called_once_with(str(art.id))
        self.assertIsNotNone(art.images_fetched_at)  # stamped so a 2nd view does not re-queue

    async def test_no_backfill_when_already_fetched(self):
        art = article(images_fetched_at=datetime.now(timezone.utc), images=[{"url": "https://i/a.jpg", "caption": "c", "after_paragraph": 1}])
        db = fake_db(art)
        with mock.patch.object(editorial.refresh_editorial_images, "delay") as delay, mock.patch.object(
            editorial.generate_study_pack, "delay"
        ):
            out = await editorial.get_editorial(art.id, db)
        delay.assert_not_called()
        self.assertFalse(out.images_pending)
        self.assertEqual(out.images[0].after_paragraph, 1)

    async def test_queue_failure_never_breaks_reading(self):
        art = article()
        db = fake_db(art)
        with mock.patch.object(
            editorial.refresh_editorial_images, "delay", side_effect=RuntimeError("redis down")
        ), mock.patch.object(editorial.generate_study_pack, "delay"):
            out = await editorial.get_editorial(art.id, db)
        self.assertEqual(out.body_ko.split("\n"), [P1, P2])
        self.assertFalse(out.images_pending)


class AudioPathTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(headers={})

    async def test_voices_cleaned_title_and_body(self):
        art = article()
        db = fake_db(art)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            out = await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        self.assertEqual(out.status, "queued")
        args = delay.call_args.args
        spoken = args[1]
        self.assertEqual(spoken.split("\n"), ["미 국채시장.", P1, P2])
        self.assertNotIn("보기 설정", spoken)
        self.assertNotIn("지금 많이 보는", spoken)
        self.assertTrue(args[4].startswith(f"article-audio:{art.id}:ko-female-1:article-v1:"))

    async def test_cache_hit_does_not_call_tts(self):
        art = article()
        hit = SimpleNamespace(cache_key="k", opus_path="/o", aac_path="/a", duration_sec=42)
        db = fake_db(art, cached_audio=hit)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            out = await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        self.assertEqual(out.status, "succeeded")
        delay.assert_not_called()

    async def test_dead_running_job_is_replaced(self):
        art = article()
        dead = SimpleNamespace(status="running", updated_at=datetime.now(timezone.utc) - timedelta(minutes=30))
        db = fake_db(art, existing_job=dead)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        db.delete.assert_awaited_once_with(dead)
        delay.assert_called_once()

    async def test_fresh_running_job_is_reused(self):
        art = article()
        live = SimpleNamespace(
            id=uuid.uuid4(), status="running", progress=0.4, updated_at=datetime.now(timezone.utc) - timedelta(seconds=30)
        )
        db = fake_db(art, existing_job=live)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            out = await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        delay.assert_not_called()
        self.assertEqual(out.job_id, live.id)

    async def test_empty_article_is_409(self):
        art = article(body_ko="닫기\n로그인")
        db = fake_db(art)
        with self.assertRaises(Exception) as ctx:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        self.assertEqual(ctx.exception.status_code, 409)


def ready_pack(art):
    """A study pack valid for `art`'s current cleaned text."""
    cleaned = clean_article_text(art.body_ko)
    paras = split_paragraphs(cleaned)
    return {
        "version": sp.STUDY_VERSION,
        "text_sig": sp.study_text_sig(cleaned),
        "summary_vi": "Tóm tắt.",
        "key_points_vi": ["Ý chính"],
        "key_terms": [{"ko": "국채", "vi": "trái phiếu chính phủ"}],
        "paragraphs": [
            {
                "easy_ko": f"쉬운 {i}.",
                "sentences": [
                    {"ko": s_, "vi": f"dịch {i}", "words": [{"surface": "국채시장이", "base": None, "pos": None, "meaning_vi": "thị trường trái phiếu"}], "grammar_notes_vi": []}
                    for s_ in sp.split_sentences(p)
                ],
            }
            for i, p in enumerate(paras)
        ],
    }


class StudyPackReadTests(unittest.IsolatedAsyncioTestCase):
    async def get(self, art):
        db = fake_db(art)
        with mock.patch.object(editorial.refresh_editorial_images, "delay"), mock.patch.object(
            editorial.generate_study_pack, "delay"
        ) as study_delay:
            out = await editorial.get_editorial(art.id, db)
        return out, study_delay

    async def test_first_read_queues_generation_once_and_reports_pending(self):
        art = article()
        out, study_delay = await self.get(art)
        study_delay.assert_called_once_with(str(art.id))
        self.assertEqual(out.study_status, "pending")
        self.assertIsNone(out.study)
        self.assertEqual(art.study_status, "pending")  # a 2nd view will not re-queue

    async def test_pending_with_fresh_heartbeat_is_not_requeued(self):
        art = article(study_status="pending", study_updated_at=datetime.now(timezone.utc) - timedelta(minutes=2))
        out, study_delay = await self.get(art)
        study_delay.assert_not_called()
        self.assertEqual(out.study_status, "pending")

    async def test_pending_with_dead_heartbeat_is_requeued(self):
        art = article(study_status="pending", study_updated_at=datetime.now(timezone.utc) - timedelta(minutes=45))
        out, study_delay = await self.get(art)
        study_delay.assert_called_once()

    async def test_recent_failure_not_retried_old_failure_is(self):
        recent = article(study_status="failed", study_updated_at=datetime.now(timezone.utc) - timedelta(minutes=3))
        out, study_delay = await self.get(recent)
        study_delay.assert_not_called()
        self.assertEqual(out.study_status, "failed")
        old = article(study_status="failed", study_updated_at=datetime.now(timezone.utc) - timedelta(hours=2))
        _, study_delay = await self.get(old)
        study_delay.assert_called_once()

    async def test_ready_pack_is_returned(self):
        art = article()
        art.study_pack = ready_pack(art)
        art.study_status = "ready"
        out, study_delay = await self.get(art)
        study_delay.assert_not_called()
        self.assertEqual(out.study_status, "ready")
        self.assertEqual(out.study.summary_vi, "Tóm tắt.")
        self.assertEqual(len(out.study.paragraphs), len(out.body_ko.split("\n")))
        self.assertEqual(out.study.paragraphs[0].sentences[0].words[0].meaning_vi, "thị trường trái phiếu")

    async def test_pack_for_other_text_is_not_shown_and_is_regenerated(self):
        art = article()
        pack = ready_pack(art)
        pack["text_sig"] = "someoldsig"
        art.study_pack = pack
        art.study_status = "ready"
        out, study_delay = await self.get(art)
        self.assertIsNone(out.study)
        self.assertEqual(out.study_status, "pending")
        study_delay.assert_called_once()

    async def test_queue_failure_never_breaks_reading(self):
        art = article()
        db = fake_db(art)
        with mock.patch.object(editorial.refresh_editorial_images, "delay"), mock.patch.object(
            editorial.generate_study_pack, "delay", side_effect=RuntimeError("redis down")
        ):
            out = await editorial.get_editorial(art.id, db)
        self.assertEqual(out.study_status, "none")
        self.assertEqual(out.body_ko.split("\n"), [P1, P2])

    async def test_article_without_body_is_never_queued(self):
        art = article(body_ko="닫기\n로그인")
        out, study_delay = await self.get(art)
        study_delay.assert_not_called()
        self.assertEqual(out.study_status, "none")


class EasyAudioTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(headers={})

    async def test_easy_variant_voices_simplified_text_with_own_cache_key(self):
        art = article()
        art.study_pack = ready_pack(art)
        art.study_status = "ready"
        db = fake_db(art)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(variant="easy"), self.request(), db)
        args = delay.call_args.args
        self.assertEqual(args[1].split("\n"), ["미 국채시장.", "쉬운 0.", "쉬운 1."])
        self.assertTrue(args[4].endswith(":easy"))

    async def test_easy_variant_before_pack_is_ready_is_409(self):
        art = article()
        db = fake_db(art)
        with self.assertRaises(Exception) as ctx:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(variant="easy"), self.request(), db)
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_original_cache_key_unchanged(self):
        art = article()
        db = fake_db(art)
        with mock.patch.object(editorial.generate_article_audio, "delay") as delay:
            await editorial.request_article_audio(art.id, ArticleAudioRequest(), self.request(), db)
        self.assertFalse(delay.call_args.args[4].endswith(":easy"))


if __name__ == "__main__":
    unittest.main()
