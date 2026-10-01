"""Re-uploading a lesson file that an older extraction read.

The regression: the upload endpoint treats the same file (same hash) as "already
done" and hands back the old batch and its finished job, so after the extraction
was fixed an editor re-uploading the same note still saw the old, incomplete
proposals. A lesson's job key now carries the extraction version; an unreviewed
batch is read again (its old proposals replaced), a confirmed one never is."""
import hashlib
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from sqlalchemy.dialects import postgresql

from app.api.routers import ingest
from app.services import ingestion, lesson_extract

RAW = "### 날씨와 계절\n* **날씨**: Thời tiết".encode("utf-8")
HASH = hashlib.sha256(RAW).hexdigest()
PLAIN = f"lesson:{HASH}"
VERSIONED = f"lesson:{HASH}:{lesson_extract.PROMPT_VERSION}"


class KeysTests(unittest.TestCase):
    def test_a_lesson_key_carries_the_extraction_version(self):
        lookup, new = ingest.upload_job_keys("lesson", HASH, "awaiting_review")
        self.assertEqual((lookup, new), ([VERSIONED], VERSIONED))
        for state in ("rolled_back", "failed", "cancelled", "queued"):
            self.assertEqual(ingest.upload_job_keys("lesson", HASH, state)[0], [VERSIONED])

    def test_a_confirmed_lesson_still_answers_to_any_earlier_job(self):
        lookup, new = ingest.upload_job_keys("lesson", HASH, "confirmed")
        self.assertEqual(lookup, [VERSIONED, PLAIN])
        self.assertEqual(new, VERSIONED)

    def test_other_kinds_keep_their_plain_key(self):
        self.assertEqual(ingest.upload_job_keys("corpus", HASH, "awaiting_review"), ([f"corpus:{HASH}"], f"corpus:{HASH}"))


class UploadTests(unittest.IsolatedAsyncioTestCase):
    def make(self, batch_status, job=None):
        batch = SimpleNamespace(id=uuid.UUID(int=1), status=batch_status)
        first, second = mock.MagicMock(), mock.MagicMock()
        first.scalar_one_or_none.return_value = batch
        second.scalars.return_value.first.return_value = job
        db = mock.MagicMock()
        db.execute = mock.AsyncMock(side_effect=[first, second])
        db.add = mock.MagicMock()
        db.commit = mock.AsyncMock()
        db.flush = mock.AsyncMock()
        db.delete = mock.AsyncMock()

        async def refresh(obj):
            if not hasattr(obj, "id") or obj.id is None:
                obj.id = uuid.UUID(int=2)
            if getattr(obj, "status", None) is None:
                obj.status = "queued"

        db.refresh = mock.AsyncMock(side_effect=refresh)
        return db, batch

    async def upload(self, db):
        upload = SimpleNamespace(read=mock.AsyncMock(return_value=RAW), content_type="text/plain", filename="bai-hoc.txt")
        with mock.patch.object(ingest.extract_lesson_import, "delay") as delay:
            result = await ingest.create_import_batch(db=db, profile=SimpleNamespace(id=uuid.UUID(int=9)), kind="lesson", file=upload)
        return result, delay

    def looked_up(self, db):
        stmt = db.execute.call_args_list[1].args[0]
        params = stmt.compile(dialect=postgresql.dialect()).params
        return [v for v in params.values() if isinstance(v, list)][0]

    async def test_an_unreviewed_batch_is_read_again_by_the_current_extraction(self):
        # the old job (plain key, 19 cards) is NOT found under the versioned key
        db, batch = self.make("awaiting_review", job=None)
        result, delay = await self.upload(db)
        self.assertEqual(self.looked_up(db), [VERSIONED])
        delay.assert_called_once()
        self.assertEqual(delay.call_args.args[1], str(batch.id))  # onto the same batch
        new_job = db.add.call_args.args[0]
        self.assertEqual(new_job.idempotency_key, VERSIONED)
        self.assertEqual(result.import_batch_id, batch.id)

    async def test_the_same_upload_twice_does_not_run_gemini_twice(self):
        running = SimpleNamespace(id=uuid.UUID(int=7), status="running")
        db, _batch = self.make("awaiting_review", job=running)
        result, delay = await self.upload(db)
        delay.assert_not_called()
        self.assertEqual(result.job_id, running.id)

    async def test_a_confirmed_lesson_is_never_read_again(self):
        done = SimpleNamespace(id=uuid.UUID(int=8), status="succeeded")
        db, _batch = self.make("confirmed", job=done)
        result, delay = await self.upload(db)
        self.assertEqual(self.looked_up(db), [VERSIONED, PLAIN])
        delay.assert_not_called()
        self.assertEqual(result.job_id, done.id)

    async def test_a_failed_job_is_cleared_and_retried(self):
        failed = SimpleNamespace(id=uuid.UUID(int=6), status="failed")
        db, _batch = self.make("failed", job=failed)
        _result, delay = await self.upload(db)
        db.delete.assert_awaited_once_with(failed)
        delay.assert_called_once()


class ClearTests(unittest.TestCase):
    def test_the_old_proposals_of_an_unconfirmed_batch_are_dropped(self):
        db = mock.MagicMock()
        db.execute.return_value.rowcount = 19
        batch = SimpleNamespace(id=uuid.UUID(int=1), status="awaiting_review")
        self.assertEqual(ingestion.clear_staged_items(db, batch), 19)
        sql = str(db.execute.call_args.args[0])
        self.assertIn("DELETE FROM", sql)
        self.assertIn("import_item", sql)

    def test_a_confirmed_batch_is_left_alone(self):
        db = mock.MagicMock()
        batch = SimpleNamespace(id=uuid.UUID(int=1), status="confirmed")
        self.assertEqual(ingestion.clear_staged_items(db, batch), 0)
        db.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
