"""Putting answers on an exam batch, and staging what the extraction read.

Answers come only from a key (a file, the paper's own pages, or numbers an editor
typed). A key that arrives late — after the batch was confirmed — still reaches the
questions in the library; a missing key never turns a question red."""
import unittest
from types import SimpleNamespace
from unittest import mock

from app.api.routers import ingest
from app.models import ImportItem
from app.services import exam_extract as ex, ingestion


def question(n, *, status="pending", answer=None, flags=None, confidence=0.9):
    return ImportItem(
        kind="exam_item",
        status=status,
        confidence=confidence,
        payload={"number": n, "stem_ko": f"문제 {n}", "options": ["가", "나", "다", "라"], "answer": answer, "answer_from_key": answer is not None, "flags": flags or []},
    )


def live(n, answer=None):
    return SimpleNamespace(number=n, answer=answer, answer_source=None)


class AnswerRowsTests(unittest.TestCase):
    def test_answers_land_on_the_matching_questions(self):
        items = [question(1), question(2)]
        report = ingestion._answer_rows(items, [], {1: 3, 2: 4}, commit=True)
        self.assertEqual([i.payload["answer"] for i in items], [3, 4])
        self.assertTrue(all(i.payload["answer_from_key"] for i in items))
        self.assertEqual((report.applied, report.unmatched, report.missing, report.changed), ([1, 2], [], [], []))

    def test_a_question_the_key_lacks_says_so_and_one_it_has_stops_saying_it(self):
        items = [question(1, flags=["no_answer"], status="flagged_yellow"), question(2)]
        report = ingestion._answer_rows(items, [], {1: 2}, commit=True)
        self.assertEqual(items[0].payload["flags"], [])
        self.assertEqual(items[0].status, "pending")  # nothing left to look at
        self.assertEqual(items[1].payload["flags"], ["no_answer"])
        self.assertEqual(items[1].status, "flagged_yellow")
        self.assertEqual(report.missing, [2])

    def test_the_editors_decisions_are_kept(self):
        confirmed, rejected = question(1, status="confirmed"), question(2, status="rejected")
        report = ingestion._answer_rows([confirmed, rejected], [], {1: 1, 2: 2}, commit=True)
        self.assertEqual(confirmed.status, "confirmed")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(report.unmatched, [2])  # a rejected question is not part of the paper any more

    def test_a_new_answer_over_a_different_one_is_reported(self):
        items = [question(1, answer=1), question(2, answer=2)]
        report = ingestion._answer_rows(items, [], {1: 3, 2: 2}, commit=True)
        self.assertEqual(report.changed, [1])
        self.assertEqual(items[0].payload["answer"], 3)

    def test_numbers_in_the_key_without_a_question_are_unmatched(self):
        report = ingestion._answer_rows([question(1)], [], {1: 1, 40: 2}, commit=True)
        self.assertEqual(report.unmatched, [40])

    def test_a_dry_run_changes_nothing(self):
        items, rows = [question(1)], [live(1)]
        report = ingestion._answer_rows(items, rows, {1: 3}, commit=False)
        self.assertEqual(report.applied, [1])
        self.assertIsNone(items[0].payload["answer"])
        self.assertEqual(items[0].payload["flags"], [])
        self.assertIsNone(rows[0].answer)

    def test_questions_already_in_the_library_get_the_answer_as_a_key_answer(self):
        rows = [live(1), live(2)]
        ingestion._answer_rows([question(1), question(2)], rows, {1: 4}, commit=True)
        self.assertEqual((rows[0].answer, rows[0].answer_source), (4, "editor"))
        self.assertIsNone(rows[1].answer)

    def test_the_status_of_a_paper_s_answers(self):
        self.assertEqual(ingestion._answer_status([question(1), question(2)]), "pending")
        self.assertEqual(ingestion._answer_status([question(1, answer=1), question(2)]), "partial")
        self.assertEqual(ingestion._answer_status([question(1, answer=1), question(2, answer=2)]), "complete")
        self.assertEqual(ingestion._answer_status([question(1, answer=1), question(2, status="rejected")]), "complete")
        self.assertEqual(ingestion._answer_status([]), "pending")

    def test_the_part_of_the_exam_a_batch_is(self):
        summary = ImportItem(kind=ingestion.EXAM_SUMMARY, status="confirmed", confidence=1.0, payload={"section": "nghe"})
        self.assertEqual(ingestion.paper_skill([summary], "102회 읽기"), "đọc")  # the editor's own label wins
        self.assertEqual(ingestion.paper_skill([summary], "102회"), "nghe")
        self.assertIsNone(ingestion.paper_skill([], "102회"))


class StagingTests(unittest.TestCase):
    def draft(self):
        passage = {"payload": {"local_ref": "P1", "kind": "đọc hiểu", "body_ko": "본문", "withheld": False, "source_page": 3, "flags": []}, "confidence": 0.9, "flags": []}
        ok = {"payload": {"number": 1, "stem_ko": "가", "options": ["a", "b", "c", "d"], "answer": None, "flags": []}, "confidence": 0.9, "flags": []}
        bad = {"payload": {"number": 2, "stem_ko": "나", "options": ["a"], "answer": None, "flags": ["withheld"]}, "confidence": 0.9, "flags": ["withheld"]}
        return ex.Draft(passages=[passage], items=[ok, bad], summary={"section": "đọc", "gaps": []}, key={})

    def test_each_entry_becomes_an_item_with_its_light_and_a_summary_follows(self):
        db = mock.MagicMock()
        staged, flagged = ingestion.stage_exam_draft(db, SimpleNamespace(id="b1"), self.draft())
        added = [c.args[0] for c in db.add.call_args_list]
        self.assertEqual([(i.kind, i.status) for i in added], [("exam_passage", "pending"), ("exam_item", "pending"), ("exam_item", "flagged_red"), ("exam_summary", "confirmed")])
        self.assertEqual((staged, flagged), (3, 1))  # the summary is neither staged nor flagged

    def test_a_key_on_the_papers_own_pages_is_applied_for_the_right_part(self):
        draft = self.draft()
        draft.key = {"đọc": {1: 2}, "nghe": {1: 3}}
        db = mock.MagicMock()
        with mock.patch.object(ingestion, "apply_answers") as apply:
            ingestion.stage_exam_draft(db, SimpleNamespace(id="b1"), draft)
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[2], {1: 2})
        self.assertEqual(apply.call_args.kwargs["source"], "paper")

    def test_a_key_for_another_part_is_not_applied(self):
        draft = self.draft()
        draft.key = {"nghe": {1: 3}}
        with mock.patch.object(ingestion, "apply_answers") as apply:
            ingestion.stage_exam_draft(mock.MagicMock(), SimpleNamespace(id="b1"), draft)
        apply.assert_not_called()


class UploadKeyTests(unittest.TestCase):
    def test_an_exam_paper_job_key_carries_the_extraction_version(self):
        lookup, new = ingest.upload_job_keys("exam_paper", "h", "awaiting_review")
        self.assertEqual((lookup, new), (["exam_paper:h:exam-v3"], "exam_paper:h:exam-v3"))
        # a paper read by the first version (plain key) is read again
        self.assertNotIn("exam_paper:h", lookup)

    def test_a_confirmed_paper_is_not_read_again(self):
        lookup, new = ingest.upload_job_keys("exam_paper", "h", "confirmed")
        self.assertEqual(lookup, ["exam_paper:h:exam-v3", "exam_paper:h"])
        self.assertEqual(new, "exam_paper:h:exam-v3")


if __name__ == "__main__":
    unittest.main()
