"""The listening corpus browser: what counts as the same sentence, which speech
level a line is, how a session walks the corpus without repeats, and the three
learner endpoints built on those. Pure functions are tested directly; the
endpoints get a fake DB session (it answers by looking at the SQL text), the
way tests/test_progress.py does it."""
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from app.api.routers import content
from app.services import corpus_browse
from app.services.corpus_browse import (
    Sentence,
    count_facets,
    dedupe,
    detect_register,
    effective_register,
    matches,
    normalize_key,
    seeded_order,
)
from app.services.ingestion import drop_duplicate_cues
from app.services.subtitles import Cue


def sentence(n: int, text: str, *, film=1, level=2, register="존댓말", topics=(), grammar=()):
    return Sentence(
        id=uuid.UUID(int=n),
        film_id=film,
        text_ko=text,
        kind="câu",
        level=level,
        register=register,
        topic_ids=tuple(topics),
        grammar_ids=tuple(grammar),
    )


class NormalizeKeyTests(unittest.TestCase):
    def test_punctuation_spacing_and_case_do_not_make_a_new_sentence(self):
        a = normalize_key("아니요, 저는 괜찮아요.")
        self.assertEqual(a, normalize_key("아니요 저는   괜찮아요"))
        self.assertEqual(a, normalize_key("“아니요, 저는 괜찮아요!”"))
        self.assertEqual(normalize_key("OK, ok!"), normalize_key("ok ok"))

    def test_other_word_forms_stay_different(self):
        # same meaning, another ending: a variant to teach, not a copy to hide
        self.assertNotEqual(normalize_key("괜찮아요."), normalize_key("괜찮습니다."))
        self.assertNotEqual(normalize_key("괜찮아요."), normalize_key("괜찮아."))

    def test_nothing_but_punctuation_has_no_identity(self):
        self.assertEqual(normalize_key("... ♪ !!"), "")


class DetectRegisterTests(unittest.TestCase):
    CASES = {
        # the importer's label was wrong for these (real lines from the corpus)
        "집은 이미 감염되었습니다.": "존댓말",
        "안타깝습니다. 그게 전부입니다. 정말 그렇습니다.": "존댓말",
        "당신은 좋은 문제 해결사입니다.": "존댓말",
        "당신은 무엇을 할 것인가?": "반말",
        # speech levels
        "시대가 변했습니다.": "존댓말",
        "그들은 어떻습니까?": "존댓말",
        "갑시다.": "존댓말",
        "결과가 나오면 알려주세요.": "존댓말",
        "아니요.": "존댓말",
        "나에겐 계획이 있다.": "반말",
        "못 들었어?": "반말",
        "아직도 대리야?": "반말",
        # "아니다" ends in 니다 but is a plain verb, not -ㅂ니다
        "그것은 사실이 아니다.": "반말",
        # a vocative after the comma must not hide the ending before it
        "안녕하세요, 공씨.": "존댓말",
        "부인, 한 병 더 드릴까요?": "존댓말",
        # one line, two sentences, two levels
        "담배 피우러 나가자. 좋아요.": "hỗn hợp",
        # an interjection after a polite sentence says nothing
        "돼지고기를 좋아해요? 예.": "존댓말",
        # nothing to go on: fragments, interjections, a noun that ends in 요
        "예.": None,
        "그게 필요.": None,
        "더 좋은 계약으로.": None,
        "": None,
    }

    def test_known_lines(self):
        for text, expected in self.CASES.items():
            with self.subTest(text=text):
                self.assertEqual(detect_register(text), expected)

    def test_the_ending_beats_the_importers_label_but_only_when_it_is_conclusive(self):
        self.assertEqual(effective_register("집은 이미 감염되었습니다.", "반말"), "존댓말")
        self.assertEqual(effective_register("더 좋은 계약으로.", "hỗn hợp"), "hỗn hợp")


class DedupeTests(unittest.TestCase):
    def test_keeps_one_copy_and_the_same_one_every_time(self):
        a = sentence(2, "아니요, 저는 괜찮아요.", film=3)
        b = sentence(1, "아니요 저는 괜찮아요", film=3)
        c = sentence(9, "아니요, 저는 괜찮아요!", film=2)
        d = sentence(4, "죄송합니다.")
        kept = dedupe([a, b, c, d])
        self.assertEqual({s.id for s in kept}, {c.id, d.id})  # lowest film id wins
        self.assertEqual({s.id for s in dedupe([d, c, b, a])}, {c.id, d.id})  # input order is irrelevant

    def test_sentences_without_letters_are_dropped(self):
        self.assertEqual(dedupe([sentence(1, "...")]), [])


class SessionOrderTests(unittest.TestCase):
    def test_paging_with_one_seed_visits_every_sentence_once(self):
        items = [sentence(n, f"문장 {n}") for n in range(1, 38)]
        seen = []
        for offset in range(0, len(items), 10):
            seen += seeded_order(items, "abc")[offset : offset + 10]
        self.assertEqual(sorted(s.id for s in seen), sorted(s.id for s in items))

    def test_same_seed_same_order_other_seed_other_order(self):
        items = [sentence(n, f"문장 {n}") for n in range(1, 30)]
        self.assertEqual(seeded_order(items, "x"), seeded_order(list(reversed(items)), "x"))
        self.assertNotEqual([s.id for s in seeded_order(items, "x")], [s.id for s in seeded_order(items, "y")])


class FilterAndFacetTests(unittest.TestCase):
    ITEMS = [
        sentence(1, "시대가 변했습니다.", level=2, topics=[10, 11], grammar=[5]),
        sentence(2, "기다려요.", level=1, topics=[11], film=2),
        sentence(3, "나에겐 계획이 있다.", level=1, register="반말", topics=[11], film=2),
    ]

    def test_filters_combine(self):
        ids = lambda **kw: sorted(s.id.int for s in self.ITEMS if matches(s, **kw))  # noqa: E731
        self.assertEqual(ids(), [1, 2, 3])
        self.assertEqual(ids(level=1), [2, 3])
        self.assertEqual(ids(level=1, register="반말"), [3])
        self.assertEqual(ids(topic_id=10), [1])
        self.assertEqual(ids(grammar_id=5), [1])
        self.assertEqual(ids(film_id=2, topic_id=11), [2, 3])
        self.assertEqual(ids(level=5), [])

    def test_counts(self):
        c = count_facets(self.ITEMS)
        self.assertEqual(c.total, 3)
        self.assertEqual(c.levels, [(1, 2), (2, 1)])
        self.assertEqual(c.registers, [("존댓말", 2), ("반말", 1)])
        self.assertEqual(c.topics, [(11, 3), (10, 1)])  # biggest first
        self.assertEqual(c.grammar, [(5, 1)])
        self.assertEqual(c.films, [(2, 2), (1, 1)])


class ImportDedupeTests(unittest.TestCase):
    def test_known_lines_and_repeats_inside_the_file_are_not_staged(self):
        known = corpus_browse.corpus_keys(["아니요, 저는 괜찮아요.", "..."])
        cues = [
            Cue("1-2", "아니요 저는 괜찮아요"),  # already in the corpus
            Cue("3-4", "죄송합니다."),
            Cue("5-6", "죄송합니다!"),  # repeat of the line above
            Cue("7-8", "♪ ... ♪"),  # no letters: ignored, not counted
            Cue("9-10", "알겠습니다."),
        ]
        fresh, dropped = drop_duplicate_cues(cues, known)
        self.assertEqual([c.source_ref for c in fresh], ["3-4", "9-10"])
        self.assertEqual(dropped, 2)


# ------------------------------------------------------------------ endpoints --
def corpus_row(n, text, *, film=1, level=2, register="존댓말", topics=(), grammar=(), meaning=None, usage=None):
    return (uuid.UUID(int=n), film, text, "câu", level, register, list(topics), list(grammar), meaning, usage)


ROWS = [
    corpus_row(1, "시대가 변했습니다.", topics=[10], grammar=[5]),
    corpus_row(2, "시대가 변했습니다!", film=2, topics=[10]),  # repeat of 1
    corpus_row(3, "기다려요.", level=1, topics=[11], film=2),
    corpus_row(4, "집은 이미 감염되었습니다.", level=3, register="반말", film=2),  # label is wrong
    corpus_row(5, "나에겐 계획이 있다.", level=1, register="반말"),
]
FILMS = [(1, "Phim A"), (2, "Phim B")]
TOPICS = [(10, "Công việc"), (11, "Khẩu ngữ")]
GRAMMAR = [(5, "V/A + -았/었-")]


class FakeDB:
    def __init__(self, rows=ROWS, similar=None, target=None):
        self.rows = rows
        self.similar = similar or []
        self.target = target
        self.snapshot_reads = 0

    async def execute(self, stmt):
        text = str(stmt)
        res = mock.MagicMock()
        if "<=>" in text:
            res.all.return_value = self.similar
        elif "FROM corpus.film" in text:
            res.all.return_value = FILMS
        elif "FROM content.topic" in text:
            res.all.return_value = TOPICS
        elif "FROM content.grammar_point" in text:
            res.all.return_value = GRAMMAR
        else:  # corpus.corpus_item snapshot
            self.snapshot_reads += 1
            res.all.return_value = self.rows
        return res

    async def get(self, model, item_id):
        return self.target


class CorpusEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        content.reset_corpus_snapshot()

    async def test_facets_count_distinct_sentences_with_corrected_registers(self):
        out = await content.corpus_facets(FakeDB())
        self.assertEqual(out.total, 4)  # the repeat is counted once
        self.assertEqual([(x.level, x.count) for x in out.levels], [(1, 2), (2, 1), (3, 1)])
        # lines 4 and 5 were stored as 반말; 4 ends in 습니다, so it is 존댓말 now
        self.assertEqual([(x.register, x.count) for x in out.registers], [("존댓말", 3), ("반말", 1)])
        self.assertEqual([(x.name, x.count) for x in out.topics], [("Công việc", 1), ("Khẩu ngữ", 1)])
        self.assertEqual([(x.pattern, x.count) for x in out.grammar], [("V/A + -았/었-", 1)])
        self.assertEqual({x.title for x in out.films}, {"Phim A", "Phim B"})

    async def test_browse_pages_without_repeats_and_reports_the_total(self):
        db = FakeDB()
        first = await content.browse_corpus(db, seed="s1", offset=0, limit=2)
        second = await content.browse_corpus(db, seed="s1", offset=2, limit=2)
        self.assertEqual((first.total, second.total), (4, 4))
        ids = [i.id for i in first.items + second.items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 4)
        self.assertEqual(db.snapshot_reads, 1)  # the snapshot is reused across pages

    async def test_browse_filters_and_fills_in_names(self):
        out = await content.browse_corpus(FakeDB(), level=1, register="반말")
        self.assertEqual([i.text_ko for i in out.items], ["나에겐 계획이 있다."])
        self.assertEqual(out.total, 1)
        item = out.items[0]
        self.assertEqual((item.film_title, item.register), ("Phim A", "반말"))
        out = await content.browse_corpus(FakeDB(), topic_id=10)
        self.assertEqual(out.items[0].topics, ["Công việc"])
        self.assertEqual(out.items[0].grammar_patterns, ["V/A + -았/었-"])

    async def test_browse_carries_the_meaning_and_usage_note(self):
        rows = [
            corpus_row(1, "기다려요.", meaning="Tôi đợi đây.", usage="Lịch sự, dùng với người lớn."),
            corpus_row(2, "기다려", register="반말"),  # not enriched yet: both are null, not missing
        ]
        out = await content.browse_corpus(FakeDB(rows=rows), seed="s")
        by_text = {i.text_ko: i for i in out.items}
        self.assertEqual(by_text["기다려요."].meaning_vi, "Tôi đợi đây.")
        self.assertEqual(by_text["기다려요."].usage_note_vi, "Lịch sự, dùng với người lớn.")
        self.assertIsNone(by_text["기다려"].meaning_vi)
        self.assertIsNone(by_text["기다려"].usage_note_vi)

    async def test_unnatural_lines_are_filtered_in_the_query_not_in_python(self):
        seen = []

        class SpyDB(FakeDB):
            async def execute(self, stmt):
                seen.append(str(stmt))
                return await super().execute(stmt)

        await content.corpus_facets(SpyDB())
        snapshot_sql = next(q for q in seen if "FROM corpus.corpus_item" in q)
        self.assertIn("naturalness IS DISTINCT FROM", snapshot_sql)  # NULL (not judged yet) stays visible

    def test_a_repeat_with_a_meaning_wins_over_one_still_waiting_for_it(self):
        waiting = sentence(1, "기다려요.", film=1)
        done = Sentence(**{**waiting.__dict__, "id": uuid.UUID(int=2), "film_id": 2, "meaning_vi": "Tôi đợi."})
        self.assertEqual(dedupe([waiting, done])[0].id, done.id)
        self.assertEqual(dedupe([done, waiting])[0].id, done.id)  # whatever the input order

    async def test_browse_past_the_end_is_an_empty_page(self):
        out = await content.browse_corpus(FakeDB(), offset=40)
        self.assertEqual((out.total, out.items), (4, []))

    async def test_similar_skips_repeats_and_the_sentence_itself(self):
        target = SimpleNamespace(text_ko="기다려요.", embedding=[0.1, 0.2])
        near = [
            (uuid.UUID(int=2), 0.05),  # a stored copy that dedupe removed
            (uuid.UUID(int=3), 0.01),  # the target's own text under another id
            (uuid.UUID(int=5), 0.2),
            (uuid.UUID(int=4), 0.3),
        ]
        out = await content.similar_corpus_items(uuid.UUID(int=3), FakeDB(similar=near, target=target), limit=2)
        self.assertEqual([i.text_ko for i in out], ["나에겐 계획이 있다.", "집은 이미 감염되었습니다."])
        self.assertEqual([i.distance for i in out], [0.2, 0.3])

    async def test_similar_without_an_embedding_is_empty_and_unknown_id_is_404(self):
        no_vec = SimpleNamespace(text_ko="기다려요.", embedding=None)
        self.assertEqual(await content.similar_corpus_items(uuid.UUID(int=3), FakeDB(target=no_vec)), [])
        with self.assertRaises(HTTPException) as ctx:
            await content.similar_corpus_items(uuid.UUID(int=99), FakeDB(target=None))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
