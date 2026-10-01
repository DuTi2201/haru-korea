"""What a confirmed proposal becomes. A reviewer can edit a staged card by hand
in Studio, so the chunk layers are tidied again when the row is written, and
every key must be a real column (a typo here would only surface at confirm)."""
import unittest

from app.models import GrammarPoint, VocabItem
from app.schemas import GrammarPointOut, VocabItemOut
from app.services import ingestion


class RowFieldsTests(unittest.TestCase):
    def test_a_good_chunk_payload_becomes_a_vocab_row_with_every_layer(self):
        payload = {
            "hangul": "비가 오다", "meaning_vi": "trời mưa", "level": 1, "pos": "cụm",
            "family": "Động từ đi với thời tiết", "node_word": "오다", "register": "neutral",
            "usage_note_vi": "Dùng 오다, không dùng 내리다 ở đây.",
            "collocations": [{"ko": "눈이 오다", "vi": "tuyết rơi"}], "distractors": ["불다", "들다"],
            "confidence": 0.9, "junk": "ignored",
        }
        fields = ingestion._row_fields("vocab_item", payload)
        row = VocabItem(lesson_id=1, import_item_id=None, **fields)
        self.assertEqual((row.family, row.node_word, row.register), ("Động từ đi với thời tiết", "오다", "neutral"))
        self.assertEqual(row.collocations, [{"ko": "눈이 오다", "vi": "tuyết rơi"}])
        self.assertEqual(row.distractors, ["불다", "들다"])
        self.assertFalse(hasattr(row, "junk"))
        self.assertNotIn("confidence", fields)

    def test_a_hand_edit_that_breaks_a_layer_is_repaired_not_published(self):
        payload = {
            "hangul": "비가 오다", "meaning_vi": "trời mưa", "level": 1,
            "node_word": "내리다", "distractors": ["불다"], "register": "văn nói", "collocations": "눈이 오다",
        }
        fields = ingestion._row_fields("vocab_item", payload)
        self.assertEqual((fields["node_word"], fields["distractors"], fields["register"], fields["collocations"]),
                         (None, None, None, None))

    def test_an_old_payload_without_any_layer_still_publishes(self):
        fields = ingestion._row_fields("vocab_item", {"hangul": "봄", "meaning_vi": "mùa xuân", "level": 1})
        row = VocabItem(lesson_id=1, import_item_id=None, **fields)
        self.assertEqual((row.hangul, row.family, row.node_word, row.collocations), ("봄", None, None, None))

    def test_a_grammar_payload_keeps_its_group_and_well_formed_contrasts_only(self):
        payload = {
            "pattern": "V + -(으)ㄹ 것 같다", "meaning_vi": "có vẻ sẽ", "level": 2, "contrast_group": "Phỏng đoán",
            "contrasts": [{"pattern": "-는 것 같다", "diff_vi": "hiện tại"}, {"pattern": "x"}, "y"],
        }
        fields = ingestion._row_fields("grammar_point", payload)
        row = GrammarPoint(lesson_id=1, import_item_id=None, **fields)
        self.assertEqual(row.contrast_group, "Phỏng đoán")
        self.assertEqual(row.contrasts, [{"pattern": "-는 것 같다", "diff_vi": "hiện tại"}])

    def test_overlong_labels_are_cut_to_the_column_width(self):
        fields = ingestion._row_fields("vocab_item", {"hangul": "봄", "meaning_vi": "m", "level": 1, "family": "x" * 500})
        self.assertEqual(len(fields["family"]), 120)


class OutSchemaTests(unittest.TestCase):
    def test_rows_without_layers_serialise_with_nulls(self):
        row = VocabItem(id=1, lesson_id=1, hangul="봄", meaning_vi="mùa xuân", level=1)
        out = VocabItemOut.model_validate(row)
        self.assertEqual((out.family, out.collocations, out.distractors), (None, None, None))

    def test_layers_serialise_as_typed_objects(self):
        row = VocabItem(id=1, lesson_id=1, hangul="비가 오다", meaning_vi="mưa", level=1, node_word="오다",
                        collocations=[{"ko": "눈이 오다", "vi": "tuyết rơi"}], distractors=["불다"])
        out = VocabItemOut.model_validate(row).model_dump()
        self.assertEqual(out["collocations"], [{"ko": "눈이 오다", "vi": "tuyết rơi"}])
        grammar = GrammarPointOut.model_validate(
            GrammarPoint(id=2, lesson_id=1, pattern="p", meaning_vi="m", level=2,
                         contrast_group="g", contrasts=[{"pattern": "q", "diff_vi": "d"}])
        )
        self.assertEqual(grammar.contrasts[0].diff_vi, "d")


if __name__ == "__main__":
    unittest.main()
