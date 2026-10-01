"""Lesson extraction: an inventory, then the way the items go together, then one card per item.

Why not one call: asking a (Flash-Lite) model to read a lesson and return the
whole thing — lesson info, a long free-text field, every vocabulary card with
seven fields and every grammar card — in ONE structured answer makes it stop
early. A 4.6 kB lesson with ~40 words in bold and six grammar points came back
as 14 words and 4 patterns; nothing downstream drops items (every card the
model returned is staged), the model simply stopped listing. It also skipped
whatever sat nested inside another bullet (the seasons and their activities,
the clothing words, the place names) and the "supplementary" grammar, because
"every word that appears in the lesson" never said what counts as a card.

So:

1. INVENTORY — a cheap call that only lists terms and patterns (short strings,
   nothing to get lazy about), with a precise definition of what a card is.
   For a plain-text lesson the words printed in bold are added from the text
   itself, so what the author highlighted can never be lost to the model.
2. ORGANISE — a second short call that sorts those terms into *families* (the
   verbs that go with weather, the temperature scale) and the grammar patterns
   into *contrast groups* (the look-alike "것 같다" forms). TOPIK tests which
   words go together and which pattern is closest in meaning, so a card is
   learned with its set, not alone. A failed call only means no sets.
3. CARDS — the terms are cut into small groups (families stay side by side);
   each group is a call that must return exactly one card per `ref`, now with
   the chunk layers: the node word, the collocates, the register, a usage note,
   tempting wrong choices (grammar: how it differs from its group-mates). A ref
   that comes back missing is asked for again; one that still fails is staged as
   a red, empty card so the editor sees the gap instead of the item silently
   vanishing.
4. OVERVIEW — the lesson's `content` text, in a call of its own so its length
   does not eat the effort of the lists.

Nothing here touches the database; ingestion.run_lesson_extraction stages the
result.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.services import gemini_client

PROMPT_VERSION = "lesson-v4"

VOCAB_GROUP = 12  # cards per call: small enough that none is skipped
GRAMMAR_GROUP = 3  # grammar cards are long (usage, tips), so fewer per call
MAX_TERMS = 150  # a sanity cap on what one lesson can ask the model to write
MAX_FAMILY = 10  # members of one family / contrast group; a bigger set is not a set anymore
MAX_LABEL = 120
MAX_COLLOCATIONS = 3
MAX_DISTRACTORS = 3
REGISTERS = ("spoken", "written", "neutral")

Generate = Callable[..., dict[str, Any]]

_HANGUL_RE = re.compile("[\uAC00-\uD7A3]")  # a syllable block; a lone jamo (the ㄹ of a 받침 note) is not a word
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_PAREN_RE = re.compile(r"[(（][^)）]*[)）]")
_NUMBERING_RE = re.compile(r"^\s*\d+\s*[.)]\s*")
_KEY_STRIP_RE = re.compile(r"[\s·\-~()（）.,:;!?]")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# ------------------------------------------------------------------ schemas --
INVENTORY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "description": "Tên bài học, ngắn gọn"},
        "level": {"type": "INTEGER", "description": "1-6, ước lượng theo cấp TOPIK tương ứng"},
        "topics": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Tên các chủ đề liên quan"},
        "confidence": {"type": "NUMBER", "description": "0-1, độ rõ ràng của tài liệu gốc"},
        "vocab_terms": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Mọi từ/cụm từ tiếng Hàn được dạy trong bài, mỗi mục một lần, chỉ chữ Hàn",
        },
        "grammar_patterns": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Mọi mẫu ngữ pháp được dạy, dạng 'V/A + hình thái', mỗi mẫu một lần",
        },
    },
    "required": ["title", "level", "vocab_terms", "grammar_patterns", "confidence"],
}

_VOCAB_ITEM_PROPS: dict[str, Any] = {
    "ref": {"type": "STRING", "description": "Chép y nguyên ref của mục (v1, v2...)"},
    "hangul": {"type": "STRING"},
    "pos": {"type": "STRING", "nullable": True, "description": "từ loại, vd 동사/명사/형용사"},
    "meaning_vi": {"type": "STRING"},
    "definition_ko": {"type": "STRING", "nullable": True},
    "level": {"type": "INTEGER"},
    "hanja": {"type": "STRING", "nullable": True},
    "sino_vietnamese": {"type": "STRING", "nullable": True, "description": "âm Hán Việt nếu có"},
    "example_ko": {"type": "STRING", "nullable": True},
    "node_word": {
        "type": "STRING",
        "nullable": True,
        "description": "Chỉ với cụm: phần người học dễ chọn sai (thường là động từ/tính từ), chép y nguyên một đoạn của hangul. Từ đơn thì null",
    },
    "register": {
        "type": "STRING",
        "nullable": True,
        "enum": list(REGISTERS),
        "description": "spoken = chủ yếu khi nói, written = chủ yếu văn viết/bài thi, neutral = dùng chung; null nếu không chắc",
    },
    "usage_note_vi": {
        "type": "STRING",
        "nullable": True,
        "description": "Tối đa hai câu tiếng Việt: dùng với ai/ở đâu và lỗi người Việt hay mắc với từ này; null nếu không có gì đáng nói",
    },
    "collocations": {
        "type": "ARRAY",
        "nullable": True,
        "items": {
            "type": "OBJECT",
            "properties": {"ko": {"type": "STRING"}, "vi": {"type": "STRING"}},
            "required": ["ko", "vi"],
        },
        "description": "Tối đa 3 cách kết hợp tự nhiên khác của từ trung tâm, ưu tiên lấy từ tài liệu",
    },
    "distractors": {
        "type": "ARRAY",
        "nullable": True,
        "items": {"type": "STRING"},
        "description": "Chỉ với cụm có node_word: 2-3 từ trông có vẻ đúng nhưng KHÔNG đi với cụm này",
    },
    "confidence": {"type": "NUMBER", "description": "0-1, độ tự tin"},
}

VOCAB_CARDS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": _VOCAB_ITEM_PROPS,
                "required": ["ref", "hangul", "meaning_vi", "level", "confidence"],
            },
        }
    },
    "required": ["items"],
}

GRAMMAR_CARDS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ref": {"type": "STRING", "description": "Chép y nguyên ref của mẫu (g1, g2...)"},
                    "pattern": {
                        "type": "STRING",
                        "description": "Mẫu ngữ pháp viết dạng V/A + hình thái, vd 'V + -(으)ㄹ 뿐만 아니라'",
                    },
                    "meaning_vi": {"type": "STRING"},
                    "level": {"type": "INTEGER"},
                    "example_ko": {"type": "STRING", "nullable": True},
                    "usage_context_vi": {
                        "type": "STRING",
                        "description": "Giải thích bằng tiếng Việt: dùng KHI NÀO, trong HOÀN CẢNH/tình huống nào, với sắc thái gì — không chỉ định nghĩa suông",
                    },
                    "topik_tip_vi": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Nhận xét bằng tiếng Việt (câu khẳng định, không phải lời ra lệnh): mẫu này hay xuất hiện ở dạng bài/ngữ cảnh TOPIK nào và dấu hiệu nhận biết là gì",
                    },
                    "contrasts": {
                        "type": "ARRAY",
                        "nullable": True,
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "pattern": {"type": "STRING", "description": "Một mẫu khác trong nhóm đối chiếu, chép y nguyên từ danh sách nhóm"},
                                "diff_vi": {"type": "STRING", "description": "Khác mẫu này ở đâu: sắc thái, thời điểm, chủ ngữ, văn nói/viết"},
                            },
                            "required": ["pattern", "diff_vi"],
                        },
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["ref", "pattern", "meaning_vi", "level", "usage_context_vi", "confidence"],
            },
        }
    },
    "required": ["items"],
}

_GROUP_LIST: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING", "description": "Tên nhóm bằng tiếng Việt, 3-8 từ, nói rõ các mục có gì chung"},
            "refs": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Các ref thuộc nhóm (v1, v5...)"},
        },
        "required": ["label", "refs"],
    },
}

ORGANISE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {"vocab_families": _GROUP_LIST, "grammar_groups": _GROUP_LIST},
    "required": ["vocab_families", "grammar_groups"],
}

OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "content": {
            "type": "STRING",
            "description": "Nội dung bài học dạng văn bản thuần, hiển thị lại cho người học",
        }
    },
    "required": ["content"],
}


# ------------------------------------------------------------------ prompts --
INVENTORY_PROMPT = """Bạn là biên tập viên nội dung học tiếng Hàn cho người Việt. Việc của bạn lúc này
CHỈ là lập danh mục: đọc kỹ tài liệu bài học đính kèm và liệt kê MỌI mục học được
có trong đó, để bước sau viết thẻ cho từng mục. Danh mục thiếu mục là lỗi nặng
nhất, vì người học sẽ không bao giờ thấy mục bị sót; thừa một mục thì biên tập
viên chỉ cần xóa.

Cách đếm mục (mỗi mục chỉ liệt kê một lần):
1. Từ vựng: mọi từ hoặc cụm từ tiếng Hàn được dạy, dù nằm trong bảng, trong danh
   sách, được in đậm, hay chỉ đi kèm nghĩa tiếng Việt. Kể cả mục nằm lồng trong
   một dòng khác: nếu dòng giới thiệu mùa xuân có ghi 따뜻하다, 벚꽃이 피다,
   소풍을 가다 thì mỗi cụm là một mục, và chính từ 봄 cũng là một mục.
2. Dòng ghi nhiều biến thể cách nhau bởi dấu gạch chéo thì mỗi biến thể là một
   mục riêng (ví dụ "우산 / 우산을 쓰다" là hai mục).
3. Động từ và tính từ ghi ở dạng từ điển (kết thúc bằng -다). Cụm cố định giữ
   nguyên cả trợ từ ("비가 오다", "소풍을 가다"): khi bài dạy một cụm thì ghi
   CẢ CỤM, đừng tách ra danh từ rời và động từ rời.
4. Địa danh và tên riêng được nêu trong bài cũng là mục từ vựng.
5. Ngữ pháp: mọi mẫu được dạy, kể cả mục ghi là "bổ trợ", "liên quan" hay "mở
   rộng", viết dạng "V/A + hình thái" (ví dụ "V + -(으)면서").
6. Câu ví dụ không phải là mục riêng: chúng sẽ được gắn vào từ hoặc mẫu tương
   ứng ở bước sau. Từ chỉ có trong câu ví dụ mà không được dạy riêng thì không
   cần đưa vào.
Chỉ lấy những gì có trong tài liệu, không thêm từ ngoài tài liệu. Chỉ ghi chữ Hàn
của từ, không ghi nghĩa.

Ngoài danh mục, trả về tiêu đề bài, cấp độ (1-6 theo TOPIK), các chủ đề liên quan
và độ tự tin (0-1) dựa trên độ rõ ràng của tài liệu. Trả về đúng JSON schema đã cho."""

_VOCAB_CARD_RULES = """Quy tắc:
- Trả về ĐÚNG MỘT thẻ cho MỖI mục trong danh sách, mang `ref` của mục đó (chép y
  nguyên). Không bỏ mục nào, không thêm mục nào, không gộp hai mục.
- hangul: chép đúng mục từ như đã cho.
- Nghĩa, từ loại, ví dụ: ưu tiên lấy từ tài liệu (nghĩa tiếng Việt có sẵn thì
  dùng lại, đừng dịch khác đi). Nếu tài liệu không ghi thì tự viết bằng kiến thức
  của bạn và đặt confidence không quá 0,7 để biên tập viên kiểm tra.
- example_ko: chép nguyên câu ví dụ trong tài liệu nếu có. Nếu không có, viết
  một câu ngắn, tự nhiên, dùng đúng mục từ.
- hanja và sino_vietnamese chỉ điền khi là từ Hán Hàn và bạn chắc chắn.
- level: cấp TOPIK 1-6 ước lượng riêng cho mục này.
- Đề thi TOPIK kiểm tra TỪ NÀO ĐI VỚI TỪ NÀO, nên với mỗi mục hãy viết thêm:
  * node_word: chỉ với cụm (từ hai thành phần trở lên, như danh từ + trợ từ + động
    từ). Đó là phần người học dễ chọn sai, thường là động từ hoặc tính từ (오다 trong
    "비가 오다"), và phải là một đoạn chép y nguyên của hangul. Từ đơn thì để null.
  * distractors: chỉ với cụm có node_word. 2-3 từ cùng loại trông có vẻ đúng, nhất
    là những từ người Việt chọn nhầm khi dịch từng chữ, nhưng KHÔNG đi với phần còn
    lại của cụm. Trước khi đưa một từ vào, hãy ghép thử nó với phần còn lại của cụm:
    nếu người Hàn vẫn nói được (dù ít dùng hơn, ví dụ hai động từ cùng đi với một danh
    từ) thì KHÔNG đưa vào, vì người học chọn nó sẽ bị chấm sai oan. Không trùng
    node_word. Không chắc thì bỏ từ đó; có ít hơn 2 từ chắc chắn sai thì để null.
  * collocations: tối đa 3 cách kết hợp tự nhiên khác của từ trung tâm (ưu tiên lấy
    từ tài liệu), mỗi cách gồm ko và vi.
  * register: spoken (chủ yếu khi nói), written (chủ yếu văn viết, bài thi) hoặc
    neutral (dùng chung). Không chắc thì null.
  * usage_note_vi: tối đa hai câu tiếng Việt về cách dùng với ai, ở đâu, và lỗi
    người Việt hay mắc với từ này. Không có gì đáng nói thì null.
  Điều gì không chắc thì để null hoặc bỏ trống và đặt confidence không quá 0,7;
  không bịa.
- Nếu có phần "Nhóm của các mục", mục thuộc nhóm nào thì viết thẻ nhất quán với các
  mục cùng nhóm (ví dụ các cụm cùng nhóm không lặp lại nhau trong collocations).
Trả về đúng JSON schema đã cho."""

_GRAMMAR_CARD_RULES = """Quy tắc:
- Trả về ĐÚNG MỘT thẻ cho MỖI mẫu trong danh sách, mang `ref` của mẫu đó (chép y
  nguyên). Không bỏ mẫu nào, không thêm mẫu nào.
- pattern: viết dạng "V/A + hình thái" (ví dụ "V + -(으)ㄹ 뿐만 아니라").
- Nghĩa và ví dụ: lấy từ tài liệu; mẫu mà tài liệu chỉ nhắc ngắn thì tự bổ sung
  bằng kiến thức của bạn và đặt confidence không quá 0,7.
- QUAN TRỌNG: đừng chỉ ghi công thức như sách giáo khoa. Viết usage_context_vi
  bằng tiếng Việt: dùng khi nào, trong hoàn cảnh nào, với sắc thái hay thái độ gì
  so với các mẫu gần nghĩa (người học cần biết ÁP DỤNG chứ không chỉ nhớ công thức).
- Nếu mẫu hay gặp trong đề thi TOPIK, thêm topik_tip_vi: viết như MỘT NHẬN XÉT
  VỀ NGÔN NGỮ (câu khẳng định: mẫu này hay xuất hiện ở dạng bài hay ngữ cảnh
  nào, dấu hiệu nhận biết là gì), KHÔNG viết như lời ra lệnh hay lời dặn làm bài.
- Đề TOPIK hay hỏi "chọn câu có nghĩa gần nhất", nên mẫu thuộc một nhóm đối chiếu
  (xem phần "Nhóm của các mục") phải học cùng các mẫu anh em của nó. Với mẫu như
  vậy điền contrasts: MỖI mẫu khác trong nhóm một dòng gồm pattern (chép y nguyên
  từ danh sách nhóm) và diff_vi (khác mẫu này ở đâu: sắc thái, thời điểm, chủ ngữ,
  văn nói hay văn viết). Mẫu không thuộc nhóm nào thì contrasts để trống.
Trả về đúng JSON schema đã cho."""

ORGANISE_PROMPT = """Bạn là giảng viên luyện thi TOPIK cho người Việt. Tài liệu bài học đính kèm đã được
lập danh mục từ vựng (v1, v2...) và ngữ pháp (g1, g2...). Việc của bạn là xếp
chúng thành NHÓM để người học học theo nhóm chứ không học từng mục rời, vì đề thi
TOPIK kiểm tra từ nào đi với từ nào và mẫu ngữ pháp nào gần nghĩa với mẫu nào.

Họ từ vựng (vocab_families): gom những mục đi cùng nhau, ví dụ (chỉ để minh họa):
- cùng một kiểu kết hợp: 비가 오다, 눈이 오다, 바람이 불다 (động từ đi với hiện tượng thời tiết);
- cùng một thang hoặc một cặp đối nghĩa: 덥다, 따뜻하다, 쌀쌀하다, 춥다 (thang nhiệt độ);
- cùng một tình huống nhỏ trong bài (mọi thứ gắn với một mùa, một hoạt động).
Mỗi họ có từ 2 đến 8 mục và một `label` tiếng Việt ngắn nói rõ các mục có gì chung.
Mục nào không hợp với họ nào thì để ngoài, đừng ép.

Nhóm đối chiếu ngữ pháp (grammar_groups): chỉ gom các mẫu dễ bị nhầm với nhau vì gần
nghĩa hoặc gần hình thức (ví dụ -(으)ㄹ 것 같다 và -(으)ㄴ/는 것 같다). Mỗi nhóm có
từ 2 mẫu trở lên và một `label` tiếng Việt nói rõ điểm chung. Mẫu nào không có
"anh em" thì để ngoài.

Quy tắc: chỉ dùng ref có trong danh sách và chép y nguyên; mỗi ref thuộc nhiều nhất
một nhóm; không thêm ref, không đổi ref. Trả về đúng JSON schema đã cho."""

OVERVIEW_PROMPT = """Bạn là biên tập viên nội dung học tiếng Hàn cho người Việt. Đọc tài liệu bài học
đính kèm và viết lại thành `content`: văn bản thuần tiếng Việt (xen từ tiếng Hàn khi
cần) nêu đủ ý chính của bài — chủ đề, các nhóm từ, các mẫu ngữ pháp được dạy, và
ghi chú văn hóa hay cách dùng thực tế nếu có. Không dùng markdown, không giữ số
trích dẫn kiểu [1, 3]. Trả về đúng JSON schema đã cho."""


def organise_prompt(terms: list[str], patterns: list[str]) -> str:
    """The numbered inventory, for the call that sorts it into families / groups."""
    vocab = "\n".join(f"v{i + 1}. {t}" for i, t in enumerate(terms)) or "(không có)"
    grammar = "\n".join(f"g{i + 1}. {p}" for i, p in enumerate(patterns)) or "(không có)"
    return f"{ORGANISE_PROMPT}\n\nDanh sách từ vựng:\n{vocab}\n\nDanh sách ngữ pháp:\n{grammar}\n"


def cards_prompt(
    kind: str,
    title: str,
    refs: list[tuple[str, str]],
    retry: bool = False,
    groups: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Instructions + the numbered items of one group. `groups` is the families
    (vocab) or contrast groups (grammar) these items belong to, as (label,
    every member of that set) — the whole set, so a card written in a call that
    holds only part of it still knows its group-mates."""
    rules = _VOCAB_CARD_RULES if kind == "vocab" else _GRAMMAR_CARD_RULES
    what = "từ vựng" if kind == "vocab" else "ngữ pháp"
    lines = "\n".join(f"{ref}. {term}" for ref, term in refs)
    again = (
        "\nLần trước các mục này bị bỏ sót; hãy viết thẻ cho từng mục dưới đây.\n" if retry else "\n"
    )
    sets = ""
    if groups:
        listed = "\n".join(f"- {label}: {' | '.join(members)}" for label, members in groups)
        sets = f"Nhóm của các mục (cả nhóm, kể cả mục không nằm trong lần viết này):\n{listed}\n\n"
    return (
        "Bạn là biên tập viên nội dung học tiếng Hàn cho người Việt. Tài liệu bài học đính kèm đã được "
        f"lập danh mục; việc của bạn là viết thẻ {what} cho các mục dưới đây.\n\n"
        f"{rules}\n{again}"
        f"Chủ đề bài học: {title}\n"
        f"Các mục cần viết thẻ:\n{lines}\n\n"
        f"{sets}"
    )


# ---------------------------------------------------------- term cleaning --
def term_key(term: str) -> str:
    """What two spellings of one item share: spaces and punctuation ignored."""
    return _KEY_STRIP_RE.sub("", term).lower()


def _clean_term(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip(" \t.,:;!?\"'“”‘’•*_")


def clean_terms(terms: list[str], *, need_hangul: bool = True) -> list[str]:
    """Trim, drop empties and duplicates (by `term_key`), keep order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in terms:
        term = _clean_term(str(raw))
        if not term or (need_hangul and not _HANGUL_RE.search(term)):
            continue
        key = term_key(term)
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def bold_terms(text: str) -> list[str]:
    """The Korean words and phrases a plain-text lesson prints in **bold**.

    An author bolds what is being taught, so these are candidates the model must
    not be able to lose. Grammar formulas (they start with "-" or "~") are left
    to the grammar list; a gloss in parentheses is dropped; "A / B" becomes two
    terms."""
    found: list[str] = []
    for segment in _BOLD_RE.findall(text or ""):
        segment = _NUMBERING_RE.sub("", segment).strip()
        if not segment or segment[0] in "-~–—" or "+" in segment:
            continue
        segment = _PAREN_RE.sub("", segment)
        for part in re.split(r"\s*/\s*", segment):
            part = _clean_term(part)
            if not part or part[0] in "-~" or not _HANGUL_RE.search(part):
                continue
            if len(part.split()) > 5:  # a sentence, not a term
                continue
            found.append(part)
    return clean_terms(found)


def merge_terms(model_terms: list[str], text_terms: list[str]) -> list[str]:
    """The model's list, then every term the text itself marks that the model
    did not list (matched ignoring spaces and punctuation)."""
    merged = clean_terms(model_terms)
    have = {term_key(t) for t in merged}
    for term in text_terms:
        if term_key(term) not in have:
            merged.append(term)
            have.add(term_key(term))
    return merged[:MAX_TERMS]


# ------------------------------------------------------------------- result --
class Collocation(BaseModel):
    ko: str
    vi: str


class Contrast(BaseModel):
    pattern: str
    diff_vi: str


class VocabCard(BaseModel):
    hangul: str
    pos: str | None = None
    meaning_vi: str
    definition_ko: str | None = None
    level: int = 1
    hanja: str | None = None
    sino_vietnamese: str | None = None
    example_ko: str | None = None
    # the chunk layers (lesson-v4); all optional, so an old payload still loads
    family: str | None = None  # set by code from the ORGANISE step, never by the card call
    node_word: str | None = None
    register: str | None = None
    usage_note_vi: str | None = None
    collocations: list[Collocation] | None = None
    distractors: list[str] | None = None
    confidence: float = 0.5


class GrammarCard(BaseModel):
    pattern: str
    meaning_vi: str
    level: int = 1
    example_ko: str | None = None
    usage_context_vi: str | None = None
    topik_tip_vi: str | None = None
    contrast_group: str | None = None  # set by code from the ORGANISE step
    contrasts: list[Contrast] | None = None
    confidence: float = 0.5


@dataclass
class LessonDraft:
    title: str
    level: int
    content: str
    topics: list[str]
    confidence: float
    vocab: list[VocabCard] = field(default_factory=list)
    grammar: list[GrammarCard] = field(default_factory=list)
    # items the model never produced a card for even after a second ask; they are
    # in `vocab` / `grammar` as empty cards with confidence 0.2 (shown in red)
    missing: list[str] = field(default_factory=list)


def _clamp_level(value: Any, default: int = 1) -> int:
    try:
        return min(6, max(1, int(value)))
    except (TypeError, ValueError):
        return default


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str | None:
    """A trimmed one-line string, or None for anything else (or empty)."""
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def tidy_vocab(data: dict[str, Any]) -> None:
    """Keeps the chunk layers only where they make sense. The model is told not
    to invent, but a node word that is not part of the phrase, an unknown
    register or a distractor equal to the answer would each break a card or an
    exercise built from it, so they are dropped here rather than trusted."""
    hangul = _text(data.get("hangul")) or ""
    node = _text(data.get("node_word"))
    if node is not None and not (node in hangul and node != hangul):
        node = None
    data["node_word"] = node
    data["family"] = (_text(data.get("family")) or "")[:MAX_LABEL] or None

    register = (_text(data.get("register")) or "").lower()
    data["register"] = register if register in REGISTERS else None
    data["usage_note_vi"] = _text(data.get("usage_note_vi"))

    collocations: list[dict[str, str]] = []
    seen = {term_key(hangul)}
    raw_collocations = data.get("collocations")
    for entry in raw_collocations if isinstance(raw_collocations, list) else []:
        if not isinstance(entry, dict):
            continue
        ko, vi = _text(entry.get("ko")), _text(entry.get("vi"))
        if not ko or not vi or not _HANGUL_RE.search(ko) or term_key(ko) in seen:
            continue
        seen.add(term_key(ko))
        collocations.append({"ko": ko, "vi": vi})
        if len(collocations) == MAX_COLLOCATIONS:
            break
    data["collocations"] = collocations or None

    distractors: list[str] = []
    seen = {term_key(node or ""), term_key(hangul)}
    raw_distractors = data.get("distractors")
    for entry in raw_distractors if isinstance(raw_distractors, list) and node else []:
        word = _text(entry)
        if not word or not _HANGUL_RE.search(word) or term_key(word) in seen:
            continue
        seen.add(term_key(word))
        distractors.append(word)
        if len(distractors) == MAX_DISTRACTORS:
            break
    data["distractors"] = distractors or None


def tidy_grammar(data: dict[str, Any]) -> None:
    """Contrasts keep their shape here; which sibling each one names is checked
    once every card of the group exists (`_link_contrasts`)."""
    data["contrast_group"] = (_text(data.get("contrast_group")) or "")[:MAX_LABEL] or None
    contrasts: list[dict[str, str]] = []
    raw = data.get("contrasts")
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        pattern, diff = _text(entry.get("pattern")), _text(entry.get("diff_vi"))
        if pattern and diff:
            contrasts.append({"pattern": pattern, "diff_vi": diff})
    data["contrasts"] = contrasts or None


def _card_from_raw(kind: str, raw: dict[str, Any]) -> BaseModel | None:
    """One card from the model's answer, tolerant of a stray level or
    confidence (one bad number must not cost the whole group). None when the
    mandatory text is missing."""
    data = dict(raw)
    data.pop("ref", None)
    # the sets are decided by the ORGANISE step, whatever the card call says
    data.pop("family", None)
    data.pop("contrast_group", None)
    data["level"] = _clamp_level(data.get("level"))
    data["confidence"] = _clamp_confidence(data.get("confidence"))
    if kind == "vocab":
        tidy_vocab(data)
    else:
        tidy_grammar(data)
    try:
        return VocabCard.model_validate(data) if kind == "vocab" else GrammarCard.model_validate(data)
    except ValidationError:
        return None


def _default_generate(**kwargs: Any) -> dict[str, Any]:
    return gemini_client.generate_structured(**kwargs)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    """Gemini's JSON mode is normally bare JSON; a stray ``` fence is stripped."""
    data = json.loads(_JSON_FENCE_RE.sub("", result["text"].strip()))
    if not isinstance(data, dict):
        raise ValueError("Gemini returned JSON that is not an object")
    return data


def _ask(generate: Generate, document: Any, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    result = generate(
        model=settings.GEMINI_MODEL_LESSON_INGEST,
        prompt=[document, prompt],
        response_schema=schema,
        prompt_version=PROMPT_VERSION,
    )
    return _json(result)


# A set: (label, indexes of its members into the term list, ascending).
Groups = list[tuple[str, list[int]]]


def parse_groups(data: dict[str, Any], key: str, prefix: str, count: int) -> Groups:
    """The families / contrast groups the ORGANISE call answered with, kept only
    where they hold up: refs must exist, a ref belongs to the first set that
    names it, a set needs a label and at least two members and is cut at
    MAX_FAMILY. Anything else is ignored — a bad set costs that set, not the
    lesson."""
    groups: Groups = []
    used: set[int] = set()
    raw_groups = data.get(key)
    for raw in raw_groups if isinstance(raw_groups, list) else []:
        if not isinstance(raw, dict):
            continue
        label = (_text(raw.get("label")) or "").strip(" .:;-")[:MAX_LABEL]
        members: list[int] = []
        refs = raw.get("refs")
        for ref in refs if isinstance(refs, list) else []:
            match = re.fullmatch(rf"{prefix}(\d+)", str(ref).strip().lower())
            if match is None:
                continue
            index = int(match.group(1)) - 1
            if 0 <= index < count and index not in used and index not in members:
                members.append(index)
        members = sorted(members)[:MAX_FAMILY]
        if not label or len(members) < 2:
            continue
        used.update(members)
        groups.append((label, members))
    return groups


def cluster_order(count: int, groups: Groups) -> list[int]:
    """The order that keeps the lesson's own order but puts the members of a set
    side by side, at the place its first member had."""
    group_of = {i: g for g, (_label, members) in enumerate(groups) for i in members}
    order: list[int] = []
    placed: set[int] = set()
    for i in range(count):
        if i in placed:
            continue
        g = group_of.get(i)
        for j in groups[g][1] if g is not None else [i]:
            order.append(j)
            placed.add(j)
    return order


def _organise(generate: Generate, document: Any, terms: list[str], patterns: list[str]) -> tuple[Groups, Groups]:
    """(vocab families, grammar contrast groups). A failed call means no sets."""
    try:
        data = _ask(generate, document, organise_prompt(terms, patterns), ORGANISE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - the sets are a layer on top of the cards, never a reason to fail
        print(f"[lesson] organise failed: {str(exc)[:200]}", flush=True)
        return [], []
    return (
        parse_groups(data, "vocab_families", "v", len(terms)),
        parse_groups(data, "grammar_groups", "g", len(patterns)),
    )


def _link_contrasts(cards: list[BaseModel], terms: list[str], labels: list[str | None]) -> None:
    """Points each contrast at the group-mate it names. The model was shown the
    mates' inventory strings and told to copy one; a name that matches none of
    them is dropped, except in a pair, where there is only one mate it can mean."""
    for i, card in enumerate(cards):
        contrasts = getattr(card, "contrasts", None)
        if not contrasts:
            continue
        mates = [j for j in range(len(cards)) if j != i and labels[i] is not None and labels[j] == labels[i]]
        linked: list[Contrast] = []
        used: set[int] = set()
        own = term_key(terms[i])
        for contrast in contrasts:
            key = term_key(contrast.pattern)
            if key == own or key == term_key(getattr(card, "pattern", "")):
                continue  # a pattern is not its own look-alike
            hit = next((j for j in mates if j not in used and term_key(terms[j]) == key), None)
            if hit is None:
                hit = next(
                    (j for j in mates if j not in used and len(key) >= 3 and (key in term_key(terms[j]) or term_key(terms[j]) in key)),
                    None,
                )
            if hit is None and len(mates) == 1 and len(contrasts) == 1:
                hit = mates[0]
            if hit is None:
                continue
            used.add(hit)
            linked.append(Contrast(pattern=getattr(cards[hit], "pattern", "") or terms[hit], diff_vi=contrast.diff_vi))
        card.contrasts = linked or None  # type: ignore[attr-defined]


def _write_cards(
    generate: Generate,
    document: Any,
    kind: str,
    title: str,
    level: int,
    terms: list[str],
    tick: Callable[[], None] | None = None,
    labels: list[str | None] | None = None,
) -> tuple[list[BaseModel], list[str], int]:
    """Cards for `terms`, in the order of `terms`. Returns (cards, missing_terms,
    failed_calls). A card for a missing term is an empty red placeholder, so the
    returned list always lines up with `terms`. `labels[i]` is the family (vocab)
    or contrast group (grammar) of `terms[i]`, if it has one."""
    prefix = "v" if kind == "vocab" else "g"
    schema = VOCAB_CARDS_SCHEMA if kind == "vocab" else GRAMMAR_CARDS_SCHEMA
    size = VOCAB_GROUP if kind == "vocab" else GRAMMAR_GROUP
    sets = labels or [None] * len(terms)
    by_term: dict[int, BaseModel] = {}
    failed_calls = 0

    def context(indexes: list[int]) -> list[tuple[str, list[str]]]:
        """The whole set of every family these items belong to."""
        out: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        for i in indexes:
            label = sets[i]
            if label is not None and label not in seen:
                seen.add(label)
                out.append((label, [terms[j] for j in range(len(terms)) if sets[j] == label]))
        return out

    def run(indexes: list[int], retry: bool) -> None:
        nonlocal failed_calls
        refs = [(f"{prefix}{i + 1}", terms[i]) for i in indexes]
        try:
            prompt = cards_prompt(kind, title, refs, retry, context(indexes) or None)
            data = _ask(generate, document, prompt, schema)
        except Exception as exc:  # noqa: BLE001 - a failed group is retried once, then reported
            failed_calls += 1
            print(f"[lesson] {kind} group failed: {str(exc)[:200]}", flush=True)
            return
        wanted = {ref: i for ref, i in zip((r for r, _ in refs), indexes)}
        for raw in data.get("items") or []:
            if not isinstance(raw, dict):
                continue
            i = wanted.get(str(raw.get("ref", "")).strip())
            if i is None or i in by_term:
                continue
            card = _card_from_raw(kind, raw)
            if card is not None:
                by_term[i] = card

    for start in range(0, len(terms), size):
        run(list(range(start, min(start + size, len(terms)))), retry=False)
        if tick:
            tick()
    # whatever came back short is asked for again, in groups of the same size
    left = [i for i in range(len(terms)) if i not in by_term]
    for start in range(0, len(left), size):
        run(left[start : start + size], retry=True)

    missing: list[str] = []
    cards: list[BaseModel] = []
    for i, term in enumerate(terms):
        card = by_term.get(i)
        if card is None:
            missing.append(term)
            placeholder = {"meaning_vi": "", "level": level, "confidence": 0.2}
            card = (
                VocabCard(hangul=term, **placeholder)
                if kind == "vocab"
                else GrammarCard(pattern=term, usage_context_vi="", **placeholder)
            )
        cards.append(card)
    for i, card in enumerate(cards):
        if kind == "vocab":
            card.family = sets[i]  # type: ignore[attr-defined]
        else:
            card.contrast_group = sets[i]  # type: ignore[attr-defined]
    if kind == "grammar":
        _link_contrasts(cards, terms, sets)
    return cards, missing, failed_calls


def extract(
    file_bytes: bytes,
    mime_type: str,
    *,
    generate: Generate | None = None,
    fallback: Callable[[], LessonDraft] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> LessonDraft:
    """Read a lesson (image, PDF or plain text) into cards, one per item it
    teaches. `fallback` (the old one-call extraction) is used only when the
    inventory finds nothing at all."""
    gen: Generate = generate or _default_generate
    document = gemini_client.part_from_bytes(file_bytes, mime_type)

    inventory = _ask(gen, document, INVENTORY_PROMPT, INVENTORY_SCHEMA)
    text_terms: list[str] = []
    if mime_type.startswith("text/"):
        text_terms = bold_terms(file_bytes.decode("utf-8", errors="ignore"))
    listed = clean_terms([str(t) for t in inventory.get("vocab_terms") or []])
    terms = merge_terms(listed, text_terms)
    patterns = clean_terms([str(p) for p in inventory.get("grammar_patterns") or []], need_hangul=False)
    title = str(inventory.get("title") or "").strip() or "Bài học"
    level = _clamp_level(inventory.get("level"))
    print(
        f"[lesson] inventory vocab={len(terms)} (listed by the model {len(listed)}, "
        f"added from the text's bold {len(terms) - len(listed)}) grammar={len(patterns)}",
        flush=True,
    )

    if not terms and not patterns:
        if fallback is not None:
            print("[lesson] empty inventory; using the one-call extraction", flush=True)
            return fallback()
        raise RuntimeError("Không tìm thấy từ vựng hay ngữ pháp nào trong tài liệu")

    calls = -(-len(terms) // VOCAB_GROUP) + -(-len(patterns) // GRAMMAR_GROUP)
    organise = len(terms) >= 2 or len(patterns) >= 2  # nothing to group in a one-word lesson
    total = calls + 2 + (1 if organise else 0)  # + inventory, overview, organise
    done = 0

    def report(step: str) -> None:
        if on_progress:
            on_progress(min(done, total), total, step)

    def tick() -> None:
        nonlocal done
        done += 1
        report(f"Đang viết thẻ ({min(done - (2 if organise else 1), calls)}/{calls})")

    done = 1  # the inventory
    report(f"Đã lập danh mục: {len(terms)} từ, {len(patterns)} mẫu ngữ pháp")

    vocab_labels: list[str | None] = [None] * len(terms)
    grammar_labels: list[str | None] = [None] * len(patterns)
    if organise:
        families, contrast_groups = _organise(gen, document, terms, patterns)
        # members of a set sit side by side, so a day's new cards come as a set
        term_order = cluster_order(len(terms), families)
        pattern_order = cluster_order(len(patterns), contrast_groups)
        for label, members in families:
            for i in members:
                vocab_labels[i] = label
        for label, members in contrast_groups:
            for i in members:
                grammar_labels[i] = label
        terms = [terms[i] for i in term_order]
        vocab_labels = [vocab_labels[i] for i in term_order]
        patterns = [patterns[i] for i in pattern_order]
        grammar_labels = [grammar_labels[i] for i in pattern_order]
        done = 2
        print(
            f"[lesson] organised: {len(families)} families, {len(contrast_groups)} contrast groups",
            flush=True,
        )
        report(f"Đã xếp {len(families)} họ cụm và {len(contrast_groups)} nhóm ngữ pháp")

    vocab, vocab_missing, vocab_failed = _write_cards(gen, document, "vocab", title, level, terms, tick, vocab_labels)
    grammar, grammar_missing, grammar_failed = _write_cards(
        gen, document, "grammar", title, level, patterns, tick, grammar_labels
    )
    if calls and vocab_failed + grammar_failed >= calls * 2:
        # every group failed twice: a model/quota outage, not a lesson problem
        raise RuntimeError("Gemini không viết được thẻ nào cho bài học này, thử lại sau")

    try:
        content = str(_ask(gen, document, OVERVIEW_PROMPT, OVERVIEW_SCHEMA).get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001 - the overview is a nicety; the cards matter more
        print(f"[lesson] overview failed: {str(exc)[:200]}", flush=True)
        content = ""

    missing = vocab_missing + grammar_missing
    if missing:
        print(f"[lesson] no card for {len(missing)} item(s): {missing[:10]}", flush=True)
    return LessonDraft(
        title=title,
        level=level,
        content=content,
        topics=clean_terms([str(t) for t in inventory.get("topics") or []], need_hangul=False),
        confidence=_clamp_confidence(inventory.get("confidence")),
        vocab=vocab,  # type: ignore[arg-type]
        grammar=grammar,  # type: ignore[arg-type]
        missing=missing,
    )
