"""Exam-paper extraction, version 2: copy what is printed, check it twice, never guess an answer.

What went wrong with the first version (exam-v1), read off a real paper (TOPIK II
102회 읽기): the whole paper went to the model in ONE call and came back as a single
`stem_ko` string per question. The model then wrote whatever it liked into that
string — the group instruction instead of the question, the first sentence of the
passage, a banner's text — and it paraphrased ("고르십시오" became "고르시오",
"우표 박물관" became "우체 박물관"). There was no answer key, so it guessed one, and
because a guessed answer is always flagged, EVERY card was red and the flag meant
nothing.

So:

1. PAGES, NOT THE WHOLE FILE. A PDF is split into single pages and the model reads
   a window of a few consecutive pages at a time (WINDOW_PAGES, overlapping, so a
   passage and its questions that straddle a page break are whole in some window).
   Small windows keep each request far below Gemini's inline-size limit — a 14 MB
   scan as one request is not safe — and keep the model from getting lazy over a
   long answer.
2. THE QUESTION IS SPLIT INTO ITS PARTS: the group instruction ("[9~12] 다음 글 또는
   도표의 내용과 같은 것을 고르십시오."), the line printed beside the number, the
   passage (shared by the questions that use it, with <보기> and ㉠–㉣ kept), the
   options. Underlined words are kept as <u>…</u>. The prompt forbids paraphrasing.
3. TWO READS. The pages are read a second time, with the windows shifted, and the
   two results are compared character by character. A difference is not corrected —
   it is SHOWN: the card says which words differ and what each read saw.
4. NO ANSWER IS INVENTED. Answers come only from a printed answer key (a table of
   번호/정답 on the pages, a separate key file, or numbers an editor types). Until
   then a question has no answer and is simply not offered for practice; a missing
   key does not make anything red.
5. FLAGS HAVE REASONS. Each question or passage carries the reasons it needs a look
   (the two reads disagree, the passage is withheld, a word marked unreadable ...),
   and only the serious ones turn it red.

Nothing here touches the database; ingestion.stage_exam_draft stages the result.
"""
from __future__ import annotations

import difflib
import io
import json
import re
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.services import exam_drill, gemini_client

PROMPT_VERSION = "exam-v2"

WINDOW_PAGES = 5  # consecutive pages the model reads in one call
WINDOW_STRIDE = 3  # the next window starts this many pages later (overlap = 2)
VERIFY_PHASE = 2  # the second read starts its windows here, so they cut the paper elsewhere
MAX_PAGES = 80  # a sanity cap: a TOPIK paper is ~25 pages
WORKERS = 3  # windows read at the same time
OPTION_COUNT = 4  # every TOPIK multiple-choice question has four choices
KEY_WINDOW_PAGES = 6  # an answer key is a page or three; longer ones are cut into groups of this many
PASSAGE_SIMILARITY = 0.9  # two readings of a passage this alike are the same passage
UNREADABLE = "[?]"
LOW_CONFIDENCE = 0.8

SKILL_BY_SECTION = {"읽기": "đọc", "듣기": "nghe", "쓰기": "viết"}

# ------------------------------------------------------------------- schemas --
EXAM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "paper_section": {
            "type": "STRING",
            "nullable": True,
            "description": "phần thi in ở bìa/đầu trang: '읽기', '듣기' hoặc '쓰기'; null nếu các trang này không ghi",
        },
        "passages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "local_ref": {"type": "STRING", "description": "id tạm, vd 'P1', để câu hỏi tham chiếu"},
                    "kind": {"type": "STRING", "enum": ["đọc hiểu", "nghe", "biểu đồ"]},
                    "body_ko": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "toàn văn chép nguyên văn; null nếu bị che bản quyền",
                    },
                    "withheld": {"type": "BOOLEAN", "description": "true nếu đề in thông báo không công bố đoạn văn"},
                    "page": {"type": "INTEGER", "description": "trang (1..n trong tập này) nơi đoạn văn bắt đầu"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["local_ref", "kind", "withheld", "page", "confidence"],
            },
        },
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "number": {"type": "INTEGER", "description": "số câu in trong đề"},
                    "passage_ref": {"type": "STRING", "nullable": True},
                    "qtype_code": {"type": "STRING", "nullable": True},
                    "instruction_ko": {"type": "STRING", "nullable": True},
                    "group_from": {"type": "INTEGER", "nullable": True},
                    "group_to": {"type": "INTEGER", "nullable": True},
                    "stem_ko": {"type": "STRING", "description": "chữ in ngay sau số câu; chuỗi rỗng nếu không có"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "answer_guess": {"type": "INTEGER", "nullable": True},
                    "page": {"type": "INTEGER"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["number", "stem_ko", "options", "page", "confidence"],
            },
        },
        "answer_key": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "section": {"type": "STRING", "nullable": True},
                    "number": {"type": "INTEGER"},
                    "answer": {"type": "INTEGER"},
                },
                "required": ["number", "answer"],
            },
        },
    },
    "required": ["passages", "items", "answer_key"],
}

KEY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "sections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "section": {"type": "STRING", "description": "'읽기', '듣기' hoặc '쓰기' (dòng 영역 của bảng)"},
                    "answers": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"number": {"type": "INTEGER"}, "answer": {"type": "INTEGER"}},
                            "required": ["number", "answer"],
                        },
                    },
                },
                "required": ["section", "answers"],
            },
        }
    },
    "required": ["sections"],
}


# ------------------------------------------------------------------- prompts --
def build_prompt(page_count: int, qtypes: list[tuple[str, str, str]]) -> str:
    qtype_hint = "\n".join(f"- {code} ({skill}): {name_vi}" for code, name_vi, skill in qtypes) or (
        "(chưa có loại câu hỏi nào — để qtype_code là null)"
    )
    return f"""Bạn đang số hóa đề thi TOPIK thành ngân hàng câu hỏi học thuật. Người học sẽ tin
những gì bạn chép, nên chép ĐÚNG TỪNG CHỮ quan trọng hơn mọi thứ khác.

Các tệp đính kèm là {page_count} trang LIÊN TIẾP của đề (tệp đầu tiên là trang 1). Chỉ
trích xuất những gì in trên các trang này.

QUY TẮC CHÉP
1. Chép NGUYÊN VĂN từng ký tự như in. Không diễn đạt lại, không tóm tắt, không sửa
   chính tả, không đổi đuôi câu (giữ "고르십시오", không viết "고르시오"), không thêm
   hay bớt dấu câu. Không dịch.
2. Không đoán chữ. Nhìn kỹ từng nét: các chữ gần giống nhau (우표/우체, 낳/났) rất dễ
   nhầm. Chỗ mờ hoặc không chắc → ghi [?] ngay tại chỗ đó và hạ confidence xuống
   dưới 0.6.
3. Phần bị GẠCH CHÂN trong đề → bọc bằng <u>…</u>, đúng đoạn bị gạch, vd:
   가을에 부는 바람이 <u>시원하다</u>.
4. Chỗ trống in là ( ) → giữ "( )". Các ký hiệu ㉠ ㉡ ㉢ ㉣, (가) (나) (다) (라), <보기>
   giữ nguyên.
5. Phương án: chỉ ghi nội dung, KHÔNG ghi ①②③④ ở đầu; giữ thứ tự. Phương án là hình
   ảnh → "[hình]".
6. Khung quảng cáo / biển báo / biểu đồ / bảng: chép MỌI chữ và số in trong khung theo
   thứ tự đọc. Bảng: mỗi hàng một dòng, các ô cách nhau bằng " | ". Biểu đồ: mỗi nhãn
   kèm số liệu, vd "가격: 48%". Chỉ chép những gì in, không diễn giải.

CÁCH TÁCH
- instruction_ko: dòng chỉ dẫn CHUNG của nhóm, in ở đầu nhóm câu, vd
  "※ [9~12] 다음 글 또는 도표의 내용과 같은 것을 고르십시오. (각 2점)". Chép nguyên văn nhưng
  bỏ "※" ở đầu và "(각 2점)" ở cuối; giữ "[9~12]". MỌI câu trong nhóm đều mang cùng
  instruction_ko (lặp lại cho từng câu). Nếu nhóm bắt đầu ở trang không có trong tập
  này thì để null. group_from / group_to là hai số trong [ ].
- stem_ko: chữ in ngay sau số câu. Vd câu 1: "이 동네로 이사를 ( ) 일 년이 됐다." Nếu sau
  số câu không có chữ (câu chỉ gồm khung quảng cáo/biểu đồ rồi đến phương án) → "".
  Đừng đưa chữ trong khung quảng cáo hay đoạn văn vào stem_ko.
- passages: đoạn văn / bài đọc / khung quảng cáo / biển báo / biểu đồ / các câu (가)~(라)
  cần sắp xếp — phần nằm giữa chỉ dẫn và các phương án mà câu hỏi dựa vào. Mỗi đoạn một
  local_ref ("P1", "P2"...) DÙNG CHUNG cho mọi câu dựa vào nó, không chép lại cùng một
  đoạn nhiều lần. Dạng <보기>: body_ko bắt đầu bằng dòng "<보기>", rồi câu trong khung
  <보기>, rồi đoạn văn có ㉠~㉣.
- Nếu đề in thông báo đoạn văn KHÔNG được công bố vì bản quyền (vd "저작권 관련 법령에
  따라 본 문항의 지문은 공개하지 않습니다", "NOTICE ... NOT disclosed") → tạo passage đó
  với withheld=true, body_ko=null; câu hỏi của nhóm vẫn trích xuất bình thường.
- Câu hỏi bị cắt ở mép tập trang (thiếu phương án, hoặc đoạn văn nằm ở trang không có
  ở đây) → vẫn trích xuất phần thấy được và đặt confidence dưới 0.5; hệ thống sẽ lấy
  bản đầy đủ từ lượt đọc khác.
- Trang trắng, bìa, trang "유의사항", tiêu đề phần thi: bỏ qua. paper_section = phần thi in
  trên bìa/đầu trang (읽기, 듣기 hoặc 쓰기), null nếu không thấy.
- Nếu trang là BẢNG ĐÁP ÁN (정답표: cột 번호 / 정답 / 배점) → KHÔNG tạo câu hỏi; ghi mọi
  cặp (section, number, answer) vào answer_key (① ② ③ ④ → 1 2 3 4). Nếu không phải bảng
  đáp án thì answer_key là [].

ĐÁP ÁN: KHÔNG tự giải đề. answer_guess để null trừ khi bạn thật sự chắc chắn (nó chỉ là
gợi ý cho người duyệt, không bao giờ được dùng làm đáp án).

qtype_code CHỈ chọn trong danh sách sau; null nếu không khớp (không tự đặt mã mới):
{qtype_hint}

Trả về đúng JSON schema, không thêm lời giải thích. confidence (0-1) phản ánh việc bạn chắc
chắn đã chép đúng từng chữ."""


def build_key_prompt(page_count: int) -> str:
    return f"""Các tệp đính kèm là {page_count} trang BẢNG ĐÁP ÁN chính thức (정답표) của một đề thi TOPIK.

Mỗi bảng thuộc một phần thi, ghi ở dòng "수준 : TOPIK II  영역 : 읽기" (읽기 / 듣기 / 쓰기).
Với mỗi bảng đáp án TRẮC NGHIỆM, trả về section và mọi cặp (번호, 정답).
- Bảng chia thành hai nhóm cột (vd 1–25 bên trái, 26–50 bên phải): đọc CẢ HAI nhóm.
- ① ② ③ ④ → 1 2 3 4. Bỏ qua cột 배점 (điểm).
- Bảng đáp án mẫu của phần viết (쓰기, 서답형, câu 51~54 là đoạn văn) → bỏ qua.
- Chép đúng như in. Ô không đọc rõ thì bỏ cặp đó, đừng suy luận và đừng điền cho đủ số.

Trả về đúng JSON schema, không thêm giải thích."""


# ----------------------------------------------------------------- text tools --
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CIRCLED = "①②③④⑤"
_CIRCLED_PREFIX_RE = re.compile(rf"^\s*[{_CIRCLED}]\s*")
_SCORE_SUFFIX_RE = re.compile(r"\s*\(\s*각\s*\d+\s*점\s*\)\s*$")
_GROUP_RE = re.compile(r"\[\s*(\d+)\s*[~∼～\-–]\s*(\d+)\s*\]")
_UNDERLINE_TAG_RE = re.compile(r"</?u>", re.IGNORECASE)


def _txt(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def _int(value: Any, low: int | None = None, high: int | None = None) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if (low is not None and number < low) or (high is not None and number > high):
        return None
    return number


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def tidy_text(value: Any, *, multiline: bool = False) -> str:
    """The model's text with its markup made uniform: <u> in lower case, no empty
    underline, no runs of blank space. Wording is never touched."""
    text = _txt(value)
    text = _UNDERLINE_TAG_RE.sub(lambda m: m.group(0).lower(), text)
    text = text.replace("<u></u>", "")
    if multiline:
        lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    else:
        text = re.sub(r"\s+", " ", text)
    return text


def underline_balanced(text: str) -> bool:
    return len(re.findall(r"<u>", text)) == len(re.findall(r"</u>", text))


def squash(text: str | None) -> str:
    """The form two readings are compared in: no white space (line breaks and
    spacing are layout, not content). Every other character counts, <u> included."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text or ""))


def clean_option(value: Any) -> str:
    return _CIRCLED_PREFIX_RE.sub("", tidy_text(value))


def clean_instruction(value: Any) -> str | None:
    text = tidy_text(value).lstrip("※").strip()
    text = _SCORE_SUFFIX_RE.sub("", text).strip()
    return text or None


def group_of(instruction: str | None) -> tuple[int, int] | None:
    match = _GROUP_RE.search(instruction or "")
    if match is None:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    return (low, high) if low <= high else None


def _json(result: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(_JSON_FENCE_RE.sub("", result["text"].strip()))
    if not isinstance(data, dict):
        raise ValueError("Gemini returned JSON that is not an object")
    return data


def span_diffs(a: str, b: str, *, context: int = 4, limit: int = 3) -> list[tuple[str, str]]:
    """Where two readings of the same text differ: (what read A saw, what read B
    saw) around each difference, found on the white-space-free text."""
    sa, sb = squash(a), squash(b)
    if sa == sb:
        return []
    out: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out.append(
            (
                sa[max(0, i1 - context) : i2 + context],
                sb[max(0, j1 - context) : j2 + context],
            )
        )
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------- pages ----
def split_pdf(data: bytes) -> list[bytes] | None:
    """One single-page PDF per page, or None when the file cannot be split (then
    the caller keeps it whole)."""
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages: list[bytes] = []
        for page in reader.pages:
            writer = PdfWriter()
            writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            pages.append(buffer.getvalue())
        return pages or None
    except Exception:  # noqa: BLE001 — a PDF we cannot split is still readable whole
        return None


def load_pages(files: list[tuple[bytes, str]]) -> list[tuple[bytes, str]]:
    """Every uploaded file as a list of pages, in upload order. A paper split over
    several files is one document: its pages simply follow each other."""
    pages: list[tuple[bytes, str]] = []
    for data, mime in files:
        if mime == "application/pdf":
            split = split_pdf(data)
            if split:
                pages.extend((p, mime) for p in split)
                continue
        pages.append((data, mime))
    if len(pages) > MAX_PAGES:
        raise ValueError(f"Tài liệu có {len(pages)} trang, vượt giới hạn {MAX_PAGES} trang")
    return pages


def windows(count: int, size: int = WINDOW_PAGES, stride: int = WINDOW_STRIDE, phase: int = 0) -> list[tuple[int, int]]:
    """Page spans [start, end) the model reads, each overlapping the next. The first
    starts at 0; the others start at `phase` (default: `stride`), then every
    `stride` pages, until one reaches the last page. A shifted read (phase > 0) also
    begins one page shorter, so none of its windows equals a window of the first read."""
    if count <= 0:
        return []
    if count <= size:
        return [(0, count)]
    spans = [(0, size - 1 if phase else size)]
    start = phase or stride
    while True:
        spans.append((start, min(start + size, count)))
        if start + size >= count:
            break
        start += stride
    return spans


# -------------------------------------------------------------- parsed reads ---
@dataclass
class _Passage:
    key: tuple[int, str]  # (read window, the model's local ref)
    kind: str
    body: str | None
    withheld: bool
    page: int  # absolute, 1-based
    confidence: float


@dataclass
class _Question:
    number: int
    stem: str
    options: list[str]
    instruction: str | None
    group: tuple[int, int] | None
    passage_key: tuple[int, str] | None
    qtype: str | None
    guess: int | None
    page: int
    confidence: float
    window: int


@dataclass
class _Read:
    index: int
    span: tuple[int, int]
    questions: list[_Question] = field(default_factory=list)
    passages: dict[tuple[int, str], _Passage] = field(default_factory=dict)
    section: str | None = None
    key_rows: list[tuple[str | None, int, int]] = field(default_factory=list)


def section_skill(value: Any) -> str | None:
    text = _txt(value)
    for ko, skill in SKILL_BY_SECTION.items():
        if ko in text or skill in text.lower():
            return skill
    return None


def parse_read(data: dict[str, Any], index: int, span: tuple[int, int]) -> _Read:
    """One window's answer, tolerant: a malformed passage or question costs only
    itself, never the window."""
    read = _Read(index=index, span=span, section=section_skill(data.get("paper_section")))
    start = span[0]

    for raw in _list(data.get("passages")):
        if not isinstance(raw, dict):
            continue
        ref = _txt(raw.get("local_ref"))
        if not ref:
            continue
        kind = _txt(raw.get("kind"))
        if kind not in ("đọc hiểu", "nghe", "biểu đồ"):
            kind = "đọc hiểu"
        body = tidy_text(raw.get("body_ko"), multiline=True) or None
        withheld = bool(raw.get("withheld"))
        if withheld:
            body = None
        page = start + (_int(raw.get("page"), 1) or 1)
        read.passages[(index, ref)] = _Passage(
            key=(index, ref),
            kind=kind,
            body=body,
            withheld=withheld,
            page=page,
            confidence=_confidence(raw.get("confidence")),
        )

    for raw in _list(data.get("items")):
        if not isinstance(raw, dict):
            continue
        number = _int(raw.get("number"), 1)
        if number is None:
            continue
        instruction = clean_instruction(raw.get("instruction_ko"))
        group = group_of(instruction)
        if group is None:
            low, high = _int(raw.get("group_from"), 1), _int(raw.get("group_to"), 1)
            group = (low, high) if low is not None and high is not None and low <= high else None
        ref = _txt(raw.get("passage_ref"))
        options = [clean_option(o) for o in _list(raw.get("options")) if isinstance(o, (str, int, float))]
        read.questions.append(
            _Question(
                number=number,
                stem=tidy_text(raw.get("stem_ko"), multiline=True),
                options=options,
                instruction=instruction,
                group=group,
                passage_key=(index, ref) if ref else None,
                qtype=_txt(raw.get("qtype_code")) or None,
                guess=_int(raw.get("answer_guess"), 1, 5),
                page=start + (_int(raw.get("page"), 1) or 1),
                confidence=_confidence(raw.get("confidence")),
                window=index,
            )
        )

    for raw in _list(data.get("answer_key")):
        if not isinstance(raw, dict):
            continue
        number, answer = _int(raw.get("number"), 1), _int(raw.get("answer"), 1, 5)
        if number is not None and answer is not None:
            read.key_rows.append((section_skill(raw.get("section")), number, answer))
    return read


# ------------------------------------------------------------------ merging ----
@dataclass
class _Merged:
    questions: dict[int, _Question]
    passages: dict[tuple[int, str], _Passage]  # canonical passages only
    alias: dict[tuple[int, str], tuple[int, str]]  # any read's passage -> its canonical one
    duplicates: set[int]
    section: str | None
    key_rows: list[tuple[str | None, int, int]]

    def passage_of(self, question: _Question) -> _Passage | None:
        if question.passage_key is None:
            return None
        canonical = self.alias.get(question.passage_key)
        return self.passages.get(canonical) if canonical else None


def same_passage(a: _Passage, b: _Passage) -> bool:
    """Two reads of one passage, whole or cut by a window edge."""
    if a.withheld or b.withheld:
        return a.withheld and b.withheld and a.page == b.page
    sa, sb = squash(a.body), squash(b.body)
    if not sa or not sb:
        return False
    if sa in sb or sb in sa:
        return True
    return difflib.SequenceMatcher(None, sa, sb, autojunk=False).ratio() >= PASSAGE_SIMILARITY


def _score(question: _Question, passage: _Passage | None) -> float:
    """How complete a reading of a question is, to pick among the windows that
    each saw it. A window that begins in the middle of a group lacks the
    instruction or the passage; one that ends there lacks options."""
    score = question.confidence
    score += 3.0 if len(question.options) == OPTION_COUNT else (1.0 if question.options else 0.0)
    if passage is not None and (passage.body or passage.withheld):
        score += 2.0
    if question.instruction:
        score += 1.0
    unreadable = sum(o.count(UNREADABLE) for o in question.options) + question.stem.count(UNREADABLE)
    return score - 2.0 * min(unreadable, 3)


def _spans_touch(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def merge(reads: list[_Read]) -> _Merged:
    """The best reading of every question, from the windows that saw it, with each
    passage once. A question number found twice in windows that do not overlap, with
    different content, is a sign the file holds two papers: flagged as duplicate."""
    spans = {r.index: r.span for r in reads}
    all_passages = {k: p for r in reads for k, p in r.passages.items()}

    def passage_for(q: _Question) -> _Passage | None:
        return all_passages.get(q.passage_key) if q.passage_key else None

    by_number: dict[int, list[_Question]] = {}
    for read in reads:
        for q in read.questions:
            by_number.setdefault(q.number, []).append(q)

    winners: dict[int, _Question] = {}
    duplicates: set[int] = set()
    for number, candidates in by_number.items():
        ranked = sorted(candidates, key=lambda q: (-_score(q, passage_for(q)), q.window))
        winner = ranked[0]
        winners[number] = winner
        for other in ranked[1:]:
            if _spans_touch(spans[winner.window], spans[other.window]):
                continue
            if squash(other.stem) != squash(winner.stem) or [squash(o) for o in other.options] != [
                squash(o) for o in winner.options
            ]:
                duplicates.add(number)
                break

    # Passages: only those a winning question uses, each once.
    canonical: list[_Passage] = []
    alias: dict[tuple[int, str], tuple[int, str]] = {}
    for number in sorted(winners):
        q = winners[number]
        if q.passage_key is None or q.passage_key in alias:
            continue
        passage = all_passages.get(q.passage_key)
        if passage is None:
            continue
        for existing in canonical:
            if same_passage(existing, passage):
                # keep the fuller reading as the canonical one
                if len(squash(passage.body)) > len(squash(existing.body)):
                    canonical[canonical.index(existing)] = passage
                    for k, v in list(alias.items()):
                        if v == existing.key:
                            alias[k] = passage.key
                    alias[passage.key] = passage.key
                    alias[existing.key] = passage.key
                else:
                    alias[passage.key] = existing.key
                break
        else:
            canonical.append(passage)
            alias[passage.key] = passage.key

    sections = [r.section for r in reads if r.section]
    section = max(set(sections), key=sections.count) if sections else None
    return _Merged(
        questions=winners,
        passages={p.key: p for p in canonical},
        alias=alias,
        duplicates=duplicates,
        section=section,
        key_rows=[row for r in reads for row in r.key_rows],
    )


def adopt_passage(into: _Merged, passage: _Passage) -> tuple[int, str]:
    """Bring a passage from another reading into `into`, once: if it is the same
    passage as one already there, that one is used."""
    for existing in into.passages.values():
        if same_passage(existing, passage):
            into.alias[passage.key] = existing.key
            return existing.key
    into.passages[passage.key] = passage
    into.alias[passage.key] = passage.key
    return passage.key


def compare(first: _Merged, second: _Merged) -> tuple[dict[int, list[dict[str, str]]], dict[tuple[int, str], list[dict[str, str]]]]:
    """Differences between two independent readings, by question number and by
    passage (keyed by the FIRST reading's passage). A passage one read cut short
    (a window edge) is not a difference; a changed character is."""
    question_diffs: dict[int, list[dict[str, str]]] = {}
    passage_diffs: dict[tuple[int, str], list[dict[str, str]]] = {}

    def note(target: list[dict[str, str]], name: str, a: str, b: str) -> None:
        for seen, other in span_diffs(a, b):
            target.append({"field": name, "a": seen, "b": other})

    for number, qa in first.questions.items():
        qb = second.questions.get(number)
        if qb is None:
            continue
        diffs: list[dict[str, str]] = []
        if qa.instruction and qb.instruction:
            note(diffs, "instruction", qa.instruction, qb.instruction)
        if qa.stem or qb.stem:
            note(diffs, "stem", qa.stem, qb.stem)
        if len(qa.options) == len(qb.options):
            for i, (oa, ob) in enumerate(zip(qa.options, qb.options), start=1):
                note(diffs, f"option {i}", oa, ob)
        elif qa.options and qb.options and len(qb.options) == OPTION_COUNT:
            diffs.append({"field": "options", "a": f"{len(qa.options)} phương án", "b": f"{len(qb.options)} phương án"})
        if diffs:
            question_diffs[number] = diffs
        pa, pb = first.passage_of(qa), second.passage_of(qb)
        if pa is not None and pb is not None and pa.body and pb.body:
            sa, sb = squash(pa.body), squash(pb.body)
            if sa != sb and sa not in sb and sb not in sa:
                found: list[dict[str, str]] = []
                note(found, "passage", pa.body, pb.body)
                if found:
                    passage_diffs.setdefault(pa.key, found)
    return question_diffs, passage_diffs


# ------------------------------------------------------------------- flags -----
RED_FLAGS = frozenset(
    {"withheld", "missing_passage", "missing_qtype", "options_count", "unclear", "no_underline", "duplicate_number", "empty"}
)
FLAG_LABELS_VI = {
    "withheld": "Đoạn văn không được công bố (bản quyền)",
    "missing_passage": "Thiếu đoạn văn mà câu hỏi cần",
    "missing_qtype": "Chưa xác định loại câu hỏi",
    "options_count": "Số phương án khác 4",
    "unclear": "Có chữ đọc không rõ ([?]) hoặc gạch chân lỗi",
    "no_underline": "Đề nói “밑줄” nhưng không thấy phần gạch chân",
    "duplicate_number": "Số câu xuất hiện hai lần với nội dung khác nhau",
    "empty": "Đoạn văn trống",
    "text_mismatch": "Hai lượt đọc khác nhau",
    "low_confidence": "Mô hình không chắc chữ chép đúng",
    "single_read": "Chỉ một lượt đọc thấy câu này",
    "no_answer": "Bảng đáp án không có câu này",
}


def status_for(flags: list[str], confidence: float) -> str:
    """The review-queue light. Red = cannot be right as it stands; yellow = look
    at it; green = nothing found. A missing answer key is none of these."""
    if confidence < 0.5 or any(f in RED_FLAGS for f in flags):
        return "flagged_red"
    if flags or confidence < LOW_CONFIDENCE:
        return "flagged_yellow"
    return "pending"


_QTYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"순서에 맞게|배열"), "read_order"),
    (re.compile(r"밑줄 친 부분과 의미"), "read_grammar_choice"),
    (re.compile(r"무엇에 대한 글|중심 (?:생각|내용)|주제|제목"), "read_title_topic"),
    (re.compile(r"도표|안내문|광고|그래프"), "read_chart_info"),
    (re.compile(r"들어갈 (?:말|곳|내용)|빈칸"), "read_blank"),
]


def guess_qtype(instruction: str | None, stem: str, passage_body: str | None, valid: set[str]) -> str | None:
    """A question type from the printed instruction, for when the model left it
    empty or named a code that does not exist."""
    text = f"{instruction or ''} {stem}"
    for pattern, code in _QTYPE_RULES:
        if pattern.search(text) and code in valid:
            return code
    if passage_body:
        code = "read_short_passage" if len(squash(passage_body)) < 350 else "read_long_passage"
        return code if code in valid else None
    return "read_grammar_choice" if "read_grammar_choice" in valid else None


def needs_passage(instruction: str | None, stem: str, qtype: str | None) -> bool:
    return qtype in exam_drill.NEEDS_PASSAGE or bool(exam_drill.PASSAGE_REFERENCE.search(f"{instruction or ''} {stem}"))


def question_flags(
    question: _Question, passage: _Passage | None, qtype: str | None, *, diffs: list[dict[str, str]] | None, single: bool
) -> list[str]:
    flags: list[str] = []
    if qtype is None:
        flags.append("missing_qtype")
    if len(question.options) != OPTION_COUNT:
        flags.append("options_count")
    texts = [question.stem, question.instruction or "", *question.options]
    if any(UNREADABLE in t for t in texts) or not all(underline_balanced(t) for t in texts):
        flags.append("unclear")
    if passage is not None and passage.withheld:
        flags.append("withheld")
    elif needs_passage(question.instruction, question.stem, qtype) and (passage is None or not passage.body):
        flags.append("missing_passage")
    withheld = passage is not None and passage.withheld
    shown = f"{question.stem} {passage.body if passage is not None and passage.body else ''}"
    if "밑줄" in f"{question.instruction or ''} {question.stem}" and "<u>" not in shown and not withheld:
        flags.append("no_underline")
    if diffs:
        flags.append("text_mismatch")
    if single:
        flags.append("single_read")
    if question.confidence < LOW_CONFIDENCE:
        flags.append("low_confidence")
    return flags


# --------------------------------------------------------------- the draft -----
@dataclass
class Draft:
    """What staging needs: each entry is {"payload", "confidence", "flags"}."""

    passages: list[dict[str, Any]]
    items: list[dict[str, Any]]
    summary: dict[str, Any]
    key: dict[str, dict[int, int]]


Generate = Callable[..., dict[str, Any]]


def _default_generate(**kwargs: Any) -> dict[str, Any]:
    return gemini_client.generate_structured(**kwargs)


def _model() -> str:
    return settings.GEMINI_MODEL_EXAM_INGEST or settings.GEMINI_MODEL_LESSON_INGEST


def _ask(generate: Generate, parts: list[Any], prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """One structured call, asked again once when the answer is not usable JSON."""
    last: Exception | None = None
    for _ in range(2):
        try:
            return _json(
                generate(model=_model(), prompt=[*parts, prompt], response_schema=schema, prompt_version=PROMPT_VERSION)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last = exc
    assert last is not None
    raise last


def _read_windows(
    generate: Generate,
    pages: list[tuple[bytes, str]],
    spans: list[tuple[int, int]],
    qtypes: list[tuple[str, str, str]],
    on_done: Callable[[], None],
) -> tuple[list[_Read], list[int]]:
    """Read every window (a few at a time). Returns the reads and the indexes of
    windows that failed — a failed window only means its questions come from a
    neighbouring window or are reported missing."""

    def one(index: int) -> _Read | None:
        start, end = spans[index]
        parts = [gemini_client.part_from_bytes(data, mime) for data, mime in pages[start:end]]
        try:
            return parse_read(_ask(generate, parts, build_prompt(end - start, qtypes), EXAM_SCHEMA), index, spans[index])
        except Exception as exc:  # noqa: BLE001
            print(f"[exam] window {index} (pages {start + 1}-{end}) failed: {exc}", flush=True)
            return None

    reads: list[_Read] = []
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, result in enumerate(pool.map(one, range(len(spans)))):
            on_done()
            if result is None:
                failed.append(index)
            else:
                reads.append(result)
    return reads, failed


def extract(
    files: list[tuple[bytes, str]],
    qtypes: list[tuple[str, str, str]],
    *,
    session_label: str = "",
    generate: Generate | None = None,
    verify: bool | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Draft:
    """Read an exam paper (PDF / images / text) into a draft: passages and
    questions with the reasons each needs a look. `qtypes` is
    (code, name_vi, skill) per known question type."""
    gen = generate or _default_generate
    do_verify = settings.EXAM_VERIFY_PASS if verify is None else verify
    pages = load_pages(files)
    first_spans = windows(len(pages))
    second_spans = windows(len(pages), phase=VERIFY_PHASE) if do_verify else []
    total = len(first_spans) + len(second_spans)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(done, total, f"Đang đọc đề ({done}/{total} lượt)")

    reads_a, failed_a = _read_windows(gen, pages, first_spans, qtypes, tick)
    if not reads_a:
        raise RuntimeError("Gemini không đọc được trang nào của đề (mọi lượt đều lỗi)")
    first = merge(reads_a)

    second: _Merged | None = None
    failed_b: list[int] = []
    if second_spans:
        reads_b, failed_b = _read_windows(gen, pages, second_spans, qtypes, tick)
        second = merge(reads_b) if reads_b else None

    q_diffs: dict[int, list[dict[str, str]]] = {}
    p_diffs: dict[tuple[int, str], list[dict[str, str]]] = {}
    extra: set[int] = set()
    if second is not None:
        q_diffs, p_diffs = compare(first, second)
        for number, question in second.questions.items():
            if number not in first.questions:
                passage = second.passage_of(question)
                question.passage_key = adopt_passage(first, passage) if passage is not None else None
                first.questions[number] = question
                extra.add(number)

    label_skill = section_skill(session_label)
    section = label_skill or first.section or (second.section if second else None)
    key = collect_key(first.key_rows + (second.key_rows if second else []), section)
    return build_draft(
        first,
        qtypes,
        q_diffs=q_diffs,
        p_diffs=p_diffs,
        single=extra,
        section=section,
        key=key,
        stats={
            "pages": len(pages),
            "windows": len(first_spans),
            "failed_windows": failed_a,
            "verify_windows": len(second_spans),
            "verify_failed": failed_b,
            "verified": second is not None,
        },
    )


def build_draft(
    merged: _Merged,
    qtypes: list[tuple[str, str, str]],
    *,
    q_diffs: dict[int, list[dict[str, str]]],
    p_diffs: dict[tuple[int, str], list[dict[str, str]]],
    single: set[int],
    section: str | None,
    key: dict[str, dict[int, int]],
    stats: dict[str, Any],
) -> Draft:
    valid = {code for code, _name, _skill in qtypes}
    skill_of = {code: skill for code, _name, skill in qtypes}

    # passages get tidy refs in the order their first question appears
    refs: dict[tuple[int, str], str] = {}
    for number in sorted(merged.questions):
        passage = merged.passage_of(merged.questions[number])
        if passage is not None and passage.key not in refs:
            refs[passage.key] = f"P{len(refs) + 1}"

    passages: list[dict[str, Any]] = []
    for passage in sorted(refs, key=lambda k: refs[k]):
        p = merged.passages[passage]
        flags: list[str] = []
        if p.withheld:
            flags.append("withheld")
        elif not p.body:
            flags.append("empty")
        elif UNREADABLE in p.body or not underline_balanced(p.body):
            flags.append("unclear")
        diffs = p_diffs.get(p.key)
        if diffs:
            flags.append("text_mismatch")
        if not p.withheld and p.confidence < LOW_CONFIDENCE:
            flags.append("low_confidence")
        payload: dict[str, Any] = {
            "local_ref": refs[p.key],
            "kind": p.kind,
            "body_ko": p.body,
            "withheld": p.withheld,
            "source_page": p.page,
            "flags": flags,
        }
        if diffs:
            payload["alt"] = diffs
        passages.append({"payload": payload, "confidence": p.confidence, "flags": flags})

    items: list[dict[str, Any]] = []
    for number in sorted(merged.questions):
        q = merged.questions[number]
        passage = merged.passage_of(q)
        qtype = q.qtype if q.qtype in valid else None
        if qtype is None:
            qtype = guess_qtype(q.instruction, q.stem, passage.body if passage else None, valid)
        flags = question_flags(q, passage, qtype, diffs=q_diffs.get(number), single=number in single)
        if number in merged.duplicates:
            flags.append("duplicate_number")
        payload = {
            "number": number,
            "passage_ref": refs.get(passage.key) if passage is not None else None,
            "qtype_code": qtype,
            "instruction_ko": q.instruction,
            "group_range": list(q.group) if q.group else None,
            "stem_ko": q.stem,
            "options": q.options,
            "answer": None,
            "answer_from_key": False,
            "answer_guess": q.guess,
            "source_page": q.page,
            "flags": flags,
        }
        if number in q_diffs:
            payload["alt"] = q_diffs[number]
        items.append({"payload": payload, "confidence": q.confidence, "flags": flags})

    numbers = sorted(merged.questions)
    gaps = [n for n in range(numbers[0], numbers[-1] + 1) if n not in merged.questions] if numbers else []
    if section is None:
        votes = [skill_of.get(i["payload"]["qtype_code"]) for i in items if i["payload"]["qtype_code"]]
        votes = [v for v in votes if v]
        section = max(set(votes), key=votes.count) if votes else None
    summary = {
        "prompt_version": PROMPT_VERSION,
        "section": section,
        "numbers": [numbers[0], numbers[-1]] if numbers else None,
        "questions": len(numbers),
        "gaps": gaps,
        "duplicates": sorted(merged.duplicates),
        "key": None,
        **stats,
    }
    return Draft(passages=passages, items=items, summary=summary, key=key)


# ------------------------------------------------------------- answer keys -----
def collect_key(rows: list[tuple[str | None, int, int]], default_section: str | None) -> dict[str, dict[int, int]]:
    """Answer-key rows a paper's own pages carried, by skill. A number given two
    different answers is dropped (unreadable, not a choice)."""
    seen: dict[str, dict[int, set[int]]] = {}
    for section, number, answer in rows:
        skill = section or default_section or "?"
        seen.setdefault(skill, {}).setdefault(number, set()).add(answer)
    return {
        skill: {n: next(iter(a)) for n, a in sorted(numbers.items()) if len(a) == 1}
        for skill, numbers in seen.items()
    }


@dataclass
class KeyRead:
    sections: dict[str, dict[int, int]]
    conflicts: dict[str, list[int]]
    pages: int


def _read_key_once(generate: Generate, pages: list[tuple[bytes, str]]) -> dict[str, dict[int, set[int]]]:
    """Every (skill, number) -> the answers this read gave for it."""
    result: dict[str, dict[int, set[int]]] = {}
    for start in range(0, len(pages), KEY_WINDOW_PAGES):
        chunk = pages[start : start + KEY_WINDOW_PAGES]
        parts = [gemini_client.part_from_bytes(data, mime) for data, mime in chunk]
        data = _ask(generate, parts, build_key_prompt(len(chunk)), KEY_SCHEMA)
        for section in _list(data.get("sections")):
            if not isinstance(section, dict):
                continue
            skill = section_skill(section.get("section"))
            if skill is None or skill == "viết":
                continue  # the writing table is model essays, not numbered choices
            for row in _list(section.get("answers")):
                if not isinstance(row, dict):
                    continue
                number, answer = _int(row.get("number"), 1), _int(row.get("answer"), 1, 5)
                if number is not None and answer is not None:
                    result.setdefault(skill, {}).setdefault(number, set()).add(answer)
    return result


def extract_key(
    files: list[tuple[bytes, str]],
    *,
    generate: Generate | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> KeyRead:
    """Read a printed answer key (정답표) twice. A number the two reads do not agree
    on (different answers, or only one read saw it) is not used: it comes back as a
    conflict, for the editor to type."""
    gen = generate or _default_generate
    pages = load_pages(files)
    reads: list[dict[str, dict[int, set[int]]]] = []
    for attempt in range(2):
        reads.append(_read_key_once(gen, pages))
        if on_progress:
            on_progress(attempt + 1, 2, f"Đang đọc bảng đáp án (lượt {attempt + 1}/2)")
    sections: dict[str, dict[int, int]] = {}
    conflicts: dict[str, list[int]] = {}
    for skill in sorted({s for r in reads for s in r}):
        numbers = {n for r in reads for n in r.get(skill, {})}
        agreed: dict[int, int] = {}
        disputed: list[int] = []
        for n in sorted(numbers):
            seen = [r.get(skill, {}).get(n, set()) for r in reads]
            union = set().union(*seen)
            if all(len(s) == 1 for s in seen) and len(union) == 1:
                agreed[n] = next(iter(union))
            else:
                disputed.append(n)
        if agreed:
            sections[skill] = agreed
        if disputed:
            conflicts[skill] = disputed
    return KeyRead(sections=sections, conflicts=conflicts, pages=len(pages))


_PAIR_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\s*[-:=.)]+\s*)+([1-5])(?!\d)")
_CIRCLED_BLACK = "❶❷❸❹❺"


class AnswerTextError(ValueError):
    pass


def parse_answer_text(text: str, *, start: int = 1) -> dict[int, int]:
    """Answers an editor typed or pasted: `1-2, 2-1` / `1:2` / `1 ②` pairs, or one
    unbroken run of digits (`2113…`) taken as the answers from question `start`.
    Anything it cannot read exactly raises AnswerTextError — a guess here would put
    a wrong answer in the bank."""
    normalized = text
    for i, ch in enumerate(_CIRCLED, start=1):
        normalized = normalized.replace(ch, f"-{i} ")
    for i, ch in enumerate(_CIRCLED_BLACK, start=1):
        normalized = normalized.replace(ch, f"-{i} ")
    normalized = unicodedata.normalize("NFKC", normalized)  # full-width digits and signs
    pairs = _PAIR_RE.findall(normalized)
    if pairs:
        if re.search(r"\d", _PAIR_RE.sub(" ", normalized)):
            raise AnswerTextError("Có số không đọc được thành cặp “câu-đáp án”: " + normalized.strip()[:60])
        answers: dict[int, int] = {}
        for number, answer in pairs:
            n, a = int(number), int(answer)
            if answers.get(n, a) != a:
                raise AnswerTextError(f"Câu {n} được ghi hai đáp án khác nhau")
            answers[n] = a
        return answers
    compact = re.sub(r"[\s,;]+", "", normalized)
    if compact and re.fullmatch(r"[1-5]+", compact):
        return {start + i: int(ch) for i, ch in enumerate(compact)}
    raise AnswerTextError("Không hiểu định dạng. Dùng “1-2, 2-1, 3-4”, “1:2”, hoặc một chuỗi liền “2134…”.")


@dataclass
class AnswerReport:
    skill: str | None
    applied: list[int]
    changed: list[int]  # applied where a different answer was already there
    unmatched: list[int]  # in the key but not in the paper
    missing: list[int]  # in the paper but not in the key
    conflicts: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "applied": len(self.applied),
            "changed": self.changed,
            "unmatched": self.unmatched,
            "missing": self.missing,
            "conflicts": self.conflicts,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def section_from_label(label: str | None) -> str | None:
    """The skill an editor's session label names ("102회 읽기", "đề đọc"), or None
    when it names none or several."""
    text = (label or "").lower()
    found = {
        skill
        for skill, words in (
            ("đọc", ("읽기", "đọc", "doc", "reading")),
            ("nghe", ("듣기", "nghe", "listening")),
            ("viết", ("쓰기", "viết", "viet", "writing")),
        )
        if any(w in text for w in words)
    }
    return found.pop() if len(found) == 1 else None


def choose_key_section(key: dict[str, dict[int, int]], paper_skill: str | None) -> str | None:
    """Which table of a multi-section key belongs to this paper. When the paper's part
    is known and the key has no table for it, the answer is None: listening answers
    must never be put on a reading paper because they are the only table there is."""
    if paper_skill:
        return paper_skill if paper_skill in key else None
    return next(iter(key)) if len(key) == 1 else None
