"""News-article extraction: raw HTML -> (clean paragraphs, inline images).

Why this module exists: the first version of fetch_article_text just
called `get_text()` on the whole <body> minus a list of class-name hints,
which left every outlet's page chrome in the stored text — view-settings
modals ("보기 설정 / 닫기 / 글자 크기 / 보통 / 크게 ..."), the AI-summary
disclaimer, photo captions, "지금 많이 보는 기사" ranking widgets, login
prompts. That text went straight to the reader screen AND to TTS, so the
"listen to the whole article" button literally read the menu aloud
(confirmed in the worker log of a real job).

Three layers, each usable on its own:

1. `clean_article_text(text)` — PURE text -> text. Drops known page chrome
   and photo-credit lines, then keeps only the span from the first real
   paragraph to the last one. Idempotent, so it is also applied at READ
   time to articles stored before this module existed (no data migration).
2. `extract_article(html, url)` — picks the real article container in the
   DOM (explicit selectors + a readability-style density score), walks it
   block by block (inline tags like <b>/<a> no longer split a sentence
   across lines), and pulls out <img>/<figure> with their captions so the
   captions stop polluting the text.
3. `article_tts_text(title, body)` — what the "Nghe toàn văn" button
   actually voices.

A "real paragraph" throughout = a line of >= 30 characters that ends like
a sentence. Menu items ("닫기"), rankings and related-article headlines
essentially never do; captions end in a photo credit; body paragraphs
essentially always do.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# Bump when the cleaning rules change so cached TTS audio (keyed off the
# text it voiced) is regenerated instead of serving a stale recording.
TEXT_CLEAN_VERSION = "clean-v1"

MAX_IMAGES = 8
_MIN_PARAGRAPH_CHARS = 30
_WS_RE = re.compile(r"[\s ​　]+")

_SENTENCE_END = (".", "!", "?", "…", '"', "”", "’", "'", "」", "』", "。")


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _ends_sentence(line: str) -> bool:
    line = line.rstrip()
    if not line:
        return False
    if line.endswith(_SENTENCE_END):
        return True
    # closing bracket right after a full stop: "...했다.)" / "...했다.]"
    return len(line) >= 2 and line[-1] in ")]" and line[-2] in ".!?…"


def _is_paragraph(line: str) -> bool:
    return len(line) >= _MIN_PARAGRAPH_CHARS and _ends_sentence(line)


# --------------------------------------------------------------- noise rules --
def _squash(s: str) -> str:
    """Lowercase + drop all whitespace, so "글자 크기" == "글자크기"."""
    return re.sub(r"\s+", "", s).lower()


# Whole-line UI chrome (compared after _squash). Real body paragraphs are
# never one of these words on their own line.
_CHROME_EXACT = {
    _squash(x)
    for x in (
        "보기 설정", "닫기", "글자 크기", "글자크기", "보통", "크게", "아주 크게", "작게", "컬러 모드",
        "라이트", "다크", "베이지", "그린", "본문 요약", "본문요약", "내 뉴스플리에 저장",
        "내 뉴스플레이에 저장", "로그인", "로그아웃", "회원가입", "회원정보수정", "회원탈퇴", "구독신청",
        "구독하기", "mPaper", "RSS", "인쇄", "메일", "url 공유", "공유", "공유하기", "스크랩", "크게보기",
        "확대", "축소", "가", "가-", "가+", "-", "+", "구글 선호 매체 등록", "관련이슈", "기사원문",
        "기사 원문", "기사원문 보기", "목록", "맨위로", "TOP", "포토", "영상", "댓글", "댓글쓰기",
        "좋아요", "싫어요", "북마크", "저장", "페이스북", "트위터", "카카오톡", "카카오스토리", "네이버 블로그",
        "밴드", "URL 복사", "링크 복사", "입력 :", "입력:", "수정 :", "수정:", "입력", "수정",
        "이전기사", "다음기사", "이전 기사", "다음 기사", "더보기", "전체보기", "뉴스룸 PICK", "에디터 PICK",
    )
}

# Lines that are page furniture even though they are sentence-like.
_CHROME_RE = re.compile(
    r"자동\s*요약된\s*내용|NAVER\s*MEDIA\s*API|로그인\s*후\s*이용|회원이\s*아니|"
    r"구독은\s*로그인|뉴스레터\s*(구독|신청)|앱을?\s*(다운로드|설치)|이용약관|개인정보\s*(처리|취급)|"
    r"기사\s*제보|이\s*기사를\s*(공유|스크랩|추천)|홈으로\s*이동|^\d{4}[-./]\d{1,2}[-./]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?$|"
    r"^(입력|수정|등록|승인|발행)\s*[:：]?\s*\d{4}",
    re.IGNORECASE,
)

# Copyright / redistribution notices.
_COPYRIGHT_RE = re.compile(
    r"무단\s*(전재|복제)|재배포\s*금지|All\s*rights\s*reserved|저작권자\s*[ⓒ©(<\[]|[ⓒ©]\s*\d{4}|^\s*(Copyright|ⓒ|©)",
    re.IGNORECASE,
)

# Photo captions / credits that leaked into the text stream.
_CAPTION_RE = re.compile(
    r"^(크게\s*보기|\[?\s*(자료\s*)?사진\s*[=:/]|사진\s*(제공|출처))|"
    r"[/=]\s*(연합뉴스|뉴스1|뉴시스|게티이미지(뱅크)?|Getty\s*Images|AP|AFP|EPA|로이터|셔터스톡|픽사베이)\s*$|"
    r"[\s.](연합뉴스|뉴스1|뉴시스|AP연합뉴스|AFP연합뉴스|로이터연합뉴스|게티이미지뱅크|게티이미지코리아)\s*$|"
    r"<\s*저작권자.*>\s*$",
    re.IGNORECASE,
)

# Inline "related article" teasers that sit in the middle of a body.
_INLINE_TEASER_RE = re.compile(r"^\s*([▶▷☞►■◆●※]\s*)?[\[\(]?\s*(관련\s*기사|관련\s*뉴스|함께\s*읽|이어\s*읽|더\s*읽어)")

# Everything from one of these lines to the end is widget/footer, never body
# (ranking boxes, related lists). Only honoured AFTER the first real
# paragraph, so a header widget can never swallow the article.
_TAIL_MARKER_RE = re.compile(
    r"^(지금\s*)?많이\s*(보는|본)\s*(기사|뉴스)?|^읽어볼\s*만한|^HOT\s*뉴스|^핫\s*뉴스|^인기\s*(기사|뉴스)|"
    r"^랭킹\s*뉴스|^실시간\s*(인기|랭킹)|^추천\s*(기사|뉴스)|^주요\s*뉴스|^관련\s*(기사|뉴스|이슈)\s*$|"
    r"^기사\s*(공유|스크랩)|^댓글\s*\d*$|^이\s*시각\s*(주요|인기)|^[\[\(]?\s*오늘의\s*(운세|뉴스|핫)|"
    r"^에서\s*.{1,12}\s*팔로우$|^(다른|같은)\s*(기사|칼럼)|^뉴스레터",
)

# Author credit line worth keeping right after the last paragraph.
_BYLINE_RE = re.compile(r"(기자|위원|교수|논설|칼럼니스트|작가|대표|실장|연구원|박사|변호사|센터장|소장|이사장|위원장|장관|총장|의원|@[\w.-]+)")


def _is_noise_line(line: str) -> bool:
    if _squash(line) in _CHROME_EXACT:
        return True
    # The sentence-like patterns are only trusted on SHORT lines: a long
    # real paragraph of an op-ed about privacy/copyright may legitimately
    # mention "개인정보 처리" or "무단 전재".
    if len(line) < 120 and _CHROME_RE.search(line):
        return True
    if len(line) < 200 and _COPYRIGHT_RE.search(line):
        return True
    if _CAPTION_RE.search(line):
        return True
    if _INLINE_TEASER_RE.match(line):
        return True
    return False


def select_body_lines(lines: list[str]) -> list[int]:
    """Indices (into `lines`) that belong to the article body. `lines` are
    already whitespace-normalised, non-empty strings.

    1. drop chrome / copyright / caption / teaser lines
    2. cut at the first ranking/related widget marker (after the first
       real paragraph)
    3. keep only [first paragraph .. last paragraph], extended by adjacent
       short sentence-ending lines and ONE trailing author-credit line
    Falls back to step 1's result untouched when no line qualifies as a
    paragraph (poems, dialogue-only pieces) or when trimming would throw
    away most of the text — better a slightly dirty article than an empty
    one."""
    alive = [i for i, ln in enumerate(lines) if not _is_noise_line(ln)]
    if not alive:
        return []

    first_para = next((k for k, i in enumerate(alive) if _is_paragraph(lines[i])), None)
    if first_para is None:
        return alive

    # 2. tail cut
    end = len(alive)
    for k in range(first_para + 1, len(alive)):
        if _TAIL_MARKER_RE.search(lines[alive[k]]):
            end = k
            break
    alive = alive[:end]

    last_para = max(k for k, i in enumerate(alive) if _is_paragraph(lines[i]))

    # 3. trim head/tail, tolerating short sentence-ending lines at the edges
    start = first_para
    while start > 0 and _ends_sentence(lines[alive[start - 1]]):
        start -= 1
    stop = last_para
    while stop + 1 < len(alive) and _ends_sentence(lines[alive[stop + 1]]):
        stop += 1
    if (
        stop + 1 < len(alive)
        and len(lines[alive[stop + 1]]) <= 40
        and _BYLINE_RE.search(lines[alive[stop + 1]])
    ):
        stop += 1

    kept = alive[start : stop + 1]
    total_before = sum(len(lines[i]) for i in alive)
    total_after = sum(len(lines[i]) for i in kept)
    if total_before and total_after < 0.4 * total_before:
        return alive  # trimming looks wrong for this text — don't trust it
    return kept


def clean_article_text(text: str) -> str:
    """Pure text -> text version of the pipeline above; safe on already
    clean text (idempotent). One paragraph per line in, one per line out."""
    if not text:
        return ""
    lines: list[str] = []
    for raw in text.splitlines():
        ln = _norm_ws(raw)
        if ln and not (lines and lines[-1] == ln and len(ln) < 20):
            lines.append(ln)
    kept = select_body_lines(lines)
    return "\n".join(lines[i] for i in kept)


def split_paragraphs(body: str) -> list[str]:
    return [p for p in (ln.strip() for ln in (body or "").splitlines()) if p]


# ------------------------------------------------------------- image cleanup --
_CAP_COPYRIGHT_TAIL_RE = re.compile(r"\s*[<\[(（]?\s*저작권자.*$|\s*[ⓒ©]\s*\d{0,4}.*$|\s*무단\s*(전재|복제).*$")
_CAP_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_CAP_STAMP_RE = re.compile(r"/?\s*\d{4}[-./]\s?\d{1,2}[-./]\s?\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*/?")
_CAP_END_RE = re.compile(r"[.!?…\"”’'」』。)]$")
_CAPTION_MAX = 400


def clean_image_caption(caption: str | None) -> str | None:
    """A caption as the reader should see it: copyright tags, photographer
    e-mails and upload timestamps ("…jieunlee@yna.co.kr/2026-09-17 19:48:00/
    <저작권자 ⓒ …>") dropped, whitespace normalised, capped in length.
    Pure and idempotent; None when nothing readable is left."""
    if not caption:
        return None
    text = _norm_ws(caption)
    text = _CAP_COPYRIGHT_TAIL_RE.sub("", text)
    text = _CAP_EMAIL_RE.sub("", text)
    text = _CAP_STAMP_RE.sub(" ", text)
    text = _norm_ws(text).strip(" /|·-")
    if len(text) > _CAPTION_MAX:
        cut = text[:_CAPTION_MAX]
        end = max(cut.rfind(". "), cut.rfind("다. "))
        text = cut[: end + 1] if end >= _CAPTION_MAX // 2 else cut.rstrip() + "…"
    return text or None


def _is_author_portrait(caption: str | None) -> bool:
    """Columnists' head-shots carry just "김준기 논설위원" as their caption —
    that is a byline picture, not a picture of the story."""
    if not caption:
        return False
    return (
        len(caption) <= 24
        and len(caption.split()) <= 4
        and _BYLINE_RE.search(caption) is not None
        and _CAP_END_RE.search(caption) is None
    )


def clean_images(images: list[dict] | None) -> list[dict]:
    """Normalise stored/extracted image dicts: clean captions and drop
    author portraits. Pure and idempotent, so it runs both when an article
    is scraped and again at read time (legacy rows get the same treatment
    with no migration)."""
    out: list[dict] = []
    for im in images or []:
        url = im.get("url")
        if not url:
            continue
        cap = clean_image_caption(im.get("caption"))
        if _is_author_portrait(cap):
            continue
        out.append({"url": url, "caption": cap, "after_paragraph": int(im.get("after_paragraph") or 0)})
    return out


# ------------------------------------------------------------------- TTS text --
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_TITLE_TAG_RE = re.compile(r"^\s*[\[［【〈<(（][^\]］】〉>)）]{1,12}[\]］】〉>)）]\s*")


def article_tts_text(title: str | None, body: str) -> str:
    """Exactly what "Nghe toàn văn" voices: the cleaned title (leading
    "[사설]"-style tag dropped) as its own paragraph, then the cleaned body,
    one paragraph per line. URLs/emails are removed — a TTS voice reads
    them character by character."""
    paras: list[str] = []
    if title and title.strip():
        t = _TITLE_TAG_RE.sub("", title.strip()).strip()
        if t:
            paras.append(t if _ends_sentence(t) else t + ".")
    for p in split_paragraphs(clean_article_text(body)):
        p = _EMAIL_RE.sub("", _URL_RE.sub("", p)).strip()
        if p:
            paras.append(p)
    return "\n".join(paras)


# ------------------------------------------------------------ HTML extraction --
_DROP_TAGS = {
    "script", "style", "noscript", "iframe", "form", "svg", "button", "select", "input",
    "template", "nav", "footer", "aside", "canvas", "video", "audio", "object", "embed",
}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
    "h6", "blockquote", "tr", "table", "td", "th", "dd", "dt", "dl", "pre", "hr", "address",
    "center", "header",
}

# Class/id fragments of containers that are never article text.
_BOILERPLATE_HINTS = (
    "comment", "reply", "disqus", "share", "sharing", "social", "sns-", "_sns", "sns_",
    "relate", "recommend", "popular", "ranking", "rank_", "hot_news", "hotnews",
    "tag_area", "taglist", "tag-list", "hashtag", "byline", "reporter", "journalist",
    "author-info", "copyright", "ad_area", "ad-area", "adbanner", "banner", "promotion",
    "subscribe", "newsletter", "print", "btn_area", "util", "font-size", "fontsize", "viewset",
    "view_set", "modal", "popup", "layer", "toolbar", "breadcrumb", "pagination", "paging",
)
_CAPTION_HINTS = ("caption", "img_desc", "photo_desc", "img-desc", "imgdesc", "imgcap", "img_cap", "photo_txt", "thumb_txt", "cap_")

_ARTICLE_SELECTORS = (
    "[itemprop=articleBody]", "#articleBody", "#article-view-content-div", "#article_body",
    "#articleBodyContents", "#newsct_article", "#dic_area", "#article_txt", ".article_txt",
    "#cont_newstext", "#newsEndContents", "#CmAdContent", ".art_body", ".article-body",
    ".article_body", ".article-text", ".article_text", ".news_body", ".news-body", ".view_con",
    ".view_cont", ".viewBox2", ".story-news", ".article_view", ".news_view", "article",
)

_IMG_URL_ATTRS = ("data-src", "data-original", "data-lazy-src", "data-lazy", "data-echo", "data-url", "src")
_IMG_BAD_HINTS = (
    "icon", "logo", "btn", "button", "banner", "sprite", "blank", "pixel", "loading", "spacer",
    "profile", "reporter", "byline", "emoji", "emoticon", "thumb_reporter", "/ad/", "/ads/",
    "adserver", "doubleclick", "share", "sns", "avatar", "favicon", "gravatar", "1x1",
)


@dataclass
class ExtractedArticle:
    title: str | None
    site_name: str | None
    body: str
    images: list[dict] = field(default_factory=list)
    strategy: str = "fallback"


def _attr_ident(el: Tag) -> str:
    cls = el.get("class") or []
    if isinstance(cls, str):
        cls = [cls]
    return f"{' '.join(cls)} {el.get('id') or ''}".lower()


def _is_alive(el) -> bool:
    return getattr(el, "attrs", None) is not None


def _parse_srcset(value: str) -> str | None:
    best, best_w = None, -1.0
    for part in value.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0.0
        if len(bits) > 1:
            try:
                w = float(bits[1].rstrip("wx"))
            except ValueError:
                w = 0.0
        if w >= best_w:
            best, best_w = bits[0], w
    return best


def _img_url(img: Tag, base_url: str) -> str | None:
    cands: list[str] = []
    for attr in ("data-srcset", "srcset"):
        v = img.get(attr)
        if v:
            picked = _parse_srcset(v)
            if picked:
                cands.append(picked)
    for attr in _IMG_URL_ATTRS:
        v = img.get(attr)
        if v:
            cands.append(v)
    for c in cands:
        c = c.strip()
        if not c or c.startswith("data:") or c.startswith("javascript:"):
            continue
        return urljoin(base_url, c)
    return None


def _img_ok(img: Tag, url: str) -> bool:
    low = url.lower()
    path = urlparse(low).path
    if path.endswith((".svg", ".ico")):
        return False
    ident = f"{_attr_ident(img)} {(img.get('alt') or '').lower()} {low}"
    if any(h in ident for h in _IMG_BAD_HINTS):
        return False
    for dim in ("width", "height"):
        v = str(img.get(dim) or "").strip().rstrip("px")
        if v.isdigit() and int(v) < 120:
            return False
    return True


def _img_key(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc}{p.path}".lower()


class _Walker:
    """Block-aware text walk: inline tags join, block tags/<br> break.
    Emits one line per paragraph and records images (with caption) at the
    line index where they sit."""

    def __init__(self, base_url: str, collect_images: bool):
        self.base_url = base_url
        self.collect_images = collect_images
        self.lines: list[str] = []
        self.images: list[dict] = []  # {url, caption, slot}
        self._buf: list[str] = []
        self._seen: set[str] = set()

    def _flush(self) -> None:
        if self._buf:
            text = _norm_ws("".join(self._buf))
            if text:
                self.lines.append(text)
            self._buf = []

    def _add_image(self, img: Tag, caption: str | None) -> None:
        if not self.collect_images:
            return
        url = _img_url(img, self.base_url)
        if not url or not _img_ok(img, url):
            return
        key = _img_key(url)
        if key in self._seen:
            return
        self._seen.add(key)
        cap = _norm_ws(caption) if caption else None
        if not cap:
            alt = _norm_ws(img.get("alt") or "")
            cap = alt if len(alt) >= 8 else None
        self._flush()
        self.images.append({"url": url, "caption": cap, "slot": len(self.lines)})

    def walk(self, node: Tag) -> None:
        for child in list(node.children):
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                self._buf.append(str(child))
                continue
            if not isinstance(child, Tag) or not _is_alive(child):
                continue
            name = (child.name or "").lower()
            if name in _DROP_TAGS:
                continue
            if name == "br":
                self._flush()
                continue
            if name == "img":
                self._add_image(child, None)
                continue
            if name == "figure" or name == "picture":
                caption_el = child.find("figcaption")
                cap = caption_el.get_text(" ") if caption_el else None
                if not cap:
                    for c in child.find_all(True):
                        if any(h in _attr_ident(c) for h in _CAPTION_HINTS):
                            cap = c.get_text(" ")
                            break
                imgs = child.find_all("img")
                for img in imgs:
                    self._add_image(img, cap)
                continue  # captions/links inside a figure never become text
            ident = _attr_ident(child)
            if ident.strip() and any(h in ident for h in _CAPTION_HINTS):
                # caption block sitting next to a bare <img>: attach it to
                # the previous image if that one has none yet, never text
                text = child.get_text(" ")
                if self.images and not self.images[-1]["caption"] and text.strip():
                    self.images[-1]["caption"] = _norm_ws(text)
                continue
            if name in _BLOCK_TAGS:
                self._flush()
                self.walk(child)
                self._flush()
            else:
                self.walk(child)
        # (buffer is flushed by the enclosing block)

    def finish(self) -> None:
        self._flush()


def _lines_of(container: Tag, base_url: str, collect_images: bool = False) -> _Walker:
    w = _Walker(base_url, collect_images)
    w.walk(container)
    w.finish()
    return w


def _score(container: Tag, base_url: str) -> int:
    w = _lines_of(container, base_url)
    return sum(len(ln) for ln in w.lines if _is_paragraph(ln) and not _is_noise_line(ln))


def _strip_boilerplate_inside(container: Tag) -> None:
    """Remove widget containers INSIDE the chosen article container. Never
    the container itself or its ancestors (a wrapper that merely happens
    to carry a hinted class must not delete the whole article), and never
    a hinted element that holds most of the container's text."""
    total = len(container.get_text(" ", strip=True)) or 1
    for el in list(container.find_all(True)):
        if not _is_alive(el) or el is container:
            continue
        ident = _attr_ident(el)
        if ident.strip() and any(h in ident for h in _BOILERPLATE_HINTS):
            if len(el.get_text(" ", strip=True)) > 0.6 * total:
                continue
            el.decompose()


def _json_ld_articles(soup: BeautifulSoup) -> list[dict]:
    found: list[dict] = []

    def visit(obj) -> None:
        if isinstance(obj, list):
            for o in obj:
                visit(o)
        elif isinstance(obj, dict):
            t = obj.get("@type")
            types = t if isinstance(t, list) else [t]
            if any(isinstance(x, str) and x.endswith("Article") for x in types):
                found.append(obj)
            for k in ("@graph", "mainEntity", "mainEntityOfPage"):
                if k in obj:
                    visit(obj[k])

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        try:
            visit(json.loads(raw))
        except (ValueError, TypeError):
            continue
    return found


def _json_ld_image_urls(article: dict, base_url: str) -> list[str]:
    img = article.get("image") or article.get("thumbnailUrl")
    out: list[str] = []
    items = img if isinstance(img, list) else [img]
    for it in items:
        if isinstance(it, str):
            out.append(urljoin(base_url, it))
        elif isinstance(it, dict) and it.get("url"):
            out.append(urljoin(base_url, str(it["url"])))
    return out


def _pick_container(root: Tag, base_url: str) -> tuple[Tag | None, str]:
    candidates: dict[int, tuple[Tag, str]] = {}

    for sel in _ARTICLE_SELECTORS:
        try:
            for el in root.select(sel):
                candidates.setdefault(id(el), (el, f"selector:{sel}"))
        except Exception:  # noqa: BLE001 — a bad selector must never break extraction
            continue

    # readability-style: parents/grandparents of long <p> and of <br>-separated blocks
    counts: dict[int, tuple[Tag, int]] = {}
    for p in root.find_all(["p", "div", "td"]):
        if not _is_alive(p):
            continue
        if p.name == "p":
            txt = p.get_text(" ", strip=True)
            weight = len(txt) if len(txt) >= 40 else 0
        else:
            weight = 200 if len(p.find_all("br", recursive=False)) >= 3 else 0
        if not weight:
            continue
        for level, anc in enumerate([p.parent, p.parent.parent if p.parent else None]):
            if isinstance(anc, Tag) and anc.name not in ("html", "[document]"):
                prev = counts.get(id(anc), (anc, 0))[1]
                counts[id(anc)] = (anc, prev + weight // (level + 1))
        if p.name == "div":
            counts[id(p)] = (p, counts.get(id(p), (p, 0))[1] + weight)
    for el, _ in sorted(counts.values(), key=lambda t: -t[1])[:12]:
        candidates.setdefault(id(el), (el, "density"))

    if not candidates:
        return None, "none"

    scored = [(_score(el, base_url), el, how) for el, how in candidates.values()]
    best = max(s for s, _, _ in scored)
    if best < 100:
        return None, "none"
    good = [(s, el, how) for s, el, how in scored if s >= 0.85 * best]

    def depth(el: Tag) -> int:
        return sum(1 for _ in el.parents)

    # among near-best candidates prefer an explicit selector match, then the deepest
    good.sort(key=lambda t: (0 if t[2].startswith("selector") else 1, -depth(t[1])))
    _, el, how = good[0]
    return el, how


def _clean_title(title: str | None, site_name: str | None) -> str | None:
    if not title:
        return None
    t = _norm_ws(title)
    for sep in (" - ", " | ", " : ", " :: ", " – ", " — "):
        if sep in t:
            head, _, tail = t.rpartition(sep)
            tail_l = tail.lower()
            if head and (
                (site_name and site_name.lower() in tail_l)
                or (site_name and tail_l in site_name.lower())
                or len(tail) <= 12
            ):
                t = head.strip()
                break
    return t or None


def extract_article(html: str, url: str, site_name: str | None = None) -> ExtractedArticle:
    soup = BeautifulSoup(html, "html.parser")

    ld_articles = _json_ld_articles(soup)

    title: str | None = None
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"]
    elif ld_articles and ld_articles[0].get("headline"):
        title = str(ld_articles[0]["headline"])
    elif soup.title and soup.title.string:
        title = soup.title.string
    site = site_name
    if not site:
        for attrs in ({"property": "og:site_name"}, {"name": "application-name"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                site = tag["content"].strip()
                break
    title = _clean_title(title, site)

    og_images: list[str] = []
    for attrs in ({"property": "og:image"}, {"name": "twitter:image"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            og_images.append(urljoin(url, tag["content"].strip()))
    for art in ld_articles:
        og_images.extend(_json_ld_image_urls(art, url))

    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()
    root = soup.body or soup

    container, how = _pick_container(root, url)
    images: list[dict] = []
    lines: list[str] = []
    strategy = "fallback"

    if container is not None:
        _strip_boilerplate_inside(container)
        w = _lines_of(container, url, collect_images=True)
        lines, images, strategy = w.lines, w.images, how
    else:
        ld_body = next((str(a["articleBody"]) for a in ld_articles if len(str(a.get("articleBody") or "")) >= 200), None)
        if ld_body:
            lines = [_norm_ws(x) for x in re.split(r"\n+|(?<=[.!?…])\s{2,}", ld_body) if _norm_ws(x)]
            strategy = "json-ld"
        else:
            _strip_boilerplate_inside(root)
            w = _lines_of(root, url, collect_images=True)
            lines, images = w.lines, w.images

    kept = select_body_lines(lines)
    body = "\n".join(lines[i] for i in kept)

    out_images: list[dict] = []
    seen_keys: set[str] = set()
    for im in images:
        n_before = sum(1 for i in kept if i < im["slot"])
        out_images.append({"url": im["url"], "caption": im["caption"], "after_paragraph": n_before})
        seen_keys.add(_img_key(im["url"]))
    if not out_images:
        for u in og_images:
            if _img_key(u) not in seen_keys and not any(h in u.lower() for h in _IMG_BAD_HINTS):
                out_images.append({"url": u, "caption": None, "after_paragraph": 0})
                break
    return ExtractedArticle(
        title=title, site_name=site, body=body, images=clean_images(out_images)[:MAX_IMAGES], strategy=strategy
    )
