"""Run: python -m unittest discover -s tests -v   (from the repo root)

The text fixtures below are REAL page-chrome noise copied from articles
stored in production (경향신문 / 세계일보 / 동아일보), the case this module
was written for. The HTML fixtures are small synthetic pages with the
shapes seen on Korean news CMSes.
"""
import unittest

from app.services.article_extract import (
    article_tts_text,
    clean_article_text,
    clean_image_caption,
    clean_images,
    extract_article,
    select_body_lines,
)

P1 = "미 국채시장이 불안하다. 10년물 금리가 5%, 30년물은 5.3%를 넘나들고 있다. 이란전쟁과 인플레 우려가 직접적 계기지만, 문제의 뿌리는 깊다."
P2 = "이 상황에서 떠올리게 되는 것이 스티븐 마이런의 2024년 보고서다. 세계 무역체제 재편을 위한 안내서인데, 트럼프 2기의 관세 및 달러 정책을 이해하는 이론적 배경이 되어왔다."
P3 = "금리를 이기는 시장이 없음에 유의하고, 겸손과 균형감으로 국채시장을 주시할 필요가 있겠다."

KHAN_HEAD = """보기 설정
닫기
글자 크기
보통
보통
크게
크게
아주 크게
아주 크게
컬러 모드
라이트
라이트
다크
다크
베이지
베이지
그린
그린
컬러 모드
닫기
본문 요약
닫기
인공지능 기술로 자동 요약된 내용입니다. 전체 내용을 이해하기 위해 본문과 함께 읽는 것을 추천합니다.
(제공 = 경향신문&NAVER MEDIA API)
내 뉴스플리에 저장
닫기
서울 강서구 여명학교에서 지난 17일 열린 주민설명회에서 주민들이 건립 반대 손팻말을 들고 있다. 여명학교는 북한이탈주민 청소년 대안교육기관이다. 연합뉴스"""

KHAN_TAIL = """이호승 전 대통령비서실 정책실장
지금 많이 보는 기사
읽어볼 만한 기사
뉴스룸 PICK
에디터 PICK
카메라 워크 K
아름다운 언덕에서 들려오는 세이렌 노래, 제리 율스만 기념관
오마주
타노스에게 죽은 로키, 왜 다시 나오지?···12월 ‘둠스데이’ 준비하려면 필수 시청
연재 레터 구독은 로그인 후 이용 가능합니다.
로그인
경향신문 홈으로 이동
아직 회원이 아니신가요?
닫기
닫기
닫기"""

SEGYE_HEAD = """스포츠월드
세계비즈
구독신청
mPaper
RSS
로그인
회원가입
로그아웃
회원정보수정
회원탈퇴
관련이슈
사설
입력 :
2026-09-23 21:22
인쇄
메일
url 공유
-
+
구글 선호 매체 등록
'주민동의 없는 여명학교 건립 반대' (서울=연합뉴스) 이지은 기자 = 17일 서울 강서구 여명학교에서 열린 주민설명회에서 주민들이 손팻말을 들고 있다. 2026.9.17 jieunlee@yna.co.kr/2026-09-17 19:48:00/ <저작권자 ⓒ 1980-2026 ㈜연합뉴스. 무단 전재 재배포 금지, AI 학습 및 활용 금지>"""

SEGYE_TAIL = """G
o
o
g
l
e
News
에서 세계일보 팔로우
20260923515724
0101100300000
0
2026-09-23 21:22
0
[사설] 여명학교 교장 '눈물' 호소… 더는 외면 말아야
세계일보
-
HOT뉴스
1
아이스커피 들고 구직 면접 갔다가…美직장가 달군 '커피게이
2
중식당 페인트 테러 용의자 4명 모두 중국인…사건 당일 출
포토"""

DONGA_HEAD = """크게보기
24일 대구 달서구 두류공원 무료급식소에서 노인들이 점심 배식을 기다리고 있다. 행정안전부에 따르면 23일 65세 이상 인구 비율이 20.0%가 되며 한국은 유엔이 정한 '초고령사회'에 진입했다. 대구=뉴스1"""

DONGA_TAIL = """1
[오늘의 운세/9월 28일]
2
량현량하 김량하, 내년 1월 결혼…상대는 레이싱모델 김희
3
강남역 이름 11억에 낙찰…지하철역 이름 팔아 280억 수익 봤다
4
김도영 결승타로 日에 설욕…웃을 수만은 없는 韓야구 AG 5연패
5"""


class CleanTextTests(unittest.TestCase):
    def check(self, head, paras, tail, must_not):
        raw = "\n".join([head, *paras, tail])
        out = clean_article_text(raw)
        self.assertEqual(out.split("\n")[: len(paras)], paras, out)
        for bad in must_not:
            self.assertNotIn(bad, out)
        return out

    def test_khan_real_noise(self):
        out = self.check(
            KHAN_HEAD, [P1, P2, P3], KHAN_TAIL,
            ["보기 설정", "글자 크기", "본문 요약", "자동 요약", "NAVER", "뉴스플리", "연합뉴스", "지금 많이 보는", "PICK", "닫기", "로그인"],
        )
        # the author credit right after the body is kept, nothing after it
        self.assertTrue(out.endswith("이호승 전 대통령비서실 정책실장"), out[-80:])

    def test_segye_real_noise(self):
        out = self.check(
            SEGYE_HEAD, [P1, P2], SEGYE_TAIL,
            ["스포츠월드", "구독신청", "회원가입", "구글 선호", "저작권자", "연합뉴스", "팔로우", "HOT뉴스", "20260923515724", "아이스커피"],
        )
        self.assertEqual(out.split("\n"), [P1, P2])

    def test_donga_real_noise(self):
        out = self.check(
            DONGA_HEAD, [P1, P3], DONGA_TAIL,
            ["크게보기", "뉴스1", "오늘의 운세", "레이싱모델", "강남역"],
        )
        self.assertEqual(out.split("\n"), [P1, P3])

    def test_idempotent_and_clean_text_untouched(self):
        clean = "\n".join([P1, P2, P3])
        self.assertEqual(clean_article_text(clean), clean)
        raw = "\n".join([KHAN_HEAD, P1, P2, KHAN_TAIL])
        once = clean_article_text(raw)
        self.assertEqual(clean_article_text(once), once)

    def test_keeps_short_lead_sentence_and_subheading_inside_body(self):
        lead = "국민 5명 중 1명이 노인이다."
        sub = "고령사회의 그늘"
        raw = "\n".join(["닫기", lead, P1, sub, P2, "지금 많이 보는 기사", "닫기"])
        out = clean_article_text(raw).split("\n")
        self.assertEqual(out, [lead, P1, sub, P2])

    def test_body_mentioning_privacy_terms_is_not_dropped(self):
        long_para = (
            "개인정보 처리 방침과 이용약관을 개정하려는 기업이 늘고 있지만 정작 이용자가 동의 내용을 읽는 경우는 드물어 "
            "제도의 실효성이 의문이라는 지적이 끊이지 않는다. 전문가들은 동의 절차를 단순화하고 핵심 내용을 "
            "한눈에 볼 수 있도록 표준화된 요약본을 의무적으로 제공해야 한다고 입을 모은다."
        )
        self.assertGreaterEqual(len(long_para), 120)
        out = clean_article_text("\n".join([long_para, P1]))
        self.assertIn("개인정보 처리 방침", out)

    def test_no_paragraph_at_all_falls_back_to_filtered_text(self):
        raw = "닫기\n봄\n여름\n가을\n겨울"
        self.assertEqual(clean_article_text(raw), "봄\n여름\n가을\n겨울")

    def test_select_body_lines_empty(self):
        self.assertEqual(select_body_lines([]), [])
        self.assertEqual(clean_article_text(""), "")


class TtsTextTests(unittest.TestCase):
    def test_title_first_tag_stripped_and_urls_removed(self):
        body = "\n".join([KHAN_HEAD, P1 + " https://example.com/x 참고.", P2, KHAN_TAIL])
        text = article_tts_text("[정동칼럼]미 국채시장, 마이런 보고서의 관점", body)
        lines = text.split("\n")
        self.assertEqual(lines[0], "미 국채시장, 마이런 보고서의 관점.")
        self.assertNotIn("http", text)
        self.assertNotIn("보기 설정", text)
        self.assertNotIn("지금 많이 보는", text)


KHAN_HTML = f"""<html><head>
<title>[정동칼럼]미 국채시장, 마이런 보고서의 관점 : 경향신문</title>
<meta property="og:title" content="[정동칼럼]미 국채시장, 마이런 보고서의 관점">
<meta property="og:site_name" content="경향신문">
<meta property="og:image" content="https://img.khan.co.kr/hero.jpg">
</head><body>
<header><a href="/">경향신문 홈으로 이동</a></header>
<div class="view-set"><div class="modal"><h3>보기 설정</h3><button>닫기</button>
<ul><li>글자 크기</li><li>보통</li><li>크게</li><li>컬러 모드</li></ul></div>
<div class="summary-layer"><h3>본문 요약</h3><p>인공지능 기술로 자동 요약된 내용입니다. 전체 내용을 이해하기 위해 본문과 함께 읽는 것을 추천합니다.</p></div></div>
<div class="art_cont">
  <h1>[정동칼럼]미 국채시장, 마이런 보고서의 관점</h1>
  <div class="art_body" id="articleBody">
    <figure class="art_photo"><img src="//img.khan.co.kr/a.jpg" width="800" alt="">
      <figcaption>미국 국채 이미지. 연합뉴스</figcaption></figure>
    <p class="content_text">{P1}</p>
    <p class="content_text">이 상황에서 떠올리게 되는 것이 <b>스티븐 마이런</b>의 <a href="/x">2024년 보고서</a>다. 세계 무역체제 재편을 위한 안내서인데, 트럼프 2기의 관세 및 달러 정책을 이해하는 이론적 배경이 되어왔다.</p>
    <figure><img data-src="/photos/b.jpg" src="data:image/gif;base64,R0lGOD"><figcaption>두 번째 사진 설명입니다. 게티이미지뱅크</figcaption></figure>
    <p class="content_text">{P3}</p>
  </div>
  <div class="byline"><p>이호승 전 대통령비서실 정책실장</p></div>
</div>
<div class="related"><h3>지금 많이 보는 기사</h3><ul><li><a>타노스에게 죽은 로키, 왜 다시 나오지?</a></li></ul></div>
<div id="comment_area"><p>댓글을 남겨주세요. 회원만 작성할 수 있습니다.</p></div>
<footer><p>ⓒ 경향신문, 무단 전재 및 재배포 금지</p></footer>
</body></html>"""


class ExtractHtmlTests(unittest.TestCase):
    def test_khan_like_page(self):
        art = extract_article(KHAN_HTML, "https://www.khan.co.kr/article/202609231846025/")
        self.assertEqual(art.site_name, "경향신문")
        self.assertEqual(art.title, "[정동칼럼]미 국채시장, 마이런 보고서의 관점")
        paras = art.body.split("\n")
        self.assertEqual(len(paras), 3, paras)
        self.assertEqual(paras[0], P1)
        # inline <b>/<a> must NOT split the sentence into several lines
        self.assertIn("스티븐 마이런의 2024년 보고서다.", paras[1])
        self.assertEqual(paras[2], P3)
        for bad in ["보기 설정", "자동 요약", "닫기", "연합뉴스", "게티이미지", "지금 많이 보는", "댓글", "무단 전재"]:
            self.assertNotIn(bad, art.body)
        self.assertTrue(art.strategy.startswith("selector"), art.strategy)

    def test_images_with_captions_and_positions(self):
        art = extract_article(KHAN_HTML, "https://www.khan.co.kr/article/202609231846025/")
        self.assertEqual(len(art.images), 2, art.images)
        first, second = art.images
        self.assertEqual(first["url"], "https://img.khan.co.kr/a.jpg")
        self.assertEqual(first["caption"], "미국 국채 이미지. 연합뉴스")
        self.assertEqual(first["after_paragraph"], 0)  # before the first paragraph
        self.assertEqual(second["url"], "https://www.khan.co.kr/photos/b.jpg")  # lazy data-src wins over the data: placeholder
        self.assertEqual(second["after_paragraph"], 2)  # between paragraph 2 and 3

    def test_density_fallback_without_known_selectors(self):
        html = f"""<html><body>
        <div class="top"><a>홈</a><a>정치</a><a>경제</a></div>
        <div class="col-main"><div class="xyz"><h1>제목</h1>
          <p>{P1}</p><p>{P2}</p><p>{P3}</p></div>
          <div class="other"><p>구독 문의</p><ul><li>인기 기사 하나</li></ul></div></div>
        <div class="side"><p>짧은 광고 문구</p></div></body></html>"""
        art = extract_article(html, "https://example.com/a")
        self.assertEqual(art.body.split("\n"), [P1, P2, P3])
        self.assertEqual(art.strategy, "density")

    def test_br_separated_body(self):
        html = f"""<html><body><div id="wrap"><div id="content">
        {P1}<br><br>{P2}<br>{P3}<br><br>
        </div><div id="foot">연락처 02-000-0000</div></div></body></html>"""
        art = extract_article(html, "https://example.com/b")
        self.assertEqual(art.body.split("\n"), [P1, P2, P3])

    def test_hinted_wrapper_does_not_delete_the_article(self):
        html = f"""<html><body><div class="sns_on print_view"><article>
        <p>{P1}</p><p>{P2}</p></article></div></body></html>"""
        art = extract_article(html, "https://example.com/c")
        self.assertEqual(art.body.split("\n"), [P1, P2])

    def test_og_image_used_when_no_inline_images(self):
        html = f"""<html><head><meta property="og:image" content="/cover.jpg"></head>
        <body><article><p>{P1}</p><p>{P2}</p></article></body></html>"""
        art = extract_article(html, "https://example.com/d")
        self.assertEqual(art.images, [{"url": "https://example.com/cover.jpg", "caption": None, "after_paragraph": 0}])

    def test_icons_and_tiny_images_skipped(self):
        html = f"""<html><body><article>
        <img src="/img/icon_share.png"><img src="/x/logo.png"><img src="/p.jpg" width="40">
        <p>{P1}</p><p>{P2}</p><img src="/real.jpg" alt="현장 사진 설명입니다"></article></body></html>"""
        art = extract_article(html, "https://example.com/e")
        self.assertEqual([i["url"] for i in art.images], ["https://example.com/real.jpg"])
        self.assertEqual(art.images[0]["caption"], "현장 사진 설명입니다")
        self.assertEqual(art.images[0]["after_paragraph"], 2)


class ImageCleanupTests(unittest.TestCase):
    # Real caption pulled from a 세계일보 article on production.
    SEGYE_CAP = (
        "'주민동의 없는 여명학교 건립 반대' (서울=연합뉴스) 이지은 기자 = 17일 서울 강서구 여명학교에서 열린 "
        "주민설명회에서 주민들이 손팻말을 들고 있다. 2026.9.17 jieunlee@yna.co.kr/2026-09-17 19:48:00/ "
        "<저작권자 ⓒ 1980-2026 ㈜연합뉴스. 무단 전재 재배포 금지, AI 학습 및 활용 금지>"
    )

    def test_caption_loses_copyright_email_and_timestamps(self):
        cap = clean_image_caption(self.SEGYE_CAP)
        self.assertTrue(cap.endswith("손팻말을 들고 있다."), cap)
        for junk in ("jieunlee@", "저작권자", "19:48", "무단"):
            self.assertNotIn(junk, cap)

    def test_plain_caption_untouched_and_idempotent(self):
        cap = "이헌석 정책위원이 지난 15일 인터뷰에서 문제점을 설명하고 있다. 강윤중 선임기자"
        self.assertEqual(clean_image_caption(cap), cap)
        once = clean_image_caption(self.SEGYE_CAP)
        self.assertEqual(clean_image_caption(once), once)

    def test_empty_or_only_junk_caption_is_none(self):
        self.assertIsNone(clean_image_caption(None))
        self.assertIsNone(clean_image_caption("  "))
        self.assertIsNone(clean_image_caption("<저작권자 ⓒ 경향신문, 무단 전재 및 재배포 금지>"))

    def test_long_caption_capped_at_a_sentence(self):
        long_cap = "이것은 매우 긴 설명이다. " * 60
        cap = clean_image_caption(long_cap)
        self.assertLessEqual(len(cap), 401)
        self.assertTrue(cap.endswith(".") or cap.endswith("…"))

    def test_author_portrait_dropped_but_real_photos_kept(self):
        imgs = [
            {"url": "https://x/a.jpg", "caption": "이헌석 정책위원이 인터뷰에서 문제점을 설명하고 있다. 강윤중 선임기자", "after_paragraph": 0},
            {"url": "https://x/b.jpg", "caption": "김준기 논설위원", "after_paragraph": 43},
            {"url": "https://x/c.jpg", "caption": None, "after_paragraph": 5},
            {"url": "", "caption": "빈 주소", "after_paragraph": 1},
        ]
        out = clean_images(imgs)
        self.assertEqual([i["url"] for i in out], ["https://x/a.jpg", "https://x/c.jpg"])
        self.assertEqual(clean_images(out), out)  # idempotent

    def test_extract_applies_cleanup(self):
        html = f"""<html><body><article>
        <figure><img src="/p/1.jpg"><figcaption>{self.SEGYE_CAP}</figcaption></figure>
        <p>{P1}</p><p>{P2}</p>
        <figure><img src="/p/2.jpg"><figcaption>김준기 논설위원</figcaption></figure></article></body></html>"""
        art = extract_article(html, "https://example.com/f")
        self.assertEqual([i["url"] for i in art.images], ["https://example.com/p/1.jpg"])
        self.assertNotIn("저작권자", art.images[0]["caption"])


if __name__ == "__main__":
    unittest.main()
