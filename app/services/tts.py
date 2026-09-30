"""Text-to-speech + ffmpeg transcode — the real implementation behind
`audio.lecture_audio` (whole-lesson narration), `audio.corpus_item_audio`
(single-sentence listening clips) and the article read-aloud.

Voices: Google Chirp 3 HD (app/services/google_tts.py) is the PRIMARY voice
for Korean-only text whenever the `TTS_GG_Chirp` credential is configured
(`synthesize_article_chirp`, `synthesize_korean_tts_chirp_first`); Gemini TTS
below is the fallback, and the only voice for the podcast script, which mixes
Vietnamese with Korean (a ko-KR voice cannot read Vietnamese). One call
here == one TTS request, so every caller MUST go through the
content-addressed cache (cache_key = f"{content_id}:{voice}:{prompt_version}",
already the pattern audio.py's lecture-audio route uses) rather than
calling this on every playback — that's what keeps repeat listens free.

Gemini's TTS models return headerless 16-bit signed little-endian PCM
(default 24kHz mono for a unary/non-streaming request); ffmpeg transcodes
that once into two containers so the frontend can always play something:
Opus/Ogg (small, good quality, Chrome/Firefox/Android) and AAC/ADTS
(larger, but the one that also plays on Safari/iOS, since Ogg-Opus
support there is inconsistent).
"""
from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable

from google import genai
from google.genai import errors as genai_errors

from app.core.config import settings
from app.services import google_tts

# Friendly ids (already the shape of LectureAudioRequest.voice's default,
# "ko-female-1") mapped to one of Gemini's prebuilt voice names. Kept as a
# small indirection layer (SDD §5 "cấu hình thay cho hard-code") so the
# API contract doesn't have to change if Gemini's voice catalog does.
VOICE_MAP: dict[str, str] = {
    "ko-female-1": "Kore",
    "ko-female-2": "Leda",
    "ko-male-1": "Charon",
    "ko-male-2": "Orus",
}
DEFAULT_GEMINI_VOICE = "Kore"

_RATE_RE = re.compile(r"rate=(\d+)")


class TransientTTSError(RuntimeError):
    """A failure that is worth retrying (Gemini 5xx / empty audio) — as
    opposed to quota exhaustion or a content rejection, where a retry only
    burns more quota. Still a RuntimeError, so every existing caller that
    catches RuntimeError is unaffected."""

# The actual cause of the podcast 500s turned out to be the (now-removed)
# system_instruction field, not length — a ~1600-char chunk reproduced the
# same 500 regardless. Kept as a defensive ceiling anyway: Google doesn't
# document a hard input limit for native TTS, and a whole-lesson script
# can run to several thousand characters, so this keeps each request to
# roughly a minute of speech. Below this, text is sent as one chunk
# exactly like before (so short callers are unaffected); above it, text is
# split at sentence boundaries and synthesized as multiple chunks whose
# raw PCM gets concatenated before a single transcode.
_MAX_TTS_CHARS = 1600
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _resolve_voice(voice: str) -> str:
    return VOICE_MAP.get(voice) or (voice if voice in VOICE_MAP.values() else DEFAULT_GEMINI_VOICE)


def _split_for_tts(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentence-like pieces into chunks no larger than
    max_chars, splitting only at a sentence boundary (a punctuation mark
    the podcast script already uses for pauses) so a chunk edge never
    lands mid-clause. Falls back to a hard split for the rare single
    "sentence" that alone exceeds max_chars (e.g. a long compound example
    with no terminal punctuation)."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
        else:
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start : start + max_chars])
    if current:
        chunks.append(current)
    return chunks or [text.strip()]


def _run_ffmpeg(pcm_bytes: bytes, sample_rate: int, *, codec_args: list[str]) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *codec_args, "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(codec_args)}): {proc.stderr.decode('utf-8', errors='replace')[:500]}")
    return proc.stdout


def transcode_pcm(pcm_bytes: bytes, sample_rate: int = 24000) -> tuple[bytes, bytes, int]:
    """Pure transcode step, split out from `synthesize_korean_tts` so it can
    be exercised in tests without a live Gemini call (feed it synthetic PCM
    instead). Returns (opus_bytes, aac_bytes, duration_sec)."""
    if not pcm_bytes:
        raise RuntimeError("empty pcm_bytes")
    opus_bytes = _run_ffmpeg(pcm_bytes, sample_rate, codec_args=["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"])
    aac_bytes = _run_ffmpeg(pcm_bytes, sample_rate, codec_args=["-c:a", "aac", "-b:a", "96k", "-f", "adts"])
    duration_sec = round(len(pcm_bytes) / (sample_rate * 2))  # 16-bit mono PCM: 2 bytes/sample
    return opus_bytes, aac_bytes, duration_sec


def _synthesize_chunk_pcm(client: genai.Client, text: str, config: dict, models_to_try: list[str]) -> tuple[bytes, int]:
    """One chunk of text -> (pcm_bytes, sample_rate), trying each model in
    order and falling through to the next only on a quota (429) error.
    Raises RuntimeError with a clean, user-facing message otherwise."""
    response = None
    for i, model_name in enumerate(models_to_try):
        try:
            response = client.models.generate_content(model=model_name, contents=text, config=config)
            break
        except genai_errors.APIError as exc:
            is_quota = exc.status == "RESOURCE_EXHAUSTED" or exc.code == 429
            if is_quota and i < len(models_to_try) - 1:
                continue
            if is_quota:
                # Surface a clean, actionable message instead of the raw SDK
                # repr (a huge nested-dict string) — this is what ends up
                # verbatim in job.error.message and is shown to the user.
                raise RuntimeError(
                    "Đã hết hạn mức Gemini TTS miễn phí trong hôm nay (đã thử cả model dự phòng). "
                    "Thử lại vào ngày mai, hoặc bật billing (pay-as-you-go) cho API key trong "
                    "Google AI Studio / Google Cloud Console để tăng hạn mức."
                ) from exc
            if exc.code == 400 and "tried to generate text" in (exc.message or ""):
                # Content-shape rejection, not a transient failure: some
                # phrasing in this particular chunk (imperative/exam-style
                # wording like "hãy chọn đáp án...") reads to the TTS model
                # like a task to perform rather than a transcript to voice.
                # Confirmed in production against a TOPIK-tip sentence
                # phrased as a command; build_podcast_prompt now steers the
                # script generator away from that phrasing, but since each
                # retry regenerates a fresh script, retrying can still help.
                raise RuntimeError(
                    "Một đoạn trong kịch bản bị Gemini hiểu nhầm là câu lệnh thay vì lời thoại cần "
                    "đọc (thường do câu mẹo TOPIK viết theo kiểu ra lệnh). Thử lại để kịch bản được "
                    "viết lại theo cách diễn đạt khác."
                ) from exc
            raise TransientTTSError(
                f"Gemini TTS tạm thời gặp lỗi ({exc.code} {exc.status or 'unknown'}), thử lại sau ít phút."
            ) from exc
    assert response is not None  # loop always either returns via break or raises

    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else None
    inline = parts[0].inline_data if parts else None
    if inline is None or not inline.data:
        raise TransientTTSError("Gemini TTS returned no inline audio data")

    match = _RATE_RE.search(inline.mime_type or "")
    sample_rate = int(match.group(1)) if match else 24000
    return inline.data, sample_rate


def _tts_config(voice: str) -> tuple[dict, str]:
    """(request config, resolved Gemini voice name). Plain dict config (not
    typed types.GenerateContentConfig(...)) to match this pinned SDK's
    proven-working style in gemini_client.py — the SDK's dict->proto
    conversion accepts snake_case keys here.

    NOTE: no `system_instruction`. It was once added to stop a long
    conversational script from being *answered* instead of voiced (400
    "Model tried to generate text, but it should only be used for TTS") and
    turned out to be the wrong fix: Gemini's native TTS documents only
    response_modalities/speech_config, and every podcast call started
    failing with a generic 500 the moment it was added. The 400 is
    prevented at the source instead — script prompts forbid instruction-like
    phrasing, and article audio voices real news prose."""
    voice_name = _resolve_voice(voice)
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": voice_name}},
        },
    }
    return config, voice_name


def _tts_models() -> list[str]:
    """Gemini's free-tier TTS quota is a hard per-model daily cap (observed:
    10 requests/day for gemini-2.5-flash-tts) shared by every TTS caller
    (lecture/corpus/vocab/podcast/article audio). The quota is tracked per
    model, so a same-shape fallback model is a genuinely separate bucket.
    Only a RESOURCE_EXHAUSTED (429) falls through to the next model — any
    other APIError is the same regardless of model, so it is raised
    immediately instead of burning a second call."""
    models = [settings.GEMINI_MODEL_TTS]
    if settings.GEMINI_MODEL_TTS_FALLBACK and settings.GEMINI_MODEL_TTS_FALLBACK not in models:
        models.append(settings.GEMINI_MODEL_TTS_FALLBACK)
    return models


def synthesize_korean_tts(text_ko: str, voice: str = "ko-female-1") -> tuple[bytes, bytes, int]:
    """Returns (opus_bytes, aac_bytes, duration_sec). Raises RuntimeError on
    any failure (missing key, empty/blocked response, ffmpeg error) — the
    caller (a Celery task) turns that into a job.failed row, same as every
    other AI-backed task in this codebase.

    A long single-shot input (the podcast script — a whole lesson's
    vocab+grammar, easily several thousand characters) reliably 500s the
    native TTS endpoint; short callers (one word/sentence) never do. Text is
    chunked at sentence boundaries and the raw PCM concatenated before ONE
    transcode, so long scripts get the same reliability short callers have.
    For real article text (paragraph pauses, retries, progress) use
    synthesize_article_tts instead."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    text_ko = text_ko.strip()
    if not text_ko:
        raise RuntimeError("empty text_ko")

    client = _get_client()
    config, voice_name = _tts_config(voice)
    models_to_try = _tts_models()
    chunks = _split_for_tts(text_ko, _MAX_TTS_CHARS)
    pcm_parts: list[bytes] = []
    sample_rate = 24000
    print(f"[tts] chars={len(text_ko)} chunks={len(chunks)} voice={voice_name}", flush=True)
    for chunk in chunks:
        pcm, sample_rate = _synthesize_chunk_pcm(client, chunk, config, models_to_try)
        pcm_parts.append(pcm)

    return transcode_pcm(b"".join(pcm_parts), sample_rate)


# ------------------------------------------------------ article read-aloud --
# Paragraph-aligned chunks of ~1200 chars: small enough that one bad
# request costs little and the voice does not drift over a very long input,
# large enough to keep the request count (= quota) low.
_ARTICLE_CHUNK_CHARS = 1200
_ARTICLE_MAX_CHARS = 8000  # ~ a 20-minute read; beyond this the tail is not voiced
_PAUSE_AFTER_PARAGRAPH_MS = 700
_PAUSE_AFTER_SENTENCE_MS = 250
_CHUNK_RETRY_DELAYS_SEC = (2.0, 6.0)


def plan_article_chunks(
    text: str, max_chars: int = _ARTICLE_CHUNK_CHARS, max_total: int = _ARTICLE_MAX_CHARS
) -> list[tuple[str, int]]:
    """text (one paragraph per line) -> [(chunk_text, pause_after_ms)].

    Chunks end on a paragraph boundary whenever possible, so the silence
    inserted between chunks is a real paragraph pause; paragraphs inside one
    chunk are separated by a blank line (the model's own pacing). A single
    paragraph longer than max_chars is split at sentence boundaries and
    gets the shorter sentence pause between its pieces. `max_total` caps the
    characters voiced (the tail of a very long text is dropped)."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    size = 0
    total = 0

    def flush(pause_ms: int) -> None:
        nonlocal current, size
        if current:
            chunks.append(("\n\n".join(current), pause_ms))
            current, size = [], 0

    for para in paragraphs:
        if total + len(para) > max_total and chunks:
            break
        total += len(para)
        if len(para) > max_chars:
            flush(_PAUSE_AFTER_PARAGRAPH_MS)
            pieces = _split_for_tts(para, max_chars)
            for i, piece in enumerate(pieces):
                last = i == len(pieces) - 1
                chunks.append((piece, _PAUSE_AFTER_PARAGRAPH_MS if last else _PAUSE_AFTER_SENTENCE_MS))
            continue
        if current and size + len(para) + 2 > max_chars:
            flush(_PAUSE_AFTER_PARAGRAPH_MS)
        current.append(para)
        size += len(para) + 2
    flush(0)
    if chunks:
        chunks[-1] = (chunks[-1][0], 0)
    return chunks


def _silence_pcm(ms: int, sample_rate: int) -> bytes:
    return b"\x00" * (int(sample_rate * ms / 1000) * 2)  # 16-bit mono


def synthesize_article_tts(
    text: str,
    voice: str = "ko-female-1",
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[bytes, bytes, int]:
    """Read a whole news article aloud. Differences from synthesize_korean_tts:
    paragraph-aligned chunking, real silence between chunks (so paragraph
    breaks are audible instead of being flattened into one run-on blob),
    per-chunk retry on transient Gemini errors (a 5xx on chunk 4 used to
    throw away chunks 1-3 and their quota), and an `on_progress(done,
    total)` callback so the UI can show "2/4" instead of a bare spinner.
    Returns (opus_bytes, aac_bytes, duration_sec)."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    plan = plan_article_chunks(text)
    if not plan:
        raise RuntimeError("empty text_ko")

    client = _get_client()
    config, voice_name = _tts_config(voice)
    models_to_try = _tts_models()
    print(
        f"[tts] article chars={len(text)} chunks={len(plan)} chunk_lens={[len(c) for c, _ in plan]} voice={voice_name}",
        flush=True,
    )
    if on_progress:
        on_progress(0, len(plan))

    pcm_parts: list[bytes] = []
    sample_rate = 24000
    for idx, (chunk, pause_ms) in enumerate(plan):
        attempt = 0
        while True:
            try:
                pcm, sample_rate = _synthesize_chunk_pcm(client, chunk, config, models_to_try)
                break
            except TransientTTSError:
                if attempt >= len(_CHUNK_RETRY_DELAYS_SEC):
                    raise
                time.sleep(_CHUNK_RETRY_DELAYS_SEC[attempt])
                attempt += 1
        pcm_parts.append(pcm)
        if pause_ms:
            pcm_parts.append(_silence_pcm(pause_ms, sample_rate))
        if on_progress:
            on_progress(idx + 1, len(plan))

    return transcode_pcm(b"".join(pcm_parts), sample_rate)


# ------------------------------------------------- Google Chirp 3 HD (primary)
# Chirp is cheap per character and its request limit (5,000 bytes) is enforced
# by google_tts.CHUNK_CHARS, so a longer read-aloud cap than Gemini's is fine.
_CHIRP_ARTICLE_MAX_CHARS = 12000


def chirp_enabled() -> bool:
    """True when the TTS_GG_Chirp credential is present in the environment."""
    return google_tts.is_configured()


def chirp_spec() -> str:
    """Voice + pace id (e.g. "gc3-iapetus-r85") for cache keys."""
    return google_tts.spec()


def describe_error(exc: BaseException) -> str:
    """One-line, log-safe description of a failure. Never includes the
    credential: it travels in a request header, not in any exception text."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}"[:400] if text else type(exc).__name__


def _chirp_chunk_pcm(text: str) -> tuple[bytes, int]:
    """One Chirp request; a retryable failure is mapped onto TransientTTSError
    so the same retry loop as the Gemini path applies. A non-retryable
    GoogleTTSError is already a RuntimeError and simply propagates."""
    try:
        return google_tts.synthesize_pcm(text)
    except google_tts.GoogleTTSTransient as exc:
        raise TransientTTSError(str(exc)) from exc


def _retry_transient(fn: Callable[[str], tuple[bytes, int]], text: str) -> tuple[bytes, int]:
    attempt = 0
    while True:
        try:
            return fn(text)
        except TransientTTSError:
            if attempt >= len(_CHUNK_RETRY_DELAYS_SEC):
                raise
            time.sleep(_CHUNK_RETRY_DELAYS_SEC[attempt])
            attempt += 1


def _join_pcm(parts: list[tuple[bytes, int, int]]) -> tuple[bytes, int]:
    """[(pcm, sample_rate, pause_after_ms)] -> one PCM stream with the pauses
    inserted as silence. All chunks come from one voice, so one sample rate;
    a mismatch would garble the join, so it is an error, not a shrug."""
    rates = {rate for _, rate, _ in parts}
    if len(rates) != 1:
        raise RuntimeError(f"Google TTS returned mixed sample rates {sorted(rates)}")
    rate = rates.pop()
    out: list[bytes] = []
    for pcm, _rate, pause_ms in parts:
        out.append(pcm)
        if pause_ms:
            out.append(_silence_pcm(pause_ms, rate))
    return b"".join(out), rate


def synthesize_article_chirp(
    text: str, on_progress: Callable[[int, int], None] | None = None
) -> tuple[bytes, bytes, int]:
    """Read a whole article aloud with Google Chirp 3 HD: same paragraph-aligned
    plan, real silence between chunks, per-chunk retry of transient errors and
    progress callback as the Gemini path, but ~900-character chunks (the
    request limit is in bytes). Raises RuntimeError on any failure — the
    caller (the Celery task) decides whether to fall back to Gemini; this never
    mixes two voices inside one recording."""
    if not google_tts.is_configured():
        raise RuntimeError("TTS_GG_Chirp is not set")
    plan = plan_article_chunks(text, google_tts.CHUNK_CHARS, _CHIRP_ARTICLE_MAX_CHARS)
    if not plan:
        raise RuntimeError("empty text_ko")
    print(
        f"[tts] article provider=google-chirp spec={google_tts.spec()} chars={len(text)} chunks={len(plan)}",
        flush=True,
    )
    if on_progress:
        on_progress(0, len(plan))

    parts: list[tuple[bytes, int, int]] = []
    for idx, (chunk, pause_ms) in enumerate(plan):
        pcm, rate = _retry_transient(_chirp_chunk_pcm, chunk)
        parts.append((pcm, rate, pause_ms))
        if on_progress:
            on_progress(idx + 1, len(plan))

    pcm_all, rate = _join_pcm(parts)
    return transcode_pcm(pcm_all, rate)


def _synthesize_korean_chirp(text: str) -> tuple[bytes, bytes, int]:
    chunks = _split_for_tts(text, google_tts.CHUNK_CHARS)
    parts = []
    for chunk in chunks:
        pcm, rate = _retry_transient(_chirp_chunk_pcm, chunk)
        parts.append((pcm, rate, 0))
    print(f"[tts] chars={len(text)} chunks={len(chunks)} provider=google-chirp spec={google_tts.spec()}", flush=True)
    pcm_all, rate = _join_pcm(parts)
    return transcode_pcm(pcm_all, rate)


def synthesize_korean_tts_chirp_first(text_ko: str, voice: str = "ko-female-1") -> tuple[bytes, bytes, int]:
    """Korean-ONLY text (a lesson passage, one listening sentence, one word):
    Google Chirp 3 HD when configured, Gemini when it is not or when Chirp
    fails. Not for the podcast script — that mixes in Vietnamese, which a
    ko-KR voice cannot read; it stays on synthesize_korean_tts."""
    text = text_ko.strip()
    if not text:
        raise RuntimeError("empty text_ko")
    if not google_tts.is_configured():
        return synthesize_korean_tts(text, voice)
    try:
        return _synthesize_korean_chirp(text)
    except Exception as exc:  # noqa: BLE001 - any Chirp failure falls back rather than losing the audio
        reason = describe_error(exc)
        print(f"[tts] google-chirp failed ({reason}); falling back to Gemini", flush=True)
        try:
            return synthesize_korean_tts(text, voice)
        except RuntimeError as gem_exc:
            raise RuntimeError(f"Google Chirp lỗi ({reason}); Gemini dự phòng cũng lỗi: {gem_exc}") from gem_exc
