"""Gemini text-to-speech + ffmpeg transcode — the real implementation
behind both `audio.lecture_audio` (whole-lesson narration) and
`audio.corpus_item_audio` (single-sentence listening clips). One call
here == one Gemini TTS request, so every caller MUST go through the
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

from google import genai
from google.genai import errors as genai_errors

from app.core.config import settings

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
            raise RuntimeError(
                f"Gemini TTS tạm thời gặp lỗi ({exc.code} {exc.status or 'unknown'}), thử lại sau ít phút."
            ) from exc
    assert response is not None  # loop always either returns via break or raises

    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else None
    inline = parts[0].inline_data if parts else None
    if inline is None or not inline.data:
        raise RuntimeError("Gemini TTS returned no inline audio data")

    match = _RATE_RE.search(inline.mime_type or "")
    sample_rate = int(match.group(1)) if match else 24000
    return inline.data, sample_rate


def synthesize_korean_tts(text_ko: str, voice: str = "ko-female-1") -> tuple[bytes, bytes, int]:
    """Returns (opus_bytes, aac_bytes, duration_sec). Raises RuntimeError on
    any failure (missing key, empty/blocked response, ffmpeg error) — the
    caller (a Celery task) turns that into a job.failed row, same as every
    other AI-backed task in this codebase."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    text_ko = text_ko.strip()
    if not text_ko:
        raise RuntimeError("empty text_ko")

    client = _get_client()
    voice_name = _resolve_voice(voice)
    # Plain dict config (not typed types.GenerateContentConfig(...)) to
    # match this pinned SDK's proven-working style in gemini_client.py —
    # the SDK's dict->proto conversion accepts snake_case keys here.
    #
    # NOTE: this used to also send `system_instruction`, added to stop a
    # long conversational script (the podcast feature) from reading enough
    # like a request/task that the model tried to *answer* it instead of
    # voicing it (400 "Model tried to generate text, but it should only be
    # used for TTS"). That turned out to be the wrong fix: Gemini's native
    # TTS docs (response_modalities=["AUDIO"]) document only
    # response_modalities/speech_config for these models — no
    # system_instruction — and every podcast call started failing with a
    # generic 500 INTERNAL the moment system_instruction was added,
    # reproducing on the very first ~1600-char chunk regardless of content.
    # Removed; the actual fix for the 400 is at the source — the script
    # PROMPT (build_podcast_prompt in ingestion.py) already forbids the
    # model from writing instruction-like phrasing into the script itself,
    # so the text handed to TTS is already plain narration.
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": voice_name}},
        },
    }
    # Gemini's free-tier TTS quota is a hard per-model daily cap (observed:
    # 10 requests/day for gemini-2.5-flash-tts) — every TTS caller
    # (lecture/corpus/vocab/podcast audio) shares this same call, so one
    # busy test day exhausts it for all of them at once. The quota is
    # tracked per model, so a same-shape fallback model is a genuinely
    # separate bucket, not just a retry of the same failure. Only a
    # RESOURCE_EXHAUSTED (429) falls through to the next model — any other
    # APIError (bad request, transient 5xx, etc.) is the same regardless of
    # model, so it's raised immediately instead of burning a second call.
    models_to_try = [settings.GEMINI_MODEL_TTS]
    if settings.GEMINI_MODEL_TTS_FALLBACK and settings.GEMINI_MODEL_TTS_FALLBACK not in models_to_try:
        models_to_try.append(settings.GEMINI_MODEL_TTS_FALLBACK)

    # A long single-shot input (the podcast script — a whole lesson's
    # vocab+grammar, easily several thousand characters) reliably 500s the
    # native TTS endpoint; short callers (one word/sentence) never do. Chunk
    # at sentence boundaries and concatenate the raw PCM before transcoding
    # once, so long scripts get the same reliability short callers already
    # have without changing this function's signature or callers.
    chunks = _split_for_tts(text_ko, _MAX_TTS_CHARS)
    pcm_parts: list[bytes] = []
    sample_rate = 24000
    # TEMPORARY diagnostic logging (Railway captures worker stdout): the
    # 500 INTERNAL from Gemini is generic and gives no hint of *why*, so
    # this pins down exactly what was sent on the chunk that fails —
    # length, chunk count, and a text preview — without needing to guess
    # again. Safe to remove once the actual cause is confirmed.
    print(
        f"[tts-debug] total_len={len(text_ko)} chunks={len(chunks)} "
        f"chunk_lens={[len(c) for c in chunks]} voice={voice_name}",
        flush=True,
    )
    for idx, chunk in enumerate(chunks):
        print(f"[tts-debug] chunk {idx + 1}/{len(chunks)} len={len(chunk)} text={chunk!r}", flush=True)
        pcm, sample_rate = _synthesize_chunk_pcm(client, chunk, config, models_to_try)
        pcm_parts.append(pcm)

    return transcode_pcm(b"".join(pcm_parts), sample_rate)
