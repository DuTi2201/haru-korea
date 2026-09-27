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

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _resolve_voice(voice: str) -> str:
    return VOICE_MAP.get(voice) or (voice if voice in VOICE_MAP.values() else DEFAULT_GEMINI_VOICE)


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
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": voice_name}},
        },
    }
    response = client.models.generate_content(model=settings.GEMINI_MODEL_TTS, contents=text_ko, config=config)

    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else None
    inline = parts[0].inline_data if parts else None
    if inline is None or not inline.data:
        raise RuntimeError("Gemini TTS returned no inline audio data")

    match = _RATE_RE.search(inline.mime_type or "")
    sample_rate = int(match.group(1)) if match else 24000
    return transcode_pcm(inline.data, sample_rate)
