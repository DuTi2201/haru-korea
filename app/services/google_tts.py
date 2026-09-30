"""Google Cloud Text-to-Speech (Chirp 3 HD) over plain REST.

Why this exists: Gemini's TTS has a tiny free-tier daily quota and 500s on
long input, which made whole-article read-aloud unreliable. Chirp 3 HD is
billed per character (the first 1M characters/month are free), takes up to
5,000 bytes per request, and has a fixed `speakingRate`, so a slower,
shadowing-friendly pace is one parameter away.

Only the transport lives here — chunking, retry, silence and the Gemini
fallback are in app/services/tts.py, and the DB/cache bookkeeping is in the
Celery task. Returns raw 16-bit mono PCM + its sample rate, the same shape the
Gemini path produces, so the existing ffmpeg transcode is reused unchanged.

SECURITY: the credential comes only from settings.TTS_GG_CHIRP (the Railway
variable `TTS_GG_Chirp`). It is sent in a request HEADER, never in the URL, so
it cannot end up in an httpx exception message or an access log, and nothing
in this module prints or returns it. Error text surfaced to callers is
Google's own error message (which never echoes the key) or a fixed string.
"""
from __future__ import annotations

import base64
import json
import re
import struct
import threading

import httpx

from app.core.config import settings

ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# The API accepts 5,000 bytes of input per request. A Korean syllable is 3
# bytes in UTF-8 (worst case 4 for an emoji), so 900 characters is at most
# 3,600 bytes — comfortably inside the limit for any text.
CHUNK_CHARS = 900
MAX_REQUEST_BYTES = 5000

_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-Chirp3-HD-[A-Za-z]+$")


class GoogleTTSError(RuntimeError):
    """The request cannot succeed as sent (bad key, API not enabled, bad
    voice, ...). Retrying will not help; callers fall back to Gemini."""


class GoogleTTSTransient(GoogleTTSError):
    """Worth retrying: rate limit (429), 5xx, network timeout."""


# ---------------------------------------------------------------- settings --
def _credential() -> str:
    raw = (settings.TTS_GG_CHIRP or "").strip()
    # A value pasted into a Railway field sometimes keeps its quotes.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    return raw


def is_configured() -> bool:
    return bool(_credential())


def voice_name() -> str:
    return (settings.TTS_CHIRP_VOICE or "").strip() or "ko-KR-Chirp3-HD-Iapetus"


def speaking_rate() -> float:
    # The API accepts 0.25–2.0; stay inside it whatever the env says.
    return min(2.0, max(0.25, float(settings.TTS_CHIRP_SPEAKING_RATE)))


def language_code() -> str:
    parts = voice_name().split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else "ko-KR"


def spec() -> str:
    """Short, cache-key-safe id of the voice + pace in use, e.g.
    "gc3-iapetus-r85". Part of the article-audio cache key, so changing the
    voice or the rate on Railway makes recordings regenerate instead of
    serving the old voice."""
    short = re.sub(r"[^a-z0-9]", "", voice_name().split("-")[-1].lower()) or "voice"
    return f"gc3-{short}-r{int(round(speaking_rate() * 100))}"


# -------------------------------------------------------------------- auth --
_sa_lock = threading.Lock()
_sa_credentials = None  # cached google.oauth2 service-account credentials


def _service_account_token(raw: str) -> str:
    """Bearer token for a service-account JSON pasted into the variable
    (an API key is the simple, recommended form; this just means pasting the
    JSON instead does not silently break)."""
    global _sa_credentials
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover - google-auth ships with google-genai
        raise GoogleTTSError("google-auth is not installed") from exc
    with _sa_lock:
        if _sa_credentials is None:
            try:
                info = json.loads(raw)
                _sa_credentials = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
            except (ValueError, KeyError) as exc:
                raise GoogleTTSError(
                    "TTS_GG_Chirp trông như JSON nhưng không phải service account hợp lệ."
                ) from exc
        if not _sa_credentials.valid:
            try:
                _sa_credentials.refresh(Request())
            except Exception as exc:  # noqa: BLE001 - google-auth raises its own hierarchy
                raise GoogleTTSTransient("Không lấy được access token từ service account.") from exc
        return _sa_credentials.token


def _auth_headers() -> dict[str, str]:
    raw = _credential()
    if not raw:
        raise GoogleTTSError("TTS_GG_Chirp is not set")
    if raw.startswith("{"):
        return {"Authorization": f"Bearer {_service_account_token(raw)}"}
    return {"X-Goog-Api-Key": raw}


# ------------------------------------------------------------------- audio --
def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    """LINEAR16 responses come wrapped in a WAV container. Strip it and return
    (16-bit mono PCM, sample_rate). Walks the RIFF chunks instead of assuming a
    44-byte header, and tolerates a data-chunk size that is 0/oversized."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data, 24000  # already raw PCM
    rate, channels, bits = 24000, 1, 16
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body_start = pos + 8
        if chunk_id == b"fmt " and size >= 16:
            _fmt, channels, rate, _brate, _align, bits = struct.unpack("<HHIIHH", data[body_start : body_start + 16])
        elif chunk_id == b"data":
            end = body_start + size
            pcm = data[body_start:] if size == 0 or end > len(data) else data[body_start:end]
            if channels != 1 or bits != 16:
                raise GoogleTTSError(f"Unexpected audio format from Google TTS ({channels}ch/{bits}bit)")
            if not pcm:
                raise GoogleTTSTransient("Google TTS returned empty audio")
            return pcm, rate
        pos = body_start + size + (size & 1)
    raise GoogleTTSError("Google TTS returned a WAV without audio data")


def _google_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        msg = err.get("message") or ""
        status = err.get("status") or ""
        return f"{status}: {msg}".strip(": ")[:300]
    except Exception:  # noqa: BLE001 - a non-JSON error body
        return (resp.text or "")[:200]


def synthesize_pcm(text: str, *, client: httpx.Client | None = None) -> tuple[bytes, int]:
    """One request -> (pcm_bytes, sample_rate). Raises GoogleTTSTransient for a
    failure worth retrying and GoogleTTSError for one that is not."""
    text = text.strip()
    if not text:
        raise GoogleTTSError("empty text")
    if len(text.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise GoogleTTSError("text exceeds the 5,000-byte request limit")

    voice = voice_name()
    if not _VOICE_RE.match(voice):
        raise GoogleTTSError(f"TTS_CHIRP_VOICE '{voice}' không phải tên giọng Chirp 3 HD")
    body = {
        "input": {"text": text},
        "voice": {"languageCode": language_code(), "name": voice},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": speaking_rate()},
    }
    headers = _auth_headers()
    try:
        if client is not None:
            resp = client.post(ENDPOINT, json=body, headers=headers)
        else:
            resp = httpx.post(ENDPOINT, json=body, headers=headers, timeout=_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise GoogleTTSTransient("Google TTS timed out") from exc
    except httpx.TransportError as exc:
        raise GoogleTTSTransient(f"Google TTS network error ({type(exc).__name__})") from exc

    if resp.status_code != 200:
        detail = _google_message(resp)
        msg = f"Google TTS HTTP {resp.status_code}" + (f" — {detail}" if detail else "")
        if resp.status_code in _TRANSIENT_STATUS:
            raise GoogleTTSTransient(msg)
        raise GoogleTTSError(msg)

    try:
        audio = base64.b64decode(resp.json().get("audioContent") or "")
    except (ValueError, TypeError) as exc:
        raise GoogleTTSTransient("Google TTS returned an unreadable body") from exc
    if not audio:
        raise GoogleTTSTransient("Google TTS returned no audio")
    return wav_to_pcm(audio)
