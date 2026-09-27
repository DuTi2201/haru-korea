"""Thin wrapper around the Gemini API — the SDD's "AI Gateway": every
call goes through here so prompt version, structured-output schema and
cost logging are consistent regardless of which module calls it.

This is intentionally a stub with a clear extension point
(`generate_structured`) rather than a full implementation of every
prompt in the SDD — fill in real prompts per module as each one is built.
"""
from __future__ import annotations

from typing import Any

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def generate_structured(
    *,
    model: str,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    """Call Gemini with a JSON response schema. Every caller MUST log
    (model, prompt_version, token counts) to `app_config`/analytics —
    wire that up once a real prompt lands here; omitted in this stub.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = _get_client()
    config: dict[str, Any] = {"response_mime_type": "application/json"}
    if response_schema:
        config["response_schema"] = response_schema

    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return {"text": response.text, "prompt_version": prompt_version}
