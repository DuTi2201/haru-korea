from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config, read from env vars (Railway service Variables).

    Mirrors SDD §5 "cấu hình thay cho hard-code": nothing here should be
    hard-coded into business logic — rubric versions, model names and
    prompt versions belong in the `app_config` table (see app/models.py),
    not here. This file is transport/infra config only.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENVIRONMENT: str = "development"

    # Postgres — asyncpg driver for the app, psycopg2 for Alembic (sync)
    DATABASE_URL: str = "postgresql+asyncpg://haru:haru@localhost:5432/haru"

    @property
    def SYNC_DATABASE_URL(self) -> str:
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

    # Redis — broker/result backend for Celery AND the job.progress pub/sub
    # channel that replaces the SDD's SSE-over-Redis fan-out.
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS — the Lovable frontend is a separate origin (no Supabase, no
    # same-origin cookies), so this must be explicit and not "*".
    CORS_ORIGINS: str = ""
    CORS_ORIGIN_REGEX: str = r"^https://.*\.lovable\.app$"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Auth
    JWT_SECRET: str = "dev-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TTL_MIN: int = 60
    JWT_REFRESH_TTL_DAYS: int = 30

    # Gemini — model choice is config, not hard-coded (SDD §5 principle 6);
    # override per-env on Railway if a cheaper/newer model becomes the
    # better fit. Flash-Lite for both ingestion jobs: high volume, well-
    # constrained JSON schema, no need for Pro-tier reasoning.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_LESSON_INGEST: str = "gemini-3.5-flash-lite"
    GEMINI_MODEL_CORPUS_INGEST: str = "gemini-3.5-flash-lite"
    # Reads an exam paper off its pages (app/services/exam_extract.py). Copying
    # every character exactly is the one job where a stronger model pays off;
    # empty = GEMINI_MODEL_LESSON_INGEST. Set on Railway, no deploy of code.
    GEMINI_MODEL_EXAM_INGEST: str = ""
    # Read the paper a second time and show where the two readings differ.
    EXAM_VERIFY_PASS: bool = True
    # Text model for the beginner study pack (translation / simplified Korean /
    # word breakdown) — see app/services/study_pack.py. Override via env to use
    # a stronger model without a code change.
    GEMINI_MODEL_STUDY: str = "gemini-3.5-flash-lite"
    # Model that writes the "Bài giảng tổng hợp" script (story, phrases,
    # synonyms, idioms — see app/services/podcast_script.py). Empty = use
    # GEMINI_MODEL_LESSON_INGEST; set a stronger model here for richer lectures.
    GEMINI_MODEL_PODCAST: str = ""
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    # TTS: a real Gemini call (app/services/tts.py), not a stub — always
    # cache the result (audio.lecture_audio / audio.corpus_item_audio) and
    # never call this per-playback.
    GEMINI_MODEL_TTS: str = "gemini-2.5-flash-preview-tts"
    # Gemini's free-tier TTS quota is tracked per-model (e.g. a hard 10
    # requests/day cap on gemini-2.5-flash-tts alone), so a fallback model
    # is a genuinely separate quota bucket, not just cosmetic redundancy.
    # synthesize_korean_tts only fails over to this on a RESOURCE_EXHAUSTED
    # (429) from the primary model — kept a same-generation TTS model (not
    # one of the newer 3.x models) since it's confirmed to use the exact
    # same generate_content/response_modalities/speech_config call shape.
    GEMINI_MODEL_TTS_FALLBACK: str = "gemini-2.5-pro-preview-tts"

    # Google Cloud Text-to-Speech, Chirp 3 HD — the PRIMARY voice for Korean
    # article / passage read-aloud (app/services/google_tts.py); Gemini above
    # stays as the fallback. The credential is ONLY ever read from the
    # environment (Railway variable `TTS_GG_Chirp`; env names are matched
    # case-insensitively) and must never be written into code, docs or
    # tests. Empty => Chirp is off and everything behaves exactly as before.
    TTS_GG_CHIRP: str = ""
    # Non-secret tuning: change on Railway to swap the voice or pace without a
    # deploy of code. Changing either also changes the article-audio cache key
    # (see google_tts.spec), so recordings regenerate with the new voice.
    TTS_CHIRP_VOICE: str = "ko-KR-Chirp3-HD-Iapetus"
    TTS_CHIRP_SPEAKING_RATE: float = 0.85
    # The "Bài giảng tổng hợp" lecture mixes Vietnamese explanation with Korean
    # examples. A Chirp voice belongs to ONE language, so the lecture is voiced
    # run by run: Hangul with TTS_CHIRP_VOICE (pace above, slowed for
    # shadowing) and Vietnamese with the voice below. Empty voice = the same
    # persona as the Korean one in vi-VN (e.g. vi-VN-Chirp3-HD-Iapetus), so
    # the lecturer sounds like one person. Both are cache-key inputs.
    TTS_CHIRP_VOICE_VI: str = ""
    TTS_CHIRP_SPEAKING_RATE_VI: float = 0.95

    # Content-ingestion tuning (Studio uploads — admin.py/ingest.py)
    MAX_INGEST_FILE_MB: int = 20
    # Cues per Gemini call when classifying a subtitle file — the knob
    # that keeps FR-19/Gate-G6's "prompt size không phình theo độ dài
    # kịch bản" true: each call's prompt is this many cues, however long
    # the film is; a longer film means more calls, never a bigger prompt.
    CORPUS_CHUNK_SIZE: int = 40
    # Sentences per Gemini call when adding a meaning / usage note to corpus
    # sentences (app/services/corpus_enrich.py). Smaller than the classifier's
    # chunk because the answer is long-form text per sentence; the pause keeps
    # a backfill of the whole corpus under the per-minute request limit.
    CORPUS_ENRICH_CHUNK_SIZE: int = 20
    CORPUS_ENRICH_PAUSE_SEC: float = 1.0

    # Rate limiting (Postgres-table equivalent of the SDD's Redis quota)
    AI_DAILY_JOB_LIMIT: int = 200


settings = Settings()
