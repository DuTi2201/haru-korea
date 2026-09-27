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

    # Gemini
    GEMINI_API_KEY: str = ""

    # Rate limiting (Postgres-table equivalent of the SDD's Redis quota)
    AI_DAILY_JOB_LIMIT: int = 200


settings = Settings()
