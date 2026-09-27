# Haru backend

FastAPI + Celery + Redis + PostgreSQL(pgvector) backend for the Haru
Korean-learning app, per the project's SDD (decoupled architecture:
this is the API/worker side; the Lovable project is a pure client).

Verified locally end-to-end (Postgres 16 + pgvector 0.6, Redis 7, Python
3.11): Alembic migration applies and reverses cleanly and repeatedly;
signup → JWT → `POST /api/v1/lessons/{id}/lecture` → 202 + job_id →
Celery worker → Redis-published progress → `GET /api/v1/jobs/{id}`
resolves to `succeeded` with a result payload — the exact 202+job_id+SSE
pattern the SDD requires for every AI-backed call.

## Architecture at a glance

- **api** service: `uvicorn app.main:app` — FastAPI, stateless, no direct
  Gemini calls.
- **worker** service: `celery -A app.core.celery_app worker` — every
  Gemini call and heavy transform (ffmpeg, OCR, grading, exam ingestion)
  runs here. Same image as `api`, different start command.
- **Postgres (pgvector-railway)**: source of truth, including the `jobs`
  table (durable job state) and `vector` columns for corpus/exam-passage/
  exam-item embeddings.
- **Redis**: Celery broker/result backend AND the `job:{id}:events`
  pub/sub channel the SSE endpoint (`GET /api/v1/jobs/{id}/events`)
  subscribes to. This is the direct replacement for an in-process
  SSE-over-Redis design — same event shape (`job.progress`,
  `job.succeeded`, `job.failed`), split across two Railway services.

See `app/models.py` and `app/schemas.py` for the full data model / API
contract the Lovable (or any) frontend should mirror in TypeScript.

## Deploying on Railway (two services, one image)

Both `api` and `worker` build from this same repo/Dockerfile; only the
**Custom Start Command** differs per Railway service:

- `api`: leave the Dockerfile's default `CMD` (uvicorn), or set explicitly:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- `worker`: set Custom Start Command to
  `celery -A app.core.celery_app worker --loglevel=info --concurrency=2`

Both services need the same variables (Railway lets you share them at
the environment level):

| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{ pgvector-railway.DATABASE_URL }}` but rewritten to the `postgresql+asyncpg://` scheme (Railway's template gives `postgresql://`; edit the scheme prefix) |
| `REDIS_URL` | `${{ Redis.REDIS_URL }}` |
| `CORS_ORIGINS` | the Lovable published domain, e.g. `https://haru.lovable.app` |
| `CORS_ORIGIN_REGEX` | `^https://.*\.lovable\.app$` (covers every preview subdomain) |
| `JWT_SECRET` | a long random string |
| `GEMINI_API_KEY` | server-side only, never exposed to the client |

After first deploy of `api`, run the migration once (Railway's one-off
command / shell, or a Release Command on the service):
`alembic upgrade head`

Generate a public domain for `api` only (`worker` needs none — it has
no HTTP server). Give that domain to the Lovable project as
`VITE_API_BASE_URL`.

## Local development

```
cp .env.example .env
docker compose up --build
# api on :8000, worker consuming the same Redis/Postgres
alembic upgrade head   # first time, or after a schema change
```

## What's stubbed vs. real

Real and tested: auth (signup/login/JWT/roles), the generic `jobs`
read+SSE endpoints, the full `generate_lecture_audio` job end-to-end,
the writing-submission state machine's shape, the import/review-queue
shape, problem+json error responses, CORS for the Lovable origin.

Stubbed (`TODO` in code, intentionally — this is a first scaffold, not
the finished app): real Gemini prompts for every module, ffmpeg audio
transcoding (currently a `sleep`), Storage integration for
photo/audio uploads (image_key/opus_path are placeholder strings), the
3-pass exam-ingestion pipeline, SRS scheduling logic, AI cost logging /
admin usage dashboard, and per-learner daily AI quota enforcement.
