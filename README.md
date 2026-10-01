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
| `GEMINI_MODEL_LESSON_INGEST` | optional, defaults to `gemini-3.5-flash-lite` |
| `GEMINI_MODEL_CORPUS_INGEST` | optional, defaults to `gemini-3.5-flash-lite` |
| `GEMINI_MODEL_STUDY` | optional, model for the beginner study pack (translation / simple Korean); defaults to `gemini-3.5-flash-lite` |
| `TTS_GG_Chirp` | Google Cloud Text-to-Speech API key (Chirp 3 HD, the primary Korean voice). Set ONLY as a Railway variable on the api AND worker services, never in code; empty = Gemini TTS only |
| `TTS_CHIRP_VOICE` | optional, defaults to `ko-KR-Chirp3-HD-Iapetus`; changing it regenerates article audio |
| `TTS_CHIRP_SPEAKING_RATE` | optional, defaults to `0.85` (API range 0.25–2.0); changing it regenerates article audio |
| `TTS_CHIRP_VOICE_VI` | optional; the Vietnamese voice of the "Bài giảng tổng hợp" lecture (Hangul runs use `TTS_CHIRP_VOICE`, everything else this one). Empty = the same persona in vi-VN (e.g. `vi-VN-Chirp3-HD-Iapetus`); changing it regenerates lectures |
| `TTS_CHIRP_SPEAKING_RATE_VI` | optional, defaults to `0.95`; changing it regenerates lectures |
| `GEMINI_MODEL_PODCAST` | optional; the model that WRITES the lecture script (empty = `GEMINI_MODEL_LESSON_INGEST`). A stronger model gives richer lectures |
| `GEMINI_EMBEDDING_MODEL` | optional, defaults to `gemini-embedding-2` |
| `MAX_INGEST_FILE_MB` | optional, defaults to `20` — caps a single Studio upload |
| `CORPUS_CHUNK_SIZE` | optional, defaults to `40` — subtitle cues per Gemini call |

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
the writing-submission state machine's shape, problem+json error
responses, CORS for the Lovable origin, and the full ingestion pipeline
for all three `import_batch` kinds: `POST /api/v1/imports` (multipart
upload, `kind=lesson|corpus|exam_paper`) → Gemini extraction/
classification (Celery) → `GET .../items` + `PATCH .../items/{id}`
review queue → `POST .../confirm` (Celery, writes content.lesson/
vocab_item/grammar_point, corpus.corpus_item+embedding, or content.
exam_passage/exam_item) → `POST .../rollback` (lesson and exam_paper:
full undo, even after confirm, via import_item lineage; corpus: only
before confirm — see `CorpusItem`'s docstring in `app/models.py` for
why). `kind=exam_paper` requires `exam_kind`/`session_label` form fields
and classifies each question against a starter `content.question_type`
taxonomy seeded by the `d1a4e9f2b6c7` migration (12 common TOPIK I/II
reading+listening categories — extend that table directly for more);
an answer is only trusted as `answer_source="editor"` when the model
read it off a printed answer key in the document, otherwise it's staged
as `ai_guess` and always force-flagged for review regardless of
confidence, since a wrong exam answer is worse than most other content
errors. This is a single-pass multimodal extraction (passages + items +
answer resolution in one Gemini call per paper), not literally a
"3-pass pipeline" — that phrase in an earlier pass of this README was
this project's own draft label, never a verbatim SRS/SDD requirement,
so it's corrected here.

Also real: learner-facing reads for the content this pipeline produces —
`GET /api/v1/lessons/{id}` (accepts a numeric id or the literal `today`,
which just picks the earliest confirmed lesson as a placeholder "next
lesson" — real adaptive scheduling is still stubbed, see below) returns
the lesson with its topics/vocab/grammar; `GET /api/v1/corpus/items`
returns a random sample of corpus.corpus_item for listening practice,
with film/topic/grammar-pattern names resolved via separate bulk queries
in application code rather than a cross-schema SQL join (SDD module-
boundary principle); `POST /api/v1/progress/reviews` nudges a learner's
`item_state.strength` (the readiness signal) after a quick vocab/grammar
check and moves the item's review schedule (`app/services/srs.py`: a small
SM-2-style ladder of 1 day, 3 days, then each gap times `ease`; a wrong answer
restarts it and brings the card back in ten minutes; an answer given before
the card is due does not move the schedule). `GET /api/v1/me/review-queue` is
today's sitting: the cards that are due, then new cards (at most 8 per rolling
24 h, about one grammar point in four) from the earliest lessons; a due
vocabulary card that was answered right before comes with a fill-in-the-blank
built from it without a model call (`app/services/exercises.py`: the node word
of a chunk is blanked, the wrong choices are the card's own `distractors`, then
the other chunks of its family).

Lessons are read in chunks, not word lists (`PROMPT_VERSION = "lesson-v4"` in
`app/services/lesson_extract.py`): an inventory call, a short call that sorts the
terms into families (verbs that go with weather, the temperature scale) and the
grammar patterns into contrast groups (the look-alike "것 같다" forms), then the
cards with the chunk layers (`family`, `node_word`, `register`, `usage_note_vi`,
`collocations`, `distractors`; grammar: `contrast_group`, `contrasts`). TOPIK
tests which words go together and which pattern is closest in meaning, so a card
is learned with its set.

Verified end-to-end against a local Postgres+pgvector+Redis with the
Gemini calls mocked (real network calls need a live API key, which this
sandbox doesn't have) — see the ingestion Pydantic extraction schemas
and prompts in `app/services/ingestion.py`.

File uploads are base64'd straight through the Celery/Redis message
(no object storage provisioned yet) — fine for admin-tool volumes, capped
by `MAX_INGEST_FILE_MB`; swap for real object storage if that stops
being true. Subtitle files are classified in fixed-size chunks
(`CORPUS_CHUNK_SIZE`) specifically so each Gemini call's prompt stays a
constant size regardless of film length (FR-19 / Gate G6's "kích thước
prompt không phình theo độ dài kịch bản"); exam papers don't need this
(bounded page count), so they're extracted in one call per paper.

Stubbed (`TODO` in code, intentionally — this is a first scaffold, not
the finished app): ffmpeg audio transcoding (currently a `sleep`),
Storage integration for writing-submission photo/audio uploads
(image_key/opus_path are placeholder strings — the lesson/corpus/exam
ingestion above solved this differently, see above), error-driven weakness
practice (`ErrorLog` is not written yet), AI cost logging /
admin usage dashboard, and per-learner daily AI quota enforcement.
