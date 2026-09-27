# Haru backend — FastAPI + Celery worker, single image, two entrypoints.
# Railway builds this once per service (api, worker) and each service
# overrides CMD via its Railway "Custom Start Command" (see README.md).
FROM python:3.11-slim AS base

# ffmpeg: required by the audio pipeline (lecture_audio opus/aac transcode)
# per SDD section 9 (audio module). libpq-dev/gcc: build deps for psycopg2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv

# Default: API server. The "worker" Railway service overrides this with
# `celery -A app.core.celery_app worker --loglevel=info --concurrency=2`
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
