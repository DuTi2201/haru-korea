import redis.asyncio as aioredis
import redis as redis_sync

from app.core.config import settings

# Async client — used by FastAPI's SSE endpoint to subscribe.
async_redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

# Sync client — used by Celery tasks (sync context) to publish.
sync_redis = redis_sync.from_url(settings.REDIS_URL, decode_responses=True)


def job_channel(job_id: str) -> str:
    return f"job:{job_id}:events"
