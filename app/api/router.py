from fastapi import APIRouter

from app.api.routers import admin, audio, auth, content, ingest, jobs, writing

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(audio.router)
api_router.include_router(writing.router)
api_router.include_router(ingest.router)
api_router.include_router(admin.router)
api_router.include_router(content.router)
