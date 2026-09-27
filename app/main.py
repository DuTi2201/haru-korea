from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import settings

app = FastAPI(
    title="Haru API",
    version="0.1.0",
    description="Backend for the Haru Korean-learning app (FastAPI + Celery + Redis + Postgres/pgvector).",
)

# CORS: the Lovable frontend is a separate origin; no cookies, Bearer-JWT
# only, so allow_credentials stays False. cors_origin_list covers exact
# known origins (published domain); the regex covers Lovable's rotating
# id-preview-*.lovable.app subdomains without needing a redeploy per preview.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def problem_json_handler(request: Request, exc: StarletteHTTPException):
    """Normalizes every HTTPException into RFC 9457 problem+json, per
    SDD §7. `detail` is either already a problem dict (raised via
    app.api.deps._problem) or a plain string (framework-raised errors)."""
    body = exc.detail
    if not isinstance(body, dict):
        body = {"type": "about:blank", "title": str(body), "status": exc.status_code}
    return JSONResponse(status_code=exc.status_code, content=body, media_type="application/problem+json")


@app.get("/healthz", tags=["ops"])
async def healthz():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


app.include_router(api_router)
