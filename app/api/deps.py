import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db import get_db
from app.models import Profile

bearer_scheme = HTTPBearer(auto_error=False)


def _problem(status_code: int, title: str, code: str, detail: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"type": "about:blank", "title": title, "status": status_code, "code": code, "detail": detail},
    )


async def get_current_claims(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict:
    if creds is None:
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Missing bearer token", "unauthenticated")
    try:
        return decode_token(creds.credentials)
    except ValueError as exc:
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", "invalid_token") from exc


async def get_current_profile(
    claims: Annotated[dict, Depends(get_current_claims)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Profile:
    profile = await db.get(Profile, uuid.UUID(claims["sub"]))
    if profile is None:
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Profile not found", "unauthenticated")
    return profile


def require_role(*roles: str):
    async def _dep(profile: Annotated[Profile, Depends(get_current_profile)]) -> Profile:
        if profile.role not in roles:
            raise _problem(
                status.HTTP_403_FORBIDDEN, "Insufficient role", "forbidden", f"requires one of {roles}"
            )
        return profile

    return _dep
