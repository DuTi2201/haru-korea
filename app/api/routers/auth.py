import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile
from app.core.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password
from app.db import get_db
from app.models import Profile
from app.schemas import LoginRequest, ProfileOut, RefreshRequest, SignupRequest, TokenPair

router = APIRouter(prefix="/auth", tags=["identity"])


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    existing = await db.execute(select(Profile).where(Profile.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise _problem(status.HTTP_409_CONFLICT, "Email already registered", "email_taken")

    profile = Profile(
        email=body.email,
        hashed_password=hash_password(body.password),
        display_name=body.display_name,
        role="learner",
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return TokenPair(
        access_token=create_access_token(str(profile.id), profile.role),
        refresh_token=create_refresh_token(str(profile.id), profile.role),
    )


@router.post("/login", response_model=TokenPair)
async def login(body: LoginRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(Profile).where(Profile.email == body.email))
    profile = result.scalar_one_or_none()
    if profile is None or not verify_password(body.password, profile.hashed_password):
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Invalid credentials", "invalid_credentials")
    return TokenPair(
        access_token=create_access_token(str(profile.id), profile.role),
        refresh_token=create_refresh_token(str(profile.id), profile.role),
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    """Exchanges a still-valid refresh token for a new TokenPair.

    Access tokens are short-lived (JWT_ACCESS_TTL_MIN, currently 60 min) so
    a session doesn't stay silently authenticated forever, but until now
    there was no way to actually redeem the long-lived refresh_token that
    login/signup already hand out — apiFetch had nothing to call, so every
    session became permanently logged-out ("Invalid or expired token" on
    every request) exactly one access-token TTL after login, with no way
    back in short of logging in again. This is the missing other half of
    that flow: the frontend now calls this on a 401 invalid_token and
    retries once (see client.ts) instead of surfacing the error.

    Rotates BOTH tokens (not just re-signing a new access token) so a
    leaked refresh token has a bounded lifetime of one use, same rationale
    as the access/refresh split itself.
    """
    try:
        claims = decode_token(body.refresh_token)
    except ValueError as exc:
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", "invalid_token") from exc
    if claims.get("type") != "refresh":
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", "invalid_token")
    profile = await db.get(Profile, uuid.UUID(claims["sub"]))
    if profile is None:
        raise _problem(status.HTTP_401_UNAUTHORIZED, "Profile not found", "unauthenticated")
    return TokenPair(
        access_token=create_access_token(str(profile.id), profile.role),
        refresh_token=create_refresh_token(str(profile.id), profile.role),
    )


@router.get("/me", response_model=ProfileOut)
async def me(profile: Annotated[Profile, Depends(get_current_profile)]):
    return profile
