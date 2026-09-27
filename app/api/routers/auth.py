from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _problem, get_current_profile
from app.core.security import create_access_token, create_refresh_token, hash_password, verify_password
from app.db import get_db
from app.models import Profile
from app.schemas import LoginRequest, ProfileOut, SignupRequest, TokenPair

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


@router.get("/me", response_model=ProfileOut)
async def me(profile: Annotated[Profile, Depends(get_current_profile)]):
    return profile
