"""idempotently promote/create the owner's admin profile

Owner asked for an admin account to build real content data via Studio
(POST /imports, /editorial-sources, etc — all gated on role="admin" per
the SDD's editor/admin permission model). There is no self-service way
to become admin (POST /auth/signup always sets role="learner" — see
app/api/routers/auth.py), so this has to be seeded directly.

Behavior, kept deliberately non-destructive both ways:
  - If a profile with this email already exists (e.g. from an earlier
    signup attempt), only its `role` is promoted to "admin". Its
    existing password is left untouched — we don't know it, and
    silently overwriting it would lock the owner out of whatever they
    already set.
  - If no such profile exists yet, one is created with role="admin"
    and a freshly generated, randomly-generated bcrypt password hash
    (via the same passlib/bcrypt scheme as app/core/security.py). Only
    the hash is stored here — same as every other profile's
    hashed_password column — the plaintext was shared with the owner
    directly in chat and is not written to this file or anywhere else
    in the repo.
  - downgrade() only reverts the role back to "learner"; it never
    deletes the profile row, since an automatic delete of a real
    account (and its FK-linked ItemState/WritingSubmission/etc rows)
    on a routine migration rollback would be needlessly destructive.
"""
import uuid

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9e21fa6c8b3d'
down_revision = '5a8e20e9f6d8'
branch_labels = None
depends_on = None

ADMIN_EMAIL = "vuvanduong802@gmail.com"
# bcrypt hash of a randomly generated 20-char password, shared with the
# owner directly in chat (never committed in plaintext anywhere).
ADMIN_PASSWORD_HASH = "$2b$12$WekB55BkOMZ2PeOR3wKic.5Lp06rj.HsKZI8V8e2IBF3HDpkTYooa"


def upgrade() -> None:
    bind = op.get_bind()
    existing = bind.execute(
        sa.text("SELECT id, role FROM profiles WHERE email = :email"),
        {"email": ADMIN_EMAIL},
    ).fetchone()

    if existing is not None:
        if existing.role != "admin":
            bind.execute(
                sa.text("UPDATE profiles SET role = 'admin' WHERE email = :email"),
                {"email": ADMIN_EMAIL},
            )
    else:
        bind.execute(
            sa.text(
                "INSERT INTO profiles "
                "(id, email, hashed_password, role, display_name, daily_minutes, created_at) "
                "VALUES (:id, :email, :hashed_password, 'admin', :display_name, 15, now())"
            ),
            {
                "id": str(uuid.uuid4()),
                "email": ADMIN_EMAIL,
                "hashed_password": ADMIN_PASSWORD_HASH,
                "display_name": "Admin",
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text("UPDATE profiles SET role = 'learner' WHERE email = :email"),
        {"email": ADMIN_EMAIL},
    )
