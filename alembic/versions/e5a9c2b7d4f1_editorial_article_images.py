"""Editorial article photos

Adds editorial.editorial_article.images (JSONB list of {url, caption,
after_paragraph}) and images_fetched_at. NULL images_fetched_at marks an
article imported before this migration: the API queues one background
refresh (refresh_editorial_images) the first time such an article is read,
so existing articles get their photos without a manual re-import.

Revision ID: e5a9c2b7d4f1
Revises: c3f7a2e9d5b1
Create Date: 2026-09-29 04:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'e5a9c2b7d4f1'
down_revision = 'c3f7a2e9d5b1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'editorial_article',
        sa.Column('images', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        schema='editorial',
    )
    op.add_column(
        'editorial_article',
        sa.Column('images_fetched_at', sa.DateTime(timezone=True), nullable=True),
        schema='editorial',
    )


def downgrade() -> None:
    op.drop_column('editorial_article', 'images_fetched_at', schema='editorial')
    op.drop_column('editorial_article', 'images', schema='editorial')
