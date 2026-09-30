"""Editorial article study pack

Adds editorial.editorial_article.study_pack (JSONB), study_status
(none|pending|ready|failed) and study_updated_at. The pack is the beginner
"reading ladder" (Vietnamese summary, per-sentence translation + word
breakdown, simplified-Korean paragraphs) generated once per article by the
generate_study_pack task; existing articles get it lazily the first time they
are read (status 'none').

Revision ID: a7d3e1f5b9c2
Revises: e5a9c2b7d4f1
Create Date: 2026-09-30 14:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a7d3e1f5b9c2'
down_revision = 'e5a9c2b7d4f1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'editorial_article',
        sa.Column('study_pack', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema='editorial',
    )
    op.add_column(
        'editorial_article',
        sa.Column('study_status', sa.String(length=16), nullable=False, server_default='none'),
        schema='editorial',
    )
    op.add_column(
        'editorial_article',
        sa.Column('study_updated_at', sa.DateTime(timezone=True), nullable=True),
        schema='editorial',
    )


def downgrade() -> None:
    op.drop_column('editorial_article', 'study_updated_at', schema='editorial')
    op.drop_column('editorial_article', 'study_status', schema='editorial')
    op.drop_column('editorial_article', 'study_pack', schema='editorial')
