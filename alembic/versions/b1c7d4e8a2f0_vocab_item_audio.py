"""Vocab-item audio cache

Adds `audio.vocab_item_audio` — same content-addressed cache shape as
`audio.corpus_item_audio`, but keyed to one content.vocab_item instead of a
listening sentence. Backs the "nghe" (listen) button on the vocab-study
screen, which the SRS/SDD specify but the app never actually built (a vocab
flashcard only ever showed the written hangul, with no audio at all).

Revision ID: b1c7d4e8a2f0
Revises: 9e21fa6c8b3d
Create Date: 2026-09-28 00:30:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'b1c7d4e8a2f0'
down_revision = '9e21fa6c8b3d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'vocab_item_audio',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('vocab_item_id', sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(length=160), nullable=False),
        sa.Column('opus_data', sa.LargeBinary(), nullable=False),
        sa.Column('aac_data', sa.LargeBinary(), nullable=False),
        sa.Column('prompt_version', sa.String(length=32), nullable=False),
        sa.Column('voice', sa.String(length=64), nullable=False),
        sa.Column('duration_sec', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='audio',
    )
    op.create_index('ix_audio_vocab_item_audio_vocab_item_id', 'vocab_item_audio', ['vocab_item_id'], schema='audio')
    op.create_index('ix_audio_vocab_item_audio_cache_key', 'vocab_item_audio', ['cache_key'], unique=True, schema='audio')


def downgrade() -> None:
    op.drop_index('ix_audio_vocab_item_audio_cache_key', table_name='vocab_item_audio', schema='audio')
    op.drop_index('ix_audio_vocab_item_audio_vocab_item_id', table_name='vocab_item_audio', schema='audio')
    op.drop_table('vocab_item_audio', schema='audio')
