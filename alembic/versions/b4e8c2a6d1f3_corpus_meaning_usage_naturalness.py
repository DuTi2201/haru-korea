"""Corpus sentences get a Vietnamese meaning, a usage note and a naturalness verdict

Learners could hear a film line and see its Korean text, but not what it
means, when to say it, or to whom. corpus.corpus_item gets:

- meaning_vi        natural Vietnamese translation of the line
- usage_note_vi     when / with whom / in what tone the line is used
- naturalness       'natural' | 'awkward' | 'unnatural' — an AI verdict on
                    whether a native speaker would really say it (machine-
                    translated or garbled subtitle lines are 'unnatural' and
                    are hidden from learners; NULL = not judged yet)
- enriched_version  which enrichment prompt filled the three columns above;
                    NULL = never enriched, so a backfill can resume where it
                    stopped and a better prompt can re-run later

All four are NULLable and added without a default, so this is a metadata-only
change in Postgres (no table rewrite) and the code that is already running
keeps working while the migration is applied.

Revision ID: b4e8c2a6d1f3
Revises: a7d3e1f5b9c2
Create Date: 2026-10-01 08:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'b4e8c2a6d1f3'
down_revision = 'a7d3e1f5b9c2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('corpus_item', sa.Column('meaning_vi', sa.Text(), nullable=True), schema='corpus')
    op.add_column('corpus_item', sa.Column('usage_note_vi', sa.Text(), nullable=True), schema='corpus')
    op.add_column('corpus_item', sa.Column('naturalness', sa.String(length=16), nullable=True), schema='corpus')
    op.add_column('corpus_item', sa.Column('enriched_version', sa.String(length=16), nullable=True), schema='corpus')


def downgrade() -> None:
    op.drop_column('corpus_item', 'enriched_version', schema='corpus')
    op.drop_column('corpus_item', 'naturalness', schema='corpus')
    op.drop_column('corpus_item', 'usage_note_vi', schema='corpus')
    op.drop_column('corpus_item', 'meaning_vi', schema='corpus')
