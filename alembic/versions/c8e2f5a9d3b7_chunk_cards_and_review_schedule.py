"""Chunk cards (collocation families, contrast groups) and a spaced-review schedule

Two additive changes, both for learning the way TOPIK tests:

content.vocab_item gets the layers that make a card a *chunk* rather than a word
- family          the set it is learned in ("Động từ đi với thời tiết")
- node_word       the part of `hangul` the learner must choose correctly
                  (오다 in 비가 오다); what a fill-in-the-blank hides
- register        spoken | written | neutral
- usage_note_vi   when / with what it is used, and the usual Vietnamese slip
- collocations    other natural pairings, [{"ko": ..., "vi": ...}]
- distractors     tempting wrong stand-ins for node_word

content.grammar_point gets the contrast layer
- contrast_group  the set of look-alike patterns it is learned with
- contrasts       how it differs from each other member,
                  [{"pattern": ..., "diff_vi": ...}]

item_state gets a small SM-2-style schedule next to the existing `strength`
- due_at          when the item should come back
- reps / lapses   correct answers in a row / times it was forgotten
- ease            growth factor of the gaps (starts at 2.2, floor 1.3)
- interval_days   the current gap
- introduced_at   when it was first studied (caps new items per day)
Rows that already exist are filled from `strength` so nothing starts from zero:
strength 0.2 per correct answer maps to reps 0-5 and the 1-3-7-14-30 day ladder.

Every new column is NULLable or has a constant default, so this is a
metadata-only change in Postgres (no table rewrite) and the code that is
already running keeps working while the migration is applied.

Revision ID: c8e2f5a9d3b7
Revises: b4e8c2a6d1f3
Create Date: 2026-10-01 10:30:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'c8e2f5a9d3b7'
down_revision = 'b4e8c2a6d1f3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name, col in (
        ('family', sa.String(length=120)),
        ('node_word', sa.String(length=120)),
        ('register', sa.String(length=16)),
        ('usage_note_vi', sa.Text()),
        ('collocations', postgresql.JSONB()),
        ('distractors', postgresql.JSONB()),
    ):
        op.add_column('vocab_item', sa.Column(name, col, nullable=True), schema='content')
    op.add_column('grammar_point', sa.Column('contrast_group', sa.String(length=120), nullable=True), schema='content')
    op.add_column('grammar_point', sa.Column('contrasts', postgresql.JSONB(), nullable=True), schema='content')

    op.add_column('item_state', sa.Column('due_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('item_state', sa.Column('reps', sa.Integer(), server_default='0', nullable=False))
    op.add_column('item_state', sa.Column('lapses', sa.Integer(), server_default='0', nullable=False))
    op.add_column('item_state', sa.Column('ease', sa.Float(), server_default='2.2', nullable=False))
    op.add_column('item_state', sa.Column('interval_days', sa.Float(), server_default='0', nullable=False))
    op.add_column('item_state', sa.Column('introduced_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_item_state_due_at', 'item_state', ['due_at'])

    # Existing rows: one correct answer was worth +0.2 strength, so strength
    # tells how far up the 0-1-3-7-14-30 day ladder the learner got.
    op.execute(
        """
        UPDATE item_state SET reps = LEAST(5, GREATEST(0, ROUND(strength / 0.2)::int)),
                              introduced_at = last_seen
        """
    )
    op.execute(
        """
        UPDATE item_state SET
            interval_days = CASE reps WHEN 0 THEN 0 WHEN 1 THEN 1 WHEN 2 THEN 3
                                      WHEN 3 THEN 7 WHEN 4 THEN 14 ELSE 30 END
        """
    )
    op.execute("UPDATE item_state SET due_at = last_seen + interval_days * interval '1 day'")


def downgrade() -> None:
    op.drop_index('ix_item_state_due_at', table_name='item_state')
    for name in ('introduced_at', 'interval_days', 'ease', 'lapses', 'reps', 'due_at'):
        op.drop_column('item_state', name)
    op.drop_column('grammar_point', 'contrasts', schema='content')
    op.drop_column('grammar_point', 'contrast_group', schema='content')
    for name in ('distractors', 'collocations', 'usage_note_vi', 'register', 'node_word', 'family'):
        op.drop_column('vocab_item', name, schema='content')
