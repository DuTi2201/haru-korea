"""Error loop (richer error_log), exam-drill attempts, writing drills

Three additive changes for the practice loop:

error_log gets what is needed to find the same mistake again
- item_type / item_id   the review card that was answered wrong
- mode                  how it was asked (recognize | cloze | exam | writing)
- detail                what was chosen, the right answer (JSONB)
plus an index on (learner_id, created_at) for the "last 30 days" report.

exam_attempt (new) holds a learner's answer to a real exam question, right or
wrong, so accuracy per question type can be shown.

writing_drill (new) holds one "viết câu 51–52" exercise built from the learner's
own cards: the generated text with two blanks, the answers and the check.

Nothing existing is rewritten: new columns are NULLable and the new tables are
empty, so the code that is already running keeps working while this is applied.

Revision ID: a9d4c2e6b8f1
Revises: c8e2f5a9d3b7
Create Date: 2026-10-01 13:30:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a9d4c2e6b8f1'
down_revision = 'c8e2f5a9d3b7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('error_log', sa.Column('item_type', sa.String(length=16), nullable=True))
    op.add_column('error_log', sa.Column('item_id', sa.Integer(), nullable=True))
    op.add_column('error_log', sa.Column('mode', sa.String(length=16), nullable=True))
    op.add_column('error_log', sa.Column('detail', postgresql.JSONB(), nullable=True))
    op.create_index('ix_error_log_learner_created', 'error_log', ['learner_id', 'created_at'])

    op.create_table(
        'exam_attempt',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('learner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('exam_item_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chosen', sa.SmallInteger(), nullable=False),
        sa.Column('correct', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['learner_id'], ['profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['exam_item_id'], ['content.exam_item.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_exam_attempt_learner_id', 'exam_attempt', ['learner_id'])
    op.create_index('ix_exam_attempt_item', 'exam_attempt', ['exam_item_id'])

    op.create_table(
        'writing_drill',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('learner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_items', postgresql.JSONB(), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('prompt', postgresql.JSONB(), nullable=True),
        sa.Column('answers', postgresql.JSONB(), nullable=True),
        sa.Column('result', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.String(length=16), server_default='pending', nullable=False),
        sa.Column('grade_status', sa.String(length=16), server_default='none', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['learner_id'], ['profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_writing_drill_learner_id', 'writing_drill', ['learner_id'])


def downgrade() -> None:
    op.drop_index('ix_writing_drill_learner_id', table_name='writing_drill')
    op.drop_table('writing_drill')
    op.drop_index('ix_exam_attempt_item', table_name='exam_attempt')
    op.drop_index('ix_exam_attempt_learner_id', table_name='exam_attempt')
    op.drop_table('exam_attempt')
    op.drop_index('ix_error_log_learner_created', table_name='error_log')
    for name in ('detail', 'mode', 'item_id', 'item_type'):
        op.drop_column('error_log', name)
