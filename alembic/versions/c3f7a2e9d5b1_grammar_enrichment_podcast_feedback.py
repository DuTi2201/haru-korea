"""Grammar usage enrichment, podcast script column, outline feedback

Owner feedback this round: (1) a bare "V/A + form" grammar pattern isn't
enough to actually use it — content.grammar_point gets usage_context_vi
(when/in what situation it's used) and topik_tip_vi (how it shows up in
TOPIK exam questions), both filled going forward by the lesson/editorial
extraction prompts and left NULL on existing rows until backfilled/
re-extracted. (2) audio.lecture_audio gets a script_text column so the new
Gemini-generated "bài giảng tổng hợp" (consolidated vocab+grammar podcast,
see app.workers.tasks.generate_content_podcast) can store its transcript
alongside the synthesized audio, reusing this table/route instead of a
parallel one. (3) editorial.editorial_outline_submission gets
feedback_status/feedback_text — saving a "luyện dàn ý" outline previously
never actually analyzed it; now it does (grade_editorial_outline).

Revision ID: c3f7a2e9d5b1
Revises: b1c7d4e8a2f0
Create Date: 2026-09-28 02:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'c3f7a2e9d5b1'
down_revision = 'b1c7d4e8a2f0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('grammar_point', sa.Column('usage_context_vi', sa.Text(), nullable=True), schema='content')
    op.add_column('grammar_point', sa.Column('topik_tip_vi', sa.Text(), nullable=True), schema='content')
    op.add_column('lecture_audio', sa.Column('script_text', sa.Text(), nullable=True), schema='audio')
    op.add_column(
        'editorial_outline_submission',
        sa.Column('feedback_status', sa.String(length=16), nullable=False, server_default='none'),
        schema='editorial',
    )
    op.add_column(
        'editorial_outline_submission', sa.Column('feedback_text', sa.Text(), nullable=True), schema='editorial'
    )


def downgrade() -> None:
    op.drop_column('editorial_outline_submission', 'feedback_text', schema='editorial')
    op.drop_column('editorial_outline_submission', 'feedback_status', schema='editorial')
    op.drop_column('lecture_audio', 'script_text', schema='audio')
    op.drop_column('grammar_point', 'topik_tip_vi', schema='content')
    op.drop_column('grammar_point', 'usage_context_vi', schema='content')
