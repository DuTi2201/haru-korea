"""TTS byte storage + editorial reading module

Two independent additions bundled into one migration:

1) Real TTS storage: `audio.lecture_audio` gets `opus_data`/`aac_data`
   (bytea) alongside the existing path columns (which now hold this app's
   own streaming-route URLs, not filesystem paths), and a new
   `audio.corpus_item_audio` table gives listening-screen sentences the
   same content-addressed cache. Both back app/services/tts.py's real
   Gemini-TTS-plus-ffmpeg pipeline (previously a `time.sleep` stub).

2) Editorial reading module (사설/칼럼 practice for TOPIK 쓰기 câu 54, per
   the "Đề xuất SRS/SDD" proposal doc): new `editorial` schema with
   editorial_source/editorial_article/editorial_outline_submission/
   editorial_candidate, `import_batch.kind` gains 'editorial_article'
   (+ bare `editorial_article_id`), and content.vocab_item/grammar_point's
   `lesson_id` becomes nullable so a word/pattern can be sourced from an
   article instead of a lesson.

Revision ID: f3c8a1d9e4b2
Revises: d1a4e9f2b6c7
Create Date: 2026-09-27 18:05:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'f3c8a1d9e4b2'
down_revision = 'd1a4e9f2b6c7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------- 1) audio --
    op.add_column('lecture_audio', sa.Column('opus_data', sa.LargeBinary(), nullable=True), schema='audio')
    op.add_column('lecture_audio', sa.Column('aac_data', sa.LargeBinary(), nullable=True), schema='audio')

    op.create_table(
        'corpus_item_audio',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('corpus_item_id', postgresql.UUID(as_uuid=True), nullable=False),
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
    op.create_index('ix_audio_corpus_item_audio_corpus_item_id', 'corpus_item_audio', ['corpus_item_id'], schema='audio')
    op.create_index('ix_audio_corpus_item_audio_cache_key', 'corpus_item_audio', ['cache_key'], unique=True, schema='audio')

    # ------------------------------------------------------- 2) editorial --
    op.execute("CREATE SCHEMA IF NOT EXISTS editorial")

    # Postgres forbids using a freshly-added enum value in the same
    # transaction that added it, and (pre-PG12) forbids ADD VALUE inside a
    # transaction block at all — autocommit_block() escapes Alembic's
    # wrapping transaction for just this statement.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE import_kind ADD VALUE IF NOT EXISTS 'editorial_article'")

    op.add_column('import_batch', sa.Column('editorial_article_id', postgresql.UUID(as_uuid=True), nullable=True))

    op.alter_column('vocab_item', 'lesson_id', existing_type=sa.Integer(), nullable=True, schema='content')
    op.alter_column('grammar_point', 'lesson_id', existing_type=sa.Integer(), nullable=True, schema='content')

    op.create_table(
        'editorial_source',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('base_url', sa.String(length=500), nullable=False),
        sa.Column('rss_url', sa.String(length=500), nullable=True),
        sa.Column('license_note', sa.Text(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        schema='editorial',
    )

    op.create_table(
        'editorial_article',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_name', sa.String(length=120), nullable=False),
        sa.Column('source_url', sa.String(length=1000), nullable=False),
        sa.Column('title_ko', sa.String(length=500), nullable=True),
        sa.Column('topic_tags', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('level_estimate', sa.SmallInteger(), nullable=True),
        sa.Column('published_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('body_ko', sa.Text(), nullable=True),
        sa.Column('vocab_ids', postgresql.ARRAY(sa.Integer()), nullable=False, server_default='{}'),
        sa.Column('grammar_ids', postgresql.ARRAY(sa.Integer()), nullable=False, server_default='{}'),
        sa.Column('model_outline', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('thinking_guide_text', sa.Text(), nullable=True),
        sa.Column('import_item_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_url'),
        schema='editorial',
    )

    op.create_table(
        'editorial_outline_submission',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('learner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('editorial_article_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('phenomenon_text', sa.Text(), nullable=True),
        sa.Column('cause_text', sa.Text(), nullable=True),
        sa.Column('consequence_text', sa.Text(), nullable=True),
        sa.Column('solution_text', sa.Text(), nullable=True),
        sa.Column('revision_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['learner_id'], ['profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('learner_id', 'editorial_article_id', name='uq_editorial_outline_learner_article'),
        schema='editorial',
    )
    op.create_index(
        'ix_editorial_editorial_outline_submission_learner_id',
        'editorial_outline_submission', ['learner_id'], schema='editorial',
    )
    op.create_index(
        'ix_editorial_editorial_outline_submission_editorial_article_id',
        'editorial_outline_submission', ['editorial_article_id'], schema='editorial',
    )

    op.create_table(
        'editorial_candidate',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_name', sa.String(length=120), nullable=False),
        sa.Column('source_url', sa.String(length=1000), nullable=False),
        sa.Column('title_ko', sa.String(length=500), nullable=False),
        sa.Column('snippet_ko', sa.Text(), nullable=True),
        sa.Column('topic_tags', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('published_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'status',
            sa.Enum('new', 'dismissed', 'ingested', name='editorial_candidate_status'),
            nullable=False,
            server_default='new',
        ),
        sa.Column('discovered_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_url'),
        schema='editorial',
    )


def downgrade() -> None:
    op.drop_table('editorial_candidate', schema='editorial')
    op.execute("DROP TYPE IF EXISTS editorial_candidate_status")
    op.drop_index(
        'ix_editorial_editorial_outline_submission_editorial_article_id',
        table_name='editorial_outline_submission', schema='editorial',
    )
    op.drop_index(
        'ix_editorial_editorial_outline_submission_learner_id',
        table_name='editorial_outline_submission', schema='editorial',
    )
    op.drop_table('editorial_outline_submission', schema='editorial')
    op.drop_table('editorial_article', schema='editorial')
    op.drop_table('editorial_source', schema='editorial')

    op.alter_column('grammar_point', 'lesson_id', existing_type=sa.Integer(), nullable=False, schema='content')
    op.alter_column('vocab_item', 'lesson_id', existing_type=sa.Integer(), nullable=False, schema='content')

    op.drop_column('import_batch', 'editorial_article_id')

    # Removing an enum value requires rebuilding the type; only safe if no
    # row currently uses it (same trade-off the module docstring accepts).
    op.execute("ALTER TYPE import_kind RENAME TO import_kind_old")
    op.execute("CREATE TYPE import_kind AS ENUM ('lesson', 'corpus', 'exam_paper')")
    op.execute("ALTER TABLE import_batch ALTER COLUMN kind TYPE import_kind USING kind::text::import_kind")
    op.execute("DROP TYPE import_kind_old")

    op.execute("DROP SCHEMA IF EXISTS editorial CASCADE")

    op.drop_index('ix_audio_corpus_item_audio_cache_key', table_name='corpus_item_audio', schema='audio')
    op.drop_index('ix_audio_corpus_item_audio_corpus_item_id', table_name='corpus_item_audio', schema='audio')
    op.drop_table('corpus_item_audio', schema='audio')
    op.drop_column('lecture_audio', 'aac_data', schema='audio')
    op.drop_column('lecture_audio', 'opus_data', schema='audio')
