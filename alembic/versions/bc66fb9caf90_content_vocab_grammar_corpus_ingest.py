"""content lesson/vocab/grammar + corpus film + ingest lineage

Adds the SRS §5 entity model that the initial migration deferred:
content.lesson / content.topic / content.lesson_topic / content.vocab_item /
content.grammar_point, and corpus.film. Reworks corpus.corpus_item to match
the SRS ER diagram exactly (film_id becomes a real FK now that corpus.film
exists; level/kind/register get the spec's actual types; source_ref and the
CORPUS_ITEM }o--o{ GRAMMAR_POINT bare-array tagging are added). Reworks
item_state to the SRS's polymorphic (item_type, item_id, strength,
last_seen) shape, replacing the pre-SRS SM-2 guess, and drops the old
learner-scoped `vocab_item` table it pointed at (nothing else in the repo
referenced either — grepped clean — so this is a straight replace, not a
data migration). Adds import_batch.source_file/file_hash/film_id for the
new upload+dedup flow in ingest.py.

Revision ID: bc66fb9caf90
Revises: c656a9850ad6
Create Date: 2026-09-27 16:30:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'bc66fb9caf90'
down_revision = 'c656a9850ad6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------------------------------------------------------- item_state --
    # Old shape pointed at the (now-removed) learner-scoped public.vocab_item.
    op.drop_constraint('item_state_vocab_item_id_fkey', 'item_state', type_='foreignkey')
    op.drop_constraint('uq_item_state_learner_item', 'item_state', type_='unique')
    op.drop_column('item_state', 'vocab_item_id')
    op.drop_column('item_state', 'srs_stage')
    op.drop_column('item_state', 'next_review_at')
    op.drop_column('item_state', 'ease')

    op.execute("CREATE TYPE item_state_type AS ENUM ('vocab_item', 'grammar_point')")
    op.add_column(
        'item_state',
        sa.Column(
            'item_type',
            postgresql.ENUM('vocab_item', 'grammar_point', name='item_state_type', create_type=False),
            nullable=False,
        ),
    )
    op.add_column('item_state', sa.Column('item_id', sa.Integer(), nullable=False))
    op.add_column(
        'item_state', sa.Column('strength', sa.Float(), nullable=False, server_default='0')
    )
    op.add_column(
        'item_state',
        sa.Column(
            'last_seen', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
    )
    op.create_unique_constraint(
        'uq_item_state_learner_item', 'item_state', ['learner_id', 'item_type', 'item_id']
    )
    op.create_index(op.f('ix_item_state_learner_id'), 'item_state', ['learner_id'], unique=False)

    # Old learner-scoped vocab_item — superseded by content.vocab_item below.
    op.drop_table('vocab_item')

    # --------------------------------------------------- content: lessons --
    op.create_table(
        'topic',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('quizlet_url', sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        schema='content',
    )
    op.create_table(
        'lesson',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('level', sa.SmallInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('quizlet_url', sa.String(length=500), nullable=True),
        sa.Column('import_item_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema='content',
    )
    op.create_index(
        op.f('ix_content_lesson_content_hash'), 'lesson', ['content_hash'], unique=True, schema='content'
    )
    op.create_table(
        'lesson_topic',
        sa.Column('lesson_id', sa.Integer(), nullable=False),
        sa.Column('topic_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['lesson_id'], ['content.lesson.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['topic_id'], ['content.topic.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('lesson_id', 'topic_id'),
        schema='content',
    )
    op.create_table(
        'vocab_item',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('lesson_id', sa.Integer(), nullable=False),
        sa.Column('hangul', sa.String(length=120), nullable=False),
        sa.Column('pos', sa.String(length=32), nullable=True),
        sa.Column('meaning_vi', sa.String(length=255), nullable=False),
        sa.Column('definition_ko', sa.Text(), nullable=True),
        sa.Column('level', sa.SmallInteger(), nullable=False),
        sa.Column('hanja', sa.String(length=64), nullable=True),
        sa.Column('sino_vietnamese', sa.String(length=120), nullable=True),
        sa.Column('example_ko', sa.Text(), nullable=True),
        sa.Column('import_item_id', sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(['lesson_id'], ['content.lesson.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='content',
    )
    op.create_index(
        op.f('ix_content_vocab_item_lesson_id'), 'vocab_item', ['lesson_id'], unique=False, schema='content'
    )
    op.create_table(
        'grammar_point',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('lesson_id', sa.Integer(), nullable=False),
        sa.Column('pattern', sa.String(length=255), nullable=False),
        sa.Column('meaning_vi', sa.String(length=255), nullable=False),
        sa.Column('level', sa.SmallInteger(), nullable=False),
        sa.Column('example_ko', sa.Text(), nullable=True),
        sa.Column('import_item_id', sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(['lesson_id'], ['content.lesson.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='content',
    )
    op.create_index(
        op.f('ix_content_grammar_point_lesson_id'),
        'grammar_point',
        ['lesson_id'],
        unique=False,
        schema='content',
    )

    # --------------------------------------------------------- corpus.film --
    op.create_table(
        'film',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('title'),
        schema='corpus',
    )

    # --------------------------------------------------- corpus.corpus_item --
    # film_id: text -> int FK (table is unpopulated in every deployed env —
    # ingestion never ran before this migration — so a plain USING cast is
    # safe). level: text -> smallint (1-6, matches lesson/vocab_item/
    # grammar_point). kind/register: free text -> the SRS's fixed vocab.
    op.execute("CREATE TYPE corpus_item_kind AS ENUM ('câu', 'cụm từ', 'mẫu ngữ pháp')")
    op.execute("CREATE TYPE corpus_item_register AS ENUM ('존댓말', '반말', 'hỗn hợp')")
    op.execute("ALTER TABLE corpus.corpus_item ALTER COLUMN film_id TYPE integer USING film_id::integer")
    op.execute("ALTER TABLE corpus.corpus_item ALTER COLUMN level TYPE smallint USING level::smallint")
    op.execute(
        "ALTER TABLE corpus.corpus_item ALTER COLUMN kind TYPE corpus_item_kind USING kind::corpus_item_kind"
    )
    op.execute(
        "ALTER TABLE corpus.corpus_item ALTER COLUMN register TYPE corpus_item_register "
        "USING register::corpus_item_register"
    )
    op.add_column(
        'corpus_item', sa.Column('source_ref', sa.String(length=64), nullable=True), schema='corpus'
    )
    op.add_column(
        'corpus_item',
        sa.Column(
            'grammar_point_ids', postgresql.ARRAY(sa.Integer()), nullable=False, server_default='{}'
        ),
        schema='corpus',
    )
    op.create_foreign_key(
        'corpus_item_film_id_fkey',
        'corpus_item',
        'film',
        ['film_id'],
        ['id'],
        source_schema='corpus',
        referent_schema='corpus',
        ondelete='CASCADE',
    )
    op.create_index(
        op.f('ix_corpus_corpus_item_film_id'), 'corpus_item', ['film_id'], unique=False, schema='corpus'
    )
    op.create_unique_constraint(
        'uq_corpus_item_film_source_ref', 'corpus_item', ['film_id', 'source_ref'], schema='corpus'
    )
    op.execute(
        "CREATE INDEX ix_corpus_item_grammar_point_ids_gin ON corpus.corpus_item "
        "USING gin (grammar_point_ids)"
    )

    # --------------------------------------------------------- import_batch --
    op.add_column('import_batch', sa.Column('source_file', sa.String(length=255), nullable=True))
    op.add_column('import_batch', sa.Column('file_hash', sa.String(length=64), nullable=True))
    op.add_column('import_batch', sa.Column('film_id', sa.Integer(), nullable=True))
    op.create_unique_constraint(
        'uq_import_batch_kind_file_hash', 'import_batch', ['kind', 'file_hash']
    )


def downgrade() -> None:
    # --------------------------------------------------------- import_batch --
    op.drop_constraint('uq_import_batch_kind_file_hash', 'import_batch', type_='unique')
    op.drop_column('import_batch', 'film_id')
    op.drop_column('import_batch', 'file_hash')
    op.drop_column('import_batch', 'source_file')

    # --------------------------------------------------- corpus.corpus_item --
    op.execute("DROP INDEX IF EXISTS corpus.ix_corpus_item_grammar_point_ids_gin")
    op.drop_constraint('uq_corpus_item_film_source_ref', 'corpus_item', type_='unique', schema='corpus')
    op.drop_index(op.f('ix_corpus_corpus_item_film_id'), table_name='corpus_item', schema='corpus')
    op.drop_constraint('corpus_item_film_id_fkey', 'corpus_item', type_='foreignkey', schema='corpus')
    op.drop_column('corpus_item', 'grammar_point_ids', schema='corpus')
    op.drop_column('corpus_item', 'source_ref', schema='corpus')
    op.execute(
        "ALTER TABLE corpus.corpus_item ALTER COLUMN register TYPE varchar(32) USING register::text"
    )
    op.execute("ALTER TABLE corpus.corpus_item ALTER COLUMN kind TYPE varchar(32) USING kind::text")
    op.execute("ALTER TABLE corpus.corpus_item ALTER COLUMN level TYPE varchar(16) USING level::text")
    op.execute("ALTER TABLE corpus.corpus_item ALTER COLUMN film_id TYPE varchar(64) USING film_id::text")
    op.execute("DROP TYPE IF EXISTS corpus_item_register")
    op.execute("DROP TYPE IF EXISTS corpus_item_kind")

    # --------------------------------------------------------- corpus.film --
    op.drop_table('film', schema='corpus')

    # --------------------------------------------------- content: lessons --
    op.drop_index(op.f('ix_content_grammar_point_lesson_id'), table_name='grammar_point', schema='content')
    op.drop_table('grammar_point', schema='content')
    op.drop_index(op.f('ix_content_vocab_item_lesson_id'), table_name='vocab_item', schema='content')
    op.drop_table('vocab_item', schema='content')
    op.drop_table('lesson_topic', schema='content')
    op.drop_index(op.f('ix_content_lesson_content_hash'), table_name='lesson', schema='content')
    op.drop_table('lesson', schema='content')
    op.drop_table('topic', schema='content')

    # ---------------------------------------------------------- item_state --
    op.drop_index(op.f('ix_item_state_learner_id'), table_name='item_state')
    op.drop_constraint('uq_item_state_learner_item', 'item_state', type_='unique')
    op.drop_column('item_state', 'last_seen')
    op.drop_column('item_state', 'strength')
    op.drop_column('item_state', 'item_id')
    op.drop_column('item_state', 'item_type')
    op.execute("DROP TYPE IF EXISTS item_state_type")

    op.create_table(
        'vocab_item',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('learner_id', sa.UUID(), nullable=False),
        sa.Column('term_ko', sa.String(length=120), nullable=False),
        sa.Column('meaning_vi', sa.String(length=255), nullable=False),
        sa.Column('topic_id', sa.Integer(), nullable=True),
        sa.Column('level', sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(['learner_id'], ['profiles.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.add_column('item_state', sa.Column('vocab_item_id', sa.UUID(), nullable=False))
    op.add_column('item_state', sa.Column('srs_stage', sa.Integer(), nullable=False, server_default='0'))
    op.add_column(
        'item_state',
        sa.Column(
            'next_review_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
    )
    op.add_column('item_state', sa.Column('ease', sa.Float(), nullable=False, server_default='2.5'))
    op.create_foreign_key(
        'item_state_vocab_item_id_fkey', 'item_state', 'vocab_item', ['vocab_item_id'], ['id']
    )
    op.create_unique_constraint(
        'uq_item_state_learner_item', 'item_state', ['learner_id', 'vocab_item_id']
    )
