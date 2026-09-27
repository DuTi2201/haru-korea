"""exam_paper ingest wiring + starter question_type taxonomy

Adds import_batch.exam_paper_id (bare id into content.exam_paper, same
convention as the existing film_id — set inside the new
extract_exam_paper_import Celery task, mirroring find_or_create_film's
timing). Seeds content.question_type with a starter set of common TOPIK
I/II reading+listening question categories so the exam_paper Gemini
extraction has a controlled vocabulary to classify against — this is a
reasonable starting taxonomy, not a verbatim transcription from the SRS
(which only gave question_type's column shape, not its row data); the
admin can add/edit rows directly in this table as real exam papers surface
categories this starter set doesn't cover.

Revision ID: d1a4e9f2b6c7
Revises: bc66fb9caf90
Create Date: 2026-09-27 17:10:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd1a4e9f2b6c7'
down_revision = 'bc66fb9caf90'
branch_labels = None
depends_on = None

_QUESTION_TYPES = [
    # (id, skill, code, name_ko, name_vi)
    ('9208e737-cf14-4e86-b8a6-742f61a5cad0', 'đọc', 'read_blank', '빈칸에 알맞은 것 고르기', 'Chọn từ/cụm từ điền vào chỗ trống'),
    ('7cc1ddc4-3c32-4142-85a0-50f13c070d7b', 'đọc', 'read_grammar_choice', '문법/어휘 고르기', 'Chọn ngữ pháp/từ vựng đúng'),
    ('88e13931-15a9-408d-a093-87f56fd20fdf', 'đọc', 'read_short_passage', '짧은 글 읽고 답하기', 'Đọc hiểu đoạn văn ngắn'),
    ('c875c6df-f843-411e-a4e6-ffcc9120c215', 'đọc', 'read_long_passage', '긴 글 읽고 답하기', 'Đọc hiểu đoạn văn dài'),
    ('38fa1dc6-f607-4c87-b880-441838663747', 'đọc', 'read_order', '순서대로 배열하기', 'Sắp xếp câu theo đúng thứ tự'),
    ('ebe8ac3f-6df8-406a-b718-a7bb66fde18e', 'đọc', 'read_title_topic', '제목/주제 고르기', 'Chọn tiêu đề/chủ đề chính'),
    ('bf8a1d23-d6b9-4dbb-8f3c-f0c201918271', 'đọc', 'read_chart_info', '도표/안내문 읽기', 'Đọc bảng biểu/quảng cáo/thông báo'),
    ('e5f6f663-88d9-4d92-979a-3b328b1874db', 'nghe', 'listen_picture', '알맞은 그림 고르기', 'Nghe và chọn tranh/hành động đúng'),
    ('616ca5fd-bd41-41cf-b5b8-e37513915220', 'nghe', 'listen_short_dialog', '짧은 대화 듣고 답하기', 'Nghe hội thoại ngắn, chọn ý đúng'),
    ('ca0867d1-64a9-49b0-934c-325ed35d64a9', 'nghe', 'listen_long_dialog', '긴 대화 듣고 답하기', 'Nghe hội thoại dài, trả lời nhiều câu'),
    ('661db1a9-8dc4-4f0d-accd-e9c4989b1d8f', 'nghe', 'listen_main_idea', '중심 내용 고르기', 'Nghe và chọn ý chính'),
    ('3fecb3b0-f3ab-4811-9aa0-588014b651cb', 'nghe', 'listen_news_talk', '뉴스/강연 듣기', 'Nghe tin tức/bài phát biểu'),
]


def upgrade() -> None:
    op.add_column('import_batch', sa.Column('exam_paper_id', postgresql.UUID(as_uuid=True), nullable=True))

    question_type = sa.table(
        'question_type',
        sa.column('id', postgresql.UUID(as_uuid=True)),
        sa.column('skill', sa.String),
        sa.column('code', sa.String),
        sa.column('name_ko', sa.String),
        sa.column('name_vi', sa.String),
        sa.column('active', sa.Boolean),
        schema='content',
    )
    op.bulk_insert(
        question_type,
        [
            {'id': qid, 'skill': skill, 'code': code, 'name_ko': name_ko, 'name_vi': name_vi, 'active': True}
            for qid, skill, code, name_ko, name_vi in _QUESTION_TYPES
        ],
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM content.question_type WHERE code IN ("
        + ", ".join(f"'{c[2]}'" for c in _QUESTION_TYPES)
        + ")"
    )
    op.drop_column('import_batch', 'exam_paper_id')
