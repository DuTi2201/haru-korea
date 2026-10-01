"""exam_item.instruction_ko: the instruction printed once above a group of questions

An exam question is not only its stem. "[9~12] 다음 글 또는 도표의 내용과 같은 것을
고르십시오." is printed once and applies to questions 9 to 12; the first version kept
only one text field per question, so the instruction and the question got mixed up.
The extraction (exam-v2) now copies the group's instruction onto each of its questions.

One additive, NULLable column: questions already in the library keep working, they
just have no instruction.

Revision ID: b6e2d9a4c7f3
Revises: a9d4c2e6b8f1
Create Date: 2026-10-01 17:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'b6e2d9a4c7f3'
down_revision = 'a9d4c2e6b8f1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('exam_item', sa.Column('instruction_ko', sa.Text(), nullable=True), schema='content')


def downgrade() -> None:
    op.drop_column('exam_item', 'instruction_ko', schema='content')
