"""Task dispatch generations and fencing counters."""

from alembic import op
import sqlalchemy as sa

revision = "0002_task_execution"
down_revision = "0001_platform"
branch_labels = None
depends_on = None


def upgrade():
    for name, kind, default in [
        ("generation", sa.Integer(), 1),
        ("execution_epoch", sa.BigInteger(), 0),
        ("attempt_count", sa.Integer(), 0),
        ("next_attempt_at", sa.BigInteger(), 0),
        ("event_seq", sa.BigInteger(), 0),
    ]:
        op.add_column(
            "research_tasks", sa.Column(name, kind, nullable=False, server_default=str(default))
        )
    op.add_column("research_tasks", sa.Column("error_code", sa.String(128)))


def downgrade():
    for name in [
        "generation",
        "execution_epoch",
        "attempt_count",
        "next_attempt_at",
        "event_seq",
        "error_code",
    ]:
        op.drop_column("research_tasks", name)
