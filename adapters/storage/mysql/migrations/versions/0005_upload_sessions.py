"""Bounded private upload sessions."""

from alembic import op
import sqlalchemy as sa

revision = "0005_upload_sessions"
down_revision = "0004_parsed_reference"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "upload_sessions",
        sa.Column("id", sa.String(36, collation="utf8mb4_bin"), primary_key=True),
        sa.Column("workspace_id", sa.String(36, collation="utf8mb4_bin"), nullable=False),
        sa.Column("project_id", sa.String(36, collation="utf8mb4_bin"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("object_key", sa.String(512), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("expires_at", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.String(36, collation="utf8mb4_bin")),
        sa.Column("file_id", sa.String(36, collation="utf8mb4_bin")),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.UniqueConstraint("workspace_id", "id", name="uq_upload_sessions_workspace_id_id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.ForeignKeyConstraint(
            ["workspace_id", "project_id"], ["projects.workspace_id", "projects.id"]
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "document_id"], ["documents.workspace_id", "documents.id"]
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "file_id"], ["file_objects.workspace_id", "file_objects.id"]
        ),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )


def downgrade():
    op.drop_table("upload_sessions")
