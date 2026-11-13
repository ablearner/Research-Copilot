"""Measured ordered document listing and latest-version lookup."""

from alembic import op

revision = "0006_document_indexes"
down_revision = "0005_upload_sessions"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "ix_document_listing", "documents", ["workspace_id", "deleted_at", "created_at", "id"]
    )
    op.create_index(
        "ix_document_version_latest",
        "document_versions",
        ["workspace_id", "document_id", "created_at", "id"],
    )


def downgrade():
    op.drop_index("ix_document_version_latest", "document_versions")
    op.drop_index("ix_document_listing", "documents")
