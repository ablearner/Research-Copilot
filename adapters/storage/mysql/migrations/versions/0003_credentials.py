"""Personal credentials and active document version ownership."""

from alembic import op
from adapters.storage.mysql.migrations.schema_v3 import api_credentials

revision = "0003_credentials"
down_revision = "0002_task_execution"
branch_labels = None
depends_on = None


def upgrade():
    api_credentials.create(op.get_bind(), checkfirst=True)
    op.create_unique_constraint(
        "uq_document_version_owner", "document_versions", ["workspace_id", "document_id", "id"]
    )
    op.create_foreign_key(
        "fk_document_active_version",
        "documents",
        "document_versions",
        ["workspace_id", "id", "active_version"],
        ["workspace_id", "document_id", "id"],
    )


def downgrade():
    op.drop_constraint("fk_document_active_version", "documents", type_="foreignkey")
    op.drop_constraint("uq_document_version_owner", "document_versions", type_="unique")
    op.drop_table("api_credentials")
