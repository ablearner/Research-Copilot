"""Publish an immutable attempt artifact reference with the active version."""

from alembic import op
import sqlalchemy as sa

revision = "0004_parsed_reference"
down_revision = "0003_credentials"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("document_versions", sa.Column("parsed_ref", sa.String(512), nullable=True))
    # Preserve artifacts produced by the preceding schema before switching to immutable keys.
    op.execute(
        "UPDATE document_versions SET parsed_ref = CONCAT('workspaces/', workspace_id, '/documents/', document_id, '/', id, '/parsed.json') WHERE parse_status = 'succeeded'"
    )


def downgrade():
    op.drop_column("document_versions", "parsed_ref")
