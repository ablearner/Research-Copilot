"""Initial multi-user platform schema."""

from alembic import op
from adapters.storage.mysql.migrations.schema_v1 import metadata

revision = "0001_platform"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    metadata.create_all(op.get_bind())


def downgrade():
    metadata.drop_all(op.get_bind())
