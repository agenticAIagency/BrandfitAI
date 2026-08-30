"""Create the evidence, analysis, persona, memory, and export domains."""

from alembic import op
from sqlalchemy import text

from brandfit_core.db import models
from brandfit_core.db.base import Base
from brandfit_core.taxonomy import seed_rows

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=connection)
    connection.execute(
        text(
            """
            INSERT INTO workspaces (id, name)
            VALUES ('00000000-0000-0000-0000-000000000001', 'Default workspace')
            ON CONFLICT (id) DO NOTHING
            """
        )
    )
    op.bulk_insert(models.TaxonomyTerm.__table__, seed_rows())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
