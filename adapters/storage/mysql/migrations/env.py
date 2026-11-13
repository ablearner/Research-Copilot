"""Explicit async migration runner. PLATFORM_DATABASE_URL supplies credentials."""

import asyncio
import os
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from adapters.storage.mysql.tables import metadata

config = context.config
url = os.environ.get("PLATFORM_DATABASE_URL", config.get_main_option("sqlalchemy.url"))


def migrate(connection):
    context.configure(connection=connection, target_metadata=metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=url, target_metadata=metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
