"""Async SQLAlchemy Core connection owner; production schema changes use Alembic."""

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from .tables import metadata


class Database:
    def __init__(self, url: str):
        options = {"isolation_level": "READ COMMITTED"} if url.startswith("mysql") else {}
        self.engine = create_async_engine(url, pool_pre_ping=True, **options)
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine.sync_engine, "connect")
            def foreign_keys(connection, _):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    async def initialize(self):
        """Create schema for isolated tests; deploy with explicit Alembic upgrade."""
        if self.engine.dialect.name != "sqlite":
            raise RuntimeError("Production schemas require explicit Alembic migrations")
        async with self.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    def transaction(self):
        return self.engine.begin()

    async def close(self):
        await self.engine.dispose()
