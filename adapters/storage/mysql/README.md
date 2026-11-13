# Platform relational storage

Production uses MySQL 8 / InnoDB / utf8mb4_bin. IDs are UUID strings named `id`; tenant-owned rows carry `workspace_id` with composite relational foreign keys. All timestamps are UTC Unix milliseconds, convertible with `datetime.fromtimestamp(value / 1000, timezone.utc)`.

From repository root, migrate explicitly (never from API startup):

```bash
PLATFORM_DATABASE_URL='mysql+asyncmy://user:password@host/database' python -m alembic -c adapters/storage/mysql/alembic.ini upgrade head
```

`Database.initialize()` is intentionally limited to SQLite test databases. `Database.transaction()` returns an async SQLAlchemy Core transaction context and rolls back on errors. SQLite enables foreign keys. Production migration rollback drops platform tables and must only be used deliberately on disposable deployments.
