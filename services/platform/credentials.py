import secrets
from uuid import uuid4
from sqlalchemy import select, insert, update
from adapters.storage.mysql import tables as t
from services.identity import token_hash, public_user, InvalidCredentials, ResourceNotFound


class CredentialService:
    def __init__(self, db):
        self.db = db

    async def create(self, user_id, name, scopes, ttl_seconds=86400 * 30):
        if (
            not name
            or len(name) > 128
            or not scopes
            or not set(scopes) <= {"read", "write"}
            or not 60 <= ttl_seconds <= 86400 * 90
        ):
            raise ValueError("Invalid credential settings")
        raw = "rc_" + secrets.token_urlsafe(32)
        id = str(uuid4())
        async with self.db.transaction() as c:
            user = (
                await c.execute(
                    select(t.users).where(t.users.c.id == user_id, t.users.c.status == "active")
                )
            ).first()
            if not user:
                raise InvalidCredentials()
            await c.execute(
                insert(t.api_credentials).values(
                    id=id,
                    user_id=user_id,
                    name=name,
                    token_hash=token_hash(raw),
                    scopes=scopes,
                    expires_at=t.now_ms() + ttl_seconds * 1000,
                )
            )
        return {"id": id, "token": raw, "name": name, "scopes": scopes}

    async def authenticate(self, raw, scope):
        async with self.db.transaction() as c:
            row = (
                (
                    await c.execute(
                        select(t.users, t.api_credentials.c.scopes)
                        .join(t.api_credentials, t.api_credentials.c.user_id == t.users.c.id)
                        .where(
                            t.api_credentials.c.token_hash == token_hash(raw),
                            t.api_credentials.c.revoked_at.is_(None),
                            t.api_credentials.c.expires_at > t.now_ms(),
                            t.users.c.status == "active",
                            t.users.c.email_verified_at.is_not(None),
                        )
                    )
                )
                .mappings()
                .first()
            )
        if not row or scope not in row["scopes"]:
            raise InvalidCredentials()
        result = public_user(row)
        result.pop("scopes")
        return result

    async def list(self, user_id):
        async with self.db.transaction() as c:
            return [
                dict(r)
                for r in (
                    await c.execute(
                        select(
                            t.api_credentials.c.id,
                            t.api_credentials.c.name,
                            t.api_credentials.c.scopes,
                            t.api_credentials.c.expires_at,
                            t.api_credentials.c.revoked_at,
                        ).where(t.api_credentials.c.user_id == user_id)
                    )
                ).mappings()
            ]

    async def revoke(self, user_id, id):
        async with self.db.transaction() as c:
            result = await c.execute(
                update(t.api_credentials)
                .where(t.api_credentials.c.id == id, t.api_credentials.c.user_id == user_id)
                .values(revoked_at=t.now_ms())
            )
            if not result.rowcount:
                raise ResourceNotFound()
