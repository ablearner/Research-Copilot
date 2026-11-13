"""Database-backed identity, with no request-local state on service instances."""

import asyncio
import hashlib
import secrets
from uuid import uuid4
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from sqlalchemy import select, insert, update
from sqlalchemy.exc import IntegrityError
from adapters.storage.mysql.tables import (
    users,
    workspaces,
    workspace_members,
    auth_sessions,
    now_ms,
)
from domain.schemas.identity import RequestContext


class InvalidCredentials(Exception):
    def __init__(self):
        super().__init__("Invalid credentials")


class DuplicateEmail(Exception):
    pass


class ResourceNotFound(Exception):
    pass


class PermissionDenied(Exception):
    pass


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def public_user(user):
    return {k: v for k, v in dict(user).items() if k != "password_hash"}


async def require_membership(conn, context):
    member = (
        await conn.execute(
            select(workspace_members.c.role)
            .join(users, users.c.id == workspace_members.c.user_id)
            .where(
                workspace_members.c.workspace_id == context.workspace_id,
                workspace_members.c.user_id == context.user_id,
                users.c.status == "active",
            )
        )
    ).scalar_one_or_none()
    if member is None:
        raise ResourceNotFound("Resource not found")
    return member


class IdentityService:
    def __init__(self, db, *, require_email_verification=True, session_ttl_seconds=86400):
        self.db = db
        self.require_email_verification = require_email_verification
        self.session_ttl_seconds = session_ttl_seconds
        self.hasher = PasswordHasher()

    async def register(self, email, password):
        email = email.strip().casefold()
        if len(email) > 254 or "@" not in email or not email.split("@")[0] or len(password) < 8:
            raise ValueError("Valid email and password of at least 8 characters required")
        password_hash = await asyncio.to_thread(self.hasher.hash, password)
        uid, wid = str(uuid4()), str(uuid4())
        try:
            async with self.db.transaction() as conn:
                await conn.execute(
                    insert(users).values(
                        id=uid,
                        email=email,
                        password_hash=password_hash,
                        email_verified_at=None if self.require_email_verification else now_ms(),
                    )
                )
                await conn.execute(insert(workspaces).values(id=wid, name="Personal workspace"))
                await conn.execute(
                    insert(workspace_members).values(
                        id=str(uuid4()), workspace_id=wid, user_id=uid, role="owner"
                    )
                )
                user = (await conn.execute(select(users).where(users.c.id == uid))).mappings().one()
                workspace = (
                    (await conn.execute(select(workspaces).where(workspaces.c.id == wid)))
                    .mappings()
                    .one()
                )
        except IntegrityError as exc:
            raise DuplicateEmail("Email already registered") from exc
        return {"user": public_user(user), "workspace": dict(workspace)}

    async def login(self, email, password):
        async with self.db.transaction() as conn:
            user = (
                (await conn.execute(select(users).where(users.c.email == email.strip().casefold())))
                .mappings()
                .first()
            )
        if not user:
            # Equal expensive password work for missing accounts.
            await asyncio.to_thread(self.hasher.hash, password)
            raise InvalidCredentials()
        try:
            await asyncio.to_thread(self.hasher.verify, user["password_hash"], password)
        except (VerificationError, InvalidHashError):
            raise InvalidCredentials() from None
        token = secrets.token_urlsafe(32)
        verified_hash = user["password_hash"]
        async with self.db.transaction() as conn:
            user = (
                (
                    await conn.execute(
                        select(users).where(users.c.id == user["id"]).with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if (
                user["password_hash"] != verified_hash
                or user["status"] != "active"
                or (self.require_email_verification and user["email_verified_at"] is None)
            ):
                raise InvalidCredentials()
            workspace = (
                (
                    await conn.execute(
                        select(workspaces)
                        .join(
                            workspace_members, workspace_members.c.workspace_id == workspaces.c.id
                        )
                        .where(workspace_members.c.user_id == user["id"])
                        .order_by(workspaces.c.created_at, workspaces.c.id)
                    )
                )
                .mappings()
                .first()
            )
            if not workspace:
                raise InvalidCredentials()
            await conn.execute(
                insert(auth_sessions).values(
                    id=str(uuid4()),
                    user_id=user["id"],
                    token_hash=token_hash(token),
                    expires_at=now_ms() + self.session_ttl_seconds * 1000,
                )
            )
        return {"token": token, "user": public_user(user), "workspace": dict(workspace)}

    async def authenticate(self, token):
        async with self.db.transaction() as conn:
            user = (
                (
                    await conn.execute(
                        select(users)
                        .join(auth_sessions, auth_sessions.c.user_id == users.c.id)
                        .where(
                            auth_sessions.c.token_hash == token_hash(token),
                            auth_sessions.c.revoked_at.is_(None),
                            auth_sessions.c.expires_at > now_ms(),
                            users.c.status == "active",
                        )
                    )
                )
                .mappings()
                .first()
            )
        if not user or (self.require_email_verification and user["email_verified_at"] is None):
            raise InvalidCredentials()
        return public_user(user)

    async def logout(self, token):
        async with self.db.transaction() as conn:
            await conn.execute(
                update(auth_sessions)
                .where(
                    auth_sessions.c.token_hash == token_hash(token),
                    auth_sessions.c.revoked_at.is_(None),
                )
                .values(revoked_at=now_ms())
            )

    async def resolve_workspace(self, token, workspace_id, request_id=""):
        user = await self.authenticate(token)
        context = RequestContext(user["id"], workspace_id, "", request_id)
        async with self.db.transaction() as conn:
            role = await require_membership(conn, context)
        return RequestContext(user["id"], workspace_id, role, request_id)
