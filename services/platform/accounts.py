import asyncio
import secrets
import smtplib
from email.message import EmailMessage
from uuid import uuid4
from sqlalchemy import select, insert, update
from adapters.storage.mysql import tables as t
from services.identity import token_hash, InvalidCredentials


class AccountActions:
    def __init__(self, db, identity, settings):
        self.db, self.identity, self.settings = db, identity, settings

    async def request(self, email, kind):
        if kind not in ("verify", "reset"):
            raise ValueError("Invalid action")
        async with self.db.transaction() as conn:
            user = (
                (
                    await conn.execute(
                        select(t.users).where(
                            t.users.c.email == email.strip().casefold(),
                            t.users.c.status == "active",
                        )
                    )
                )
                .mappings()
                .first()
            )
            if not user:
                return
            raw = secrets.token_urlsafe(32)
            await conn.execute(
                insert(t.email_action_tokens).values(
                    id=str(uuid4()),
                    user_id=user["id"],
                    kind=kind,
                    token_hash=token_hash(raw),
                    expires_at=t.now_ms() + 1800000,
                )
            )
        msg = EmailMessage()
        msg["From"] = self.settings.smtp_sender
        msg["To"] = user["email"]
        msg["Subject"] = "Research Copilot account action"
        msg.set_content(
            f"{self.settings.public_origin}/?action={kind}&token={raw}\nExpires in 30 minutes."
        )

        def send():
            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=10) as smtp:
                if self.settings.smtp_starttls:
                    smtp.starttls()
                if self.settings.smtp_username:
                    smtp.login(self.settings.smtp_username, self.settings.smtp_password)
                smtp.send_message(msg)

        await asyncio.to_thread(send)

    async def consume(self, token, kind, password=None):
        hashed = None
        if kind == "reset":
            if not password or not 8 <= len(password) <= 256:
                raise ValueError("Password must be 8–256 characters")
            hashed = await asyncio.to_thread(self.identity.hasher.hash, password)
        async with self.db.transaction() as conn:
            action = (
                (
                    await conn.execute(
                        select(t.email_action_tokens)
                        .where(
                            t.email_action_tokens.c.token_hash == token_hash(token),
                            t.email_action_tokens.c.kind == kind,
                            t.email_action_tokens.c.consumed_at.is_(None),
                            t.email_action_tokens.c.expires_at > t.now_ms(),
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if not action:
                raise InvalidCredentials()
            user = (
                (
                    await conn.execute(
                        select(t.users).where(t.users.c.id == action["user_id"]).with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if user["status"] != "active":
                raise InvalidCredentials()
            await conn.execute(
                update(t.email_action_tokens)
                .where(
                    t.email_action_tokens.c.id == action["id"],
                    t.email_action_tokens.c.consumed_at.is_(None),
                )
                .values(consumed_at=t.now_ms())
            )
            if kind == "verify":
                await conn.execute(
                    update(t.users)
                    .where(t.users.c.id == user["id"])
                    .values(email_verified_at=t.now_ms())
                )
            else:
                await conn.execute(
                    update(t.users).where(t.users.c.id == user["id"]).values(password_hash=hashed)
                )
                await conn.execute(
                    update(t.auth_sessions)
                    .where(t.auth_sessions.c.user_id == user["id"])
                    .values(revoked_at=t.now_ms())
                )
                await conn.execute(
                    update(t.api_credentials)
                    .where(t.api_credentials.c.user_id == user["id"])
                    .values(revoked_at=t.now_ms())
                )
