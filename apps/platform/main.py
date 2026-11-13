"""Production entrypoint: no legacy anonymous routes or shared research service singleton."""

import asyncio
import hashlib
import json
import logging
import secrets
import smtplib
import time
from collections import deque
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select, update, insert, func
from redis.asyncio import Redis
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database, tables as t
from adapters.object_store.s3 import S3ObjectStore
from services.identity import (
    IdentityService,
    InvalidCredentials,
    DuplicateEmail,
    ResourceNotFound,
    PermissionDenied,
)
from services.projects import ProjectService, VersionConflict
from services.platform.business import BusinessService
from services.platform.accounts import AccountActions
from services.platform.credentials import CredentialService
from services.identity import require_membership
from domain.schemas.identity import RequestContext

logger = logging.getLogger(__name__)


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=10000)


class MessageInput(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    document_ids: list[str] = Field(default_factory=list, max_length=20)


class UploadInput(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PaperInput(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    metadata: dict = Field(default_factory=dict)


class ImportInput(BaseModel):
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def selected(self):
        if not self.document_ids and not self.paper_ids:
            raise ValueError("Select at least one paper or document")
        if len(self.document_ids) + len(self.paper_ids) > 20:
            raise ValueError("At most 20 items per import")
        return self


class ProjectPatch(BaseModel):
    version: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=10000)


class EmailInput(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ActionInput(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    password: str | None = Field(default=None, min_length=8, max_length=256)


class CredentialInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    ttl_seconds: int = Field(default=2592000, ge=60, le=7776000)


def create_app(settings=None, *, db=None, objects=None):
    settings = settings or PlatformSettings()
    db = db or Database(settings.database_url)
    objects = objects or S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    identity = IdentityService(db, require_email_verification=settings.require_email_verification)
    projects = ProjectService(db)
    business = BusinessService(db, objects, settings.max_file_bytes)
    accounts = AccountActions(db, identity, settings)
    credentials = CredentialService(db)
    redis = Redis.from_url(settings.redis_url)

    @asynccontextmanager
    async def lifespan(app):
        if hasattr(objects, "initialize"):
            await objects.initialize()
        async with db.transaction() as conn:
            await conn.execute(select(t.users.c.id).limit(1))
        yield
        await redis.aclose()
        await db.close()

    app = FastAPI(title="Research Copilot Platform", lifespan=lifespan)
    app.state.db = db
    app.state.business = business
    app.state.identity = identity
    app.state.settings = settings
    app.state.accounts = accounts
    recent_http = deque(maxlen=2048)

    def tasks():
        from services.task_execution import TaskService

        return TaskService(db, max_running=settings.max_running, max_queued=settings.max_queued)

    @app.middleware("http")
    async def boundary(request, call_next):
        request.state.request_id = str(uuid4())
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_file_bytes + 1048576):
            return JSONResponse({"code": "request_too_large"}, 413)
        if (
            settings.environment == "production"
            and request.method == "POST"
            and request.url.path.endswith("/uploads")
        ):
            return JSONResponse({"code": "use_signed_upload"}, 410)
        unsafe = request.method not in ("GET", "HEAD", "OPTIONS")
        origin = request.headers.get("origin")
        if unsafe and origin and origin.rstrip("/") != settings.public_origin.rstrip("/"):
            return JSONResponse({"code": "origin_forbidden"}, 403)
        if unsafe and request.cookies.get("rc_session"):
            cookie = request.cookies.get("rc_csrf", "")
            header = request.headers.get("x-csrf-token", "")
            if not cookie or not secrets.compare_digest(cookie, header):
                return JSONResponse({"code": "csrf_required"}, 403)
        if settings.rate_limit_enabled:
            # Authentication endpoints are always limited by connection IP, never caller-chosen tokens.
            subject = "ip:" + (request.client.host if request.client else "unknown")
            if not request.url.path.startswith("/api/v1/auth/"):
                try:
                    user = await authenticated(request)
                    subject = "user:" + user["id"]
                except InvalidCredentials:
                    pass
            subject = hashlib.sha256(subject.encode()).hexdigest()
            try:
                value = await redis.eval(
                    "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],60) end; return n",
                    1,
                    "rc:rate:" + subject,
                )
                if value > 120:
                    return JSONResponse(
                        {"code": "rate_limited"}, 429, headers={"Retry-After": "60"}
                    )
            except Exception:
                if unsafe:
                    return JSONResponse({"code": "rate_limit_unavailable"}, 503)
        started = time.perf_counter()
        response = await call_next(request)
        recent_http.append((round((time.perf_counter() - started) * 1000, 2), response.status_code))
        route = request.scope.get("route")
        logger.info(
            json.dumps(
                {
                    "event": "http.request",
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "route": getattr(route, "path", "unmatched"),
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def error_handler(request, exc):
        status = (
            401
            if isinstance(exc, InvalidCredentials)
            else 404
            if isinstance(exc, ResourceNotFound)
            else 403
            if isinstance(exc, PermissionDenied)
            else 409
            if isinstance(exc, (DuplicateEmail, VersionConflict))
            else 422
        )
        return JSONResponse(
            {
                "code": type(exc).__name__,
                "detail": str(exc) or "Request rejected",
                "request_id": getattr(request.state, "request_id", ""),
            },
            status,
        )

    for error in (
        InvalidCredentials,
        ResourceNotFound,
        PermissionDenied,
        DuplicateEmail,
        VersionConflict,
        ValueError,
    ):
        app.add_exception_handler(error, error_handler)
    # Task exceptions are registered when the subsystem is installed.
    try:
        from services.task_execution import IdempotencyConflict, QuotaExceeded, TaskConflict

        async def task_error(request, exc):
            return JSONResponse(
                {"code": type(exc).__name__, "detail": str(exc)},
                429 if isinstance(exc, QuotaExceeded) else 409,
            )

        app.add_exception_handler(IdempotencyConflict, task_error)
        app.add_exception_handler(QuotaExceeded, task_error)
        app.add_exception_handler(TaskConflict, task_error)
    except ImportError:
        pass

    def token(request):
        auth = request.headers.get("authorization", "")
        return auth[7:] if auth.startswith("Bearer ") else request.cookies.get("rc_session", "")

    async def authenticated(request):
        raw = token(request)
        if raw.startswith("rc_"):
            return await credentials.authenticate(
                raw, "read" if request.method in ("GET", "HEAD") else "write"
            )
        return await identity.authenticate(raw)

    async def ctx(request, wid):
        user = await authenticated(request)
        context = RequestContext(user["id"], wid, "", request.state.request_id)
        async with db.transaction() as c:
            role = await require_membership(c, context)
        return RequestContext(user["id"], wid, role, request.state.request_id)

    def key(request):
        value = request.headers.get("idempotency-key", "")
        if not value or len(value) > 128:
            raise HTTPException(422, "Idempotency-Key required (maximum 128 characters)")
        return value

    @app.get("/health")
    async def health():
        async with db.transaction() as conn:
            await conn.execute(select(1))
        return {"status": "ok", "service": "platform-api"}

    @app.post("/api/v1/auth/register", status_code=201)
    async def register(body: Credentials):
        result = await identity.register(body.email, body.password)
        if settings.require_email_verification:
            try:
                await accounts.request(body.email, "verify")
            except Exception:
                logger.exception("Verification delivery failed; user can request resend")
        return result

    @app.post("/api/v1/auth/login")
    async def login(body: Credentials, response: Response):
        result = await identity.login(body.email, body.password)
        response.set_cookie(
            "rc_session",
            result.pop("token"),
            httponly=True,
            secure=settings.environment == "production",
            samesite="lax",
            max_age=86400,
        )
        response.set_cookie(
            "rc_csrf",
            secrets.token_urlsafe(32),
            secure=settings.environment == "production",
            samesite="lax",
            max_age=86400,
        )
        return result

    @app.post("/api/v1/auth/logout", status_code=204)
    async def logout(request: Request, response: Response):
        await identity.logout(token(request))
        response.delete_cookie("rc_session")
        response.delete_cookie("rc_csrf")

    @app.post("/api/v1/auth/{kind}/request", status_code=202)
    async def request_action(kind: str, body: EmailInput):
        try:
            await accounts.request(body.email, kind)
        except (OSError, smtplib.SMTPException):
            logger.warning("Account email unavailable")
        return {"status": "accepted"}

    @app.post("/api/v1/auth/{kind}/confirm")
    async def confirm_action(kind: str, body: ActionInput):
        if kind not in ("verify", "reset"):
            raise HTTPException(404)
        await accounts.consume(body.token, kind, body.password)
        return {"status": "ok"}

    @app.get("/api/v1/me")
    async def me(request: Request):
        return await authenticated(request)

    @app.get("/api/v1/workspaces")
    async def workspaces(request: Request):
        user = await authenticated(request)
        async with db.transaction() as conn:
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        select(t.workspaces)
                        .join(
                            t.workspace_members,
                            t.workspace_members.c.workspace_id == t.workspaces.c.id,
                        )
                        .where(t.workspace_members.c.user_id == user["id"])
                    )
                ).mappings()
            ]

    @app.get("/api/v1/credentials")
    async def list_credentials(request: Request):
        user = await identity.authenticate(token(request))
        return await credentials.list(user["id"])

    @app.post("/api/v1/credentials", status_code=201)
    async def create_credential(body: CredentialInput, request: Request):
        user = await identity.authenticate(token(request))
        return await credentials.create(user["id"], body.name, body.scopes, body.ttl_seconds)

    @app.delete("/api/v1/credentials/{credential_id}", status_code=204)
    async def revoke_credential(credential_id: str, request: Request):
        user = await identity.authenticate(token(request))
        await credentials.revoke(user["id"], credential_id)

    @app.get("/api/v1/sessions")
    async def sessions(request: Request):
        user = await identity.authenticate(token(request))
        async with db.transaction() as c:
            return [
                dict(r)
                for r in (
                    await c.execute(
                        select(
                            t.auth_sessions.c.id,
                            t.auth_sessions.c.created_at,
                            t.auth_sessions.c.expires_at,
                            t.auth_sessions.c.revoked_at,
                        ).where(t.auth_sessions.c.user_id == user["id"])
                    )
                ).mappings()
            ]

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def revoke_session(session_id: str, request: Request):
        user = await identity.authenticate(token(request))
        async with db.transaction() as c:
            result = await c.execute(
                update(t.auth_sessions)
                .where(t.auth_sessions.c.user_id == user["id"], t.auth_sessions.c.id == session_id)
                .values(revoked_at=t.now_ms())
            )
            if not result.rowcount:
                raise ResourceNotFound()

    @app.get("/api/v1/workspaces/{wid}/projects")
    async def list_projects(wid: str, request: Request):
        return await projects.list(await ctx(request, wid))

    @app.post("/api/v1/workspaces/{wid}/projects", status_code=201)
    async def create_project(wid: str, body: ProjectInput, request: Request):
        return await projects.create(await ctx(request, wid), body.name, body.description)

    @app.patch("/api/v1/workspaces/{wid}/projects/{pid}")
    async def edit_project(wid: str, pid: str, body: ProjectPatch, request: Request):
        return await projects.update(
            await ctx(request, wid), pid, body.version, name=body.name, description=body.description
        )

    @app.delete("/api/v1/workspaces/{wid}/projects/{pid}", status_code=204)
    async def delete_project(wid: str, pid: str, version: int, request: Request):
        await projects.delete(await ctx(request, wid), pid, version)

    @app.get("/api/v1/workspaces/{wid}/projects/{pid}/papers")
    async def project_papers(wid: str, pid: str, request: Request, offset: int = 0):
        from services.platform.papers import PaperService

        return await PaperService(db).list(await ctx(request, wid), project_id=pid, offset=offset)

    @app.post("/api/v1/workspaces/{wid}/projects/{pid}/papers", status_code=201)
    async def save_paper(wid: str, pid: str, body: PaperInput, request: Request):
        from services.platform.papers import PaperService

        return await PaperService(db).save(
            await ctx(request, wid), body.title, body.metadata, project_id=pid
        )

    @app.put("/api/v1/workspaces/{wid}/projects/{pid}/papers/{paper_id}")
    async def attach_paper(wid: str, pid: str, paper_id: str, request: Request):
        from services.platform.papers import PaperService

        return await PaperService(db).attach(await ctx(request, wid), pid, paper_id)

    @app.delete("/api/v1/workspaces/{wid}/projects/{pid}/papers/{paper_id}", status_code=204)
    async def detach_paper(wid: str, pid: str, paper_id: str, request: Request):
        from services.platform.papers import PaperService

        await PaperService(db).detach(await ctx(request, wid), pid, paper_id)

    @app.post("/api/v1/workspaces/{wid}/projects/{pid}/uploads/init", status_code=201)
    async def begin_upload(wid: str, pid: str, body: UploadInput, request: Request):
        from services.platform.uploads import UploadService

        return await UploadService(business).begin(
            await ctx(request, wid), pid, body.filename, body.size, body.sha256
        )

    @app.post("/api/v1/workspaces/{wid}/uploads/{upload_id}/complete")
    async def complete_upload(wid: str, upload_id: str, request: Request):
        from services.platform.uploads import UploadService

        return await UploadService(business).complete(await ctx(request, wid), upload_id)

    @app.post("/api/v1/workspaces/{wid}/projects/{pid}/uploads", status_code=201)
    async def upload(wid: str, pid: str, request: Request, file: UploadFile = File(...)):
        context = await ctx(request, wid)
        data = await file.read(settings.max_file_bytes + 1)
        return await business.upload(context, pid, file.filename or "paper.pdf", data)

    @app.get("/api/v1/workspaces/{wid}/projects/{pid}/documents")
    async def documents(wid: str, pid: str, request: Request):
        return await business.documents(await ctx(request, wid), pid)

    @app.get("/api/v1/workspaces/{wid}/files/{fid}/content")
    async def file(wid: str, fid: str, request: Request):
        data, mime = await business.file(await ctx(request, wid), fid)
        return Response(
            data,
            media_type=mime,
            headers={"Content-Disposition": 'inline; filename="document.pdf"'},
        )

    @app.delete("/api/v1/workspaces/{wid}/documents/{did}", status_code=204)
    async def delete_document(wid: str, did: str, request: Request):
        await business.delete_document(await ctx(request, wid), did)

    @app.post("/api/v1/workspaces/{wid}/projects/{pid}/conversations", status_code=201)
    async def new_conversation(wid: str, pid: str, request: Request):
        return await business.create_conversation(await ctx(request, wid), pid)

    @app.get("/api/v1/workspaces/{wid}/projects/{pid}/conversations")
    async def conversations(wid: str, pid: str, request: Request):
        return await business.conversations(await ctx(request, wid), pid)

    @app.get("/api/v1/workspaces/{wid}/conversations/{cid}/messages")
    async def messages(
        wid: str,
        cid: str,
        request: Request,
        after: int = 0,
        limit: int = 50,
        latest: bool = False,
        before: int | None = None,
    ):
        rows = await business.messages(
            await ctx(request, wid), cid, after, limit, latest=latest, before=before
        )
        return list(reversed(rows)) if latest else rows

    @app.post("/api/v1/workspaces/{wid}/conversations/{cid}/messages", status_code=202)
    async def send(wid: str, cid: str, body: MessageInput, request: Request):
        context = await ctx(request, wid)
        async with db.transaction() as conn:
            conv = await business._resource(conn, context, t.conversations, cid)
        if body.document_ids:
            available = await business.documents(
                context, conv["project_id"], document_ids=body.document_ids
            )
            if set(body.document_ids) - {d["id"] for d in available}:
                raise ResourceNotFound()
        return await tasks().submit(
            context,
            conv["project_id"],
            "research",
            body.model_dump(),
            idempotency_key=key(request),
            conversation_id=cid,
        )

    @app.post("/api/v1/workspaces/{wid}/projects/{pid}/imports", status_code=202)
    async def imports(wid: str, pid: str, body: ImportInput, request: Request):
        context = await ctx(request, wid)
        available = await business.documents(context, pid, document_ids=body.document_ids)
        if set(body.document_ids) - {d["id"] for d in available}:
            raise ResourceNotFound()
        if body.paper_ids:
            from services.platform.papers import PaperService

            linked = await PaperService(db).list(context, project_id=pid, paper_ids=body.paper_ids)
            if set(body.paper_ids) - {p["id"] for p in linked}:
                raise ResourceNotFound()
        return await tasks().submit(
            context, pid, "import", body.model_dump(), idempotency_key=key(request)
        )

    @app.get("/api/v1/workspaces/{wid}/tasks")
    async def list_tasks(wid: str, request: Request, project_id: str | None = None):
        return await tasks().list(await ctx(request, wid), project_id=project_id)

    @app.get("/api/v1/workspaces/{wid}/tasks/{tid}")
    async def task(wid: str, tid: str, request: Request):
        return await tasks().get(await ctx(request, wid), tid)

    @app.post("/api/v1/workspaces/{wid}/tasks/{tid}/cancel")
    async def cancel(wid: str, tid: str, request: Request):
        return await tasks().cancel(await ctx(request, wid), tid)

    @app.post("/api/v1/workspaces/{wid}/tasks/{tid}/retry", status_code=202)
    async def retry(wid: str, tid: str, request: Request):
        return await tasks().retry(await ctx(request, wid), tid)

    @app.get("/api/v1/workspaces/{wid}/tasks/{tid}/events")
    async def events(wid: str, tid: str, request: Request, after: int = 0):
        context = await ctx(request, wid)
        await tasks().get(context, tid)
        cursor = max(after, int(request.headers.get("last-event-id", "0")))

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                # Reauthorize on each batch: revocation affects open connections.
                await ctx(request, wid)
                rows = await tasks().events(context, tid, after=cursor)
                for event in rows:
                    cursor = event["event_seq"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                state = await tasks().get(context, tid)
                if state["status"] in (
                    "succeeded",
                    "failed",
                    "cancelled",
                    "partially_succeeded",
                ) and cursor >= state.get("event_seq", 0):
                    break
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/v1/workspaces/{wid}/projects/{pid}/reports")
    async def reports(wid: str, pid: str, request: Request):
        return await business.reports(await ctx(request, wid), pid)

    async def admin(request):
        user = await identity.authenticate(token(request))
        if user["email"] not in {
            e.strip().casefold() for e in settings.admin_emails.split(",") if e.strip()
        }:
            raise PermissionDenied()
        return user

    @app.get("/api/v1/admin/tasks")
    async def admin_tasks(request: Request):
        await admin(request)
        async with db.transaction() as c:
            return [
                dict(r)
                for r in (
                    await c.execute(
                        select(
                            t.research_tasks.c.id,
                            t.research_tasks.c.workspace_id,
                            t.research_tasks.c.kind,
                            t.research_tasks.c.status,
                            t.research_tasks.c.error_code,
                            t.research_tasks.c.created_at,
                        )
                        .order_by(t.research_tasks.c.created_at.desc())
                        .limit(100)
                    )
                ).mappings()
            ]

    @app.get("/api/v1/admin/tasks/{tid}")
    async def admin_task_detail(tid: str, request: Request):
        await admin(request)
        async with db.transaction() as c:
            task = (
                (
                    await c.execute(
                        select(
                            t.research_tasks.c.id,
                            t.research_tasks.c.status,
                            t.research_tasks.c.error_code,
                            t.research_tasks.c.execution_epoch,
                            t.research_tasks.c.attempt_count,
                        ).where(t.research_tasks.c.id == tid)
                    )
                )
                .mappings()
                .first()
            )
            if not task:
                raise ResourceNotFound()
            attempts = (
                (
                    await c.execute(
                        select(
                            t.task_attempts.c.attempt_no,
                            t.task_attempts.c.status,
                            t.task_attempts.c.error_code,
                            t.task_attempts.c.lease_expires_at,
                        ).where(t.task_attempts.c.task_id == tid)
                    )
                )
                .mappings()
                .all()
            )
            stages = (
                (
                    await c.execute(
                        select(
                            t.task_stages.c.scope_key, t.task_stages.c.stage, t.task_stages.c.status
                        ).where(t.task_stages.c.task_id == tid)
                    )
                )
                .mappings()
                .all()
            )
        return {
            **dict(task),
            "attempts": [dict(r) for r in attempts],
            "stages": [dict(r) for r in stages],
        }

    @app.post("/api/v1/admin/tasks/{tid}/{operation}")
    async def admin_control(tid: str, operation: str, request: Request):
        actor = await admin(request)
        from services.platform.administration import Administration

        return await Administration(tasks()).control(
            actor["id"], tid, operation, request.state.request_id
        )

    @app.get("/api/v1/admin/metrics")
    async def admin_metrics(request: Request):
        await admin(request)
        async with db.transaction() as c:
            rows = (
                await c.execute(
                    select(t.research_tasks.c.status, func.count()).group_by(
                        t.research_tasks.c.status
                    )
                )
            ).all()
            pending = (
                await c.execute(
                    select(func.count())
                    .select_from(t.outbox_events)
                    .where(t.outbox_events.c.published_at.is_(None))
                )
            ).scalar_one()
            oldest = (
                await c.execute(
                    select(func.min(t.research_tasks.c.created_at)).where(
                        t.research_tasks.c.status.in_(["queued", "retry_wait"])
                    )
                )
            ).scalar_one()
            retries = (
                await c.execute(
                    select(func.sum(t.research_tasks.c.attempt_count)).where(
                        t.research_tasks.c.attempt_count > 1
                    )
                )
            ).scalar_one() or 0
            expired = (
                await c.execute(
                    select(func.count())
                    .select_from(t.task_attempts)
                    .where(
                        t.task_attempts.c.status == "running",
                        t.task_attempts.c.lease_expires_at < await tasks()._now(c),
                    )
                )
            ).scalar_one()
            errors = dict(
                (
                    await c.execute(
                        select(t.research_tasks.c.error_code, func.count())
                        .where(t.research_tasks.c.error_code.is_not(None))
                        .group_by(t.research_tasks.c.error_code)
                    )
                ).all()
            )
        durations = sorted(ms for ms, status in recent_http)
        return {
            "tasks": dict(rows),
            "outbox_pending": pending,
            "oldest_queued_age_ms": max(0, t.now_ms() - oldest) if oldest else 0,
            "attempts_on_retried_tasks": retries,
            "expired_running_leases": expired,
            "errors_by_code": errors,
            "database_pool": db.engine.pool.status(),
            "recent_process_http": {
                "samples": len(durations),
                "p95_ms": durations[max(0, int(len(durations) * 0.95) - 1)] if durations else None,
                "server_errors": sum(status >= 500 for ms, status in recent_http),
            },
        }

    @app.post("/api/v1/admin/users/{uid}/disable", status_code=204)
    async def disable_user(uid: str, request: Request):
        actor = await admin(request)
        if actor["id"] == uid:
            raise ValueError("Cannot disable your own admin session")
        async with db.transaction() as c:
            result = await c.execute(
                update(t.users).where(t.users.c.id == uid).values(status="disabled")
            )
            if not result.rowcount:
                raise ResourceNotFound()
            await c.execute(
                update(t.auth_sessions)
                .where(t.auth_sessions.c.user_id == uid)
                .values(revoked_at=t.now_ms())
            )
            workspace = (
                await c.execute(
                    select(t.workspace_members.c.workspace_id)
                    .where(t.workspace_members.c.user_id == actor["id"])
                    .limit(1)
                )
            ).scalar_one()
            await c.execute(
                insert(t.audit_events).values(
                    id=str(uuid4()),
                    workspace_id=workspace,
                    actor_id=actor["id"],
                    action="user.disable",
                    resource_id=uid,
                    request_id=request.state.request_id,
                    outcome="success",
                )
            )

    return app


app = create_app()
