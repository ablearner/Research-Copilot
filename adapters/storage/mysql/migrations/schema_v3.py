"""Platform schema. All timestamps are UTC Unix milliseconds (time.time_ns() // 1_000_000)."""

import time
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Text,
    Integer,
    BigInteger,
    JSON,
    ForeignKeyConstraint,
    UniqueConstraint,
    Index,
)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def now_ms():
    return time.time_ns() // 1_000_000


ID = String(36).with_variant(String(36, collation="utf8mb4_bin"), "mysql")


def col(name, type_=ID, **kw):
    return Column(name, type_, **kw)


def ref(name, target, nullable=False):
    return ForeignKeyConstraint(["workspace_id", name], [target + ".workspace_id", target + ".id"])


def table(name, *fields, scoped=True):
    base = [col("id", primary_key=True)]
    if scoped:
        base += [
            col("workspace_id", nullable=False),
            ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
            UniqueConstraint("workspace_id", "id"),
        ]
    return Table(
        name,
        metadata,
        *base,
        col("created_at", BigInteger, nullable=False, default=now_ms),
        col("updated_at", BigInteger, nullable=False, default=now_ms),
        col("version", Integer, nullable=False, default=1),
        *fields,
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )


users = table(
    "users",
    col("email", String(254), unique=True, nullable=False),
    col("password_hash", String(512), nullable=False),
    col("status", String(32), default="active", nullable=False),
    col("email_verified_at", BigInteger),
    scoped=False,
)
workspaces = table("workspaces", col("name", String(255), nullable=False), scoped=False)
workspace_members = table(
    "workspace_members",
    col("user_id", ID, nullable=False),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    col("role", String(32), nullable=False),
    UniqueConstraint("workspace_id", "user_id"),
)
auth_sessions = table(
    "auth_sessions",
    col("user_id", ID, nullable=False),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    col("token_hash", String(64), unique=True, nullable=False),
    col("expires_at", BigInteger, nullable=False),
    col("revoked_at", BigInteger),
    scoped=False,
)
email_action_tokens = table(
    "email_action_tokens",
    col("user_id", ID, nullable=False),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    col("kind", String(32), nullable=False),
    col("token_hash", String(64), nullable=False, unique=True),
    col("expires_at", BigInteger, nullable=False),
    col("consumed_at", BigInteger),
    scoped=False,
)
projects = table(
    "projects",
    col("name", String(255), nullable=False),
    col("description", Text, default=""),
    col("status", String(32), nullable=False, default="active"),
)
papers = table(
    "papers",
    col("title", Text),
    col("metadata_json", JSON),
    col("status", String(32), default="active"),
)
project_papers = table(
    "project_papers",
    col("project_id", nullable=False),
    col("paper_id", nullable=False),
    ref("project_id", "projects"),
    ref("paper_id", "papers"),
    UniqueConstraint("workspace_id", "project_id", "paper_id"),
)
file_objects = table(
    "file_objects",
    col("object_key", String(512), nullable=False),
    col("sha256", String(64), nullable=False),
    col("size", BigInteger, nullable=False),
    col("mime", String(255)),
    col("status", String(32), default="pending"),
    UniqueConstraint("workspace_id", "object_key"),
)
documents = table(
    "documents",
    col("paper_id"),
    ref("paper_id", "papers"),
    col("status", String(32), default="pending"),
    col("active_version"),
    col("deleted_at", BigInteger),
)
document_versions = table(
    "document_versions",
    col("document_id", nullable=False),
    ref("document_id", "documents"),
    col("file_id", nullable=False),
    ref("file_id", "file_objects"),
    col("sha256", String(64)),
    col("parser_version", String(128)),
    col("embedding_version", String(128)),
    col("parse_status", String(32), default="pending"),
    col("index_status", String(32), default="pending"),
)
conversations = table(
    "conversations",
    col("project_id", nullable=False),
    ref("project_id", "projects"),
    col("title", Text),
    col("status", String(32), default="active"),
)
research_tasks = table(
    "research_tasks",
    col("user_id", ID, nullable=False),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    col("project_id", nullable=False),
    ref("project_id", "projects"),
    col("conversation_id"),
    ref("conversation_id", "conversations"),
    col("kind", String(64), nullable=False),
    col("status", String(32), nullable=False, default="queued"),
    col("input_json", JSON),
    col("result_ref", String(512)),
    col("cancel_requested_at", BigInteger),
    col("generation", Integer, nullable=False, default=1),
    col("execution_epoch", BigInteger, nullable=False, default=0),
    col("attempt_count", Integer, nullable=False, default=0),
    col("next_attempt_at", BigInteger, nullable=False, default=0),
    col("error_code", String(128)),
    col("event_seq", BigInteger, nullable=False, default=0),
)
messages = table(
    "messages",
    col("conversation_id", nullable=False),
    ref("conversation_id", "conversations"),
    col("seq", BigInteger, nullable=False),
    col("role", String(32), nullable=False),
    col("content", JSON, nullable=False),
    col("task_id"),
    ref("task_id", "research_tasks"),
    UniqueConstraint("workspace_id", "conversation_id", "seq"),
)
task_attempts = table(
    "task_attempts",
    col("task_id", nullable=False),
    ref("task_id", "research_tasks"),
    col("attempt_no", Integer, nullable=False),
    col("status", String(32), nullable=False),
    col("lease_owner", String(128)),
    col("lease_expires_at", BigInteger),
    col("execution_epoch", BigInteger, nullable=False, default=1),
    col("error_code", String(128)),
    UniqueConstraint("workspace_id", "task_id", "attempt_no"),
)
task_stages = table(
    "task_stages",
    col("task_id", nullable=False),
    ref("task_id", "research_tasks"),
    col("scope_key", String(128), nullable=False),
    col("stage", String(64), nullable=False),
    col("artifact_ref", String(512)),
    col("input_fingerprint", String(64), nullable=False),
    col("status", String(32), nullable=False),
    UniqueConstraint("workspace_id", "task_id", "scope_key", "stage", "input_fingerprint"),
)
task_events = table(
    "task_events",
    col("task_id", nullable=False),
    ref("task_id", "research_tasks"),
    col("event_seq", BigInteger, nullable=False),
    col("event_type", String(64), nullable=False),
    col("payload", JSON),
    UniqueConstraint("workspace_id", "task_id", "event_seq"),
)
reports = table(
    "reports",
    col("project_id", nullable=False),
    ref("project_id", "projects"),
    col("task_id", nullable=False),
    ref("task_id", "research_tasks"),
    col("output_key", String(128), nullable=False),
    col("title", Text),
    UniqueConstraint("workspace_id", "task_id", "output_key"),
)
report_versions = table(
    "report_versions",
    col("report_id", nullable=False),
    ref("report_id", "reports"),
    col("content", JSON, nullable=False),
    UniqueConstraint("workspace_id", "report_id", "version"),
)
outbox_events = table(
    "outbox_events",
    col("kind", String(64), nullable=False),
    col("payload", JSON, nullable=False),
    col("next_attempt_at", BigInteger, nullable=False, default=now_ms),
    col("published_at", BigInteger),
    col("attempt_count", Integer, nullable=False, default=0),
)
idempotency_records = table(
    "idempotency_records",
    col("actor_id", ID, nullable=False),
    ForeignKeyConstraint(["actor_id"], ["users.id"]),
    col("operation", String(64), nullable=False),
    col("key", String(128), nullable=False),
    col("request_hash", String(64), nullable=False),
    col("resource_id", ID, nullable=False),
    col("expires_at", BigInteger, nullable=False),
    UniqueConstraint("workspace_id", "actor_id", "operation", "key"),
)
audit_events = table(
    "audit_events",
    col("actor_id", ID),
    ForeignKeyConstraint(["actor_id"], ["users.id"]),
    col("action", String(64), nullable=False),
    col("resource_id", String(128)),
    col("request_id", String(128)),
    col("outcome", String(32), nullable=False),
)
Index(
    "ix_conversation_listing",
    conversations.c.workspace_id,
    conversations.c.project_id,
    conversations.c.updated_at,
    conversations.c.id,
)
Index(
    "ix_task_listing",
    research_tasks.c.workspace_id,
    research_tasks.c.status,
    research_tasks.c.created_at,
    research_tasks.c.id,
)
Index("ix_attempt_recovery", task_attempts.c.status, task_attempts.c.lease_expires_at)
Index(
    "ix_outbox_dispatch",
    outbox_events.c.published_at,
    outbox_events.c.next_attempt_at,
    outbox_events.c.id,
)

api_credentials = table(
    "api_credentials",
    col("user_id", ID, nullable=False),
    ForeignKeyConstraint(["user_id"], ["users.id"]),
    col("name", String(128), nullable=False),
    col("token_hash", String(64), nullable=False, unique=True),
    col("scopes", JSON, nullable=False),
    col("expires_at", BigInteger, nullable=False),
    col("revoked_at", BigInteger),
    scoped=False,
)
document_versions.append_constraint(
    UniqueConstraint("workspace_id", "document_id", "id", name="uq_document_version_owner")
)
documents.append_constraint(
    ForeignKeyConstraint(
        ["workspace_id", "id", "active_version"],
        ["document_versions.workspace_id", "document_versions.document_id", "document_versions.id"],
        name="fk_document_active_version",
        use_alter=True,
    )
)
