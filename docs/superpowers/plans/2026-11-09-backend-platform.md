# Multi-user Research Platform implementation ledger

Approved specification: `docs/Research-Copilot校招项目完整改进方案.md`.

## Binding constraints

- Preserve current uncommitted Agent consolidation changes.
- MySQL stores backend business facts, not Agent memory/vector data.
- Workspace authorization applies to every resource, background task and artifact.
- Personal workspaces first; no billing or collaboration UI.
- Separate API and Worker, one memory-owning Worker process initially.
- At-least-once delivery with business idempotency and epoch fencing.
- No external effects inside database transactions.
- New behavior requires failing tests before implementation and real dependency integration verification.
- Do not claim implementation or performance that was not verified.

## Execution sequence

1. M0: baseline, dependency environment, isolated dependency services and contracts.
2. M1a: SQLAlchemy business schema/migration and identity/workspace/project service with scoped CRUD tests.
3. M1b: authenticated API, CSRF, sessions, email flows and personal credentials; private documents/files/messages.
4. M2: transactional task/outbox, leases/attempts/stages, RabbitMQ, recovery/cancel/idempotency tests.
5. M3: actual existing Supervisor integration, scoped memory/artifacts/retrieval, import/research execution, SSE and frontend.
6. M4: migration tooling, deployment, admin/limits/observability, fault tests and benchmark evidence.
7. Independent review, fix findings, complete regression/integration/frontend verification.

Each task produces an independently testable result. Concrete task briefs and reports are recorded in `.superpowers/backend-platform/`; this ledger tracks completion without losing the original scope.

## Progress — 2026-11-14

- M0: preserved original Agent consolidation; baseline 979 tests. Branch remains `feat/multi-user-research-platform`; no automatic merge/push.
- M1: scoped MySQL models through migration 0006, identity/projects/papers/documents/messages/reports, signed uploads, email/PAT/session flows. Frontend uses authenticated platform API. Selected resources beyond the first 100-row page are explicitly filtered before authorization.
- M2: transactional task/outbox, generation/epoch, bounded leases/retries/quotas, cancellation, stage reuse and real queue/MySQL failure tests. Project deletion requests cancellation of in-flight work.
- M3: existing Supervisor bridge, subprocess PDF parser, immutable parsed publication, attempt-scoped vector/graph and staged SQLite memory. Retriever validates nested graph sources; corrupt memory manifests are quarantined rather than blocking startup. Logout reloads the document to discard all account-local state and callbacks.
- M4 engineering: admin/audit/limits/metrics, migration and object reconciliation tools, foreground launcher, real isolated backup/restore rehearsal, synthetic service/database benchmark and query-plan optimization. Delivery documents include architecture/ER/state machine, demo and three retrospectives.

## Evidence and outstanding acceptance

The authoritative commands, final results and coverage limits are in `docs/platform-validation.md`. Do not infer production readiness from a test count.

Backup fingerprint comparison: 24 tables / 361089 rows / 5 real file hashes identical. Benchmark reports service+MySQL performance plus a separate small-fixture real HTTP sample, neither establishing model capacity. Real users, fresh-host deployment, real model/Milvus verification, HTTP scheduling capacity and relocated multimodal artifacts remain explicitly unaccepted. These require additional external or deployment work and are not represented as completed M4 gates.
