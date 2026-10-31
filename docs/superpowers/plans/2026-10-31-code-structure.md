# Research-Copilot structure cleanup implementation plan

> For agentic workers: use subagent-driven-development task by task, with implementation and independent review gates.

**Goal:** Complete the five cleanup stages approved by the user while preserving the research application's capabilities.

**Architecture:** One collection QA implementation, one real specialist execution registry, explicit application services, and one runtime state commit boundary. Preserve public HTTP/CLI response contracts; legacy Python interfaces may remain as thin delegating adapters. Keep research orchestration separate from RAG infrastructure.

**Tech stack:** Python 3.11, FastAPI, LangGraph, Pydantic, Next.js/TypeScript.

## Global constraints

- Work in `/home/myc/Research-Copilot`, branch `refactor/code-structure-20261031`; baseline commit `807d2c472be5883c45d501ab0decee654c551cf8`.
- Do not modify `.env`, user data, resume files, or backup directories. No live model calls or external writes for tests.
- Preserve public HTTP/CLI behavior and response fields. Retain compatibility only through thin adapters with tests.
- Do not add new infrastructure, frameworks, or model features. Avoid cosmetic repository-wide rewrites.
- Regression-test behavioral changes before implementation; run focused tests per task and the full suite at the end.
- Each implementation task owns one commit or a small cohesive sequence, plus a report in `.superpowers/sdd/`.
- Application services load and persist data. Specialists return results/state updates. Registry metadata must describe the actual execution path.

## Task 1: Reproducible baseline and packaging

Files: `pyproject.toml`, `.gitignore`, `services/research/ARCHITECTURE.md`, `evaluation/README.md`, selected evaluation Python/sample sources, tests under `tests/unit/packaging/`.

- Add a regression test proving package discovery includes `runtime` and its subpackages; fix package discovery.
- Inspect prompt/skill resource loading and ensure wheel includes resources needed by installed runtime. Validate wheel contents and imports outside repository without copying secrets/data.
- Track evaluation implementation, sample fixtures and canonical documentation deliberately; keep downloaded datasets, generated reports and miscellaneous local docs ignored. Do not bulk-add local data.
- Update canonical architecture description to the actual current direct `run_action` flow (later tasks update this again).
- Record complete baseline tests; investigate pre-existing failures rather than silently ignoring them.

## Task 2: Single collection-QA implementation

Files: `agents/research_knowledge_agent.py`, `agents/research_writer_agent.py`, `tools/research/collection_qa_capability.py`, related tests.

- Compare legacy knowledge/writer collection QA behavior with capability implementation; preserve necessary semantic differences explicitly.
- Make `ResearchCollectionKnowledgeCapability` and `ResearchCollectionAnswerCapability` the authoritative implementations. Replace legacy collection QA methods with thin delegation or remove only after proving no required callers remain.
- Preserve discovery/report writing behavior, constructors and tested public compatibility methods.
- Use behavioral regression tests for query planning, evidence retrieval, language, citations and selected-paper analysis. Reuse existing fixtures where possible.
- Remove duplicate private implementations; do not move them into another large mixin.

## Task 3: One executable specialist registry and consistent Supervisor policy

Files: `runtime/research/unified_runtime.py`, `context_builder.py`, `supervisor_graph_runtime_core.py`, `result_aggregator.py`, `agents/research_supervisor_agent.py`, related schemas/tests.

- Replace phase-one skeleton adapters with a registry that binds each descriptor to the real specialist and executes `run_action` through one authoritative route.
- Use one action-to-agent catalog to derive routing/descriptor data where practical. Remove unused action executor/skeleton fields while preserving response compatibility as projections.
- Build static registry/blueprint once per runtime where safe; do not share request-specific mutable context.
- Unify no-LLM synchronous/asynchronous fallback decisions around the existing asynchronous deterministic policy, with tests for discovery-only, general chat and completion.
- Separate Supervisor fallback policy into a cohesive module if needed, without duplicating the decision builder or introducing policy plugins.
- Replace affected source-string architecture tests with meaningful behavior or AST dependency assertions.

## Task 4: Explicit services and centralized state commits

Files: `services/research/`, `memory/conversation_manager.py`, `agents/research_qa_agent.py`, `runtime/research/agent_protocol/`, `supervisor_graph_runtime_core.py`, document/chart/general agents, related tests.

- Replace the main service's inheritance-based conversation/paper/QA composition with explicit service collaborators. Keep old public service methods as thin delegates where required by API/tests.
- Move conversation application lifecycle out of memory storage; dependencies should be explicit rather than an injected whole host with implicit private methods.
- Split task-level QA execution from the RAG ReAct worker and isolate loading/commit orchestration from QA execution. Preserve existing route behavior.
- Extend typed state updates to general answer, document, chart and attempt flags. Specialists return updates; a single runtime commit boundary applies/persists them.
- Make workspace calculation pure with respect to shared context until commit; preserve actual persistence entrypoints via explicit wrappers as necessary.
- Add tests for no mutation before commit, one commit per successful action, failure paths and direct service entrypoints.

## Task 5: Boundary simplification and final documentation

Files: result serialization, remaining compatibility modules, API DTO adapters, architectural tests, canonical docs.

- Keep one internal task/result representation; generate legacy/unified public fields at a single boundary using reusable serializers. Preserve external response keys.
- Remove unused migration attributes/helpers verified against source and tests; keep explicit compatibility delegations only when needed.
- Update architecture and evaluation documentation to final behavior, ownership and verification commands. Include actual limitations, not nonexistent components.
- Replace brittle string-presence tests touched by refactor with behavior/dependency tests.
- Verify source diff, full pytest, static checks, frontend type check/build, wheel build and isolated installed import smoke. Fix failures and review the whole branch before completion.

## Completion evidence

Record per-task commits, tests and review verdicts in `.superpowers/sdd/progress.md`. Final response should report code reductions for affected existing modules separately from newly tracked evaluation sources, test counts, build results and remaining limitations.
