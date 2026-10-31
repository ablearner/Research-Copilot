# Task 4 — Explicit application services and deferred workflow state

## Result

Implemented the approved composition and commit boundaries on `refactor/code-structure-20261031` from `5f42254`.

- `LiteratureResearchService` has no mixin bases. It composes `ConversationService`, `PaperOperationsService`, and `QAApplicationService` using explicit resource/scalar dependencies and named callback ports. There is no host proxy, `__getattr__`, host dictionary copying, or generic dependency container. Public facade methods retain their signatures and delegate to their owners; private compatibility delegates retain existing call sites. Private dependency callbacks are bound methods. Public callback wrappers intentionally allow supported facade substitution; the collection-capability getter preserves existing capability replacement behavior.
- Conversation CRUD, turn recording, snapshot/message construction, and runtime conversation updates live in `services/research/conversation_service.py`. The memory module retains only the old language-helper import.
- `QAApplicationService.execute_qa` loads task/report/papers and commits successful direct application calls. `ResearchQAAgent.execute_loaded_qa` executes against loaded inputs, calculates proposed follow-up state and returns a result without storage loads/saves. Its old `execute_qa` entrypoint is a thin application delegate. The follow-up helper receives the already-loaded report. `ResearchQAToolset` receives explicit chart/cache/collection dependencies instead of a whole service.
- `RagReActQAWorker` and its models now live in `agents/rag_react_qa_worker.py`. API construction imports that module directly; old worker/model/error imports from `research_qa_agent` remain compatible. Reasoning-default exports remain intact.
- `ResearchStateDelta` now carries general answers, document/index results, chart results, attempt flags, Zotero results and replacement execution contexts. Remaining specialist assignments to shared context were removed. The existing runtime apply/persist boundary handles these fields and still commits deliberate partial deltas on failed/skipped actions.
- Parse failure returns the attempt flag; index failure returns the attempt flag and parsed document. Runtime chart failures return an attempt delta. This preserves explicit partial progress without mutating the caller's context before commit.
- Workspace calculation with `persist=False` returns a replacement execution context. Compression and paper/chart analysis propagate that replacement in their delta. The existing direct `persist=True` wrapper retains storage behavior.
- Paper import now separates `prepare_paper_import` from the direct `import_papers` commit wrapper. Specialists use the prepared response/task/papers and deferred memory update; direct HTTP/application imports still persist. Runtime commits these workflow results once.

## TDD evidence

All commands use `/home/myc/miniconda3/envs/Research-Copilot/bin/python` from `/home/myc/Research-Copilot`.

1. Before state changes:
   `python -m pytest tests/unit/services/test_research_state_commit_boundary.py -q`
   **RED: 5 failed**. General answer and document/chart attempt flags changed the shared context immediately; the workspace calculation changed the original execution context. An initial test used `model_dump` on the dataclass; this fixture error was corrected to `asdict` and the five intended assertion failures were observed before implementation.
   **GREEN: 5 passed** after deferred-state changes.
2. Before composition/QA split:
   `python -m pytest tests/unit/services/test_research_architecture_constraints.py -q -k 'composes_explicit or loaded_qa_execution'`
   **RED: 2 failed**. The facade still inherited three mixins, and the task QA module still contained the RAG worker. Both assertions pass in final verification.
3. Before import split:
   `python -m pytest tests/unit/services/test_research_state_commit_boundary.py -q -k import_preparation`
   **RED: 1 failed** because the explicit preparation operation did not exist. The final test confirms prepared imported-document state is returned without any workflow save.
4. Chart failure regression:
   `python -m pytest tests/unit/services/test_research_state_commit_boundary.py -q --tb=short`
   **RED: 1 failed, 12 passed** because `RuntimeError('chart failed')` escaped instead of returning the deliberate attempt delta. Final verification covers the returned failure and unchanged input context.

Additional behavior checks exercise QA application load/commit success, no-commit execution and failure; runtime one-save behavior for succeeded/failed/skipped deltas; compression input isolation; failed/successful Zotero result deferral; and a specialist import's no-write-before-commit and exactly-one task/memory commit. Existing conversation, paper import, figure, QA route and graph-runtime tests provide direct public-entrypoint regression coverage.

## Verification

- Initial affected sweep: `python -m pytest tests/unit/services/test_literature_research_service.py tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_research_supervisor_graph_runtime.py tests/unit/agents -q --tb=short` — **187 passed**.
- Import sweep: `python -m pytest tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_literature_research_service.py tests/unit/services/test_research_supervisor_graph_runtime.py -q --tb=short` — **107 passed**.
- Broad affected sweep: `python -m pytest tests/unit/services tests/unit/agents tests/unit/reasoning tests/unit/tools tests/unit/domain -q --tb=short` — **381 passed** in 18.48 seconds.
- After final import/signature/documentation cleanup: `python -m pytest tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_research_architecture_constraints.py tests/unit/services/test_literature_research_service.py tests/unit/agents/test_research_qa_agent.py tests/unit/reasoning tests/unit/domain/test_reasoning_defaults.py -q --tb=short` — **123 passed** in 7.77 seconds.
- Final boundary/architecture check after annotation and indentation cleanup: result appended below.
- `git diff --check` — clean before report creation.

The test runs emit two pre-existing dependency deprecation warnings (hyppo/scipy sparse import and numba `nopython=False`). No live model calls or user data modifications were performed. The controller owns the final full-suite integration run, per dispatch instructions.

## Files

- Application composition/lifecycle: `services/research/literature_research_service.py`, `conversation_service.py` (new), `paper_operations.py`, `qa_router.py`, `ARCHITECTURE.md`; compatibility shim `memory/conversation_manager.py`.
- QA execution: `agents/research_qa_agent.py`, `agents/rag_react_qa_worker.py` (new), `tools/research/qa_tools.py`, `apps/api/runtime.py`.
- Specialist state: `agents/general_answer_agent.py`, `research_document_agent.py`, `chart_analysis_agent.py`, `research_knowledge_agent.py`, `paper_analysis_agent.py`.
- Commit/workspace boundary: `runtime/research/agent_protocol/base.py`, `mixins.py`, `runtime/research/supervisor_graph_runtime_core.py`.
- Tests: `tests/unit/services/test_research_state_commit_boundary.py` (new), `test_research_architecture_constraints.py`.

## Decisions, self-review and limits

- The commit boundary covers shared workflow context and task/report/paper/memory results, not rollback of external effects. Downloads, document/vector ingestion, Zotero writes, progress events, exported figures and derived paper-figure caches remain explicit application/tool effects. The controller explicitly confirmed this distinction.
- Figure discovery can save its derived cache in the paper store before a specialist completes. Chart deltas do not save a stale papers list, so their subsequent workspace commit does not overwrite that cache. No cache transaction framework was introduced.
- Existing best-effort deferred memory-operation error handling remains unchanged. Unexpected exceptions without an explicit result delta do not invent partial workflow state.
- The old large files were split along requested ownership boundaries; conversation/paper services and loaded QA execution remain substantial. No unrelated framework or repository-wide rewrite was introduced.
- Self-review corrected the QA recovery import after the execution split, restored public facade signatures, added concrete dependency types, removed worker-only imports from task QA, and migrated source-ownership assertions to the new authoritative modules. A specialist-import test initially selected an already-ingested paper (which selection intentionally skips); its fixture now uses an importable paper and a local ingestion stub.

Final annotation/indentation follow-up command:
`python -m pytest tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_research_architecture_constraints.py -q --tb=short` — **31 passed**, the same two dependency warnings, in 7.20 seconds.

## Controller lint follow-up

Removed unused local delta imports from chart/knowledge dispatch and unused schema imports from the conversation service. Moved worker compatibility imports to the normal top-level import section with explicit re-export aliases (no blanket suppression).

- Ruff: `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m ruff check` over every Python path returned by `git diff --name-only 5f42254 -- '*.py'` — **All checks passed**.
- Focused regression: `python -m pytest tests/unit/agents/test_research_qa_agent.py tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_research_architecture_constraints.py tests/unit/reasoning/test_react_external_tools.py -q --tb=short` — **37 passed**, two existing dependency warnings, 7.29 seconds.
- `git diff --check` — clean.
