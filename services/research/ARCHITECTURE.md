# Research Supervisor Architecture

This document describes the current high-level research agent architecture.
It intentionally uses the project modules that exist in this repository.

## Main Research Flow

```text
apps/api/routers/research.py
  -> LiteratureResearchService.run_agent
  -> ResearchAgentContextBuilder
  -> ResearchSkillResolver
  -> ResearchCapabilityRegistry (inventory metadata)
  -> ResearchSupervisorGraphRuntime
  -> ResearchSupervisorAgent
  -> specialist selected from UnifiedAgentRegistry
  -> specialist.run_action(context, decision)
  -> specialist capabilities / ResearchKnowledgeAccess / ResearchMemoryGateway / ResearchExternalToolGateway
  -> RagRuntime / RetrievalTools / AnswerTools
  -> ResearchAgentGraphState
  -> ResearchAgentResultAggregator
  -> ResearchAgentRunResponse
```

## Layer Responsibilities

`LiteratureResearchService` is the application service behind the API. It
loads conversations and workspaces, builds research execution context through
the context manager, updates memory through `ResearchMemoryGateway`, starts the
supervisor runtime, and persists the final workspace state. Search-oriented
entry points such as `/research/papers/search` and TODO follow-up search now
enter the same supervisor graph via a `workflow_constraint=discovery_only`
request flag instead of bypassing the graph.

The facade composes `ConversationService`, `PaperOperationsService` and
`QAApplicationService` through explicit resources and named callback ports.
Conversation lifecycle and snapshot recording live in the application layer;
`memory.conversation_manager` retains only a language-helper import for compatibility.
QA application loading and persistence surround `ResearchQAAgent.execute_loaded_qa`;
`agents.rag_react_qa_worker` separately owns the RAG ReAct loop.

Specialists return typed `ResearchStateDelta` values. The runtime applies and
persists each returned delta once, including deliberate partial progress on a
failed action (for example, a parsed document whose indexing failed). Workspace
calculation with `persist=False` returns a replacement execution context without
changing shared input. The direct QA/import/workspace application entrypoints
still commit their own results. Downloads, document/vector ingestion, progress
events, Zotero effects and derived figure-cache writes remain explicit tool or
application effects; this boundary is not an external-effect transaction.

`ResearchDiscoveryCapability` is the shared discovery engine used under
`LiteratureDiscoveryCapability`. Topic planning, provider search, paper
curation, report drafting, and TODO generation are centralized now that the old
direct-discovery helpers are gone.

`ResearchAgentContextBuilder` prepares the per-request
`ResearchAgentToolContext`. It keeps conversation hydration, execution context,
unified agent registry, blueprint, and active skills
in one construction path.

`ResearchSkillResolver` is the Skill layer boundary. It reuses
`core.SkillRegistry`, `core.SkillMatcher`, and `core.SkillValidator`. Skills
provide workflow instructions, output rules, and validation criteria. Skills do
not call tools, retrieve documents, write memory, or replace agents.

`ResearchSupervisorGraphRuntime` is the LangGraph orchestration runtime. It
owns the graph nodes, state transitions, loop guards, and routing between the
supervisor decision node and specialist nodes.

`ResearchSupervisorAgent` is the manager decision agent. It selects the next
action and worker agent from the current `ResearchSupervisorState`, active skill
instructions, pending agent messages, and recent agent results.
For research QA, it is also the route authority: `answer_question` messages
must carry `payload.qa_route` (`collection_qa`, `document_drilldown`, or
`chart_drilldown`) plus `routing_authority=supervisor_llm`. The
`ResearchQAAgent` specialist executes that route through the QA toolset and may
return recovery observations, but lower layers do not silently pick a different
route for supervisor-authorized calls.

The action node calls `ResearchSupervisorGraphRuntime._execute_agent_run_action`.
It resolves the selected worker from `UnifiedAgentRegistry`, which binds each
descriptor to a specialist's `run_action(context, decision)`. `ResearchKnowledgeAgent` and
`ChartAnalysisAgent` also receive `task_type=action_name`. The runtime
standardizes the returned observation and execution metadata, applies any
state delta, and records a missing-agent failure if no registry entry is found.
Recoverable QA route observations return to the supervisor for an explicit
next decision.

`ResearchCapabilityRegistry` is the read-only inventory boundary over
supervisor actions, runtime/knowledge tools, and MCP server availability.
It supports skill matching and response metadata. The `UnifiedAgentRegistry`
holds the executable specialist bindings and descriptors for the supplied
graph runtime.

`ResearchKnowledgeAccess` is the Knowledge/RAG access boundary used by the
supervisor chain. It prefers registered runtime tools such as
`hybrid_retrieve`, `query_graph_summary`, `answer_with_evidence`,
`parse_document`, `index_document`, and `understand_chart`, while keeping
fallbacks to the existing `RagRuntime` methods for compatibility.

Specialist agents keep their domain responsibilities:

- `LiteratureScoutAgent`: paper discovery and search planning through
  `LiteratureDiscoveryCapability`.
- `ResearchKnowledgeAgent`: paper ingestion, Zotero sync, retrieval-support,
  and context compression through `KnowledgeOpsCapability`.
- `ResearchQAAgent`: task-level QA specialist for collection, document, and
  chart drilldown routes selected by the supervisor.
- `ResearchWriterAgent`: review/report synthesis through
  `ReviewWritingCapability`.
- `PaperAnalysisAgent`: paper structure and comparison analysis through
  `PaperAnalysisCapability`.
- `ChartAnalysisAgent`: chart and paper figure analysis through
  `ChartAnalysisCapability`.
- `GeneralAnswerAgent`: general non-research answer branch through
  `GeneralAnswerCapability`.
- `PreferenceMemoryAgent`: long-term preference observation and recommendations
  through `PreferenceRecommendationCapability` and `ResearchMemoryGateway`.

`RagRuntime`, `RetrievalTools`, `HybridRetriever`, `AnswerTools`, and
`AnswerChain` remain the underlying Knowledge/RAG implementation. They are now
internal capability providers behind `ResearchKnowledgeAccess`, not a separate
public API surface.
The RAG-layer ReAct QA worker is named `RagReActQAWorker` and is exposed as
`RagRuntime.rag_qa_worker`; it is not the task-level `ResearchQAAgent`.

`ResearchMemoryGateway` is the research-domain memory boundary. It wraps
`MemoryManager` and `GraphSessionMemory` operations used by the research
runtime, so execution-context hydration, active-paper updates, session context
saves, and research-turn persistence no longer write memory through scattered
call sites. `MemoryManager` and `GraphSessionMemory` remain the underlying
stores. Context is still the per-call visible information passed to agents.

`ResearchExternalToolGateway` is the research-domain MCP/external-tool
boundary. Research services and the RAG-layer ReAct worker call it instead of
reaching into raw MCP registries.

`ResearchAgentResultAggregator` is the final response boundary. It builds
messages, warnings, workspace metadata, delegation traces, active skill
metadata, and the `ResearchAgentRunResponse` while preserving the old response
shape.
The graph carries `AgentMessage` and `AgentResultMessage` internally. A shared
projection serializes the legacy `agent_messages`, `agent_results`, and
`delegation_plan` fields alongside the `unified_*` fields once per response;
the decision-trace notice reuses those same values. `UnifiedAgentTask` and
`UnifiedAgentResult` remain DTO adapters for callers of their public Python
interfaces, not execution envelopes.

## Compatibility

### Explicit research plans

Complex requests may return a structured plan. `domain.schemas.research_plan`
defines its steps and runtime state; `runtime.research.plan_executor` validates
dependencies and performs pure, serial transitions. `ResearchPlanCoordinator`
adapts those transitions to the supervisor's existing worker dispatch boundary.
Simple requests retain single-action routing.

The graph explicitly carries `execution_plan`, `planning_llm_calls`, and
`supervisor_decision_count`. Every dispatch binds a stable plan/step identity to
a fresh worker task ID. Results complete only the matching running task. First
and subsequent steps use the same action/instruction/parameter path; unknown
paper references stop dispatch rather than expanding scope to the collection.
Model-proposed runtime statuses, task IDs and attempt counts are reset.

Plans reject duplicate IDs, missing/cyclic dependencies, unsupported actions,
invalid supported parameters and discovery-only violations. Acceptance reuses
`ResearchEvaluationTool`: plan requirements are checked even when a worker
already returned a passing evaluation. Search needs candidates, analysis needs
its paper metadata and requested paper coverage, import needs progress, and QA
needs evidence unless its scope is explicitly metadata-only. Additional
supported requirements can be proposed per step. A skipped step only satisfies
dependencies when the plan permits skipping and the result passes acceptance.
These are structural/evidence checks, not a guarantee of factual correctness.

On failed acceptance, the planner receives completed work and failure evidence
and proposes remaining steps. The executor preserves completed steps and
archives replaced steps/results; exact repeated completed action/params are
rejected. The supervisor allows two replanning invocations by default
(`max_plan_replans`), and the existing runtime `max_steps` budget still applies.
Retries are explicit replacement-plan decisions. No tool side effect is
automatically retried or rolled back by the executor.

The finalizer reconciles the last real worker result, including a result at the
step limit. It preserves partial progress and reports incomplete plans with a
reason rather than synthesizing another successful worker result.
`research_context.current_task_plan` is a projection of the authoritative plan;
its stable display IDs are `plan_id:step_id`, and execution task IDs are retained
in metadata. Without a plan the legacy message-based projection is retained.

Final response metadata includes the plan and separate
`planning_llm_call_count`, `supervisor_decision_count`, `plan_replan_count`, and
`worker_call_count`. A planning call counts an adapter invocation (including a
failed invocation), not each provider retry inside the adapter. Compression
guardrails do not increment the plan replan counter. Legacy counters
remain available. Plan execution state is per run; this change does not add
durable graph checkpoint/resume support or external-effect transactions.

The old runtime methods remain available and delegate to the explicit layers:

- `_build_tool_context` delegates to `ResearchAgentContextBuilder`.
- The action-tool registry remains as a compatibility surface. The action node
  dispatches through `UnifiedAgentRegistry` to specialist `run_action` methods.
- `_build_response` delegates to `ResearchAgentResultAggregator`.

The public API now exposes only the high-level research surface plus health and
MCP endpoints. Low-level document/chart/ask routes were intentionally removed
and absorbed into the supervisor-driven knowledge tool layer.

The remaining research-domain search endpoints (`/research/papers/search`,
`/research/tasks/{task_id}/todos/{todo_id}/search`) no longer bypass the
supervisor runtime. They construct discovery-only `ResearchAgentRunRequest`
objects, go through `ResearchSupervisorGraphRuntime -> ResearchSupervisorAgent
-> LiteratureScoutAgent -> LiteratureDiscoveryCapability`, and then adapt the
supervisor response back to their legacy response models.

## Verification and limits

Use `python -m pytest -q -p no:cacheprovider` for offline regression checks,
`python -m ruff check .` for Python lint, and
`npm run build` in `web/` for TypeScript and the web build. The installed
wheel import smoke is described in the Task 5 report under `.superpowers/sdd/`.
The runtime state commit boundary does not roll back downloads, indexing,
progress events, Zotero writes, or derived figure-cache writes. The public
response keeps historical `unified_*` names even though execution uses the
internal message classes and direct specialist calls.
