# Research Supervisor Architecture

This document describes the current high-level research agent architecture.
It intentionally uses the project modules that exist in this repository.

## Main Research Flow

```text
apps/api/routers/research.py
  -> LiteratureResearchService.run_agent
  -> ResearchAgentContextBuilder
  -> ResearchSkillResolver
  -> ResearchCapabilityRegistry (inventory view)
  -> ResearchSupervisorGraphRuntime
  -> ResearchSupervisorAgent
  -> ResearchActionDispatcher
  -> specialist agents
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

`ResearchDiscoveryCapability` is the shared discovery engine used under
`LiteratureDiscoveryCapability`. Topic planning, provider search, paper
curation, report drafting, and TODO generation are centralized now that the old
direct-discovery helpers are gone.

`ResearchAgentContextBuilder` prepares the per-request
`ResearchAgentToolContext`. It keeps conversation hydration, execution context,
unified runtime context, unified agent registry, blueprint, and active skills
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

`ResearchActionDispatcher` is the unified delegation boundary. It builds the
`UnifiedAgentTask`, injects supervisor/runtime context, invokes the selected
specialist from the unified agent registry, and standardizes observation
metadata for state updates. The old action-tool dispatcher remains only as a
compatibility fallback when no unified specialist message is available.
When a worker reports a recoverable QA route problem, the dispatcher preserves
that observation so the next Supervisor decision can replan explicitly.

`ResearchCapabilityRegistry` is the read-only inventory boundary over
supervisor action tools, runtime/knowledge tools, and MCP server availability.
It is used both for skill matching and response metadata so the runtime no
longer has separate tool-counting logic in multiple places.

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

## Compatibility

The old runtime methods remain available and delegate to the explicit layers:

- `_build_tool_context` delegates to `ResearchAgentContextBuilder`.
- Unified worker helpers delegate to `ResearchActionDispatcher`.
- `_execute_action_tool` remains for compatibility fallback only.
- `_build_response` delegates to `ResearchAgentResultAggregator`.

The public API now exposes only the high-level research surface plus health and
MCP endpoints. Low-level document/chart/ask routes were intentionally removed
and absorbed into the supervisor-driven knowledge tool layer.

The remaining research-domain search endpoints (`/research/papers/search`,
`/research/tasks/{task_id}/todos/{todo_id}/search`) no longer bypass the
supervisor runtime. They construct discovery-only `ResearchAgentRunRequest`
objects, go through `ResearchSupervisorGraphRuntime -> ResearchSupervisorAgent
-> LiteratureScoutAgent -> LiteratureDiscoveryCapability`, and then adapt the
unified result back to their legacy response models.
