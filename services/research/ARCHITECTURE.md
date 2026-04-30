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
  -> supervisor_tools / specialist agents / ResearchKnowledgeAccess / ResearchMemoryGateway
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

`ResearchDiscoveryCapability` is the shared discovery capability used by
the supervisor's `search_literature` action. It is where topic planning,
provider search, paper curation, report drafting, and TODO generation are
centralized now that the old direct-discovery helpers are gone.

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

`ResearchActionDispatcher` is the unified supervisor action execution boundary.
It routes action execution through `ToolExecutor`, normalizes supervisor tool
outputs, preserves the unified worker fallback, and standardizes observation
metadata for state updates.

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

- `LiteratureScoutAgent`: paper discovery and search planning.
- `ResearchKnowledgeAgent`: research QA, retrieval planning, and knowledge use.
- `ResearchWriterAgent`: review/report synthesis.
- `PaperAnalysisAgent`: paper structure and comparison analysis.
- `ChartAnalysisAgent`: chart and paper figure analysis.
- `GeneralAnswerAgent`: general non-research answer branch.
- `PreferenceMemoryAgent`: long-term preference observation and recommendations.
- `ResearchQAAgent`: ReAct-style QA worker exposed through
  `RagRuntime.research_qa_agent`.

`RagRuntime`, `RetrievalTools`, `HybridRetriever`, `AnswerTools`, and
`AnswerChain` remain the underlying Knowledge/RAG implementation. They are now
internal capability providers behind `ResearchKnowledgeAccess`, not a separate
public API surface.

`ResearchMemoryGateway` is the research-domain memory boundary. It wraps
`MemoryManager` and `GraphSessionMemory` operations used by the research
runtime, so execution-context hydration, active-paper updates, session context
saves, and research-turn persistence no longer write memory through scattered
call sites. `MemoryManager` and `GraphSessionMemory` remain the underlying
stores. Context is still the per-call visible information passed to agents.

`ResearchAgentResultAggregator` is the final response boundary. It builds
messages, warnings, workspace metadata, delegation traces, active skill
metadata, and the `ResearchAgentRunResponse` while preserving the old response
shape.

## Compatibility

The old runtime methods remain available and delegate to the explicit layers:

- `_build_tool_context` delegates to `ResearchAgentContextBuilder`.
- `_execute_action_tool` and unified worker helpers delegate to
  `ResearchActionDispatcher`.
- `_build_response` delegates to `ResearchAgentResultAggregator`.

The public API now exposes only the high-level research surface plus health and
MCP endpoints. Low-level document/chart/ask routes were intentionally removed
and absorbed into the supervisor-driven knowledge tool layer.

The remaining research-domain search endpoints (`/research/papers/search`,
`/research/tasks/{task_id}/todos/{todo_id}/search`) no longer bypass the
supervisor runtime. They construct discovery-only `ResearchAgentRunRequest`
objects, go through `ResearchSupervisorGraphRuntime -> ResearchSupervisorAgent
-> SearchLiteratureTool`, and then adapt the unified result back to their
legacy response models.
