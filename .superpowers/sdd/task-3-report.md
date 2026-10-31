# Task 3 implementation report

## Changes

- Added one action-to-agent catalog and used it for Supervisor routing, specialist descriptor task support, and registry action resolution.
- Replaced metadata-only skeleton adapters with bound specialist executors that call the existing `run_action` method and return its `ResearchToolResult` directly. The graph action node now dispatches through the registry.
- Cached the registry and blueprint per graph-runtime identity. Request contexts and blueprint projections remain independent, and each graph runtime retains its own available-tool descriptors.
- Routed synchronous no-LLM decisions through the existing asynchronous deterministic fallback builder. Removed the unused action `ToolExecutor` and stale execution-handler hook; corrected response execution metadata and trace count.
- Replaced affected source-string architecture assertions with routing behavior and AST call-dependency checks. Updated the architecture description.

## TDD evidence

RED: `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q tests/unit/services/test_unified_runtime.py::test_registry_support_matches_action_catalog tests/unit/services/test_unified_runtime.py::test_registry_executes_bound_specialist_without_converting_result tests/unit/agents/test_research_supervisor_agent.py::test_research_supervisor_sync_fallback_searches_discovery_only tests/unit/agents/test_research_supervisor_agent.py::test_research_supervisor_sync_fallback_stops_general_chat_provider_failure tests/unit/agents/test_research_supervisor_agent.py::test_research_supervisor_sync_fallback_finishes_completed_discovery` produced 4 expected failures and 1 pass: catalog missing, executable adapter missing, sync discovery finalized, and sync general-chat stop reason differed. A separate RED run of the graph-runtime isolation test failed because request blueprints shared one mutable dictionary.

GREEN: `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q tests/unit/services/test_unified_runtime.py tests/unit/agents/test_research_supervisor_agent.py tests/unit/services/test_research_supervisor_graph_runtime.py tests/unit/services/test_research_architecture_constraints.py --tb=short` produced **73 passed, 2 warnings**. Warnings came from installed `hyppo`/`numba` dependencies. `compileall -q` for changed Python modules and `git diff --check` both exited 0.

## Self-review and differences

The public `build_phase1_*` entrypoint names remain as compatibility adapters; their registry results are now executable only for supplied specialist delegates. The orchestration envelope stays `AgentMessage`/`AgentResultMessage`; registry execution does not convert through `UnifiedAgentTask` or `UnifiedAgentResult`. The blueprint still includes the Supervisor descriptor, while the executable registry contains bound specialists. Full-suite verification is reserved for controller integration as directed.

## Concerns

None identified in focused validation. Registry caches retain graph-runtime references for the lifetime of the supervisor runtime; production normally supplies one graph runtime, and this preserves descriptors for multiple runtimes when present.
