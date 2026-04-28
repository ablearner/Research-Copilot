# Supervisor Stability Improvement Plan

## Goals

Preserve the current core architecture:

`user -> ResearchSupervisorGraphRuntime -> ResearchSupervisorAgent -> specialist agent -> tools/runtime`

while improving:

- stability
- robustness
- autonomy
- flexibility

without falling back to hardcoded pipelines or prompt-only routing.

## Problems In The Previous Design

The previous supervisor-first architecture already unified the main execution path, but it still had four structural weaknesses:

1. Supervisor state was too thin.
   It knew task/workspace status, but not enough about topic continuity, thread switching, or when to ignore inherited research scope.

2. Conversation state was too flat.
   A single conversation snapshot stored active papers and selected papers, but there was no explicit notion of thread or route mode. This made cross-topic contamination more likely.

3. Request hydration inherited too much.
   When a new user turn arrived inside an existing conversation, the runtime could inherit old `selected_paper_ids` / `active_paper_ids` too eagerly.

4. User intent was still partially underpowered.
   The supervisor runtime still used sync intent resolution in its state-building path, which weakened semantic routing quality compared with the async LLM-aware resolver.

## Design Principles

The improvement strategy follows five principles:

1. Make routing state explicit.
2. Isolate topic threads inside one conversation.
3. Let the supervisor see route hints, not just raw text.
4. Avoid inheriting research scope unless the current turn actually needs it.
5. Prefer structured visibility and state shaping over prompt patching.

## Target Model

The improved system keeps one supervisor, but feeds it richer state:

`user turn`
-> `conversation snapshot + thread state + route hint + async user intent`
-> `ResearchSupervisorState`
-> `action visibility + LLM decision`
-> `specialist execution`
-> `snapshot/thread update`

## Implemented Improvements

### 1. Route Mode At The Conversation Layer

Added explicit route-mode state:

- `general_chat`
- `research_discovery`
- `research_follow_up`
- `paper_follow_up`
- `document_drilldown`
- `chart_drilldown`

This gives the supervisor and the runtime a stable control signal for whether the current turn should continue the current research lane or not.

### 2. Lightweight Threading Inside A Conversation

Added thread-level snapshot state:

- `active_thread_id`
- `thread_history`
- per-thread topic / route / paper focus / last message

This makes topic continuity explicit and creates a place to track thread switches without splitting the whole product model into separate conversations.

### 3. Route-Aware Snapshot Updates

Conversation persistence now updates:

- active route mode
- active thread id
- thread history

and treats `general_chat` as a non-research turn, so ordinary chat does not keep polluting the active paper scope.

### 4. Route-Aware Request Hydration

Supervisor request hydration no longer blindly inherits old paper scope.

It now checks whether the new turn:

- is likely general chat
- looks like a fresh discovery request
- explicitly targets existing papers/documents

Only follow-up-like turns inherit prior paper focus.

### 5. Async Intent Resolution In Supervisor State Build

The supervisor runtime now uses the async user-intent resolver while building manager state, so the LLM-backed resolver participates in the main decision loop instead of only heuristics.

### 6. Richer Supervisor State

Supervisor state now includes:

- `route_mode`
- `active_thread_id`
- `active_thread_topic`
- `topic_continuity_score`
- `new_topic_detected`
- `should_ignore_research_context`

This improves the quality of single-step supervisor decisions without adding another manager layer.

### 7. Action Visibility With Priority Signals

Available supervisor actions are now exposed with:

- `priority_score`
- `visibility_reason`

so the LLM is guided by structured action visibility instead of only a flat action list.

### 8. Clarification As A Structured Stop Condition

When user intent explicitly requires clarification, the supervisor now stops with a clarification-oriented decision instead of trying to push through a low-confidence worker action.

## Why This Improves The System

### Stability

- General chat stops contaminating paper scope.
- New-topic discovery stops inheriting stale selected papers.
- The manager sees explicit route and thread state.

### Robustness

- Topic switches are treated as first-class state transitions.
- Clarification is handled earlier.
- Async intent parsing improves semantic resilience.

### Autonomy

- The supervisor still chooses the next action itself.
- We improved the state it reasons over instead of freezing more rules into the pipeline.

### Flexibility

- One conversation can contain multiple research threads.
- Discovery, follow-up, paper-level follow-up, and general chat can coexist more safely.

## Implemented Code Areas

The current implementation touches:

- `domain/schemas/research.py` — `ResearchRouteMode`, `ResearchRuntimeEventType`, `ResearchLifecycleStatus`, `ResearchContextSummary`
- `domain/schemas/research_context.py` — `ResearchContextSlice`, `QAPair`, `CompressedPaperSummary`
- `services/research/literature_research_service.py` — route-aware snapshot updates, thread management
- `services/research/research_supervisor_graph_runtime_core.py` — state building, `context_compression_needed` signal, specialist routing
- `services/research/research_context_manager.py` — context slicing, Hermes-style 3-phase compression
- `agents/research_supervisor_agent.py` — richer `ResearchSupervisorState` (91 fields), action visibility/priority, guardrail budget, `_truncate_context_slice`
- `agents/general_answer_agent.py` — general chat isolation
- `agents/preference_memory_agent.py` — preference-based recommendation
- `memory/user_profile_memory.py` — long-term interest profile for routing
- `services/research/unified_action_adapters.py` — standardized specialist I/O
- `tests/unit/services/test_literature_research_service.py`
- `tests/unit/services/test_compress_context_slice.py`

## Follow-Up Work Recommended

Completed since initial plan:

- ✅ Added explicit `clarify_request` as a first-class supervisor action (not just finalize).
- ✅ Standardized specialist observations: `progress_made`, `confidence`, `missing_inputs`, `suggested_next_actions` are now part of `ResearchSupervisorState` (`latest_result_*` fields).
- ✅ Added `GeneralAnswerAgent` for general-chat isolation.
- ✅ Added `PreferenceMemoryAgent` for preference-based routing.
- ✅ Added Hermes-style context compression as an additional stability mechanism.

Still recommended:

1. Store per-thread workspace summaries instead of only thread metadata.
2. Add more regression tests around:
   - cross-topic switching
   - paper follow-up after general chat
   - document/chart drilldown isolation
3. Add metrics for route corrections, thread switches, and clarification frequency.
4. Integrate `compress_context_slice` into the main `CompressContextTool` execution chain.
5. Add automated evaluation for supervisor decision quality (routing accuracy, action selection F1).

## Summary

This plan keeps the supervisor-specialist architecture intact, but makes it much more state-aware.

The main change is philosophical as much as technical:

we are no longer asking the supervisor to "guess better from text";
we are giving it a cleaner model of conversation state, topic continuity, and route context so it can make better autonomous decisions.
