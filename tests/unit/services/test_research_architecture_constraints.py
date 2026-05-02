from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_direct_discovery_helper_removed_from_research_service_and_paper_operations() -> None:
    literature_service = _read("services/research/literature_research_service.py")
    paper_operations = _read("services/research/paper_operations.py")

    assert "_run_direct_discovery" not in literature_service
    assert "_run_direct_discovery" not in paper_operations
    assert "research_discovery_capability.discover(" not in literature_service
    assert "research_discovery_capability.discover(" not in paper_operations

    scout_agent = _read("agents/literature_scout_agent.py")
    assert "research_discovery_capability" not in scout_agent


def test_high_level_research_modules_do_not_call_low_level_rag_handlers_directly() -> None:
    qa_router = _read("services/research/qa_router.py")
    paper_operations = _read("services/research/paper_operations.py")

    banned_markers = (
        "graph_runtime.handle_ask_document",
        "graph_runtime.handle_ask_fused",
        "graph_runtime.handle_parse_document",
        "graph_runtime.handle_index_document",
        "graph_runtime.handle_graph_backfill_document",
        "graph_runtime.handle_understand_chart",
    )
    for marker in banned_markers:
        assert marker not in qa_router
        assert marker not in paper_operations


def test_legacy_unified_bridge_methods_are_removed() -> None:
    runtime_core = _read("runtime/research/supervisor_graph_runtime_core.py")

    assert "_legacy_unified_delegates" not in runtime_core
    assert "_legacy_unified_execution_handlers" not in runtime_core


def test_supervisor_authorized_qa_requires_explicit_route() -> None:
    qa_agent = _read("agents/research_qa_agent.py")
    supervisor_agent = _read("agents/research_supervisor_agent.py")

    assert "Supervisor-authorized research QA must include preferred_qa_route" in qa_agent
    assert "_normalize_supervisor_route_payload" in supervisor_agent
    assert "routing_authority\"] = \"supervisor_llm\"" in supervisor_agent


def test_answer_question_is_owned_by_research_qa_agent_in_unified_runtime() -> None:
    supervisor_agent = _read("agents/research_supervisor_agent.py")
    runtime_core = _read("runtime/research/supervisor_graph_runtime_core.py")
    unified_runtime = _read("runtime/research/unified_runtime.py")

    assert '"answer_question": "ResearchQAAgent"' in supervisor_agent
    assert '"understand_document": "ResearchDocumentAgent"' in supervisor_agent
    assert 'self._action_descriptor("answer_question", "ResearchQAAgent"' in supervisor_agent
    assert 'self._action_descriptor("understand_document", "ResearchDocumentAgent"' in supervisor_agent
    assert '"ResearchQAAgent": self.research_qa_agent' in runtime_core
    assert '"ResearchDocumentAgent": self.research_document_agent' in runtime_core
    assert '_execute_agent_run_action' in runtime_core
    assert 'name="ResearchQAAgent"' in unified_runtime
    assert 'name="ResearchDocumentAgent"' in unified_runtime
    assert 'supported_task_types=["answer_question"]' in unified_runtime
    assert 'supported_task_types=["understand_document"]' in unified_runtime
    assert 'supported_task_types=["import_papers", "sync_to_zotero", "compress_context"]' in unified_runtime


def test_service_layer_no_longer_contains_qa_execution_subchain() -> None:
    qa_router = _read("services/research/qa_router.py")
    qa_tools = _read("tools/research/qa_tools.py")

    assert "_run_scoped_task_qa" not in qa_router
    assert "_run_direct_collection_qa" not in qa_router
    assert "_maybe_recover_qa_route" not in qa_router
    assert "ResearchQAAgent().execute_qa" in qa_router
    assert "_select_recovery_qa_route" not in qa_router
    assert "_build_answer_quality_check" not in qa_router
    assert "_rewrite_collection_question" not in qa_router
    assert "research_qa_runtime" not in qa_tools
    assert "research_knowledge_agent.plan_collection_queries" not in qa_tools
    assert "research_writer_agent.answer_collection_question" not in qa_tools
    assert "ResearchCollectionQACapability" in qa_tools

    qa_decisions = _read("tools/research/qa_decisions.py")
    assert "def select_recovery_qa_route(" in qa_decisions
    assert "def build_answer_quality_check(" in qa_decisions
    assert "def rewrite_collection_question(" in qa_decisions


def test_rag_layer_uses_rag_qa_worker_name_not_research_qa_agent_alias() -> None:
    api_runtime = _read("apps/api/runtime.py")
    rag_runtime = _read("rag_runtime/runtime.py")
    knowledge_access = _read("tools/research/knowledge_access.py")

    assert "graph_runtime.rag_qa_worker = reasoning_strategies.react_reasoning_agent" in api_runtime
    assert "self.rag_qa_worker = self.react_reasoning_agent" in rag_runtime
    assert "graph_runtime.research_qa_agent" not in api_runtime
    assert "self.research_qa_agent = self.react_reasoning_agent" not in rag_runtime
    assert 'getattr(self.graph_runtime, "rag_qa_worker", None)' in knowledge_access
    assert "RagReActQAWorker as ReActReasoningAgent" in api_runtime


def test_services_use_external_tool_gateway_instead_of_raw_registry_calls() -> None:
    paper_search_service = _read("tools/research/paper_search.py")
    function_service = _read("tools/research/research_functions.py")
    react_qa_worker = _read("agents/research_qa_agent.py")

    assert "ResearchExternalToolGateway" in paper_search_service
    assert "self.external_tool_gateway.call_tool(" in paper_search_service
    assert "self.external_tool_registry.call_tool(" not in paper_search_service
    assert "ResearchExternalToolGateway" in function_service
    assert "self.external_tool_gateway.call_tool(" in function_service
    assert "registry.call_tool(" not in function_service
    assert "ResearchExternalToolGateway" in react_qa_worker
    assert "self.external_tool_gateway.call_tool(" in react_qa_worker
    assert "mcp_client_registry.call_tool(" not in react_qa_worker


def test_specialists_use_direct_agent_run_action_dispatch() -> None:
    runtime_core = _read("runtime/research/supervisor_graph_runtime_core.py")
    unified_runtime = _read("runtime/research/unified_runtime.py")

    assert "phase1_wrapped_action_tool" not in runtime_core
    assert "_execute_agent_run_action" in runtime_core
    assert "_TASK_TYPE_AGENTS" in runtime_core
    assert "agent.run_action(context, decision" in runtime_core
    assert "unresolved_boundaries=[]" in unified_runtime
    assert "legacy_boundaries=[]" in unified_runtime


def test_workspace_persistence_uses_standalone_functions() -> None:
    mixins = _read("runtime/research/agent_protocol/mixins.py")
    knowledge_agent = _read("agents/research_knowledge_agent.py")
    paper_analysis_agent = _read("agents/paper_analysis_agent.py")

    assert "class _WorkspacePersistenceMixin" not in mixins
    assert "def persist_workspace_results(" in mixins
    assert "def comparison_scope_papers(" in mixins
    assert "_WorkspacePersistenceMixin" not in knowledge_agent
    assert "_WorkspacePersistenceMixin" not in paper_analysis_agent
    assert "persist_workspace_results(context" in knowledge_agent
    assert "persist_workspace_results(" in paper_analysis_agent
    assert "comparison_scope_papers(" in paper_analysis_agent


def test_capabilities_live_in_tools_not_services() -> None:
    tools_init = _read("tools/research/__init__.py")

    assert "PaperRanker" in tools_init
    assert "SurveyWriter" in tools_init
    assert "ResearchIntentResolver" in tools_init
    assert "PaperAnalyzer" in tools_init

    assert not (PROJECT_ROOT / "services" / "research" / "capabilities").is_dir(), \
        "Old capabilities/ directory should be deleted"
    assert not (PROJECT_ROOT / "services" / "research" / "qa").is_dir(), \
        "Old qa/ directory should be deleted"
    assert not (PROJECT_ROOT / "services" / "research" / "supervisor_tools").is_dir(), \
        "Old supervisor_tools/ directory should be deleted"


def test_intent_heuristics_are_module_level_not_instance_methods() -> None:
    intent_classifier = _read("runtime/research/intent_classifier.py")

    assert "def _looks_like_general_chat(normalized_message" in intent_classifier
    assert "def _looks_like_new_discovery(normalized_message" in intent_classifier
    assert "def _should_inherit_snapshot_scope(" in intent_classifier
    assert "def _route_mode_hint_for_request(" in intent_classifier

    runtime_core = _read("runtime/research/supervisor_graph_runtime_core.py")
    assert "self._looks_like_" not in runtime_core
    assert "self._should_inherit_snapshot_scope(" not in runtime_core
    assert "self._route_mode_hint_for_request(" not in runtime_core
    assert "self._normalize_topic_text(" not in runtime_core
