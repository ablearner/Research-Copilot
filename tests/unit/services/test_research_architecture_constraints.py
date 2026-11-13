from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from agents.research_supervisor_agent import ResearchSupervisorAgent
from runtime.research.unified_runtime import ACTION_AGENT_CATALOG, build_phase1_unified_blueprint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_workers_and_research_tools_do_not_import_peer_agents() -> None:
    worker_modules = {
        "agents." + name.removesuffix(".py")
        for name in (
            "literature_scout_agent.py", "research_writer_agent.py", "research_analyst_agent.py",
            "research_knowledge_agent.py",
        )
    }
    files = [PROJECT_ROOT / (name.replace(".", "/") + ".py") for name in worker_modules]
    files.extend((PROJECT_ROOT / "tools" / "research").rglob("*.py"))
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            assert not any(name == "agents" or name.startswith("agents.") for name in imported), (
                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} imports an Agent; "
                "shared behavior belongs in tools and dispatch belongs in Runtime"
            )


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
    qa_agent = _read("agents/research_analyst_agent.py")
    supervisor_agent = _read("agents/research_supervisor_agent.py")

    assert "Supervisor-authorized research QA must include preferred_qa_route" in qa_agent
    assert "_normalize_supervisor_route_payload" in supervisor_agent
    assert "routing_authority\"] = \"supervisor_llm\"" in supervisor_agent


def test_answer_question_is_owned_by_research_qa_agent_in_unified_runtime() -> None:
    manager = ResearchSupervisorAgent()
    blueprint = build_phase1_unified_blueprint(graph_runtime=SimpleNamespace(), research_service=SimpleNamespace())
    descriptors = {item.name: item for item in blueprint.agent_descriptors}

    for action_name, expected_agent in ACTION_AGENT_CATALOG.items():
        assert manager._worker_for_action(action_name, "WrongAgent") == expected_agent
        assert action_name in descriptors[expected_agent].supported_task_types


def test_service_layer_no_longer_contains_qa_execution_subchain() -> None:
    qa_router = _read("services/research/qa_router.py")
    qa_tools = _read("tools/research/qa_tools.py")

    assert "_run_scoped_task_qa" not in qa_router
    assert "_run_direct_collection_qa" not in qa_router
    assert "_maybe_recover_qa_route" not in qa_router
    assert ").execute_loaded_qa(" in qa_router
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
    assert "rag_qa_worker" not in knowledge_access
    assert "RagReActQAWorker as ReActReasoningAgent" in api_runtime


def test_services_use_external_tool_gateway_instead_of_raw_registry_calls() -> None:
    paper_search_service = _read("tools/research/paper_search.py")
    function_service = _read("tools/research/research_functions.py")
    react_qa_worker = _read("agents/rag_react_qa_worker.py")

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
    tree = ast.parse(_read("runtime/research/supervisor_graph_runtime_core.py"))
    execute = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_agent_run_action")
    calls = [node.func for node in ast.walk(execute) if isinstance(node, ast.Call)]
    assert any(isinstance(call, ast.Attribute) and call.attr == "resolve_for_action" for call in calls)
    assert any(isinstance(call, ast.Attribute) and call.attr == "execute_action" for call in calls)
    assert not any(isinstance(call, ast.Attribute) and call.attr == "run_action" for call in calls)


def test_workspace_persistence_uses_standalone_functions() -> None:
    mixins = _read("runtime/research/agent_protocol/mixins.py")
    knowledge_agent = _read("agents/research_knowledge_agent.py")
    paper_analysis_agent = _read("tools/research/workflows/paper_analysis.py")

    assert "class _WorkspacePersistenceMixin" not in mixins
    assert "def persist_workspace_results(" in mixins
    assert "def comparison_scope_papers(" in mixins
    assert "_WorkspacePersistenceMixin" not in knowledge_agent
    assert "_WorkspacePersistenceMixin" not in paper_analysis_agent
    assert "persist_workspace_results(" in knowledge_agent
    assert "persist_workspace_results(" in paper_analysis_agent
    assert "comparison_scope_papers(" in paper_analysis_agent


def test_capabilities_live_in_tools_not_services() -> None:
    tools_init = _read("tools/research/__init__.py")

    assert "PaperRankingTool" in tools_init
    assert "SurveyWritingTool" in tools_init
    assert "IntentResolutionTool" in tools_init
    assert "PaperAnalysisTool" in tools_init

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


def test_service_composes_explicit_application_collaborators():
    from services.research.literature_research_service import LiteratureResearchService
    assert LiteratureResearchService.__bases__ == (object,)
    for path, cls in [
        ('services/research/conversation_service.py', 'ConversationService'),
        ('services/research/paper_operations.py', 'PaperOperationsService'),
        ('services/research/qa_router.py', 'QAApplicationService'),
    ]:
        tree = ast.parse(_read(path))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        init = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        names = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
        assert not names & {'host', 'service', 'research_service'}
        assert '__getattr__' not in _read(path)
        assert '__dict__' not in _read(path)
    assert 'class ConversationMixin' not in _read('memory/conversation_manager.py')


def test_loaded_qa_execution_has_no_storage_or_commit_and_worker_is_separate():
    source = _read('agents/research_analyst_agent.py')
    assert 'class RagReActQAWorker' not in source
    assert 'class RagReActQAWorker' in _read('agents/rag_react_qa_worker.py')
    tree = ast.parse(source)
    execute = next((node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == 'execute_loaded_qa'), None)
    assert execute is not None
    calls = {node.func.attr for node in ast.walk(execute) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {'load_task', 'load_report', 'load_papers', 'build_execution_context', 'hydrate_context', 'load_user_profile', 'load_messages', 'load_conversation', 'save_task_state', 'save_report', 'persist_research_update', 'promote_conclusion_to_long_term'}
