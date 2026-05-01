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
    runtime_core = _read("services/research/research_supervisor_graph_runtime_core.py")

    assert "_legacy_unified_delegates" not in runtime_core
    assert "_legacy_unified_execution_handlers" not in runtime_core


def test_supervisor_authorized_qa_requires_explicit_route() -> None:
    qa_executor = _read("services/research/qa/executor.py")
    supervisor_agent = _read("agents/research_supervisor_agent.py")

    assert "Supervisor-authorized research QA must include preferred_qa_route" in qa_executor
    assert "_normalize_supervisor_route_payload" in supervisor_agent
    assert "routing_authority\"] = \"supervisor_llm\"" in supervisor_agent


def test_answer_question_is_owned_by_research_qa_agent_in_unified_runtime() -> None:
    supervisor_agent = _read("agents/research_supervisor_agent.py")
    runtime_core = _read("services/research/research_supervisor_graph_runtime_core.py")
    unified_runtime = _read("services/research/unified_runtime.py")

    assert '"answer_question": "ResearchQAAgent"' in supervisor_agent
    assert '"understand_document": "ResearchDocumentAgent"' in supervisor_agent
    assert 'self._action_descriptor("answer_question", "ResearchQAAgent"' in supervisor_agent
    assert 'self._action_descriptor("understand_document", "ResearchDocumentAgent"' in supervisor_agent
    assert '"ResearchQAAgent": self.research_qa_agent' in runtime_core
    assert '"ResearchDocumentAgent": self.research_document_agent' in runtime_core
    assert '"ResearchQAAgent": self._build_research_qa_execution_handler()' in runtime_core
    assert '"ResearchDocumentAgent": self._build_research_document_execution_handler()' in runtime_core
    assert 'name="ResearchQAAgent"' in unified_runtime
    assert 'name="ResearchDocumentAgent"' in unified_runtime
    assert 'supported_task_types=["answer_question"]' in unified_runtime
    assert 'supported_task_types=["understand_document"]' in unified_runtime
    assert 'supported_task_types=["import_papers", "sync_to_zotero", "compress_context"]' in unified_runtime


def test_service_layer_no_longer_contains_qa_execution_subchain() -> None:
    qa_router = _read("services/research/qa_router.py")
    qa_tools = _read("services/research/qa/tools.py")

    assert "_run_scoped_task_qa" not in qa_router
    assert "_run_direct_collection_qa" not in qa_router
    assert "_maybe_recover_qa_route" not in qa_router
    assert "ResearchQAExecutor(self).execute" in qa_router
    assert "research_qa_runtime" not in qa_tools
    assert "research_knowledge_agent.plan_collection_queries" not in qa_tools
    assert "research_writer_agent.answer_collection_question" not in qa_tools
    assert "ResearchCollectionQACapability" in qa_tools


def test_rag_layer_uses_rag_qa_worker_name_not_research_qa_agent_alias() -> None:
    api_runtime = _read("apps/api/runtime.py")
    rag_runtime = _read("rag_runtime/runtime.py")
    knowledge_access = _read("services/research/research_knowledge_access.py")
    reasoning_init = _read("reasoning/__init__.py")

    assert "graph_runtime.rag_qa_worker = reasoning_strategies.react_reasoning_agent" in api_runtime
    assert "self.rag_qa_worker = self.react_reasoning_agent" in rag_runtime
    assert "graph_runtime.research_qa_agent" not in api_runtime
    assert "self.research_qa_agent = self.react_reasoning_agent" not in rag_runtime
    assert 'getattr(self.graph_runtime, "rag_qa_worker", None)' in knowledge_access
    assert "ReActReasoningAgent = RagReActQAWorker" in reasoning_init


def test_services_use_external_tool_gateway_instead_of_raw_registry_calls() -> None:
    paper_search_service = _read("services/research/paper_search_service.py")
    function_service = _read("services/research/research_function_service.py")
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


def test_specialists_execute_capabilities_instead_of_phase1_wrapped_action_tools() -> None:
    runtime_core = _read("services/research/research_supervisor_graph_runtime_core.py")
    unified_runtime = _read("services/research/unified_runtime.py")
    capabilities = _read("services/research/research_specialist_capabilities.py")

    assert "phase1_wrapped_action_tool" not in runtime_core
    assert "_build_specialist_execution_handler" in runtime_core
    assert '"LiteratureScoutAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"ResearchKnowledgeAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"ResearchWriterAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"PaperAnalysisAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"ChartAnalysisAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"GeneralAnswerAgent": self._build_specialist_execution_handler(' in runtime_core
    assert '"PreferenceMemoryAgent": self._build_specialist_execution_handler(' in runtime_core
    assert "unresolved_boundaries=[]" in unified_runtime
    assert "legacy_boundaries=[]" in unified_runtime
    assert "class LiteratureDiscoveryCapability" in capabilities
    assert "class KnowledgeOpsCapability" in capabilities
    assert "class PaperAnalysisCapability" in capabilities
    assert "class ChartAnalysisCapability" in capabilities
    assert "class PreferenceRecommendationCapability" in capabilities
