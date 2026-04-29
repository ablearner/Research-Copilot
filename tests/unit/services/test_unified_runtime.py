from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from domain.schemas.agent_message import AgentMessage
from domain.schemas.research import PaperCandidate
from domain.schemas.unified_runtime import (
    UNIFIED_ACTION_OUTPUT_METADATA_KEY,
    UnifiedAgentResult,
    UnifiedAgentTask,
    UnifiedChartUnderstandingInput,
    UnifiedCollectionQAInput,
    UnifiedContextCompressionInput,
    UnifiedDocumentUnderstandingInput,
    UnifiedLiteratureSearchInput,
    UnifiedPaperImportInput,
    UnifiedPaperAnalysisInput,
    UnifiedReviewDraftInput,
)
from services.research.unified_action_adapters import (
    build_context_compression_input,
    build_context_compression_output,
    build_literature_search_input,
    build_literature_search_output,
    resolve_active_message,
)
from services.research.unified_runtime import (
    build_phase1_unified_agent_registry,
    build_phase1_unified_blueprint,
    build_phase1_unified_runtime_context,
    serialize_unified_agent_messages,
    serialize_unified_agent_results,
    serialize_unified_delegation_plan,
)
from tooling.registry import ToolRegistry
from tooling.schemas import ToolSpec


class DummyInput(BaseModel):
    value: str = "ok"


class DummyOutput(BaseModel):
    value: str = "ok"


async def _dummy_handler(**kwargs):
    return {"value": kwargs.get("value", "ok")}


def _build_runtime_stub():
    tool_registry = ToolRegistry()
    tool_registry.register_many(
        [
            ToolSpec(
                name="hybrid_retrieve",
                description="Retrieve evidence",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["retrieval"],
            ),
            ToolSpec(
                name="parse_document",
                description="Parse document",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["document", "parse"],
            ),
            ToolSpec(
                name="index_document",
                description="Index document",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["document", "index"],
            ),
            ToolSpec(
                name="answer_with_evidence",
                description="Answer with evidence",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["answer"],
            ),
            ToolSpec(
                name="understand_chart",
                description="Understand chart",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["chart"],
            ),
            ToolSpec(
                name="search_papers",
                description="Search papers",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["search"],
            ),
            ToolSpec(
                name="generate_review",
                description="Generate review",
                input_schema=DummyInput,
                output_schema=DummyOutput,
                handler=_dummy_handler,
                tags=["review"],
            ),
        ]
    )
    return SimpleNamespace(tool_registry=tool_registry)


def test_unified_agent_task_roundtrip_to_legacy_result():
    message = AgentMessage(
        task_id="task_1",
        agent_from="ResearchSupervisorAgent",
        agent_to="ResearchKnowledgeAgent",
        task_type="answer_question",
        instruction="answer the user question",
        metadata={"skill_name": "research_report"},
    )
    task = UnifiedAgentTask.from_agent_message(message)
    assert task.preferred_skill_name == "research_report"

    result = UnifiedAgentResult(
        task_id=task.task_id,
        agent_name="ResearchKnowledgeAgent",
        task_type=task.task_type,
        status="succeeded",
        instruction=task.instruction,
        action_output={
            "unified_input_adapter": "collection_qa_input",
            "task_id": task.task_id,
            "evidence_count": 2,
        },
        metadata={"source": "test"},
    )
    legacy = result.to_agent_result_message()
    assert legacy.task_id == "task_1"
    assert legacy.agent_from == "ResearchKnowledgeAgent"
    assert legacy.agent_to == "ResearchSupervisorAgent"
    assert legacy.status == "succeeded"
    assert legacy.metadata[UNIFIED_ACTION_OUTPUT_METADATA_KEY]["unified_input_adapter"] == "collection_qa_input"
    restored = UnifiedAgentResult.from_agent_result_message(legacy)
    assert restored.action_output is not None
    assert restored.action_output["evidence_count"] == 2


def test_build_phase1_blueprint_for_current_repo_shape():
    graph_runtime = _build_runtime_stub()
    blueprint = build_phase1_unified_blueprint(
        graph_runtime=graph_runtime,
        research_service=SimpleNamespace(),
    )

    descriptor_names = {item.name for item in blueprint.agent_descriptors}
    assert "ResearchSupervisorAgent" in descriptor_names
    assert "ResearchKnowledgeAgent" in descriptor_names
    assert "ChartAnalysisAgent" in descriptor_names
    assert "DocumentRuntime" not in descriptor_names
    assert "hybrid_retrieve" in blueprint.tool_names
    assert "parse_document" in blueprint.tool_names
    assert "index_document" in blueprint.tool_names
    assert blueprint.capability_profile_names == []
    assert blueprint.unresolved_boundaries
    knowledge_descriptor = next(
        item for item in blueprint.agent_descriptors if item.name == "ResearchKnowledgeAgent"
    )
    assert "hybrid_retrieve" in knowledge_descriptor.available_tool_names


def test_build_phase1_unified_agent_registry_uses_skeleton_adapters():
    graph_runtime = _build_runtime_stub()
    registry = build_phase1_unified_agent_registry(
        graph_runtime=graph_runtime,
        research_service=SimpleNamespace(),
    )
    executors = registry.resolve_for_task("answer_question")
    assert executors
    executor_names = {item.descriptor.name for item in executors}
    assert "ResearchKnowledgeAgent" in executor_names


def test_unified_execution_inputs_normalize_worker_payloads():
    paper = PaperCandidate(
        paper_id="paper_1",
        title="Agentic Survey",
        source="arxiv",
        pdf_url="https://example.com/paper_1.pdf",
        relevance_score=0.9,
        year=2026,
        citations=8,
    )
    review_input = UnifiedReviewDraftInput(
        topic="agentic literature review",
        task_id="task_review_1",
        curated_papers=[paper],
        max_papers=1,
    )
    assert review_input.curated_papers[0].paper_id == "paper_1"

    search_input = UnifiedLiteratureSearchInput(
        topic="agentic literature review",
        days_back=30,
        max_papers=5,
        sources=["arxiv", "openalex"],
        conversation_id="conv_1",
    )
    create_task_request = search_input.to_create_research_task_request()
    assert create_task_request.topic == "agentic literature review"
    assert create_task_request.sources == ["arxiv", "openalex"]
    assert create_task_request.run_immediately is True

    qa_input = UnifiedCollectionQAInput(
        task_id="task_qa_1",
        question="哪篇论文最值得优先阅读？",
        top_k=6,
        paper_ids=["paper_1"],
        document_ids=["doc_1"],
        image_path="/tmp/chart.png",
        page_id="page-2",
        page_number=2,
        chart_id="chart_3",
        metadata={"agent_runtime": "research_agent"},
    )
    ask_request = qa_input.to_research_task_ask_request()
    assert ask_request.question == "哪篇论文最值得优先阅读？"
    assert ask_request.paper_ids == ["paper_1"]
    assert ask_request.document_ids == ["doc_1"]
    assert ask_request.top_k == 6
    assert ask_request.image_path == "/tmp/chart.png"
    assert ask_request.page_id == "page-2"
    assert ask_request.page_number == 2
    assert ask_request.chart_id == "chart_3"

    analysis_input = UnifiedPaperAnalysisInput(
        question="对比这些论文",
        analysis_focus="compare",
        comparison_dimensions=["method", "experiment"],
        papers=[paper],
        task_topic="multi-agent literature review",
    )
    assert analysis_input.resolved_question() == "对比这些论文，重点比较：method, experiment"

    import_input = UnifiedPaperImportInput(
        task_id="task_import_1",
        selected_paper_ids=["paper_1"],
        import_top_k=1,
    )
    assert import_input.resolved_paper_ids([paper]) == ["paper_1"]

    document_input = UnifiedDocumentUnderstandingInput(
        file_path="/tmp/example.pdf",
        document_id="doc_1",
        include_graph=True,
        include_embeddings=True,
    )
    assert document_input.file_path == "/tmp/example.pdf"
    assert document_input.document_id == "doc_1"

    chart_input = UnifiedChartUnderstandingInput(
        image_path="/tmp/chart.png",
        document_id="doc_1",
        page_id="page-1",
        page_number=1,
        chart_id="chart_1",
    )
    assert chart_input.image_path == "/tmp/chart.png"
    assert chart_input.chart_id == "chart_1"

    compression_input = UnifiedContextCompressionInput(
        task_id="task_compress_1",
        selected_paper_ids=["paper_1", "paper_1"],
    )
    assert compression_input.resolved_selected_paper_ids() == ["paper_1"]


def test_unified_action_adapter_builders_roundtrip_metadata():
    request = SimpleNamespace(
        message="agentic literature review",
        days_back=30,
        max_papers=5,
        sources=["arxiv", "openalex"],
        conversation_id="conv_1",
        selected_paper_ids=["paper_1"],
        skill_name="research_report",
        reasoning_style="cot",
        metadata={"source": "test"},
    )
    context = SimpleNamespace(
        request=request,
        task=SimpleNamespace(task_id="task_1"),
    )
    active_message = AgentMessage(
        task_id="task_msg_1",
        agent_from="ResearchSupervisorAgent",
        agent_to="ResearchKnowledgeAgent",
        task_type="compress_context",
        instruction="compress",
        payload={"paper_ids": ["paper_1", "paper_1"]},
    )
    decision = SimpleNamespace(
        action_input={"paper_ids": ["paper_1", "paper_1"]},
        metadata={"active_message": active_message},
    )
    task_response = SimpleNamespace(
        task=SimpleNamespace(task_id="task_1"),
        papers=[PaperCandidate(paper_id="paper_1", title="Agentic Survey", source="arxiv")],
        report=None,
        warnings=["warn"],
    )

    search_input = build_literature_search_input(context=context, decision=decision)
    search_output = build_literature_search_output(task_response=task_response)
    compression_input = build_context_compression_input(context=context, decision=decision)
    compression_output = build_context_compression_output(
        compression_summary={
            "paper_count": 1,
            "summary_count": 2,
            "levels": ["brief"],
            "compressed_paper_ids": ["paper_1"],
        }
    )

    assert search_input.topic == "agentic literature review"
    assert search_input.to_create_research_task_request().sources == ["arxiv", "openalex"]
    assert search_output.to_metadata()["unified_input_adapter"] == "literature_search_input"
    assert compression_input.resolved_selected_paper_ids() == ["paper_1"]
    assert compression_output.to_metadata()["summary_count"] == 2
    assert resolve_active_message(decision).task_id == "task_msg_1"


def test_unified_serializers_include_agent_descriptors():
    graph_runtime = _build_runtime_stub()
    registry = build_phase1_unified_agent_registry(
        graph_runtime=graph_runtime,
        research_service=SimpleNamespace(),
    )
    message = AgentMessage(
        task_id="task_2",
        agent_from="ResearchSupervisorAgent",
        agent_to="ResearchKnowledgeAgent",
        task_type="answer_question",
        instruction="answer",
    )
    result = UnifiedAgentResult(
        task_id="task_2",
        agent_name="ResearchKnowledgeAgent",
        task_type="answer_question",
        status="succeeded",
        action_output={
            "unified_input_adapter": "collection_qa_input",
            "task_id": "task_2",
            "evidence_count": 1,
        },
        metadata={"ok": True},
    ).to_agent_result_message()

    serialized_messages = serialize_unified_agent_messages([message], registry=registry)
    serialized_results = serialize_unified_agent_results([result], registry=registry)
    serialized_plan = serialize_unified_delegation_plan([message], [result], registry=registry)

    assert serialized_messages[0]["agent_descriptor"]["name"] == "ResearchKnowledgeAgent"
    assert serialized_messages[0]["preferred_skill_name"] == "research_report"
    assert "hybrid_retrieve" in serialized_messages[0]["available_tool_names"]
    assert serialized_results[0]["agent_descriptor"]["name"] == "ResearchKnowledgeAgent"
    assert serialized_results[0]["preferred_skill_name"] == "research_report"
    assert "hybrid_retrieve" in serialized_results[0]["available_tool_names"]
    assert serialized_results[0]["execution_mode"] == "tool_native"
    assert serialized_results[0]["action_output"]["unified_input_adapter"] == "collection_qa_input"
    assert serialized_plan[0]["agent_descriptor"]["name"] == "ResearchKnowledgeAgent"
    assert serialized_plan[0]["preferred_skill_name"] == "research_report"
    assert "hybrid_retrieve" in serialized_plan[0]["available_tool_names"]
    assert serialized_plan[0]["status"] == "succeeded"
    assert serialized_plan[0]["action_output"]["unified_input_adapter"] == "collection_qa_input"


@pytest.mark.asyncio
async def test_phase1_registry_can_execute_real_handler_when_bound():
    graph_runtime = _build_runtime_stub()

    async def _execution_handler(task, context, legacy_delegate):
        return UnifiedAgentResult(
            task_id=task.task_id,
            agent_name="ResearchKnowledgeAgent",
            task_type=task.task_type,
            status="succeeded",
            instruction=task.instruction,
            metadata={
                "execution_engine": "unified_agent_registry",
                "legacy_delegate_type": legacy_delegate.__class__.__name__ if legacy_delegate is not None else None,
                "context_has_tool_registry": context.tool_registry is not None,
            },
        )

    registry = build_phase1_unified_agent_registry(
        graph_runtime=graph_runtime,
        research_service=SimpleNamespace(),
        legacy_delegates={"ResearchKnowledgeAgent": SimpleNamespace(name="knowledge")},
        execution_handlers={"ResearchKnowledgeAgent": _execution_handler},
    )
    executor = registry.get("ResearchKnowledgeAgent")
    assert executor is not None

    runtime_context = build_phase1_unified_runtime_context(
        graph_runtime=graph_runtime,
        research_service=SimpleNamespace(),
    )
    result = await executor.execute(
        UnifiedAgentTask(
            task_id="task_3",
            agent_from="ResearchSupervisorAgent",
            agent_to="ResearchKnowledgeAgent",
            task_type="answer_question",
            instruction="answer",
        ),
        runtime_context,
    )

    assert result.status == "succeeded"
    assert result.metadata["execution_engine"] == "unified_agent_registry"
    assert result.metadata["legacy_delegate_type"] == "SimpleNamespace"
    assert result.metadata["context_has_tool_registry"] is True
