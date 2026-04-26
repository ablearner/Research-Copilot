from types import SimpleNamespace

from core.config import Settings
from memory.memory_manager import MemoryManager
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from services.research.research_function_service import ResearchFunctionService
from services.research.research_report_service import ResearchReportService
from skills.research import PaperReadingSkill, ResearchEvaluationSkill, ReviewWritingSkill
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry

from apps.api.research_runtime import register_research_runtime_extensions


class PaperSearchServiceStub:
    async def search(self, **kwargs):  # pragma: no cover - not used directly here
        raise NotImplementedError


def test_register_research_runtime_extensions_registers_functions_and_local_runtime_tools(tmp_path) -> None:
    settings = Settings()
    report_service = ResearchReportService(tmp_path / "research")
    research_service = SimpleNamespace(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        memory_manager=MemoryManager(
            paper_knowledge_memory=PaperKnowledgeMemory(
                JsonPaperKnowledgeStore(tmp_path / "paper_knowledge")
            )
        ),
        paper_reading_skill=PaperReadingSkill(),
        review_writing_skill=ReviewWritingSkill(),
        evaluation_skill=ResearchEvaluationSkill(),
    )
    graph_runtime = SimpleNamespace(
        tool_registry=ToolRegistry(),
        tool_executor=ToolExecutor(ToolRegistry()),
    )
    graph_runtime.tool_executor = ToolExecutor(graph_runtime.tool_registry)

    function_service = register_research_runtime_extensions(
        settings,
        graph_runtime=graph_runtime,
        research_service=research_service,
    )

    assert isinstance(function_service, ResearchFunctionService)
    assert graph_runtime.tool_registry.get_tool("search_papers") is not None
    assert graph_runtime.tool_registry.get_tool("decompose_task") is not None
    assert graph_runtime.tool_registry.get_tool("academic_search") is not None
    assert graph_runtime.tool_registry.get_tool("search_or_import_paper") is not None
    assert graph_runtime.tool_registry.get_tool("code_execution") is None
    assert graph_runtime.tool_registry.get_tool("notification") is not None


def test_register_research_runtime_extensions_registers_code_execution_when_enabled(tmp_path) -> None:
    settings = Settings(local_code_execution_enabled=True)
    report_service = ResearchReportService(tmp_path / "research")
    research_service = SimpleNamespace(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        memory_manager=MemoryManager(
            paper_knowledge_memory=PaperKnowledgeMemory(
                JsonPaperKnowledgeStore(tmp_path / "paper_knowledge")
            )
        ),
        paper_reading_skill=PaperReadingSkill(),
        review_writing_skill=ReviewWritingSkill(),
        evaluation_skill=ResearchEvaluationSkill(),
    )
    graph_runtime = SimpleNamespace(
        tool_registry=ToolRegistry(),
        tool_executor=ToolExecutor(ToolRegistry()),
    )
    graph_runtime.tool_executor = ToolExecutor(graph_runtime.tool_registry)

    register_research_runtime_extensions(
        settings,
        graph_runtime=graph_runtime,
        research_service=research_service,
    )

    assert graph_runtime.tool_registry.get_tool("code_execution") is not None
