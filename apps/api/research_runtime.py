import logging

from core.config import Settings
from domain.schemas.research import PaperCandidate
from mcp.client.registry import MCPClientRegistry
from rag_runtime.runtime import RagRuntime
from services.research.capabilities import (
    CodeLinker,
    PaperReader,
    ResearchEvaluator,
    ReviewWriter,
    SurveyWriter,
    WritingPolisher,
)
from services.research.research_function_service import ResearchFunctionService
from services.research.literature_research_service import LiteratureResearchService
from services.research.paper_import_service import PaperImportService
from services.research.paper_search_service import PaperSearchService
from services.research.academic_search_mcp import AcademicSearchMCPDependencies
from adapters.storage.factory import create_store
from services.research.research_report_service import ResearchReportService
from tooling.research_function_registry import ResearchFunctionRegistry
from tooling.research_runtime_tool_specs import build_research_runtime_tool_spec
from tools.research import (
    ArxivSearchTool,
    IEEEMetadataSearchTool,
    OpenAlexSearchTool,
    SemanticScholarSearchTool,
)

logger = logging.getLogger(__name__)


def build_academic_search_mcp_dependencies(settings: Settings) -> AcademicSearchMCPDependencies:
    return AcademicSearchMCPDependencies(
        arxiv_tool=ArxivSearchTool(
            base_url=settings.arxiv_api_base_url,
            app_name=settings.app_name,
            timeout_seconds=settings.research_http_timeout_seconds,
        ),
        openalex_tool=OpenAlexSearchTool(
            base_url=settings.openalex_api_base_url,
            timeout_seconds=settings.research_http_timeout_seconds,
            mailto=settings.research_contact_email,
        ),
        semantic_scholar_tool=SemanticScholarSearchTool(
            base_url=settings.semantic_scholar_api_base_url,
            timeout_seconds=settings.research_http_timeout_seconds,
            api_key=settings.semantic_scholar_api_key,
        ),
        ieee_tool=IEEEMetadataSearchTool(
            base_url=settings.ieee_api_base_url,
            api_key=settings.ieee_api_key,
            timeout_seconds=settings.research_http_timeout_seconds,
        ),
    )


class ZoteroSearchTool:
    def __init__(self, *, graph_runtime: RagRuntime | None = None) -> None:
        self.graph_runtime = graph_runtime

    def _get_external_tool_registry(self):
        if self.graph_runtime is None:
            return None
        registry = getattr(self.graph_runtime, "external_tool_registry", None)
        if registry is not None:
            return registry
        return getattr(self.graph_runtime, "mcp_client_registry", None)

    async def search(self, *, query: str, max_results: int, days_back: int) -> list:
        del days_back
        registry = self._get_external_tool_registry()
        if registry is None:
            return []
        result = await registry.call_tool(
            tool_name="zotero_search_items",
            arguments={
                "query": query,
                "limit": max_results,
                "include_attachments": True,
            },
        )
        if result.status != "succeeded" or not isinstance(result.output, dict):
            logger.info(
                "Zotero search tool result | query=%s | status=%s | hits=0",
                query[:180],
                result.status,
            )
            return []
        items = result.output.get("items")
        if not isinstance(items, list):
            logger.info(
                "Zotero search tool result | query=%s | status=%s | invalid_items=1 | hits=0",
                query[:180],
                result.status,
            )
            return []
        papers = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_key = str(item.get("key") or "").strip()
            title = str(item.get("title") or "").strip()
            if not item_key or not title:
                continue
            attachments = item.get("attachments")
            pdf_url = None
            local_path = None
            if isinstance(attachments, list):
                for attachment in attachments:
                    if not isinstance(attachment, dict):
                        continue
                    content_type = str(attachment.get("content_type") or "").lower()
                    if "pdf" not in content_type and not str(attachment.get("title") or "").lower().endswith("pdf"):
                        continue
                    pdf_url = str(attachment.get("url") or "").strip() or None
                    local_path = str(attachment.get("local_path") or "").strip() or None
                    break
            papers.append(
                PaperCandidate(
                    paper_id=f"zotero:{item_key}",
                    title=title,
                    authors=list(item.get("creators") or []),
                    abstract=str(item.get("abstract") or ""),
                    year=int(item["year"]) if isinstance(item.get("year"), str) and str(item.get("year")).isdigit() else None,
                    source="zotero",
                    doi=str(item.get("doi") or "").strip() or None,
                    pdf_url=pdf_url,
                    url=str(item.get("url") or "").strip() or None,
                    metadata={
                        "zotero_item_key": item_key,
                        "zotero_collections": list(item.get("collections") or []),
                        "zotero_local_path": local_path,
                    },
                )
            )
        logger.info(
            "Zotero search tool result | query=%s | items=%s | papers=%s | titles=%s",
            query[:180],
            len(items),
            len(papers),
            " | ".join(paper.title for paper in papers[:5]),
        )
        return papers


def build_literature_research_service(
    settings: Settings,
    *,
    graph_runtime: RagRuntime | None = None,
) -> LiteratureResearchService:
    academic_search_dependencies = build_academic_search_mcp_dependencies(settings)
    external_tool_registry = None
    if graph_runtime is not None:
        external_tool_registry = getattr(graph_runtime, "external_tool_registry", None) or getattr(
            graph_runtime, "mcp_client_registry", None
        )
    reasoning_strategies = getattr(graph_runtime, "reasoning_strategies", None) if graph_runtime is not None else None
    plan_and_solve_reasoning_agent = (
        getattr(reasoning_strategies, "query_planning", None)
        or getattr(graph_runtime, "plan_and_solve_reasoning_agent", None)
        if graph_runtime is not None
        else None
    )
    llm_adapter = (
        getattr(reasoning_strategies, "llm_adapter", None)
        or getattr(plan_and_solve_reasoning_agent, "llm_adapter", None)
    )
    writing_polish_skill = WritingPolisher(llm_adapter=llm_adapter)
    review_writing_skill = ReviewWriter(
        survey_writer=SurveyWriter(llm_adapter=llm_adapter),
        polish_skill=writing_polish_skill,
    )
    paper_reading_skill = PaperReader(llm_adapter=llm_adapter)
    evaluation_skill = ResearchEvaluator()
    paper_search_service = PaperSearchService(
        arxiv_tool=academic_search_dependencies.arxiv_tool,
        openalex_tool=academic_search_dependencies.openalex_tool,
        semantic_scholar_tool=academic_search_dependencies.semantic_scholar_tool,
        ieee_tool=academic_search_dependencies.ieee_tool,
        zotero_tool=ZoteroSearchTool(graph_runtime=graph_runtime),
        external_tool_registry=external_tool_registry,
        survey_writer=review_writing_skill,
        code_linking_skill=CodeLinker(enable_remote_lookup=False),
        llm_adapter=llm_adapter,
    )
    if getattr(settings, "storage_provider", "json") == "sqlite":
        report_service = create_store(
            "sqlite",
            db_path=settings.resolve_path(settings.research_sqlite_db_path),
        )
    else:
        report_service = ResearchReportService(
            settings.resolve_path(settings.research_storage_root),
        )
    paper_import_service = PaperImportService(
        upload_dir=settings.resolve_path(settings.upload_dir),
        timeout_seconds=max(settings.research_http_timeout_seconds, 30.0),
    )
    return LiteratureResearchService(
        paper_search_service=paper_search_service,
        report_service=report_service,
        paper_import_service=paper_import_service,
        research_runtime=None,
        research_qa_runtime=None,
        paper_reading_skill=paper_reading_skill,
        evaluation_skill=evaluation_skill,
        review_writing_skill=review_writing_skill,
        writing_polish_skill=writing_polish_skill,
        import_concurrency=settings.research_import_concurrency,
    )


def register_research_runtime_extensions(
    settings: Settings,
    *,
    graph_runtime: RagRuntime,
    research_service: LiteratureResearchService,
) -> ResearchFunctionService:
    function_service = ResearchFunctionService(
        research_service=research_service,
        graph_runtime=graph_runtime,
        code_execution_enabled=settings.local_code_execution_enabled,
        zotero_api_base_url=settings.zotero_api_base_url,
        zotero_api_key=settings.zotero_api_key,
        zotero_library_type=settings.zotero_library_type,
        zotero_library_id=settings.zotero_library_id,
        allowed_file_roots=[
            settings.resolve_path(settings.research_storage_root),
            settings.resolve_path(settings.local_storage_root),
            settings.resolve_path(settings.upload_dir),
        ],
    )
    ResearchFunctionRegistry(graph_runtime.tool_registry).register_many(
        function_service.build_function_handlers(),
        replace=True,
    )
    for name, handler in function_service.build_runtime_tool_handlers().items():
        graph_runtime.tool_registry.register(
            build_research_runtime_tool_spec(name, handler),
            replace=True,
        )
    graph_runtime.research_function_service = function_service
    graph_runtime.literature_research_service = research_service
    return function_service
