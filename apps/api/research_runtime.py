import logging

from core.config import Settings
from rag_runtime.runtime import RagRuntime
from tools.research import (
    CodeLinker,
    PaperReader,
    ResearchEvaluator,
    ReviewWriter,
    SurveyWriter,
    WritingPolisher,
)
from tools.research.research_functions import ResearchFunctionService
from tools.research.external_tool_gateway import ResearchExternalToolGateway
from services.research.literature_research_service import LiteratureResearchService
from tools.research.paper_import import PaperImportService
from tools.research.paper_search import PaperSearchService
from adapters.mcp.academic_search import AcademicSearchMCPDependencies
from adapters.storage.factory import create_store
from adapters.storage.research_report_service import ResearchReportService
from tooling.research_function_registry import ResearchFunctionRegistry
from tooling.research_runtime_tool_specs import build_research_runtime_tool_spec
from tools.research import (
    ArxivSearchTool,
    IEEEMetadataSearchTool,
    OpenAlexSearchTool,
    SemanticScholarSearchTool,
)

from tools.research.zotero_search_tool import ZoteroSearchTool

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


def build_literature_research_service(
    settings: Settings,
    *,
    graph_runtime: RagRuntime | None = None,
) -> LiteratureResearchService:
    academic_search_dependencies = build_academic_search_mcp_dependencies(settings)
    external_tool_gateway = ResearchExternalToolGateway(graph_runtime=graph_runtime)
    llm_adapter = getattr(graph_runtime, "llm_adapter", None) if graph_runtime is not None else None
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
        external_tool_gateway=external_tool_gateway,
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
        paper_reading_skill=paper_reading_skill,
        evaluation_skill=evaluation_skill,
        review_writing_skill=review_writing_skill,
        writing_polish_skill=writing_polish_skill,
        import_concurrency=settings.research_import_concurrency,
        import_index_timeout_seconds=settings.research_import_index_timeout_seconds,
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
