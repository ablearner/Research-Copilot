from fastapi import HTTPException, Request, status

from core.config import Settings, get_settings
from rag_runtime.runtime import RagRuntime
from mcp.server.app import MCPServerApp
from services.research.literature_research_service import LiteratureResearchService


def get_app_settings() -> Settings:
    return get_settings()


def get_graph_runtime(request: Request) -> RagRuntime:
    graph_runtime = getattr(request.app.state, "graph_runtime", None)
    if graph_runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph runtime is not configured",
        )
    return graph_runtime


def get_mcp_server(request: Request) -> MCPServerApp:
    mcp_server = getattr(request.app.state, "mcp_server", None)
    if mcp_server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP server is not configured",
        )
    return mcp_server


def get_literature_research_service(request: Request) -> LiteratureResearchService:
    service = getattr(request.app.state, "literature_research_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Literature research service is not configured",
        )
    return service
