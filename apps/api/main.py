import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.api.runtime import build_graph_runtime, close_graph_runtime, initialize_graph_runtime
from apps.api.exception_handlers import register_exception_handlers
from apps.api.routers.ask import router as ask_router
from apps.api.routers.charts import router as charts_router
from apps.api.routers.documents import router as documents_router
from apps.api.routers.health import router as health_router
from apps.api.routers.index import router as index_router
from apps.api.routers.mcp import router as mcp_router
from apps.api.routers.parse import router as parse_router
from apps.api.routers.research import router as research_router
from apps.api.routers.upload import router as upload_router
from apps.api.research_runtime import (
    build_academic_search_mcp_dependencies,
    build_literature_research_service,
    register_research_runtime_extensions,
)
from core.config import get_settings
from core.logging import configure_logging
from services.research.zotero_local_mcp import (
    ZoteroLocalServerConfig,
    build_zotero_local_mcp_client,
)
from services.research.academic_search_mcp import build_academic_search_mcp_client
from mcp.server.app import MCPServerApp

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    logger.info(
        "Startup config: runtime_backend=%s llm_provider=%s llm_model=%s chart_vision_provider=%s chart_vision_model=%s embedding_provider=%s embedding_model=%s vector_store_provider=%s graph_store_provider=%s",
        settings.runtime_backend,
        settings.llm_provider,
        settings.llm_model,
        settings.chart_vision_provider or settings.llm_provider,
        settings.chart_vision_model or settings.vision_model or settings.llm_model,
        settings.embedding_provider,
        settings.embedding_model,
        settings.vector_store_provider,
        settings.graph_store_provider,
    )
    if settings.research_reset_on_startup:
        await app.state.literature_research_service.reset_state()
    else:
        logger.info("Research persistence preserved on startup")
    await initialize_graph_runtime(app.state.graph_runtime)
    try:
        yield
    finally:
        await close_graph_runtime(app.state.graph_runtime)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    register_exception_handlers(app)
    upload_dir = settings.resolve_path(settings.upload_dir)
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:3001",
            "http://localhost:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if str(settings.app_env).strip().lower() == "local":
        app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")
    else:
        logger.info("Upload preview endpoint disabled for app_env=%s", settings.app_env)
    app.state.settings = settings
    app.state.graph_runtime = build_graph_runtime(settings)
    app.state.graph_runtime.external_tool_registry.register_server(
        "academic-search",
        build_academic_search_mcp_client(build_academic_search_mcp_dependencies(settings)),
        replace=True,
    )
    app.state.literature_research_service = build_literature_research_service(
        settings,
        graph_runtime=app.state.graph_runtime,
    )
    app.state.research_function_service = register_research_runtime_extensions(
        settings,
        graph_runtime=app.state.graph_runtime,
        research_service=app.state.literature_research_service,
    )
    if getattr(settings, "zotero_local_enabled", False):
        app.state.graph_runtime.external_tool_registry.register_server(
            "zotero-local",
            build_zotero_local_mcp_client(
                ZoteroLocalServerConfig(
                    base_url=getattr(settings, "zotero_local_base_url", "http://127.0.0.1:23119"),
                    user_id=getattr(settings, "zotero_local_user_id", "0"),
                    timeout_seconds=getattr(settings, "zotero_local_timeout_seconds", 20.0),
                )
            ),
            replace=True,
        )
    app.state.mcp_server = MCPServerApp.from_graph_runtime(app.state.graph_runtime)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(upload_router)
    app.include_router(parse_router)
    app.include_router(index_router)
    app.include_router(charts_router)
    app.include_router(ask_router)
    app.include_router(research_router)
    app.include_router(mcp_router)
    return app


app = create_app()
