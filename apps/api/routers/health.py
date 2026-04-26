from fastapi import APIRouter, Request
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str
    runtime_backend: str
    llm_provider: str
    llm_model: str | None = None
    chart_vision_provider: str | None = None
    chart_vision_model: str | None = None
    embedding_provider: str
    embedding_model: str | None = None
    vector_store_provider: str
    graph_store_provider: str
    graph_runtime_ready: bool
    checkpointer_backend: str | None = None
    session_memory_backend: str | None = None


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    graph_runtime = getattr(request.app.state, "graph_runtime", None)
    checkpointer_backend = None
    session_memory_backend = None
    if graph_runtime is not None:
        checkpointer = getattr(graph_runtime, "checkpointer", None)
        checkpointer_backend = checkpointer.__class__.__name__ if checkpointer is not None else None
        session_memory = getattr(graph_runtime, "session_memory", None)
        store = getattr(session_memory, "store", None)
        session_memory_backend = store.__class__.__name__ if store is not None else None

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_env=settings.app_env,
        runtime_backend=settings.runtime_backend,
        llm_provider=getattr(settings, "llm_provider", "unknown"),
        llm_model=getattr(settings, "llm_model", None),
        chart_vision_provider=getattr(settings, "chart_vision_provider", None),
        chart_vision_model=getattr(settings, "chart_vision_model", None),
        embedding_provider=getattr(settings, "embedding_provider", "unknown"),
        embedding_model=getattr(settings, "embedding_model", None),
        vector_store_provider=getattr(settings, "vector_store_provider", "unknown"),
        graph_store_provider=getattr(settings, "graph_store_provider", "unknown"),
        graph_runtime_ready=graph_runtime is not None,
        checkpointer_backend=checkpointer_backend,
        session_memory_backend=session_memory_backend,
    )
