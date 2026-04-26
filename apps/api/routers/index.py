import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from apps.api.deps import get_graph_runtime
from apps.api.audit import audit_api_call
from apps.api.exception_handlers import build_error_detail
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.chart import ChartSchema
from domain.schemas.document import ParsedDocument
from fastapi import Request
from rag_runtime.runtime import RagRuntime
from rag_runtime.schemas import DocumentIndexResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["index"])
LEGACY_INDEX_ROUTE = "/index/documents"


class IndexDocumentRequest(BaseModel):
    parsed_document: ParsedDocument
    charts: list[ChartSchema] = Field(default_factory=list)
    include_graph: bool = True
    include_embeddings: bool = True
    skill_name: str | None = None


class IndexDocumentApiResponse(BaseModel):
    status: str
    result: DocumentIndexResult


async def handle_index_document_request(
    request: IndexDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> IndexDocumentApiResponse:
    try:
        trace_id = f"index_{uuid4().hex}"
        skill_context = graph_runtime.resolve_skill_context(
            task_type="index",
            preferred_skill_name=request.skill_name,
        ) if request.skill_name else None
        state = await graph_runtime.invoke(
            {
                "request_id": trace_id,
                "thread_id": trace_id,
                "task_type": "index",
                "user_input": request.parsed_document.filename,
                "document_id": request.parsed_document.id,
                "document_ids": [request.parsed_document.id],
                "parsed_document": request.parsed_document,
                "charts": request.charts,
                "include_graph": request.include_graph,
                "include_embeddings": request.include_embeddings,
                "vector_hits": [],
                "graph_hits": [],
                "summary_hits": [],
                "graph_summary_hits": [],
                "warnings": [],
                "reasoning_summary": {},
                "react_trace": [],
                "messages": [],
                "tool_traces": [],
                "errors": [],
                "metadata": {
                    "api_route": route_path,
                    "trace_id": trace_id,
                    "quota_context": quota_context,
                    **({"skill_name": request.skill_name} if request.skill_name else {}),
                },
                "retrieval_mode": "hybrid",
                "top_k": 10,
                "filters": {},
                "selected_skill": skill_context,
            }
        )
        result = DocumentIndexResult(
            document_id=request.parsed_document.id,
            graph_extraction=state.get("graph_extraction_result"),
            graph_index=graph_runtime._model_from_payload(state.get("metadata", {}).get("graph_index")),
            text_embedding_index=graph_runtime._model_from_payload(
                state.get("metadata", {}).get("text_embedding_index")
            ),
            page_embedding_index=graph_runtime._model_from_payload(
                state.get("metadata", {}).get("page_embedding_index")
            ),
            chart_embedding_index=graph_runtime._model_from_payload(
                state.get("metadata", {}).get("chart_embedding_index")
            ),
            status="failed" if state.get("errors") else "succeeded",
            metadata={"tool_traces": state.get("tool_traces", []), **state.get("metadata", {})},
        )
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="index",
            status=result.status,
            metadata={"document_id": request.parsed_document.id, **(result.metadata or {})},
        )
        return IndexDocumentApiResponse(status=result.status, result=result)
    except Exception as exc:
        logger.exception("Index document request failed", extra={"document_id": request.parsed_document.id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to index document", exc=exc),
        ) from exc


@router.post("/documents", response_model=IndexDocumentApiResponse, deprecated=True)
async def index_document(
    request: IndexDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> IndexDocumentApiResponse:
    return await handle_index_document_request(
        request=request,
        http_request=http_request,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_INDEX_ROUTE,
    )
