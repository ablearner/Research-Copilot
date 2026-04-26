import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from apps.api.audit import audit_api_call
from apps.api.deps import get_graph_runtime
from apps.api.exception_handlers import build_error_detail
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.api import ParseDocumentResponse
from rag_runtime.runtime import RagRuntime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse", tags=["parse"])
LEGACY_PARSE_ROUTE = "/parse/documents"


class ParseDocumentRequest(BaseModel):
    file_path: str
    document_id: str | None = None
    skill_name: str | None = None


async def handle_parse_document_request(
    request: ParseDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> ParseDocumentResponse:
    try:
        trace_id = f"parse_{uuid4().hex}"
        skill_context = graph_runtime.resolve_skill_context(
            task_type="parse",
            preferred_skill_name=request.skill_name,
        ) if request.skill_name else None
        state = await graph_runtime.invoke(
            {
                "request_id": trace_id,
                "thread_id": trace_id,
                "task_type": "parse",
                "user_input": request.file_path,
                "file_path": request.file_path,
                "document_id": request.document_id,
                "document_ids": [request.document_id] if request.document_id else [],
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
        parsed_document = state.get("parsed_document")
        if parsed_document is None:
            raise RuntimeError("; ".join(state.get("errors", [])) or "No parsed_document returned")
        response = ParseDocumentResponse(
            document_id=parsed_document.id,
            status=parsed_document.status,
            parsed_document=parsed_document,
            error_message=parsed_document.error_message,
        )
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="parse",
            status=response.status,
            metadata=state.get("metadata", {}),
        )
        return response
    except Exception as exc:
        logger.exception("Parse document request failed", extra={"document_id": request.document_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to parse document", exc=exc),
        ) from exc


@router.post("/documents", response_model=ParseDocumentResponse, deprecated=True)
async def parse_document(
    request: ParseDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ParseDocumentResponse:
    return await handle_parse_document_request(
        request=request,
        http_request=http_request,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_PARSE_ROUTE,
    )
