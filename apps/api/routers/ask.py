import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from apps.api.audit import audit_api_call
from apps.api.deps import get_graph_runtime
from apps.api.exception_handlers import build_error_detail
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.api import AskFusedResponse
from domain.schemas.api import AskDocumentResponse
from domain.schemas.api import QAResponse
from rag_runtime.runtime import RagRuntime

logger = logging.getLogger(__name__)
_MAX_EVIDENCES = 20
_MAX_HITS = 10

router = APIRouter(prefix="/ask", tags=["ask"])
LEGACY_ASK_DOCUMENT_ROUTE = "/ask/documents"
LEGACY_ASK_FUSED_ROUTE = "/ask/fused"


def _compact_qa_response(qa: QAResponse, document_ids: list[str]) -> QAResponse:
    allowed_document_ids = {doc_id for doc_id in document_ids if doc_id}

    def keep_document(document_id: str | None) -> bool:
        return not allowed_document_ids or document_id in allowed_document_ids

    evidence_bundle = qa.evidence_bundle.model_copy(
        update={
            "evidences": [
                evidence
                for evidence in qa.evidence_bundle.evidences
                if keep_document(evidence.document_id)
            ][: _MAX_EVIDENCES],
        }
    )

    retrieval_result = qa.retrieval_result
    if retrieval_result is not None:
        retrieval_result = retrieval_result.model_copy(
            update={
                "hits": [
                    hit
                    for hit in retrieval_result.hits
                    if keep_document(hit.document_id)
                ][: _MAX_HITS],
            }
        )

    return qa.model_copy(update={"evidence_bundle": evidence_bundle, "retrieval_result": retrieval_result})


class AskDocumentRequest(BaseModel):
    question: str
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    session_id: str | None = None
    task_intent: str | None = None
    reasoning_style: str | None = "cot"
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskFusedRequest(BaseModel):
    question: str
    image_path: str
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    page_id: str | None = None
    page_number: int = Field(default=1, ge=1)
    chart_id: str | None = None
    session_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    reasoning_style: str | None = "cot"
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def handle_ask_document_request(
    request: AskDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> AskDocumentResponse:
    try:
        trace_id = f"ask_{uuid4().hex}"
        document_ids = list(request.document_ids)
        if request.doc_id and request.doc_id not in document_ids:
            document_ids.append(request.doc_id)
        qa = await graph_runtime.handle_ask_document(
            question=request.question,
            doc_id=request.doc_id,
            document_ids=document_ids,
            top_k=request.top_k,
            filters=request.filters,
            session_id=request.session_id,
            task_intent=request.task_intent or "ask_document",
            reasoning_style=request.reasoning_style,
            metadata={
                "api_route": route_path,
                "trace_id": trace_id,
                "session_id": request.session_id,
                "quota_context": quota_context,
                **request.metadata,
            },
        )
        response = AskDocumentResponse(document_ids=document_ids, qa=_compact_qa_response(qa, document_ids))
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="ask",
            status="succeeded",
            metadata={"document_ids": document_ids, **qa.metadata},
        )
        return response
    except Exception as exc:
        logger.exception("Ask document request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to answer document question", exc=exc),
        ) from exc


async def handle_ask_fused_request(
    request: AskFusedRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> AskFusedResponse:
    try:
        trace_id = f"fused_{uuid4().hex}"
        document_ids = list(request.document_ids)
        if request.doc_id and request.doc_id not in document_ids:
            document_ids.append(request.doc_id)

        fused_result = await graph_runtime.handle_ask_fused(
            question=request.question,
            image_path=request.image_path,
            doc_id=request.doc_id,
            document_ids=document_ids,
            page_id=request.page_id,
            page_number=request.page_number,
            chart_id=request.chart_id,
            session_id=request.session_id,
            top_k=request.top_k,
            filters=request.filters,
            reasoning_style=request.reasoning_style,
            metadata={
                "api_route": route_path,
                "trace_id": trace_id,
                "session_id": request.session_id,
                "quota_context": quota_context,
                **request.metadata,
            },
        )
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="ask_fused",
            status="succeeded",
            metadata={"document_ids": document_ids, "chart_id": request.chart_id, **fused_result.qa.metadata},
        )
        return AskFusedResponse(
            document_ids=document_ids,
            qa=_compact_qa_response(fused_result.qa, document_ids),
            chart_answer=fused_result.chart_answer,
            chart_confidence=fused_result.chart_confidence,
        )
    except Exception as exc:
        logger.exception("Ask fused request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(
                http_request,
                fallback="Failed to answer fused document and chart question",
                exc=exc,
            ),
        ) from exc


@router.post("/documents", response_model=AskDocumentResponse, deprecated=True)
async def ask_document(
    request: AskDocumentRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> AskDocumentResponse:
    return await handle_ask_document_request(
        request=request,
        http_request=http_request,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_ASK_DOCUMENT_ROUTE,
    )


@router.post("/fused", response_model=AskFusedResponse, deprecated=True)
async def ask_fused(
    request: AskFusedRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> AskFusedResponse:
    return await handle_ask_fused_request(
        request=request,
        http_request=http_request,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_ASK_FUSED_ROUTE,
    )
