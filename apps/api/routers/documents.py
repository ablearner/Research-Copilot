from fastapi import APIRouter, Depends, File, Request, UploadFile

from apps.api.deps import get_graph_runtime
from apps.api.routers.ask import (
    AskDocumentRequest,
    AskFusedRequest,
    handle_ask_document_request,
    handle_ask_fused_request,
)
from apps.api.routers.index import (
    IndexDocumentApiResponse,
    IndexDocumentRequest,
    handle_index_document_request,
)
from apps.api.routers.parse import (
    ParseDocumentRequest,
    handle_parse_document_request,
)
from apps.api.routers.upload import handle_upload_document_request
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.api import (
    AskDocumentResponse,
    AskFusedResponse,
    ParseDocumentResponse,
    UploadDocumentResponse,
)
from rag_runtime.runtime import RagRuntime


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> UploadDocumentResponse:
    request.state.quota_context = quota_context
    return await handle_upload_document_request(request=request, file=file)


@router.post("/parse", response_model=ParseDocumentResponse)
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
        route_path="/documents/parse",
    )


@router.post("/index", response_model=IndexDocumentApiResponse)
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
        route_path="/documents/index",
    )


@router.post("/ask", response_model=AskDocumentResponse)
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
        route_path="/documents/ask",
    )


@router.post("/ask/fused", response_model=AskFusedResponse)
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
        route_path="/documents/ask/fused",
    )
