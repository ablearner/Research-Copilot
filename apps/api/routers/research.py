import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from apps.api.audit import audit_api_call
from apps.api.document_upload import handle_upload_document_request
from apps.api.deps import get_graph_runtime, get_literature_research_service
from apps.api.exception_handlers import build_error_detail
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.api import UploadDocumentResponse
from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    CreateResearchConversationRequest,
    CreateResearchTaskRequest,
    ImportPapersRequest,
    ImportPapersResponse,
    RenameResearchConversationRequest,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversation,
    ResearchConversationResponse,
    ResearchJob,
    ResearchPaperFigureListResponse,
    ResearchTodoActionRequest,
    ResearchTodoActionResponse,
    ResearchTaskResponse,
    SearchPapersRequest,
    SearchPapersResponse,
    UpdateResearchTodoRequest,
)
from rag_runtime.runtime import RagRuntime
from services.research.literature_research_service import LiteratureResearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])
CANONICAL_AGENT_ROUTE = "/research/agent"
LEGACY_AGENT_RUN_ROUTE = "/research/agent/run"
LEGACY_AGENT_CHAT_ROUTE = "/research/agent/chat"
CANONICAL_TASK_IMPORT_ROUTE = "/research/tasks/{task_id}/papers/import"
CANONICAL_TASK_IMPORT_JOB_ROUTE = "/research/tasks/{task_id}/papers/import/jobs"
LEGACY_PAPER_IMPORT_ROUTE = "/research/papers/import"


def _compact_qa_response(qa, document_ids: list[str]) -> Any:
    allowed_document_ids = {doc_id for doc_id in document_ids if doc_id}

    def keep_document(document_id: str | None) -> bool:
        return document_id is None or not allowed_document_ids or document_id in allowed_document_ids

    evidence_bundle = qa.evidence_bundle.model_copy(
        update={
            "evidences": [
                evidence
                for evidence in qa.evidence_bundle.evidences
                if keep_document(evidence.document_id)
            ][:20],
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
                ][:10],
            }
        )
    return qa.model_copy(update={"evidence_bundle": evidence_bundle, "retrieval_result": retrieval_result})


async def _run_research_agent_impl(
    request: ResearchAgentRunRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> ResearchAgentRunResponse:
    try:
        trace_id = f"research_agent_{uuid4().hex}"
        runtime_request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "api_route": route_path,
                    "trace_id": trace_id,
                    "quota_context": quota_context,
                }
            }
        )
        response = await research_service.run_agent(runtime_request, graph_runtime=graph_runtime)
        if request.conversation_id:
            research_service.record_agent_turn(
                request.conversation_id,
                request=runtime_request,
                response=response,
            )
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="research_agent_run",
            status=response.status,
            metadata={
                "mode": request.mode,
                "task_id": response.task.task_id if response.task else request.task_id,
                "paper_count": len(response.papers),
                "imported_count": response.import_result.imported_count if response.import_result else 0,
                "has_qa": response.qa is not None,
                "quota_context": quota_context,
            },
        )
        return response
    except Exception as exc:
        logger.exception("Research agent run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to run autonomous research agent", exc=exc),
        ) from exc


@router.post("/documents/upload", response_model=UploadDocumentResponse)
async def upload_research_document(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> UploadDocumentResponse:
    request.state.quota_context = quota_context
    return await handle_upload_document_request(request=request, file=file)


@router.post("/agent/stream")
async def run_research_agent_stream(
    request: ResearchAgentRunRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
):
    """Stream research agent progress via Server-Sent Events."""
    async def event_stream():
        progress_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def on_progress(event: dict) -> None:
            await progress_queue.put(event)

        trace_id = f"research_agent_{uuid4().hex}"
        runtime_request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "api_route": "/research/agent/stream",
                    "trace_id": trace_id,
                    "quota_context": quota_context,
                }
            }
        )
        task = asyncio.create_task(
            research_service.run_agent(
                runtime_request,
                graph_runtime=graph_runtime,
                on_progress=on_progress,
            )
        )
        try:
            while not task.done():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=2.0)
                    if event.get("type") == "token":
                        yield f"event: token\ndata: {json.dumps({'text': event['text']}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"event: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
            result = task.result()
            if request.conversation_id:
                research_service.record_agent_turn(
                    request.conversation_id,
                    request=runtime_request,
                    response=result,
                )
            audit_api_call(
                http_request,
                route="/research/agent/stream",
                trace_id=trace_id,
                task_type="research_agent_stream",
                status=result.status,
                metadata={
                    "mode": request.mode,
                    "task_id": result.task.task_id if result.task else request.task_id,
                    "paper_count": len(result.papers),
                    "quota_context": quota_context,
                },
            )
            yield f"event: complete\ndata: {json.dumps(result.model_dump(mode='json'), ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Research agent stream failed")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/agent", response_model=ResearchAgentRunResponse)
async def run_research_agent_entry(
    request: ResearchAgentRunRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchAgentRunResponse:
    return await _run_research_agent_impl(
        request=request,
        http_request=http_request,
        research_service=research_service,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=CANONICAL_AGENT_ROUTE,
    )


@router.post("/agent/run", response_model=ResearchAgentRunResponse, deprecated=True)
async def run_research_agent(
    request: ResearchAgentRunRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchAgentRunResponse:
    return await _run_research_agent_impl(
        request=request,
        http_request=http_request,
        research_service=research_service,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_AGENT_RUN_ROUTE,
    )


@router.post("/agent/chat", response_model=ResearchAgentRunResponse, deprecated=True)
async def chat_research_agent(
    request: ResearchAgentRunRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchAgentRunResponse:
    return await _run_research_agent_impl(
        request=request,
        http_request=http_request,
        research_service=research_service,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_AGENT_CHAT_ROUTE,
    )


@router.get("/conversations", response_model=list[ResearchConversation])
async def list_conversations(
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> list[ResearchConversation]:
    return research_service.list_conversations()


@router.post("/conversations", response_model=ResearchConversationResponse)
async def create_conversation(
    request: CreateResearchConversationRequest,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> ResearchConversationResponse:
    return research_service.create_conversation(request)


@router.get("/conversations/{conversation_id}", response_model=ResearchConversationResponse)
async def get_conversation(
    conversation_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> ResearchConversationResponse:
    try:
        return research_service.get_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research conversation not found: {conversation_id}") from exc


@router.patch("/conversations/{conversation_id}", response_model=ResearchConversationResponse)
async def rename_conversation(
    conversation_id: str,
    request: RenameResearchConversationRequest,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> ResearchConversationResponse:
    try:
        return research_service.rename_conversation(conversation_id, request.title)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research conversation not found: {conversation_id}") from exc


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> None:
    try:
        research_service.delete_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research conversation not found: {conversation_id}") from exc


@router.get("/jobs/{job_id}", response_model=ResearchJob)
async def get_job(
    job_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> ResearchJob:
    try:
        return research_service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research job not found: {job_id}") from exc


@router.post("/reset")
async def reset_research_workspace(
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> dict[str, str]:
    try:
        trace_id = f"research_reset_{uuid4().hex}"
        await research_service.reset_state()
        audit_api_call(
            http_request,
            route="/research/reset",
            trace_id=trace_id,
            task_type="research_reset",
            status="succeeded",
            metadata={"quota_context": quota_context},
        )
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Research workspace reset failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to reset research workspace", exc=exc),
        ) from exc


@router.post("/papers/search", response_model=SearchPapersResponse)
async def search_papers(
    request: SearchPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> SearchPapersResponse:
    try:
        trace_id = f"research_search_{uuid4().hex}"
        response = await research_service.search_papers(request, graph_runtime=graph_runtime)
        if request.conversation_id:
            research_service.record_search_turn(
                request.conversation_id,
                topic=request.topic,
                response=response,
                days_back=request.days_back,
                max_papers=request.max_papers,
                sources=request.sources,
            )
        audit_api_call(
            http_request,
            route="/research/papers/search",
            trace_id=trace_id,
            task_type="research_search",
            status="succeeded",
            metadata={
                "topic": request.topic,
                "sources": request.sources,
                "quota_context": quota_context,
                "paper_count": len(response.papers),
            },
        )
        return response
    except Exception as exc:
        logger.exception("Research paper search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to search literature papers", exc=exc),
        ) from exc


@router.post("/tasks", response_model=ResearchTaskResponse)
async def create_task(
    request: CreateResearchTaskRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchTaskResponse:
    try:
        trace_id = f"research_task_create_{uuid4().hex}"
        response = await research_service.create_task(request, graph_runtime=graph_runtime)
        if request.conversation_id:
            research_service.record_task_turn(request.conversation_id, response=response)
        audit_api_call(
            http_request,
            route="/research/tasks",
            trace_id=trace_id,
            task_type="research_task_create",
            status="succeeded",
            metadata={
                "topic": request.topic,
                "sources": request.sources,
                "quota_context": quota_context,
                "task_id": response.task.task_id,
            },
        )
        return response
    except Exception as exc:
        logger.exception("Research task creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to create literature research task", exc=exc),
        ) from exc


async def _import_papers_impl(
    request: ImportPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    quota_context: dict[str, str] = Depends(build_quota_context),
    *,
    route_path: str,
) -> ImportPapersResponse:
    try:
        trace_id = f"research_import_{uuid4().hex}"
        response = await research_service.import_papers(
            request,
            graph_runtime=graph_runtime,
        )
        if request.conversation_id:
            task_response = research_service.get_task(request.task_id) if request.task_id else None
            research_service.record_import_turn(
                request.conversation_id,
                task_response=task_response,
                import_response=response,
                selected_paper_ids=request.paper_ids,
            )
        audit_api_call(
            http_request,
            route=route_path,
            trace_id=trace_id,
            task_type="research_import",
            status="succeeded" if response.failed_count == 0 else "partial",
            metadata={
                "task_id": request.task_id,
                "paper_ids": request.paper_ids,
                "quota_context": quota_context,
                "imported_count": response.imported_count,
                "skipped_count": response.skipped_count,
                "failed_count": response.failed_count,
            },
        )
        return response
    except Exception as exc:
        logger.exception("Research paper import failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to import research papers into document pipeline", exc=exc),
        ) from exc


@router.post("/papers/import", response_model=ImportPapersResponse, deprecated=True)
async def import_papers(
    request: ImportPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ImportPapersResponse:
    return await _import_papers_impl(
        request=request,
        http_request=http_request,
        research_service=research_service,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=LEGACY_PAPER_IMPORT_ROUTE,
    )


@router.post("/tasks/{task_id}/papers/import", response_model=ImportPapersResponse)
async def import_task_papers(
    task_id: str,
    request: ImportPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ImportPapersResponse:
    return await _import_papers_impl(
        request=request.model_copy(update={"task_id": task_id}),
        http_request=http_request,
        research_service=research_service,
        graph_runtime=graph_runtime,
        quota_context=quota_context,
        route_path=CANONICAL_TASK_IMPORT_ROUTE,
    )


@router.post("/papers/import/jobs", response_model=ResearchJob)
async def import_papers_job(
    request: ImportPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchJob:
    try:
        job = await research_service.start_import_job(request, graph_runtime=graph_runtime)
        audit_api_call(
            http_request,
            route="/research/papers/import/jobs",
            trace_id=job.job_id,
            task_type="research_import_job_create",
            status="queued",
            metadata={
                "task_id": request.task_id,
                "paper_ids": request.paper_ids,
                "conversation_id": request.conversation_id,
                "quota_context": quota_context,
            },
        )
        return job
    except Exception as exc:
        logger.exception("Research import job creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to create research import job", exc=exc),
        ) from exc


@router.post("/tasks/{task_id}/papers/import/jobs", response_model=ResearchJob)
async def import_task_papers_job(
    task_id: str,
    request: ImportPapersRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchJob:
    try:
        request_with_task = request.model_copy(update={"task_id": task_id})
        job = await research_service.start_import_job(request_with_task, graph_runtime=graph_runtime)
        audit_api_call(
            http_request,
            route=CANONICAL_TASK_IMPORT_JOB_ROUTE,
            trace_id=job.job_id,
            task_type="research_import_job_create",
            status="queued",
            metadata={
                "task_id": task_id,
                "paper_ids": request_with_task.paper_ids,
                "conversation_id": request_with_task.conversation_id,
                "quota_context": quota_context,
            },
        )
        return job
    except Exception as exc:
        logger.exception("Research import job creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to create research import job", exc=exc),
        ) from exc


@router.post("/tasks/{task_id}/run", response_model=ResearchTaskResponse)
async def run_task(
    task_id: str,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchTaskResponse:
    try:
        trace_id = f"research_task_run_{uuid4().hex}"
        response = await research_service.run_task(task_id, graph_runtime=graph_runtime)
        audit_api_call(
            http_request,
            route=f"/research/tasks/{task_id}/run",
            trace_id=trace_id,
            task_type="research_task_run",
            status="succeeded",
            metadata={"task_id": task_id, "quota_context": quota_context, "paper_count": len(response.papers)},
        )
        return response
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task not found: {task_id}") from exc
    except Exception as exc:
        logger.exception("Research task run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to run literature research task", exc=exc),
        ) from exc


@router.get("/tasks/{task_id}", response_model=ResearchTaskResponse)
async def get_task(
    task_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> ResearchTaskResponse:
    try:
        return research_service.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task not found: {task_id}") from exc


@router.get("/tasks/{task_id}/report")
async def get_report(
    task_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
) -> dict[str, str]:
    try:
        response = research_service.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task not found: {task_id}") from exc
    if response.report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research report not found for task: {task_id}")
    return {"report_id": response.report.report_id, "markdown": response.report.markdown}


@router.get("/tasks/{task_id}/papers/{paper_id}/figures", response_model=ResearchPaperFigureListResponse)
async def list_paper_figures(
    task_id: str,
    paper_id: str,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
) -> ResearchPaperFigureListResponse:
    try:
        return await research_service.list_paper_figures(
            task_id,
            paper_id,
            graph_runtime=graph_runtime,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/tasks/{task_id}/papers/{paper_id}/figures/analyze",
    response_model=AnalyzeResearchPaperFigureResponse,
)
async def analyze_paper_figure(
    task_id: str,
    paper_id: str,
    request: AnalyzeResearchPaperFigureRequest,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
) -> AnalyzeResearchPaperFigureResponse:
    try:
        return await research_service.analyze_paper_figure(
            task_id,
            paper_id,
            request,
            graph_runtime=graph_runtime,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.patch("/tasks/{task_id}/todos/{todo_id}", response_model=ResearchTaskResponse)
async def update_todo_status(
    task_id: str,
    todo_id: str,
    request: UpdateResearchTodoRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchTaskResponse:
    try:
        trace_id = f"research_todo_update_{uuid4().hex}"
        response = research_service.update_todo_status(task_id, todo_id, request.status)
        audit_api_call(
            http_request,
            route=f"/research/tasks/{task_id}/todos/{todo_id}",
            trace_id=trace_id,
            task_type="research_todo_update",
            status="succeeded",
            metadata={
                "task_id": task_id,
                "todo_id": todo_id,
                "todo_status": request.status,
                "quota_context": quota_context,
            },
        )
        return response
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task or todo not found: {task_id}/{todo_id}") from exc
    except Exception as exc:
        logger.exception("Research todo update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to update research todo status", exc=exc),
        ) from exc


@router.post("/tasks/{task_id}/todos/{todo_id}/search", response_model=ResearchTodoActionResponse)
async def rerun_todo_search(
    task_id: str,
    todo_id: str,
    request: ResearchTodoActionRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchTodoActionResponse:
    try:
        trace_id = f"research_todo_search_{uuid4().hex}"
        response = await research_service.rerun_todo_search(
            task_id,
            todo_id,
            request,
            graph_runtime=graph_runtime,
        )
        audit_api_call(
            http_request,
            route=f"/research/tasks/{task_id}/todos/{todo_id}/search",
            trace_id=trace_id,
            task_type="research_todo_search",
            status="succeeded" if not response.warnings else "partial",
            metadata={
                "task_id": task_id,
                "todo_id": todo_id,
                "quota_context": quota_context,
                "paper_count": len(response.papers),
                "warning_count": len(response.warnings),
            },
        )
        return response
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task or todo not found: {task_id}/{todo_id}") from exc
    except Exception as exc:
        logger.exception("Research todo search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to rerun literature search from research todo", exc=exc),
        ) from exc


@router.post("/tasks/{task_id}/todos/{todo_id}/import", response_model=ResearchTodoActionResponse)
async def import_from_todo(
    task_id: str,
    todo_id: str,
    request: ResearchTodoActionRequest,
    http_request: Request,
    research_service: LiteratureResearchService = Depends(get_literature_research_service),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> ResearchTodoActionResponse:
    try:
        trace_id = f"research_todo_import_{uuid4().hex}"
        response = await research_service.import_from_todo(
            task_id,
            todo_id,
            request,
            graph_runtime=graph_runtime,
        )
        import_result = response.import_result
        audit_api_call(
            http_request,
            route=f"/research/tasks/{task_id}/todos/{todo_id}/import",
            trace_id=trace_id,
            task_type="research_todo_import",
            status="succeeded" if (import_result and import_result.failed_count == 0) else "partial",
            metadata={
                "task_id": task_id,
                "todo_id": todo_id,
                "quota_context": quota_context,
                "paper_count": len(response.papers),
                "imported_count": import_result.imported_count if import_result else 0,
                "skipped_count": import_result.skipped_count if import_result else 0,
                "failed_count": import_result.failed_count if import_result else 0,
            },
        )
        return response
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research task or todo not found: {task_id}/{todo_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Research todo import failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(http_request, fallback="Failed to import papers from research todo", exc=exc),
        ) from exc
