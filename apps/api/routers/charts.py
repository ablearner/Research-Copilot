import logging
import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.api.audit import audit_api_call
from apps.api.deps import get_graph_runtime
from apps.api.security import build_quota_context, require_api_key
from rag_runtime.schemas import ChartUnderstandingResult
from rag_runtime.runtime import RagRuntime
from domain.schemas.chart import ChartSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/charts", tags=["charts"])


class UnderstandChartRequest(BaseModel):
    image_path: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    chart_id: str
    context: dict[str, Any] = Field(default_factory=dict)


class UnderstandChartResponse(BaseModel):
    status: str
    result: ChartUnderstandingResult


class AskChartRequest(BaseModel):
    image_path: str
    question: str
    session_id: str | None = None
    document_id: str | None = None
    page_id: str | None = None
    page_number: int = Field(default=1, ge=1)
    chart_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class AskChartResponse(BaseModel):
    status: str
    answer: str
    session_id: str
    confidence: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _fallback_chart_result(request: UnderstandChartRequest, reason: str, detail: str | None = None) -> ChartUnderstandingResult:
    chart = ChartSchema(
        id=request.chart_id,
        document_id=request.document_id,
        page_id=request.page_id,
        page_number=request.page_number,
        chart_type="unknown",
        summary="图表结构化理解暂时不可用，但你仍然可以使用图表追问功能直接围绕图片提问。",
        confidence=0.0,
        metadata={"image_path": request.image_path, "fallback_reason": reason, "error_detail": detail},
    )
    return ChartUnderstandingResult(
        chart=chart,
        graph_text=f"chart_type: unknown\nsummary: {chart.summary}",
        metadata={"fallback_reason": reason, "error_detail": detail, "tool_traces": []},
    )


def _chart_evidence(request: AskChartRequest, *, history_count: int, answer: str) -> dict[str, Any]:
    return {
        "source": "vision_model",
        "image_path": request.image_path,
        "document_id": request.document_id,
        "page_id": request.page_id,
        "page_number": request.page_number,
        "chart_id": request.chart_id,
        "question": request.question,
        "history_turn_count": history_count,
        "answer_preview": answer[:240],
    }


def _estimate_chart_confidence(answer: str) -> float:
    lowered = answer.lower()
    if any(token in lowered for token in ["无法", "看不", "not readable", "cannot", "不确定"]):
        return 0.55
    if len(answer.strip()) < 20:
        return 0.45
    return 0.78


@router.post("/understand", response_model=UnderstandChartResponse)
async def understand_chart(
    request: UnderstandChartRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> UnderstandChartResponse:
    try:
        trace_id = f"chart_{uuid4().hex}"
        state = await graph_runtime.invoke(
            {
                "request_id": trace_id,
                "thread_id": trace_id,
                "task_type": "chart_understand",
                "user_input": request.image_path,
                "image_path": request.image_path,
                "document_id": request.document_id,
                "document_ids": [request.document_id],
                "page_id": request.page_id,
                "page_number": request.page_number,
                "chart_id": request.chart_id,
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
                    "api_route": "/charts/understand",
                    "trace_id": trace_id,
                    "quota_context": quota_context,
                    **request.context,
                },
                "retrieval_mode": "hybrid",
                "top_k": 10,
                "filters": {},
            }
        )
        chart_result = state.get("chart_result")
        if not chart_result:
            fallback_chart = await graph_runtime.chart_tools.parse_chart(
                image_path=request.image_path,
                document_id=request.document_id,
                page_id=request.page_id,
                page_number=request.page_number,
                chart_id=request.chart_id,
                context=request.context,
            )
            chart_result = {
                "chart": fallback_chart,
                "graph_text": graph_runtime.chart_tools.to_graph_text(fallback_chart),
                "metadata": {"image_path": request.image_path, "fallback_reason": "router_state_missing_chart_result"},
            }
        result = ChartUnderstandingResult(
            chart=chart_result["chart"],
            graph_text=chart_result["graph_text"],
            metadata={"tool_traces": state.get("tool_traces", []), **chart_result.get("metadata", {})},
        )
        audit_api_call(
            http_request,
            route="/charts/understand",
            trace_id=trace_id,
            task_type="chart_understand",
            status="succeeded",
            metadata={"chart_id": request.chart_id, **state.get("metadata", {})},
        )
        return UnderstandChartResponse(status="succeeded", result=result)
    except Exception as exc:
        logger.warning("Understand chart request fell back", extra={"chart_id": request.chart_id, "error": str(exc)})
        result = _fallback_chart_result(request, reason="chart_understand_error", detail=str(exc))
        return UnderstandChartResponse(status="fallback", result=result)


@router.post("/ask", response_model=AskChartResponse)
async def ask_chart(
    request: AskChartRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> AskChartResponse:
    try:
        started_at = time.perf_counter()
        session_id = request.session_id or f"chart_session_{uuid4().hex}"
        history = graph_runtime.session_memory.chart_history(session_id, image_path=request.image_path)
        answer = await graph_runtime.chart_tools.ask_chart(
            image_path=request.image_path,
            question=request.question,
            context={
                "document_id": request.document_id,
                "page_id": request.page_id,
                "page_number": request.page_number,
                "chart_id": request.chart_id,
                "quota_context": quota_context,
                **request.context,
            },
            history=history,
        )
        graph_runtime.session_memory.append_chart_turn(
            session_id=session_id,
            image_path=request.image_path,
            question=request.question,
            answer=answer,
            chart_id=request.chart_id,
            document_id=request.document_id,
            page_id=request.page_id,
        )
        audit_api_call(
            http_request,
            route="/charts/ask",
            trace_id=session_id,
            task_type="chart_qa",
            status="succeeded",
            metadata={"chart_id": request.chart_id, "session_id": session_id},
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        confidence = _estimate_chart_confidence(answer)
        return AskChartResponse(
            status="succeeded",
            answer=answer,
            session_id=session_id,
            confidence=confidence,
            evidence=_chart_evidence(request, history_count=len(history), answer=answer),
            metadata={"history_turn_count": len(history), "quota_context": quota_context, "latency_ms": latency_ms, "trace_type": "chart_qa"},
        )
    except Exception as exc:
        logger.warning("Ask chart request fell back", extra={"chart_id": request.chart_id, "error": str(exc)})
        session_id = request.session_id or f"chart_session_{uuid4().hex}"
        answer = "图表问答暂时不可用，请稍后重试，或换一个更具体的问题。"
        return AskChartResponse(
            status="fallback",
            answer=answer,
            session_id=session_id,
            confidence=0.0,
            evidence=_chart_evidence(request, history_count=0, answer=answer),
            metadata={"fallback_reason": "chart_qa_error", "error_detail": str(exc), "quota_context": quota_context},
        )


@router.post("/ask/stream")
async def ask_chart_stream(
    request: AskChartRequest,
    http_request: Request,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
    quota_context: dict[str, str] = Depends(build_quota_context),
) -> StreamingResponse:
    async def event_stream():
        session_id = request.session_id or f"chart_session_{uuid4().hex}"
        try:
            started_at = time.perf_counter()
            yield f"event: start\ndata: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            history = graph_runtime.session_memory.chart_history(session_id, image_path=request.image_path)
            answer = await graph_runtime.chart_tools.ask_chart(
                image_path=request.image_path,
                question=request.question,
                context={
                    "document_id": request.document_id,
                    "page_id": request.page_id,
                    "page_number": request.page_number,
                    "chart_id": request.chart_id,
                    "quota_context": quota_context,
                    **request.context,
                },
                history=history,
            )
            graph_runtime.session_memory.append_chart_turn(
                session_id=session_id,
                image_path=request.image_path,
                question=request.question,
                answer=answer,
                chart_id=request.chart_id,
                document_id=request.document_id,
                page_id=request.page_id,
            )
            audit_api_call(
                http_request,
                route="/charts/ask/stream",
                trace_id=session_id,
                task_type="chart_qa_stream",
                status="succeeded",
                metadata={"chart_id": request.chart_id, "session_id": session_id},
            )
            for index in range(0, len(answer), 8):
                chunk = answer[index : index + 8]
                yield f"event: token\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            done_payload = {
                "answer": answer,
                "session_id": session_id,
                "confidence": _estimate_chart_confidence(answer),
                "evidence": _chart_evidence(request, history_count=len(history), answer=answer),
                "metadata": {"history_turn_count": len(history), "quota_context": quota_context, "latency_ms": latency_ms, "trace_type": "chart_qa_stream"},
            }
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.warning("Stream ask chart request fell back", extra={"chart_id": request.chart_id, "error": str(exc)})
            answer = "图表问答暂时不可用，请稍后重试，或换一个更具体的问题。"
            for index in range(0, len(answer), 8):
                yield f"event: token\ndata: {json.dumps({'delta': answer[index:index+8]}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            done_payload = {
                "answer": answer,
                "session_id": session_id,
                "confidence": 0.0,
                "evidence": _chart_evidence(request, history_count=0, answer=answer),
                "metadata": {"fallback_reason": "chart_qa_stream_error", "error_detail": str(exc), "quota_context": quota_context},
            }
            yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
