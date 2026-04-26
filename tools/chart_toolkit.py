import logging
import asyncio
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, ImageNotVisibleLLMAdapterError, LLMAdapterError
from adapters.local_runtime import LocalLLMAdapter
from chains.chart_understanding_chain import ChartUnderstandingChain
from domain.schemas.chart import ChartSchema

logger = logging.getLogger(__name__)


class ChartAgentError(RuntimeError):
    """Raised when chart understanding fails."""


class ChartUnderstandInput(BaseModel):
    image_path: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    chart_id: str
    context: dict[str, Any] = Field(default_factory=dict)


def _error_has_cause(exc: BaseException, error_type: type[BaseException]) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, error_type):
            return True
        visited.add(id(current))
        current = current.__cause__
    return False


def explain_chart(chart: ChartSchema) -> str:
    parts: list[str] = []
    parts.append(f"Chart type: {chart.chart_type}.")
    if chart.title:
        parts.append(f"Title: {chart.title}.")
    if chart.caption:
        parts.append(f"Caption: {chart.caption}.")
    if chart.x_axis:
        parts.append(
            "X axis: "
            + ", ".join(
                value
                for value in [
                    chart.x_axis.label,
                    chart.x_axis.unit,
                    f"scale={chart.x_axis.scale}" if chart.x_axis.scale else None,
                ]
                if value
            )
            + "."
        )
    if chart.y_axis:
        parts.append(
            "Y axis: "
            + ", ".join(
                value
                for value in [
                    chart.y_axis.label,
                    chart.y_axis.unit,
                    f"scale={chart.y_axis.scale}" if chart.y_axis.scale else None,
                ]
                if value
            )
            + "."
        )
    if chart.series:
        series_names = ", ".join(series.name for series in chart.series if series.name)
        if series_names:
            parts.append(f"Series: {series_names}.")
    if chart.summary:
        parts.append(f"Summary: {chart.summary}")
    if chart.confidence is not None:
        parts.append(f"Confidence: {chart.confidence:.2f}.")
    return "\n".join(part for part in parts if part.strip())


def chart_to_graph_text(chart: ChartSchema) -> str:
    lines = [
        f"document_id: {chart.document_id}",
        f"page_id: {chart.page_id}",
        f"page_number: {chart.page_number}",
        f"chart_id: {chart.id}",
        f"chart_type: {chart.chart_type}",
    ]
    if chart.title:
        lines.append(f"title: {chart.title}")
    if chart.caption:
        lines.append(f"caption: {chart.caption}")
    if chart.summary:
        lines.append(f"summary: {chart.summary}")
    if chart.x_axis:
        lines.append(f"x_axis: {chart.x_axis.model_dump(mode='json')}")
    if chart.y_axis:
        lines.append(f"y_axis: {chart.y_axis.model_dump(mode='json')}")
    for series in chart.series:
        lines.append(f"series: {series.model_dump(mode='json')}")
    return "\n".join(lines)


def fallback_chart_summary(chart: ChartSchema) -> str | None:
    if not any([chart.title, chart.caption, chart.series, chart.x_axis, chart.y_axis]):
        return None
    return explain_chart(chart)


def visible_chart_text(chart: ChartSchema) -> str:
    lines: list[str] = []
    if chart.title:
        lines.append(f"title: {chart.title}")
    if chart.caption:
        lines.append(f"caption: {chart.caption}")
    if chart.x_axis:
        x_parts = [value for value in [chart.x_axis.label, chart.x_axis.name, chart.x_axis.unit] if value]
        if x_parts:
            lines.append(f"x_axis: {' | '.join(x_parts)}")
        if chart.x_axis.categories:
            lines.append(f"x_axis_categories: {', '.join(chart.x_axis.categories[:12])}")
    if chart.y_axis:
        y_parts = [value for value in [chart.y_axis.label, chart.y_axis.name, chart.y_axis.unit] if value]
        if y_parts:
            lines.append(f"y_axis: {' | '.join(y_parts)}")
    series_names = [series.name for series in chart.series if series.name]
    if series_names:
        lines.append(f"legend: {', '.join(series_names[:12])}")
    if chart.summary:
        lines.append(f"summary: {chart.summary}")
    return "\n".join(lines)


def normalize_chart_result(
    *,
    chart: ChartSchema,
    image_path: str,
    document_id: str,
    page_id: str,
    page_number: int,
    chart_id: str,
    context: dict[str, Any],
) -> ChartSchema:
    metadata = {
        **chart.metadata,
        "image_path": image_path,
        "context": context,
        "parsed_by": "ChartAgent",
    }
    return chart.model_copy(
        update={
            "id": chart.id or chart_id,
            "document_id": chart.document_id or document_id,
            "page_id": chart.page_id or page_id,
            "page_number": chart.page_number or page_number,
            "summary": chart.summary or fallback_chart_summary(chart),
            "metadata": metadata,
        }
    )


def fallback_timeout_chart(
    *,
    image_path: str,
    document_id: str,
    page_id: str,
    page_number: int,
    chart_id: str,
    context: dict[str, Any],
) -> ChartSchema:
    return ChartSchema(
        id=chart_id,
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        chart_type="unknown",
        summary=(
            "Chart vision analysis timed out before the provider returned a structured result. "
            "The image was received by the backend, but no visual interpretation was produced."
        ),
        confidence=0.0,
        metadata={
            "image_path": image_path,
            "context": context,
            "parsed_by": "ChartAgent",
            "fallback_reason": "vision_timeout",
        },
    )


def fallback_error_chart(
    *,
    image_path: str,
    document_id: str,
    page_id: str,
    page_number: int,
    chart_id: str,
    context: dict[str, Any],
    error_message: str,
    fallback_reason: str = "vision_error",
) -> ChartSchema:
    return ChartSchema(
        id=chart_id,
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        chart_type="unknown",
        summary=(
            "Chart vision analysis did not return a valid structured result. "
            "The backend received the image, but the vision model response could not be parsed."
        ),
        confidence=0.0,
        metadata={
            "image_path": image_path,
            "context": context,
            "parsed_by": "ChartAgent",
            "fallback_reason": fallback_reason,
            "error_message": error_message,
        },
    )


class ChartAgent:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        prompt_path: str | Path = "prompts/chart/parse_chart.txt",
        vision_timeout_seconds: float = 90.0,
    ) -> None:
        self.llm_adapter = llm_adapter or LocalLLMAdapter()
        resolved = Path(prompt_path)
        if not resolved.is_absolute():
            resolved = self._PROJECT_ROOT / resolved
        self.prompt_path = resolved
        self.vision_timeout_seconds = max(5.0, float(vision_timeout_seconds))
        self.chain = ChartUnderstandingChain(llm=self.llm_adapter, prompt_path=self.prompt_path)
        self.parse_chart_tool = StructuredTool.from_function(
            coroutine=self.parse_chart,
            name="understand_chart",
            description="Understand a chart image and return structured chart information.",
            args_schema=ChartUnderstandInput,
        )
        self.explain_chart_chain = RunnableLambda(explain_chart)
        self.graph_text_chain = RunnableLambda(chart_to_graph_text)

    async def parse_chart(
        self,
        image_path: str,
        document_id: str,
        page_id: str,
        page_number: int,
        chart_id: str,
        context: dict[str, Any] | None = None,
    ) -> ChartSchema:
        try:
            chart = await asyncio.wait_for(
                self.chain.ainvoke(
                    image_path=image_path,
                    document_id=document_id,
                    page_id=page_id,
                    page_number=page_number,
                    chart_id=chart_id,
                    context=context,
                ),
                timeout=self.vision_timeout_seconds,
            )
            return normalize_chart_result(
                chart=chart,
                image_path=image_path,
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                chart_id=chart_id,
                context=context or {},
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Chart vision analysis timed out; returning fallback chart result",
                extra={
                    "document_id": document_id,
                    "page_id": page_id,
                    "chart_id": chart_id,
                    "timeout_seconds": self.vision_timeout_seconds,
                },
            )
            return fallback_timeout_chart(
                image_path=image_path,
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                chart_id=chart_id,
                context=context or {},
            )
        except (LLMAdapterError, OSError, ValueError) as exc:
            message = str(exc)
            image_not_visible = _error_has_cause(exc, ImageNotVisibleLLMAdapterError) or "image was not visible" in message
            log_method = logger.warning if image_not_visible or "vision_error" in message else logger.exception
            log_method(
                "Failed to parse chart image: document_id=%s page_id=%s chart_id=%s reason=%s",
                document_id,
                page_id,
                chart_id,
                "vision_image_not_visible" if image_not_visible else "vision_error",
                extra={"document_id": document_id, "page_id": page_id, "chart_id": chart_id},
            )
            return fallback_error_chart(
                image_path=image_path,
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                chart_id=chart_id,
                context=context or {},
                error_message=str(exc),
                fallback_reason="vision_image_not_visible" if image_not_visible else "vision_error",
            )

    def explain_chart(self, chart: ChartSchema) -> str:
        return explain_chart(chart)

    def to_graph_text(self, chart: ChartSchema) -> str:
        return chart_to_graph_text(chart)

    async def extract_visible_text(
        self,
        *,
        image_path: str,
        context: dict[str, Any] | None = None,
        chart: ChartSchema | None = None,
    ) -> str:
        if hasattr(self.llm_adapter, "answer_image_question"):
            try:
                text = await self.llm_adapter.answer_image_question(
                    image_path=image_path,
                    question=(
                        "Extract only the visible text that is useful for chart understanding. "
                        "Return plain text with short labeled lines for title, caption, axis labels, legend, tick labels, annotations, and any visible notes. "
                        "Do not infer trends or conclusions. If text is unreadable, say so briefly."
                    ),
                    system_prompt="You are an OCR assistant for scientific charts and document figures. Read only visible text from the image.",
                    context=context or {},
                    history=[],
                )
                return text.strip()
            except Exception:
                logger.warning("Chart OCR-style extraction failed; falling back to structured chart fields", exc_info=True)
        return visible_chart_text(chart) if chart is not None else ""

    async def ask_chart(
        self,
        *,
        image_path: str,
        question: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if hasattr(self.llm_adapter, "answer_image_question"):
            return await self.llm_adapter.answer_image_question(
                image_path=image_path,
                question=question,
                context=context or {},
                history=history or [],
            )
        raise ChartAgentError("Current chart vision adapter does not support image question answering")


ChartTools = ChartAgent
ChartToolsError = ChartAgentError
