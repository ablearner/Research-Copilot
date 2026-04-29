from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.chart import ChartSchema

logger = logging.getLogger(__name__)

_CHART_ANALYSIS_PROMPT = (
    "你是科研助手里的图表分析技能。请基于结构化图表信息和论文上下文回答用户问题。\n\n"
    "用户问题：{question}\n"
    "论文/图表上下文：\n{figure_context}\n\n"
    "图表 JSON：\n{chart_json}\n\n"
    "要求：\n"
    "- 用中文回答\n"
    "- 优先结合图题、图注、论文标题、页码和视觉模型解析结果\n"
    "- 明确说明图表主要表达什么、能支持什么结论、还缺什么上下文\n"
    "- 如果图表字段和图注/问题之间存在不确定性，要明确标出不确定性\n"
    "- 不要假装看到了结构化字段以外的信息\n"
    "- 返回结构化字段"
)


class _PaperChartAnalysisLLMResponse(BaseModel):
    answer: str
    key_points: list[str] = Field(default_factory=list)


class PaperChartAnalyzer:
    name = "PaperChartAnalyzer"

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def analyze_async(
        self,
        *,
        chart: ChartSchema,
        question: str | None = None,
        figure_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        resolved_question = (question or "请解释这张图表的关键信息。").strip()
        context = figure_context or {}
        if self.llm_adapter is not None:
            try:
                result = await self.llm_adapter.generate_structured(
                    prompt=_CHART_ANALYSIS_PROMPT,
                    input_data={
                        "question": resolved_question,
                        "figure_context": context,
                        "chart_json": chart.model_dump_json(indent=2),
                    },
                    response_model=_PaperChartAnalysisLLMResponse,
                )
                if result.answer.strip():
                    return (
                        result.answer.strip(),
                        [point.strip() for point in result.key_points if point.strip()][:6],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chart analysis skill fell back to heuristic summary: %s", exc)
        return self._heuristic_analysis(chart=chart, question=resolved_question, figure_context=context)

    def _heuristic_analysis(
        self,
        *,
        chart: ChartSchema,
        question: str,
        figure_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        context = figure_context or {}
        key_points: list[str] = []
        paper_title = str(context.get("paper_title") or "").strip()
        figure_title = str(context.get("figure_title") or "").strip()
        figure_caption = str(context.get("figure_caption") or "").strip()
        page_number = context.get("page_number")
        if paper_title:
            key_points.append(f"论文：{paper_title}")
        if figure_title:
            key_points.append(f"图题：{figure_title}")
        if figure_caption:
            key_points.append(f"图注：{figure_caption}")
        if chart.summary:
            key_points.append(chart.summary)
        if chart.title:
            key_points.append(f"图题：{chart.title}")
        if chart.caption:
            key_points.append(f"图注：{chart.caption}")
        if chart.x_axis and (chart.x_axis.label or chart.x_axis.name):
            key_points.append(f"横轴：{chart.x_axis.label or chart.x_axis.name}")
        if chart.y_axis and (chart.y_axis.label or chart.y_axis.name):
            key_points.append(f"纵轴：{chart.y_axis.label or chart.y_axis.name}")
        series_names = [series.name for series in chart.series if series.name]
        if series_names:
            key_points.append(f"系列：{', '.join(series_names[:5])}")

        answer_lines = [f"围绕“{question}”，我先基于这张图的结构化信息做解释。"]
        if paper_title:
            answer_lines.append(f"它来自论文“{paper_title}”。")
        if page_number:
            answer_lines.append(f"位置大致在第 {page_number} 页。")
        if figure_caption:
            answer_lines.append(f"图注线索是：{figure_caption}")
        if chart.summary:
            answer_lines.append(f"这张图主要表达的是：{chart.summary}")
        else:
            answer_lines.append(f"这是一张 `{chart.chart_type}` 类型的图表。")
        if chart.title:
            answer_lines.append(f"标题是“{chart.title}”。")
        if chart.caption:
            answer_lines.append(f"图注提供的信息是：{chart.caption}")
        if chart.x_axis or chart.y_axis:
            axis_parts: list[str] = []
            if chart.x_axis and (chart.x_axis.label or chart.x_axis.name):
                axis_parts.append(f"横轴是 {chart.x_axis.label or chart.x_axis.name}")
            if chart.y_axis and (chart.y_axis.label or chart.y_axis.name):
                axis_parts.append(f"纵轴是 {chart.y_axis.label or chart.y_axis.name}")
            if axis_parts:
                answer_lines.append("；".join(axis_parts) + "。")
        if series_names:
            answer_lines.append(f"图中可见的主要系列包括：{', '.join(series_names[:5])}。")
        answer_lines.append("当前解释主要依赖图表结构化识别结果；如果你想继续追问趋势变化、峰值位置或和正文结论是否一致，我可以接着分析。")
        return ("\n".join(answer_lines), key_points[:6])
