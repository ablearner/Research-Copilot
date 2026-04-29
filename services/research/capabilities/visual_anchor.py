from __future__ import annotations

import re
from typing import Any, Callable

from domain.schemas.research import PaperCandidate, ResearchPaperFigurePreview


_RESULT_HINT_TERMS = {
    "result",
    "results",
    "实验",
    "结果",
    "性能",
    "导航",
    "轨迹",
    "效率",
    "成功率",
    "success",
    "rate",
    "efficiency",
    "trajectory",
    "navigation",
    "ablation",
    "benchmark",
}

_FIGURE_REFERENCE_TOKENS = ("图", "figure", "fig", "chart")


def _extract_figure_number_hints(text: str) -> set[int]:
    hints: set[int] = set()
    for pattern in (
        r"(?:figure|fig\.?|图|图表)\s*[\-#:：]?\s*(\d{1,3})",
        r"第\s*(\d{1,3})\s*(?:张|个)?\s*(?:图|图表)",
    ):
        for match in re.finditer(pattern, text.lower()):
            try:
                hints.add(int(match.group(1)))
            except (TypeError, ValueError):
                continue
    return hints


def _extract_ordinal_hints(text: str) -> set[int]:
    hints: set[int] = set()
    ordinal_words = {
        "第一": 1,
        "第二": 2,
        "第三": 3,
        "第四": 4,
        "第五": 5,
        "第六": 6,
        "第七": 7,
        "第八": 8,
        "第九": 9,
        "第十": 10,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }
    lowered = text.lower()
    for marker, value in ordinal_words.items():
        if marker in lowered and any(token in lowered for token in _FIGURE_REFERENCE_TOKENS):
            hints.add(value)
    return hints


def _keyword_terms(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower())
    terms = [term for term in normalized.split() if term]
    return [term for term in terms if len(term) >= 2 or re.search(r"[\u4e00-\u9fff]", term)]


class VisualAnchor:
    name = "VisualAnchor"

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def infer_cached_visual_anchor(
        self,
        *,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
        load_cached_figure_payload: Callable[..., dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        candidate_papers = list(papers)
        if len(document_ids) == 1:
            matched_paper = next(
                (
                    paper
                    for paper in candidate_papers
                    if str(paper.metadata.get("document_id") or "").strip() == document_ids[0]
                ),
                None,
            )
            if matched_paper is not None:
                candidate_papers = [matched_paper]
        if not candidate_papers:
            return None

        question_terms = _keyword_terms(question)
        figure_number_hints = _extract_figure_number_hints(question)
        ordinal_hints = _extract_ordinal_hints(question)
        ranked_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        global_index = 0
        for paper in candidate_papers:
            cache = load_cached_figure_payload(paper=paper)
            if cache is None:
                continue
            figures = cache.get("figures")
            if not isinstance(figures, list):
                continue
            for item in figures:
                if not isinstance(item, dict):
                    continue
                image_path = str(item.get("image_path") or "").strip()
                if not image_path:
                    continue
                candidate_text_parts = [
                    str(item.get("title") or "").strip(),
                    str(item.get("caption") or "").strip(),
                    str(item.get("chart_id") or "").strip(),
                    str(item.get("page_id") or "").strip(),
                    paper.title.strip(),
                ]
                metadata = item.get("metadata")
                if isinstance(metadata, dict):
                    candidate_text_parts.extend(
                        str(metadata.get(key) or "").strip()
                        for key in ("title", "caption", "summary", "label")
                    )
                candidate_text = " ".join(part for part in candidate_text_parts if part)
                candidate_terms = set(_keyword_terms(candidate_text))
                overlap_score = sum(
                    3 if re.search(r"[\u4e00-\u9fff]", term) else 2
                    for term in question_terms
                    if term in candidate_terms
                )
                field_bonus = 0
                lowered_title = str(item.get("title") or "").lower()
                lowered_caption = str(item.get("caption") or "").lower()
                lowered_paper_title = str(paper.title or "").lower()
                candidate_source = str(item.get("source") or "").strip().lower()
                for term in question_terms:
                    if term in lowered_title:
                        field_bonus += 3
                    if term in lowered_caption:
                        field_bonus += 4
                    if term in lowered_paper_title:
                        field_bonus += 1
                source_bonus = 6 if candidate_source == "chart_candidate" else -8
                caption_bonus = 2 if str(item.get("caption") or "").strip() else 0
                title_bonus = 1 if str(item.get("title") or "").strip() else 0
                page_penalty = 0
                raw_page_number = item.get("page_number")
                try:
                    resolved_page_number = int(raw_page_number) if raw_page_number is not None else None
                except (TypeError, ValueError):
                    resolved_page_number = None
                if candidate_source == "page_fallback" and resolved_page_number == 1:
                    page_penalty = -4
                result_bonus = 0
                if any(term in _RESULT_HINT_TERMS for term in question_terms):
                    if any(hint in lowered_title or hint in lowered_caption for hint in _RESULT_HINT_TERMS):
                        result_bonus += 4
                figure_number_bonus = self._figure_number_bonus(
                    figure_number_hints=figure_number_hints,
                    item=item,
                    metadata=metadata if isinstance(metadata, dict) else {},
                    page_number=resolved_page_number,
                )
                ordinal_bonus = 25 if ordinal_hints and global_index + 1 in ordinal_hints else 0
                score = (
                    overlap_score
                    + field_bonus
                    + source_bonus
                    + caption_bonus
                    + title_bonus
                    + result_bonus
                    + figure_number_bonus
                    + ordinal_bonus
                    + page_penalty
                    - (global_index * 0.01)
                )
                anchor = {
                    "image_path": image_path,
                    "anchor_source": "paper_figure_cache",
                    "paper_id": paper.paper_id,
                }
                document_id = str(item.get("document_id") or paper.metadata.get("document_id") or "").strip()
                if document_id:
                    anchor["document_id"] = document_id
                figure_id = str(item.get("figure_id") or "").strip()
                if figure_id:
                    anchor["figure_id"] = figure_id
                page_id = str(item.get("page_id") or "").strip()
                if page_id:
                    anchor["page_id"] = page_id
                page_number = resolved_page_number
                if page_number is not None and page_number >= 1:
                    anchor["page_number"] = page_number
                chart_id = str(item.get("chart_id") or "").strip()
                if chart_id:
                    anchor["chart_id"] = chart_id
                ranked_candidates.append((score, anchor, item))
                global_index += 1

        if not ranked_candidates:
            return None
        ranked_candidates.sort(key=lambda value: value[0], reverse=True)
        # Keep explicit figure references deterministic. Everything else can be
        # treated as candidate recall plus LLM reranking.
        if figure_number_hints or ordinal_hints:
            return {
                **ranked_candidates[0][1],
                "anchor_selection": "deterministic_figure_reference",
                "anchor_rationale": "Matched an explicit figure number or ordinal reference in the user question.",
            }
        reranked_anchor = await self._llm_rerank_cached_figures(
            question=question,
            ranked_candidates=ranked_candidates[:3],
        )
        if reranked_anchor is not None:
            return reranked_anchor
        return ranked_candidates[0][1]

    def _figure_number_bonus(
        self,
        *,
        figure_number_hints: set[int],
        item: dict[str, Any],
        metadata: dict[str, Any],
        page_number: int | None,
    ) -> int:
        if not figure_number_hints:
            return 0
        searchable = " ".join(
            str(value or "")
            for value in [
                item.get("figure_id"),
                item.get("chart_id"),
                item.get("title"),
                item.get("caption"),
                metadata.get("title"),
                metadata.get("caption"),
                metadata.get("label"),
                metadata.get("figure_label"),
            ]
        ).lower()
        for number in figure_number_hints:
            patterns = [
                rf"(?:figure|fig\.?|图|图表)\s*[\-#:：]?\s*{number}\b",
                rf"(?:chart|figure|fig|图|图表)[_\- ]?{number}\b",
                rf"\b{number}\b",
            ]
            if any(re.search(pattern, searchable) for pattern in patterns):
                return 40
            if page_number == number:
                return 8
        return 0

    def resolve_visual_anchor_figure(
        self,
        *,
        papers: list[PaperCandidate],
        qa_metadata: dict[str, Any],
        load_cached_figure_payload: Callable[..., dict[str, Any] | None],
    ) -> ResearchPaperFigurePreview | None:
        visual_anchor = qa_metadata.get("visual_anchor")
        if not isinstance(visual_anchor, dict):
            return None
        figure_id = str(visual_anchor.get("figure_id") or "").strip()
        chart_id = str(visual_anchor.get("chart_id") or "").strip()
        page_id = str(visual_anchor.get("page_id") or "").strip()
        image_path = str(visual_anchor.get("image_path") or "").strip()
        for paper in papers:
            cache = load_cached_figure_payload(paper=paper)
            if cache is None:
                continue
            figures = cache.get("figures")
            if not isinstance(figures, list):
                continue
            for item in figures:
                if not isinstance(item, dict):
                    continue
                item_figure_id = str(item.get("figure_id") or "").strip()
                item_chart_id = str(item.get("chart_id") or "").strip()
                item_page_id = str(item.get("page_id") or "").strip()
                item_image_path = str(item.get("image_path") or "").strip()
                if figure_id and item_figure_id != figure_id:
                    continue
                if not figure_id and chart_id and item_chart_id != chart_id:
                    continue
                if not figure_id and page_id and item_page_id != page_id:
                    continue
                if not figure_id and not chart_id and image_path and item_image_path != image_path:
                    continue
                try:
                    preview = ResearchPaperFigurePreview.model_validate(item)
                    enriched_metadata = dict(preview.metadata)
                    if isinstance(visual_anchor.get("anchor_selection"), str):
                        enriched_metadata["anchor_selection"] = str(visual_anchor["anchor_selection"])
                    if isinstance(visual_anchor.get("anchor_rationale"), str):
                        enriched_metadata["anchor_rationale"] = str(visual_anchor["anchor_rationale"])
                    if isinstance(visual_anchor.get("anchor_source"), str):
                        enriched_metadata["anchor_source"] = str(visual_anchor["anchor_source"])
                    return preview.model_copy(update={"metadata": enriched_metadata})
                except Exception:
                    continue
        return None

    async def _llm_rerank_cached_figures(
        self,
        *,
        question: str,
        ranked_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any] | None:
        if self.llm_adapter is None or len(ranked_candidates) < 2:
            return None
        from pydantic import BaseModel

        class _FigureSelectionLLMResponse(BaseModel):
            figure_id: str
            confidence: float = 0.0
            rationale: str = ""

        prompt = (
            "你是科研助手里的论文图表选择器。请从候选图表中选出最适合回答用户当前问题的一张。\n\n"
            "用户问题：{question}\n"
            "候选图表：\n{candidates_json}\n\n"
            "要求：\n"
            "- 优先选择和问题最直接相关的 title、caption、axis、summary 线索\n"
            "- 如果用户提到 Figure/Fig./图/第几张图，优先匹配对应编号、页码、图注或 chart_id\n"
            "- 如果问题问实验效果、消融、性能、结果，优先选择标题或图注含 evaluation/results/ablation/performance/实验/结果 的图\n"
            "- 如果没有完美匹配，也要选最可能相关的一张\n"
            "- confidence 表示你对选图的确信度，0 到 1\n"
            "- 仅返回结构化字段"
        )
        candidate_payload = [
            {
                "rank": index + 1,
                "figure_id": anchor.get("figure_id"),
                "paper_id": anchor.get("paper_id"),
                "chart_id": anchor.get("chart_id"),
                "page_id": anchor.get("page_id"),
                "page_number": anchor.get("page_number"),
                "title": item.get("title"),
                "caption": item.get("caption"),
                "source": item.get("source"),
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                "heuristic_score": round(score, 4),
            }
            for index, (score, anchor, item) in enumerate(ranked_candidates)
        ]
        try:
            result = await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data={
                    "question": question.strip(),
                    "candidates_json": candidate_payload,
                },
                response_model=_FigureSelectionLLMResponse,
            )
        except Exception:
            return None
        selected_figure_id = result.figure_id.strip()
        if not selected_figure_id:
            return None
        for _score, anchor, _item in ranked_candidates:
            if str(anchor.get("figure_id") or "").strip() == selected_figure_id:
                return {
                    **anchor,
                    "anchor_selection": "llm_rerank",
                    "anchor_rationale": result.rationale.strip(),
                    "anchor_confidence": result.confidence,
                }
        return None
