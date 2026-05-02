"""QA lifecycle mixin for LiteratureResearchService.

The service entry point delegates execution to ResearchQAAgent.execute_qa()
and keeps only route helpers, follow-up persistence, and quality checks here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskRequest,
    ResearchTaskAskResponse,
    ResearchTodoItem,
)
from tools.research.qa_decisions import is_insufficient_answer
from tools.research.qa_schemas import ResearchQARouteDecision
from domain.research_workspace import build_workspace_from_task

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.research.paper_selector import PaperSelectionScope


class QARoutingMixin:
    """Mixin providing QA route helpers and follow-up lifecycle methods.

    Assumes the host class exposes the following attributes (provided by
    ``LiteratureResearchService.__init__``):

    * ``report_service``
    * ``paper_selector_service``
    * ``qa_routing_skill``
    * ``user_intent_resolver``
    * ``chart_analysis_agent``
    * ``memory_gateway``
    * ``research_collection_qa_capability``
    * ``paper_search_service``
    * ``build_execution_context(...)``
    * ``save_task_state(...)``
    * ``_load_cached_figure_payload(...)``
    * ``list_paper_figures(...)``
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def ask_task_collection(
        self,
        task_id: str,
        request: ResearchTaskAskRequest,
        *,
        graph_runtime,
    ) -> ResearchTaskAskResponse:
        from agents.research_qa_agent import ResearchQAAgent

        result = await ResearchQAAgent().execute_qa(
            self,
            task_id,
            request,
            graph_runtime=graph_runtime,
        )
        return result.response

    # ------------------------------------------------------------------
    # Route selection
    # ------------------------------------------------------------------

    async def _select_qa_route(
        self,
        *,
        question: str,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchQARouteDecision:
        visual_anchor = self._extract_visual_anchor(request=request, metadata=metadata)
        route_result = await self.qa_routing_skill.classify_async(
            question=question,
            scope_mode=scope_mode,
            paper_ids=paper_ids,
            document_ids=document_ids,
            has_visual_anchor=visual_anchor is not None,
        )
        return ResearchQARouteDecision(
            route=route_result.route,
            confidence=route_result.confidence,
            rationale=route_result.rationale,
            visual_anchor=visual_anchor if route_result.route == "chart_drilldown" else None,
        )

    # ------------------------------------------------------------------
    # Visual anchor helpers
    # ------------------------------------------------------------------

    def _restore_visual_anchor_from_workspace(
        self,
        *,
        task: ResearchTask,
        report: ResearchReport | None,
    ) -> dict[str, Any] | None:
        workspace_candidates = [
            task.workspace.metadata if isinstance(task.workspace.metadata, dict) else {},
            report.workspace.metadata if report is not None and isinstance(report.workspace.metadata, dict) else {},
        ]
        for metadata in workspace_candidates:
            anchor = metadata.get("last_visual_anchor")
            if not isinstance(anchor, dict):
                continue
            image_path = str(anchor.get("image_path") or "").strip()
            if not image_path:
                continue
            restored: dict[str, Any] = {"image_path": image_path}
            page_id = str(anchor.get("page_id") or "").strip()
            if page_id:
                restored["page_id"] = page_id
            chart_id = str(anchor.get("chart_id") or "").strip()
            if chart_id:
                restored["chart_id"] = chart_id
            raw_page_number = anchor.get("page_number")
            try:
                page_number = int(raw_page_number) if raw_page_number is not None else None
            except (TypeError, ValueError):
                page_number = None
            if page_number is not None and page_number >= 1:
                restored["page_number"] = page_number
            for key in ("figure_id", "anchor_source", "anchor_selection", "anchor_rationale"):
                value = anchor.get(key)
                if isinstance(value, str) and value.strip():
                    restored[key] = value.strip()
            return restored
        return None

    def _extract_visual_anchor(
        self,
        *,
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        image_path = str(request.image_path or (metadata or {}).get("image_path") or "").strip()
        if not image_path:
            return None
        anchor = {"image_path": image_path}
        page_id = request.page_id or (metadata or {}).get("page_id")
        if page_id:
            anchor["page_id"] = str(page_id)
        try:
            raw_page_number = (
                request.page_number
                if request.page_number is not None
                else (metadata or {}).get("page_number")
            )
            page_number = int(raw_page_number) if raw_page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        if page_number is not None and page_number >= 1:
            anchor["page_number"] = page_number
        chart_id = request.chart_id or (metadata or {}).get("chart_id")
        if chart_id:
            anchor["chart_id"] = str(chart_id)
        return anchor

    async def _infer_cached_visual_anchor(
        self,
        *,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
    ) -> dict[str, Any] | None:
        return await self.chart_analysis_agent.infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )

    async def _infer_or_discover_visual_anchor(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
        graph_runtime: Any,
    ) -> dict[str, Any] | None:
        inferred = await self._infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
        )
        if inferred is not None:
            return inferred
        discovered = await self._discover_figures_for_scope(
            task_id=task_id,
            papers=papers,
            document_ids=document_ids,
            graph_runtime=graph_runtime,
        )
        if not discovered:
            return None
        refreshed_papers = self.report_service.load_papers(task_id)
        scoped_papers = [
            paper for paper in refreshed_papers
            if not papers or paper.paper_id in {item.paper_id for item in papers}
        ] or refreshed_papers
        return await self._infer_cached_visual_anchor(
            papers=scoped_papers,
            document_ids=document_ids,
            question=question,
        )

    async def _discover_figures_for_scope(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        graph_runtime: Any,
    ) -> bool:
        allowed_document_ids = {item for item in document_ids if item}
        discovered = False
        for paper in papers:
            document_id = str(paper.metadata.get("document_id") or "").strip()
            storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
            if paper.ingest_status != "ingested" or not document_id or not storage_uri:
                continue
            if allowed_document_ids and document_id not in allowed_document_ids:
                continue
            if self._load_cached_figure_payload(paper=paper) is not None:
                discovered = True
                continue
            try:
                await self.list_paper_figures(
                    task_id,
                    paper.paper_id,
                    graph_runtime=graph_runtime,
                )
                discovered = True
            except Exception:
                continue
        return discovered

    def _refresh_existing_pool(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        merged = self.paper_search_service._dedupe([*existing_papers, *incoming_papers])
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged,
            max_papers=max(len(merged), 1),
        )


    # ------------------------------------------------------------------
    # Follow-up & quality helpers
    # ------------------------------------------------------------------

    def _apply_qa_follow_up(
        self,
        *,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        qa,
        papers: list[PaperCandidate],
        document_ids: list[str],
        scope: PaperSelectionScope,
    ) -> tuple[ResearchTask, ResearchReport]:
        now = datetime.now(UTC).isoformat()
        report = self.report_service.load_report(task.task_id, task.report_id)
        if report is None:
            report = ResearchReport(
                report_id=task.report_id or f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                task_id=task.task_id,
                topic=task.topic,
                generated_at=now,
                markdown=f"# 文献调研报告：{task.topic}",
                paper_count=task.paper_count,
                source_counts={},
                metadata={"writer": "qa_follow_up"},
            )

        evidence_count = len(qa.evidence_bundle.evidences)
        confidence_value = qa.confidence if qa.confidence is not None else 0.0
        insufficient = is_insufficient_answer(
            answer=qa.answer,
            confidence=confidence_value,
            evidence_count=evidence_count,
        )
        question_summary = self._compact_text(request.question, limit=120)
        answer_summary = self._compact_text(qa.answer, limit=220)

        todo_items = self._upsert_follow_up_todo(
            existing_items=task.todo_items,
            question=request.question,
            question_summary=question_summary,
            answer_summary=answer_summary,
            insufficient=insufficient,
            evidence_count=evidence_count,
            confidence=confidence_value,
            created_at=now,
        )
        latest_todo = todo_items[0] if todo_items else None
        updated_markdown = self._append_qa_report_entry(
            markdown=report.markdown,
            asked_at=now,
            question=request.question,
            answer=qa.answer,
            document_count=len(document_ids),
            evidence_count=evidence_count,
            confidence=qa.confidence,
            todo_item=latest_todo,
            paper_titles=scope.selected_titles(),
            scope_mode=scope.scope_mode,
        )

        updated_highlights = list(report.highlights)
        updated_gaps = list(report.gaps)
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor = qa_metadata.get("visual_anchor") if isinstance(qa_metadata.get("visual_anchor"), dict) else None
        visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )
        if insufficient:
            gap = (
                f"关于“{question_summary}”的研究集合证据仍不足，当前仅有 {evidence_count} 条证据，"
                "建议补充更直接的论文或扩大检索范围。"
            )
            updated_gaps = self._prepend_unique_text(updated_gaps, gap)
        else:
            highlight = f"问答补充：关于“{question_summary}”，当前结论是 {answer_summary}"
            updated_highlights = self._prepend_unique_text(updated_highlights, highlight)

        report_metadata = dict(report.metadata)
        report_metadata.update(
            {
                "last_qa_at": now,
                "last_qa_question": question_summary,
                "qa_update_count": int(report.metadata.get("qa_update_count") or 0) + 1,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            }
        )
        updated_report = report.model_copy(
            update={
                "generated_at": now,
                "markdown": updated_markdown,
                "highlights": updated_highlights[:10],
                "gaps": updated_gaps[:10],
                "metadata": report_metadata,
            }
        )
        updated_task = task.model_copy(
            update={
                "updated_at": now,
                "report_id": updated_report.report_id,
                "todo_items": todo_items[:20],
            }
        )
        workspace = build_workspace_from_task(
            task=updated_task,
            report=updated_report,
            papers=papers,
            stage="qa",
            extra_questions=[request.question],
            extra_findings=[qa.answer],
            stop_reason=(
                "Collection QA found an evidence gap that should drive the next retrieval cycle."
                if insufficient
                else "Collection QA completed and committed its answer back into the research workspace."
            ),
            metadata={
                "last_qa_question": question_summary,
                "last_qa_confidence": round(confidence_value, 4),
                "last_qa_evidence_count": evidence_count,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure": (
                    visual_anchor_figure.model_dump(mode="json") if visual_anchor_figure is not None else None
                ),
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            },
        )
        updated_report = updated_report.model_copy(update={"workspace": workspace})
        updated_task = updated_task.model_copy(update={"workspace": workspace})
        return updated_task, updated_report

    def _upsert_follow_up_todo(
        self,
        *,
        existing_items: list[ResearchTodoItem],
        question: str,
        question_summary: str,
        answer_summary: str,
        insufficient: bool,
        evidence_count: int,
        confidence: float,
        created_at: str,
    ) -> list[ResearchTodoItem]:
        if insufficient:
            content = f"补充与“{question_summary}”直接相关的论文，并在扩展关键词或时间窗口后重新运行研究任务。"
            rationale = f"当前仅检索到 {evidence_count} 条可用证据，置信度 {confidence:.2f}，现有研究集合无法稳定回答该问题。"
            source = "evidence_gap"
            priority = "high"
        else:
            content = f"围绕“{question_summary}”整理一个更细粒度的对比表，并持续核验新论文是否改变当前结论。"
            rationale = f"当前已有可用答案，可继续沉淀成综述结论与实验对比材料。摘要：{answer_summary}"
            source = "qa_follow_up"
            priority = "medium"

        next_item = ResearchTodoItem(
            todo_id=f"todo_{uuid4().hex}",
            content=content,
            rationale=rationale,
            status="open",
            priority=priority,
            created_at=created_at,
            question=question,
            source=source,
            metadata={
                "evidence_count": evidence_count,
                "confidence": round(confidence, 4),
            },
        )
        updated_items: list[ResearchTodoItem] = []
        replaced = False
        for item in existing_items:
            if item.question == question and item.status == "open":
                updated_items.append(next_item)
                replaced = True
            else:
                updated_items.append(item)
        if not replaced:
            updated_items.insert(0, next_item)
        return updated_items

    def _append_qa_report_entry(
        self,
        *,
        markdown: str,
        asked_at: str,
        question: str,
        answer: str,
        document_count: int,
        evidence_count: int,
        confidence: float | None,
        todo_item: ResearchTodoItem | None,
        paper_titles: list[str] | None = None,
        scope_mode: str | None = None,
    ) -> str:
        section_heading = "## 研究集合问答补充"
        entry_lines = [
            f"### {asked_at}",
            f"问题：{question}",
        ]
        entry_lines.extend(
            [
                "",
                "回答：",
                answer.strip() or "（空）",
            ]
        )
        if section_heading in markdown:
            return f"{markdown.rstrip()}\n\n" + "\n".join(entry_lines)
        prefix = markdown.rstrip()
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{section_heading}\n\n" + "\n".join(entry_lines)

    def _prepend_unique_text(self, items: list[str], entry: str) -> list[str]:
        deduped = [item for item in items if item != entry]
        return [entry, *deduped]

    def _compact_text(self, text: str, *, limit: int) -> str:
        compacted = " ".join(text.strip().split())
        if len(compacted) <= limit:
            return compacted
        return f"{compacted[: max(limit - 1, 1)].rstrip()}…"

