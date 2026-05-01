from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
from typing import Any

from core.utils import now_iso as _now_iso
from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    ImportPapersRequest,
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTaskResponse,
)
from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.unified_runtime import UnifiedAgentResult, UnifiedAgentTask
from services.research.research_knowledge_access import ResearchKnowledgeAccess
from services.research.research_workspace import build_workspace_from_task
from services.research.unified_action_adapters import (
    build_chart_understanding_input,
    build_chart_understanding_output,
    build_context_compression_input,
    build_context_compression_output,
    build_literature_search_input,
    build_literature_search_output,
    build_paper_analysis_input,
    build_paper_analysis_output,
    build_paper_import_input,
    build_paper_import_output,
    build_review_draft_input,
    build_review_draft_output,
    resolve_active_message,
)

logger = logging.getLogger(__name__)
ResearchAgentToolContext = Any


@dataclass(slots=True)
class ResearchToolResult:
    status: str
    observation: str
    metadata: dict[str, Any]


def _update_runtime_progress(
    context: Any,
    *,
    stage: str,
    node: str,
    status: str,
    summary: str,
    extra: dict[str, Any] | None = None,
) -> None:
    progress = {
        "stage": stage,
        "node": node,
        "status": status,
        "summary": summary,
        "updated_at": _now_iso(),
        **dict(extra or {}),
    }
    context.runtime_progress = progress
    if context.progress_callback is not None:
        try:
            asyncio.get_event_loop().create_task(context.progress_callback(progress))
        except RuntimeError:
            pass
    context.research_service.append_runtime_event(
        conversation_id=context.request.conversation_id,
        event_type="memory_updated",
        task_id=context.task.task_id if context.task is not None else context.request.task_id,
        correlation_id=(
            context.task.status_metadata.correlation_id
            if context.task is not None
            else None
        ),
        payload={
            "runtime_event": "supervisor_progress",
            **progress,
        },
    )


class _WorkspacePersistenceMixin:
    def _dedupe_ids(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _dedupe_text(self, values: list[str], *, limit: int) -> list[str]:
        deduped = [value.strip() for value in values if value and value.strip()]
        return list(dict.fromkeys(deduped))[:limit]

    def _comparison_scope_papers(
        self,
        *,
        papers: list[PaperCandidate],
        selected_paper_ids: list[str],
    ) -> list[PaperCandidate]:
        if selected_paper_ids:
            allowed = set(selected_paper_ids)
            resolved = [paper for paper in papers if paper.paper_id in allowed]
            if resolved:
                return resolved
        ranked = list(papers)
        ranked.sort(
            key=lambda paper: (
                float(paper.relevance_score or 0.0),
                int(paper.citations or 0),
                int(paper.year or 0),
            ),
            reverse=True,
        )
        return ranked[:3]

    def _persist_workspace_results(
        self,
        context: Any,
        *,
        paper_analysis: AnalyzePapersFunctionOutput | None = None,
        analyzed_papers: list[PaperCandidate] | None = None,
        compression_summary: dict[str, Any] | None = None,
    ) -> None:
        task_response = context.task_response
        execution_context = context.execution_context
        if task_response is None:
            return
        task = task_response.task
        workspace_metadata = dict(task.workspace.metadata)
        key_findings = list(task.workspace.key_findings)
        next_actions = list(task.workspace.next_actions)
        must_read_ids = list(task.workspace.must_read_paper_ids)
        selected_paper_ids: list[str] = list(context.request.selected_paper_ids)
        if paper_analysis is not None:
            workspace_metadata["latest_paper_analysis"] = paper_analysis.model_dump(mode="json")
            key_findings.append(paper_analysis.answer)
            analyzed_ids = [paper.paper_id for paper in analyzed_papers or []]
            selected_paper_ids.extend(analyzed_ids)
            recommended_ids = list(paper_analysis.recommended_paper_ids)
            must_read_ids = self._dedupe_ids([*must_read_ids, *recommended_ids])
            if paper_analysis.focus == "compare":
                next_actions.append("可以继续围绕这组论文追问更细的实验差异、适用场景或失败案例。")
            elif paper_analysis.focus == "recommend":
                next_actions.append("可以直接导入推荐论文全文，或围绕推荐理由继续提问。")
            else:
                next_actions.append("可以继续针对这组论文追问方法细节、实验设置或适用边界。")
        if compression_summary is not None:
            workspace_metadata["context_compression"] = dict(compression_summary)
        updated_at = _now_iso()
        updated_workspace = task.workspace.model_copy(
            update={
                "key_findings": self._dedupe_text(key_findings, limit=6),
                "must_read_paper_ids": must_read_ids,
                "next_actions": self._dedupe_text(next_actions, limit=5),
                "metadata": workspace_metadata,
            }
        )
        updated_task = task.model_copy(update={"updated_at": updated_at, "workspace": updated_workspace})
        updated_report = (
            task_response.report.model_copy(update={"workspace": updated_workspace})
            if task_response.report is not None
            else None
        )
        context.research_service.save_task_state(
            updated_task,
            conversation_id=context.request.conversation_id,
            event_type="memory_updated",
            payload={
                "tool_name": "workspace_persist",
                "context_compression": compression_summary,
                "has_paper_analysis": paper_analysis is not None,
            },
        )
        if updated_report is not None:
            context.research_service.report_service.save_report(updated_report)
        context.task_response = task_response.model_copy(update={"task": updated_task, "report": updated_report})
        if execution_context is not None and execution_context.research_context is not None:
            research_context = context.research_service.research_context_manager.update_context(
                current_context=execution_context.research_context,
                selected_papers=self._dedupe_ids(selected_paper_ids),
                known_conclusions=self._dedupe_text(key_findings, limit=6),
                metadata={
                    **(
                        {"latest_paper_analysis": paper_analysis.model_dump(mode="json")}
                        if paper_analysis is not None
                        else {}
                    ),
                    **({"context_compression": compression_summary} if compression_summary is not None else {}),
                },
            )
            if selected_paper_ids:
                research_context.active_papers = self._dedupe_ids(selected_paper_ids)
            execution_context.research_context = research_context
            execution_context.context_slices = context.research_service.build_context_slices(
                research_context,
                selected_paper_ids=self._dedupe_ids(selected_paper_ids),
            )
            if execution_context.session_id:
                context.research_service.memory_gateway.save_context(
                    execution_context.session_id,
                    research_context,
                )


def build_specialist_unified_result(
    *,
    task: UnifiedAgentTask,
    agent_name: str,
    status: str,
    observation: str,
    metadata: dict[str, Any],
    execution_adapter: str,
    delegate_type: str | None,
) -> UnifiedAgentResult:
    return UnifiedAgentResult(
        task_id=task.task_id,
        agent_name=agent_name,
        task_type=task.task_type,
        status=status,  # type: ignore[arg-type]
        instruction=task.instruction,
        payload={
            "observation": observation,
            "tool_metadata": dict(metadata),
        },
        context_slice=task.context_slice,
        priority=task.priority,
        expected_output_schema=task.expected_output_schema,
        depends_on=task.depends_on,
        retry_count=task.retry_count,
        action_output=(
            dict(metadata)
            if UnifiedAgentResult.is_action_output_payload(metadata)
            else None
        ),
        metadata={
            "execution_engine": "unified_agent_registry",
            "execution_adapter": execution_adapter,
            "delegate_type": delegate_type,
        },
    )


class LiteratureDiscoveryCapability:
    """Literature discovery capability owned by LiteratureScoutAgent."""

    def _search_metadata(self, *, topic: str, bundle: Any) -> dict[str, object]:
        return {
            "last_search_query": topic,
            "last_search_discovered_paper_ids": [paper.paper_id for paper in bundle.papers],
            "last_search_discovered_count": len(bundle.papers),
            "search_plan": bundle.plan.model_dump(mode="json"),
        }

    def _build_discovery_execution_context(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
    ) -> Any:
        request = context.request
        return context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=task,
            report=report,
            papers=list(papers),
            document_ids=list(task.imported_document_ids),
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )

    def _persist_new_task_response(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        bundle: Any,
        topic: str,
    ) -> ResearchTaskResponse:
        search_metadata = self._search_metadata(topic=topic, bundle=bundle)
        saved_report = bundle.report.model_copy(
            update={"metadata": {**bundle.report.metadata, **search_metadata}}
        )
        completed_task = task.model_copy(
            update={
                "status": "completed",
                "paper_count": len(bundle.papers),
                "report_id": saved_report.report_id,
                "todo_items": bundle.todo_items,
                "workspace": bundle.workspace,
                "updated_at": _now_iso(),
                "metadata": {**task.metadata, **search_metadata},
            }
        )
        context.research_service.report_service.save_papers(completed_task.task_id, bundle.papers)
        context.research_service.report_service.save_report(saved_report)
        context.research_service.save_task_state(completed_task, conversation_id=context.request.conversation_id)
        return ResearchTaskResponse(
            task=completed_task,
            papers=bundle.papers,
            report=saved_report,
            warnings=bundle.warnings,
        )

    def _persist_existing_task_response(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        bundle: Any,
        topic: str,
    ) -> ResearchTaskResponse:
        existing_papers = context.research_service.report_service.load_papers(task.task_id)
        merged_papers = context.research_service._refresh_existing_pool(
            existing_papers=existing_papers,
            incoming_papers=bundle.papers,
            ranking_topic=topic,
        )
        existing_report = context.research_service.report_service.load_report(task.task_id, task.report_id)
        search_metadata = self._search_metadata(topic=topic, bundle=bundle)
        refreshed_report = bundle.report.model_copy(
            update={
                "task_id": task.task_id,
                "report_id": existing_report.report_id if existing_report is not None else bundle.report.report_id,
                "generated_at": _now_iso(),
                "metadata": {
                    **(existing_report.metadata if existing_report is not None else {}),
                    **bundle.report.metadata,
                    **search_metadata,
                },
            }
        )
        if existing_report is not None:
            markdown = refreshed_report.markdown.rstrip()
            qa_section = context.research_service._extract_markdown_section(existing_report.markdown, "## 研究集合问答补充")
            todo_section = context.research_service._extract_markdown_section(existing_report.markdown, "## TODO 执行记录")
            if qa_section:
                markdown = f"{markdown}\n\n{qa_section.strip()}"
            if todo_section:
                markdown = f"{markdown}\n\n{todo_section.strip()}"
            carry_highlights = [
                item for item in existing_report.highlights
                if item.startswith("问答补充：") or item.startswith("TODO执行：")
            ]
            refreshed_report = refreshed_report.model_copy(
                update={
                    "markdown": markdown,
                    "highlights": context.research_service._merge_text_entries(
                        refreshed_report.highlights,
                        carry_highlights,
                        limit=12,
                    ),
                    "gaps": context.research_service._merge_text_entries(
                        refreshed_report.gaps,
                        list(existing_report.gaps),
                        limit=12,
                    ),
                }
            )
        updated_task = task.model_copy(
            update={
                "status": "completed",
                "paper_count": len(merged_papers),
                "report_id": refreshed_report.report_id,
                "updated_at": _now_iso(),
                "metadata": {**task.metadata, **search_metadata},
            }
        )
        updated_workspace = build_workspace_from_task(
            task=updated_task,
            report=refreshed_report,
            papers=merged_papers,
            plan=bundle.plan,
            stop_reason="Literature discovery refreshed the current research workspace.",
            metadata={
                **dict(updated_task.workspace.metadata),
                "last_search_query": topic,
                "last_search_discovered_count": len(bundle.papers),
            },
        )
        refreshed_report = refreshed_report.model_copy(update={"workspace": updated_workspace})
        updated_task = updated_task.model_copy(update={"workspace": updated_workspace})
        context.research_service.report_service.save_papers(updated_task.task_id, merged_papers)
        context.research_service.report_service.save_report(refreshed_report)
        context.research_service.save_task_state(updated_task, conversation_id=context.request.conversation_id)
        return ResearchTaskResponse(
            task=updated_task,
            papers=merged_papers,
            report=refreshed_report,
            warnings=bundle.warnings,
        )

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        literature_scout_agent: Any,
        research_writer_agent: Any,
        curation_skill: Any,
    ) -> ResearchToolResult:
        request = context.request
        search_input = build_literature_search_input(context=context, decision=decision)
        search_request = search_input.to_create_research_task_request().model_copy(
            update={"run_immediately": False}
        )
        task_response = context.task_response
        active_task = task_response.task if task_response is not None else None
        if active_task is None:
            task_response = await context.research_service.create_task(
                search_request,
                graph_runtime=context.graph_runtime,
            )
            active_task = task_response.task
        for node, summary in (
            ("planning", "Planning literature discovery queries."),
            ("source_search", "Searching literature sources."),
            ("curation", "Curating candidate papers."),
            ("survey_writing", "Writing literature survey report."),
            ("todo_planning", "Planning follow-up research todos."),
        ):
            _update_runtime_progress(
                context,
                stage="search_literature",
                node=f"search_literature:{node}",
                status="running",
                summary=summary,
            )
        bundle = await context.research_service.research_discovery_capability.discover(
            topic=search_request.topic,
            days_back=search_request.days_back,
            max_papers=search_request.max_papers,
            sources=list(search_request.sources),
            task_id=active_task.task_id,
            execution_context=self._build_discovery_execution_context(
                context=context,
                task=active_task,
                report=task_response.report if task_response is not None else None,
                papers=task_response.papers if task_response is not None else [],
            ),
            literature_scout_agent=literature_scout_agent,
            research_writer_agent=research_writer_agent,
            curation_skill=curation_skill,
        )
        response = (
            self._persist_existing_task_response(
                context=context,
                task=active_task,
                bundle=bundle,
                topic=search_request.topic,
            )
            if context.task is not None
            else self._persist_new_task_response(
                context=context,
                task=active_task,
                bundle=bundle,
                topic=search_request.topic,
            )
        )
        context.task_response = response
        context.execution_context = self._build_discovery_execution_context(
            context=context,
            task=response.task,
            report=response.report,
            papers=response.papers,
        )
        if request.conversation_id:
            context.research_service.record_task_turn(request.conversation_id, response=response)
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:completed",
            status="completed",
            summary="Literature discovery completed.",
            extra={"paper_count": len(response.papers), "query_count": len(bundle.plan.queries)},
        )
        output = build_literature_search_output(task_response=response)
        action_label = "updated task" if task_response is not None else "created task"
        return ResearchToolResult(
            status="succeeded",
            observation=f"{action_label} {response.task.task_id}; papers={len(response.papers)}; report={bool(response.report)}",
            metadata=output.to_metadata(),
        )


class ReviewWritingCapability:
    """Review drafting capability owned by ResearchWriterAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        writer_agent: Any,
    ) -> ResearchToolResult:
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for review drafting",
                metadata={"reason": "missing_task"},
            )

        review_input = build_review_draft_input(context=context)
        existing_report = review_input.report
        candidate_report = existing_report or writer_agent.synthesize(review_input)
        retry_count = 0
        quality = self._quality_metrics(candidate_report)
        if not quality["passed"]:
            retry_count += 1
            candidate_report = writer_agent.synthesize(review_input)
            quality = self._quality_metrics(candidate_report)

        generated_at = _now_iso()
        task = task_response.task
        saved_report = candidate_report.model_copy(
            update={
                "generated_at": generated_at,
                "metadata": {
                    **candidate_report.metadata,
                    "worker_agent": "ResearchWriterAgent",
                    "write_review_retry_count": retry_count,
                    "write_review_quality_passed": quality["passed"],
                    "write_review_issue_count": len(quality["issues"]),
                },
            }
        )
        updated_task = task.model_copy(
            update={
                "updated_at": generated_at,
                "report_id": saved_report.report_id,
                "workspace": task.workspace.model_copy(
                    update={
                        "metadata": {
                            **task.workspace.metadata,
                            "write_review_retry_count": retry_count,
                        }
                    }
                ),
            }
        )
        request = context.request
        context.research_service.report_service.save_report(saved_report)
        context.research_service.save_task_state(
            updated_task,
            conversation_id=request.conversation_id,
            event_type="memory_updated",
            payload={"tool_name": "write_review", "report_id": saved_report.report_id},
        )
        context.task_response = task_response.model_copy(update={"task": updated_task, "report": saved_report})
        context.execution_context = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            report=saved_report,
            papers=review_input.curated_papers,
            document_ids=updated_task.imported_document_ids,
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )
        output = build_review_draft_output(
            task_id=updated_task.task_id,
            report_id=saved_report.report_id,
            quality=quality,
            retry_count=retry_count,
        )
        return ResearchToolResult(
            status="succeeded" if quality["passed"] else "failed",
            observation=(
                f"review drafted; words={quality['word_count']}; citations={quality['has_citations']}; retries={retry_count}"
            ),
            metadata=output.to_metadata(),
        )

    def _quality_metrics(self, report: ResearchReport) -> dict[str, Any]:
        text = report.markdown
        word_count = len([token for token in text.replace("\n", " ").split(" ") if token.strip()])
        has_citations = "[P" in text or "引用" in text
        has_key_sections = all(
            section in text
            for section in ("## 研究背景", "## 核心问题", "## 方法对比", "## 关键发现")
        ) or all(
            section in text
            for section in ("## 研究背景", "## 方法对比", "## 关键发现")
        )
        issues: list[str] = []
        if word_count < 250:
            issues.append("review_too_short")
        if not has_citations:
            issues.append("missing_citations")
        if not has_key_sections:
            issues.append("missing_required_sections")
        return {
            "passed": not issues,
            "word_count": word_count,
            "has_citations": has_citations,
            "has_key_sections": has_key_sections,
            "issues": issues,
        }


class KnowledgeOpsCapability(_WorkspacePersistenceMixin):
    """Import, sync, and context-compression capability owned by ResearchKnowledgeAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        task_type: str,
    ) -> ResearchToolResult:
        if task_type == "import_papers":
            return await self.import_papers(context=context, decision=decision)
        if task_type == "sync_to_zotero":
            return await self.sync_to_zotero(context=context, decision=decision)
        if task_type == "compress_context":
            return await self.compress_context(context=context, decision=decision)
        return ResearchToolResult(
            status="skipped",
            observation=f"ResearchKnowledgeAgent does not support task_type={task_type}",
            metadata={"reason": "unsupported_task_type"},
        )

    async def import_papers(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        task_response = context.task_response
        context.import_attempted = True
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for paper import",
                metadata={"reason": "missing_task"},
            )

        import_input = build_paper_import_input(context=context, decision=decision)
        paper_ids = import_input.resolved_paper_ids(task_response.papers)
        if not paper_ids:
            return ResearchToolResult(
                status="skipped",
                observation="no importable paper with an available PDF was found",
                metadata={"reason": "no_import_candidates"},
            )

        import_result = await context.research_service.import_papers(
            ImportPapersRequest(
                task_id=task_response.task.task_id,
                paper_ids=paper_ids,
                include_graph=import_input.include_graph,
                include_embeddings=import_input.include_embeddings,
                skill_name=import_input.skill_name,
                conversation_id=import_input.conversation_id,
            ),
            graph_runtime=context.graph_runtime,
        )
        context.import_result = import_result
        refreshed = context.research_service.get_task(task_response.task.task_id)
        context.task_response = refreshed
        request = context.request
        context.execution_context = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=refreshed.task,
            report=refreshed.report,
            papers=refreshed.papers,
            document_ids=refreshed.task.imported_document_ids,
            selected_paper_ids=paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )
        if request.conversation_id:
            context.research_service.record_import_turn(
                request.conversation_id,
                task_response=refreshed,
                import_response=import_result,
                selected_paper_ids=paper_ids,
            )
        output = build_paper_import_output(
            paper_ids=paper_ids,
            import_result=import_result,
        )
        return ResearchToolResult(
            status="succeeded" if import_result.failed_count == 0 else "failed" if import_result.imported_count == 0 and import_result.skipped_count == 0 else "succeeded",
            observation=(
                f"paper import finished; imported={import_result.imported_count}; "
                f"skipped={import_result.skipped_count}; failed={import_result.failed_count}"
            ),
            metadata=output.to_metadata(),
        )

    async def sync_to_zotero(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for zotero sync",
                metadata={"reason": "missing_task"},
            )
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        raw_paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or context.request.selected_paper_ids)
            if str(item).strip()
        ]
        papers_by_id = {paper.paper_id: paper for paper in task_response.papers}
        paper_ids = [paper_id for paper_id in raw_paper_ids if paper_id in papers_by_id]
        if not paper_ids:
            return ResearchToolResult(
                status="skipped",
                observation="no candidate papers were resolved for zotero sync",
                metadata={"reason": "no_target_papers"},
            )
        function_service = getattr(context.graph_runtime, "research_function_service", None)
        if function_service is None or not hasattr(function_service, "sync_paper_to_zotero"):
            return ResearchToolResult(
                status="failed",
                observation="research function service is unavailable for zotero sync",
                metadata={"reason": "missing_research_function_service"},
            )
        collection_name = str(payload.get("collection_name") or "").strip() or None
        results: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = papers_by_id[paper_id]
            sync_result = await function_service.sync_paper_to_zotero(
                paper,
                collection_name=collection_name,
            )
            results.append({"paper_id": paper.paper_id, "title": paper.title, **dict(sync_result)})
        context.zotero_sync_results = results
        synced_count = sum(1 for item in results if str(item.get("status") or "") in {"imported", "reused"})
        failed_count = len(results) - synced_count
        return ResearchToolResult(
            status="succeeded" if failed_count == 0 else "failed" if synced_count == 0 else "succeeded",
            observation=f"zotero sync finished; synced={synced_count}; failed={failed_count}",
            metadata={
                "paper_ids": paper_ids,
                "synced_count": synced_count,
                "failed_count": failed_count,
                "results": results,
                "collection_name": collection_name,
            },
        )

    async def compress_context(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return ResearchToolResult(
                status="skipped",
                observation="no execution context is available for compression",
                metadata={"reason": "missing_execution_context"},
            )
        compression_input = build_context_compression_input(context=context, decision=decision)
        selected_paper_ids = compression_input.resolved_selected_paper_ids()
        compressed = context.research_service.research_context_manager.compress_papers(
            papers=list(context.papers),
            selected_paper_ids=selected_paper_ids,
            paper_reading_skill=context.research_service.paper_reading_skill,
        )
        if not compressed:
            return ResearchToolResult(status="skipped", observation="no paper summary could be built for compression", metadata={"reason": "no_papers"})
        updated_context = context.research_service.research_context_manager.update_context(
            current_context=execution_context.research_context,
            selected_papers=selected_paper_ids,
            paper_summaries=compressed,
            metadata={
                "context_compression": {
                    "paper_count": len({summary.paper_id for summary in compressed}),
                    "summary_count": len(compressed),
                    "levels": sorted({summary.level for summary in compressed}),
                }
            },
        )
        execution_context.research_context = updated_context
        execution_context.context_slices = context.research_service.build_context_slices(
            updated_context,
            selected_paper_ids=selected_paper_ids,
        )
        context.compressed_context_summary = {
            "paper_count": len({summary.paper_id for summary in compressed}),
            "summary_count": len(compressed),
            "levels": sorted({summary.level for summary in compressed}),
            "compressed_paper_ids": list(dict.fromkeys(summary.paper_id for summary in compressed)),
        }
        self._persist_workspace_results(context, compression_summary=context.compressed_context_summary)
        output = build_context_compression_output(
            compression_summary=context.compressed_context_summary,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"context compressed; papers={context.compressed_context_summary['paper_count']}; summaries={context.compressed_context_summary['summary_count']}"
            ),
            metadata=output.to_metadata(),
        )


class PaperAnalysisCapability(_WorkspacePersistenceMixin):
    """Selected-paper analysis capability owned by PaperAnalysisAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        paper_analysis_agent: Any,
    ) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for selected-paper analysis",
                metadata={"reason": "missing_task"},
            )
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        selected_paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or context.request.selected_paper_ids)
            if str(item).strip()
        ]
        papers = self._comparison_scope_papers(
            papers=task_response.papers,
            selected_paper_ids=selected_paper_ids,
        )
        if not papers:
            return ResearchToolResult(
                status="skipped",
                observation="no papers are available for selected-paper analysis",
                metadata={"reason": "no_papers"},
            )
        analysis_input = build_paper_analysis_input(
            context=context,
            task_response=task_response,
            payload=payload,
            papers=papers,
        )
        evidence_hits = await self._collect_analysis_evidence(
            context=context,
            question=analysis_input.resolved_question(),
            papers=analysis_input.papers,
        )
        analysis = await paper_analysis_agent.analyze(
            question=analysis_input.resolved_question(),
            papers=analysis_input.papers,
            task_topic=analysis_input.task_topic,
            report_highlights=analysis_input.report_highlights,
            evidence_hits=evidence_hits,
        )
        context.paper_analysis_result = analysis
        self._persist_workspace_results(context, paper_analysis=analysis, analyzed_papers=analysis_input.papers)
        output = build_paper_analysis_output(
            task_id=task_response.task.task_id,
            analysis=analysis,
            analyzed_papers=analysis_input.papers,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"paper analysis completed; papers={len(analysis_input.papers)}; "
                f"focus={analysis.focus}; evidence_hits={len(evidence_hits)}"
            ),
            metadata=output.to_metadata(),
        )

    async def _collect_analysis_evidence(
        self,
        *,
        context: ResearchAgentToolContext,
        question: str,
        papers: list[PaperCandidate],
    ) -> list[RetrievalHit]:
        from agents.research_knowledge_agent import merge_retrieval_hits

        document_ids = list(
            dict.fromkeys(
                str(paper.metadata.get("document_id") or "").strip()
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip()
            )
        )
        if not document_ids:
            return []
        knowledge_access = context.knowledge_access or ResearchKnowledgeAccess.from_runtime(context.graph_runtime)
        execution_context = context.execution_context
        scope_filters = {
            "analysis_mode": "paper_analysis",
            "selected_paper_ids": [paper.paper_id for paper in papers],
            "selected_document_ids": document_ids,
        }
        try:
            retrieval_output = await knowledge_access.retrieve(
                question=question,
                document_ids=document_ids,
                top_k=max(8, min(16, len(document_ids) * 4)),
                filters={
                    "research_task_id": context.task.task_id if context.task is not None else None,
                    "research_topic": context.task.topic if context.task is not None else "",
                    **scope_filters,
                },
                session_id=getattr(execution_context, "session_id", None),
                task_id=context.task.task_id if context.task is not None else None,
                memory_hints=getattr(execution_context, "memory_hints", None) or {},
            )
        except RuntimeError:
            return []
        retrieval_hits = [
            self._attach_paper_id_to_hit(hit=hit, papers=papers)
            for hit in list(retrieval_output.retrieval_result.hits or [])
        ]
        summary_output = await knowledge_access.query_graph_summary(
            question=question,
            document_ids=document_ids,
            top_k=max(3, min(6, len(document_ids) * 2)),
            filters={
                "research_task_id": context.task.task_id if context.task is not None else None,
                "research_topic": context.task.topic if context.task is not None else "",
                **scope_filters,
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=context.task.task_id if context.task is not None else None,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
        )
        summary_hits = [
            self._attach_paper_id_to_hit(hit=hit, papers=papers)
            for hit in list(getattr(summary_output, "hits", []) or [])
        ]
        return merge_retrieval_hits(retrieval_hits, summary_hits)[:12]

    def _attach_paper_id_to_hit(
        self,
        *,
        hit: RetrievalHit,
        papers: list[PaperCandidate],
    ) -> RetrievalHit:
        document_id = str(hit.document_id or "").strip()
        matched_paper = next(
            (
                paper
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip() == document_id
            ),
            None,
        )
        if matched_paper is None:
            return hit
        metadata = dict(hit.metadata)
        metadata.setdefault("paper_id", matched_paper.paper_id)
        metadata.setdefault("title", matched_paper.title)
        return hit.model_copy(update={"metadata": metadata})


class ChartAnalysisCapability:
    """Chart and paper-figure analysis capability owned by ChartAnalysisAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        chart_analysis_agent: Any,
        task_type: str,
    ) -> ResearchToolResult:
        if task_type == "understand_chart":
            return await self.understand_chart(
                context=context,
                decision=decision,
                chart_analysis_agent=chart_analysis_agent,
            )
        if task_type == "analyze_paper_figures":
            return await self.analyze_paper_figures(
                context=context,
                decision=decision,
                chart_analysis_agent=chart_analysis_agent,
            )
        return ResearchToolResult(
            status="skipped",
            observation=f"ChartAnalysisAgent does not support task_type={task_type}",
            metadata={"reason": "unsupported_task_type"},
        )

    async def understand_chart(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        chart_analysis_agent: Any,
    ) -> ResearchToolResult:
        context.chart_attempted = True
        chart_input = build_chart_understanding_input(context=context, decision=decision)
        if not chart_input.image_path:
            return ResearchToolResult(
                status="skipped",
                observation="no chart_image_path was provided for chart understanding",
                metadata={"reason": "missing_chart_image_path"},
            )

        chart_result = await chart_analysis_agent.understand_chart(
            graph_runtime=context.graph_runtime,
            image_path=chart_input.image_path,
            document_id=chart_input.document_id,
            page_id=chart_input.page_id,
            page_number=chart_input.page_number,
            chart_id=chart_input.chart_id,
            session_id=chart_input.session_id,
            context=chart_input.context,
            skill_name=chart_input.skill_name,
        )
        context.chart_result = chart_result
        chart = getattr(chart_result, "chart", None)
        output = build_chart_understanding_output(
            chart_result=chart_result,
            chart_input=chart_input,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"chart understood; chart_id={getattr(chart, 'id', chart_input.chart_id)}; "
                f"chart_type={getattr(chart, 'chart_type', 'unknown')}"
            ),
            metadata=output.to_metadata(),
        )

    async def analyze_paper_figures(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        chart_analysis_agent: Any,
    ) -> ResearchToolResult:
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for paper figure analysis",
                metadata={"reason": "missing_task"},
            )
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("question") or context.request.message or "").strip()

        paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or context.request.selected_paper_ids)
            if str(item).strip()
        ]
        imported_papers = [
            p for p in task_response.papers
            if str(p.metadata.get("document_id") or "").strip()
            and str(p.metadata.get("storage_uri") or "").strip()
        ]
        target_papers = [p for p in imported_papers if p.paper_id in paper_ids] if paper_ids else imported_papers
        if not target_papers:
            logger.warning(
                "analyze_paper_figures: no target papers; imported_papers=%d; paper_ids=%s; all_papers=%s",
                len(imported_papers),
                paper_ids,
                [(p.paper_id, p.ingest_status, bool(p.metadata.get("document_id")), bool(p.metadata.get("storage_uri"))) for p in task_response.papers[:5]],
            )
            return ResearchToolResult(
                status="skipped",
                observation="no imported paper with a local document is available for figure analysis",
                metadata={"reason": "no_imported_papers"},
            )
        target_paper = target_papers[0]
        logger.info("analyze_paper_figures: target_paper=%s doc_id=%s", target_paper.paper_id, target_paper.metadata.get("document_id"))

        try:
            figure_list = await context.research_service.list_paper_figures(
                task_response.task.task_id,
                target_paper.paper_id,
                graph_runtime=context.graph_runtime,
            )
        except Exception as exc:
            logger.warning("Failed to list paper figures", exc_info=True)
            return ResearchToolResult(
                status="failed",
                observation=f"failed to extract figures from paper: {exc}",
                metadata={"reason": "list_figures_failed"},
            )

        if not figure_list.figures:
            return ResearchToolResult(
                status="skipped",
                observation=f"no figures found in paper '{target_paper.title}'",
                metadata={"reason": "no_figures", "paper_id": target_paper.paper_id},
            )

        best_figure = await self._select_figure_via_anchor(
            question=question or "",
            target_paper=target_paper,
            figures=figure_list.figures,
            context=context,
            chart_analysis_agent=chart_analysis_agent,
        )

        figure_request = AnalyzeResearchPaperFigureRequest(
            figure_id=best_figure.figure_id,
            page_id=best_figure.page_id,
            chart_id=best_figure.chart_id,
            image_path=best_figure.image_path,
            question=question or None,
        )
        try:
            analysis_response = await context.research_service.analyze_paper_figure(
                task_response.task.task_id,
                target_paper.paper_id,
                figure_request,
                graph_runtime=context.graph_runtime,
            )
        except Exception as exc:
            logger.warning("Failed to analyze paper figure", exc_info=True)
            return ResearchToolResult(
                status="failed",
                observation=f"figure analysis failed: {exc}",
                metadata={"reason": "analyze_figure_failed"},
            )

        exported_image_path = self._export_figure_image(
            analysis_response=analysis_response,
            task_id=task_response.task.task_id,
        )
        if exported_image_path and analysis_response.chart and hasattr(analysis_response.chart, "metadata"):
            analysis_response.chart.metadata["image_path"] = exported_image_path

        context.chart_result = analysis_response
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"paper figure analysis completed; paper='{target_paper.title}'; "
                f"figure={best_figure.figure_id}; answer_length={len(analysis_response.answer)}"
            ),
            metadata={
                "paper_id": target_paper.paper_id,
                "figure_id": best_figure.figure_id,
                "answer": analysis_response.answer,
                "key_points": analysis_response.key_points,
                "chart_type": getattr(analysis_response.chart, "chart_type", None),
                "image_path": exported_image_path,
            },
        )

    @staticmethod
    def _export_figure_image(
        analysis_response: Any,
        task_id: str,
    ) -> str | None:
        source_path = getattr(analysis_response.figure, "image_path", None) if analysis_response.figure else None
        if not source_path or not Path(source_path).is_file():
            return None
        export_dir = Path(".data/storage/figure_exports") / task_id
        export_dir.mkdir(parents=True, exist_ok=True)
        figure_id = getattr(analysis_response.figure, "figure_id", "") or "figure"
        safe_name = re.sub(r"[^\w\-_.]", "_", figure_id)
        suffix = Path(source_path).suffix or ".png"
        dest_path = export_dir / f"{safe_name}{suffix}"
        try:
            shutil.copy2(source_path, dest_path)
            logger.info("Exported figure image: %s -> %s", source_path, dest_path)
            return str(dest_path)
        except Exception:
            logger.debug("Failed to export figure image", exc_info=True)
            return str(source_path)

    async def _select_figure_via_anchor(
        self,
        *,
        question: str,
        target_paper: Any,
        figures: list[Any],
        context: ResearchAgentToolContext,
        chart_analysis_agent: Any,
    ) -> Any:
        if len(figures) == 1:
            return figures[0]
        try:
            anchor = await chart_analysis_agent.infer_cached_visual_anchor(
                papers=[target_paper],
                document_ids=[str(target_paper.metadata.get("document_id") or "")],
                question=question,
                load_cached_figure_payload=context.research_service._load_cached_figure_payload,
            )
        except Exception:
            logger.debug("infer_cached_visual_anchor failed, using fallback", exc_info=True)
            anchor = None
        if anchor is not None:
            anchor_figure_id = str(anchor.get("figure_id") or "").strip()
            if anchor_figure_id:
                matched = next((f for f in figures if f.figure_id == anchor_figure_id), None)
                if matched is not None:
                    logger.info("analyze_paper_figures: anchor selected figure_id=%s", anchor_figure_id)
                    return matched
        return figures[0]


class GeneralAnswerCapability:
    """General non-research answer capability owned by GeneralAnswerAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        general_answer_agent: Any,
    ) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        on_token = None
        if context.progress_callback is not None:
            async def on_token(text: str) -> None:
                await context.progress_callback({"type": "token", "text": text})
        result = await general_answer_agent.answer(
            question=question,
            conversation_context={
                "mode": context.request.mode,
                "task_id": context.request.task_id,
                "has_task": context.task is not None,
                "selected_paper_ids": [] if payload.get("ignore_research_context") else list(context.request.selected_paper_ids),
                "ignore_research_context": bool(payload.get("ignore_research_context")),
            },
            on_token=on_token,
        )
        warnings = list(result.warnings)
        provider_fallback = result.answer_type in {"fallback", "provider_timeout", "provider_error"}
        should_reroute = (not provider_fallback) and (
            "route_mismatch" in warnings or (
                result.answer_type == "reroute_hint"
            ) or (
                result.confidence < 0.45 and (
                    context.request.task_id is not None
                    or bool(context.request.selected_paper_ids)
                    or bool(context.request.selected_document_ids)
                    or bool(context.request.chart_image_path)
                    or bool(context.request.document_file_path)
                )
            )
        )
        if should_reroute:
            return ResearchToolResult(
                status="skipped",
                observation="general answer agent detected a likely route mismatch and requested supervisor rerouting",
                metadata={
                    **result.model_dump(mode="json"),
                    "reason": "route_mismatch",
                    "suggested_action": self._suggested_action(context=context),
                },
            )
        context.general_answer = result.answer
        context.general_answer_metadata = result.model_dump(mode="json")
        return ResearchToolResult(
            status="succeeded",
            observation="general question answered directly without research workspace tools",
            metadata=context.general_answer_metadata,
        )

    def _suggested_action(self, *, context: ResearchAgentToolContext) -> str:
        if context.request.chart_image_path:
            return "understand_chart"
        if context.request.document_file_path:
            return "understand_document"
        if context.request.task_id or context.request.selected_paper_ids or context.request.selected_document_ids:
            return "answer_question"
        return "search_literature"


class PreferenceRecommendationCapability:
    """Preference recommendation capability owned by PreferenceMemoryAgent."""

    async def run(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        preference_memory_agent: Any,
    ) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        top_k = max(
            1,
            min(
                10,
                int(
                    payload.get("top_k")
                    or context.request.recommendation_top_k
                    or 6
                ),
            ),
        )
        days_back = max(
            1,
            int(
                payload.get("days_back")
                or context.request.days_back
                or 30
            ),
        )
        raw_sources = payload.get("sources")
        sources = (
            [str(item).strip().lower() for item in raw_sources if str(item).strip()]
            if isinstance(raw_sources, list)
            else list(context.request.sources)
        )
        recommendation_output = await preference_memory_agent.recommend_recent_papers(
            question=question,
            days_back=days_back,
            top_k=top_k,
            sources=sources,
            include_notification=True,
        )
        context.preference_recommendation_result = recommendation_output
        recommendations = list(recommendation_output.recommendations)
        if not recommendations:
            return ResearchToolResult(
                status="skipped",
                observation="no personalized paper recommendations could be generated from long-term preferences",
                metadata={
                    "reason": "no_preference_recommendations",
                    **recommendation_output.model_dump(mode="json"),
                },
            )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"generated {len(recommendations)} personalized recommendations from long-term preferences"
            ),
            metadata=recommendation_output.model_dump(mode="json"),
        )
