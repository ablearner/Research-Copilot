"""Paper import, figure analysis, and TODO operations mixin.

Extracts paper-centric operations from LiteratureResearchService into
a cohesive mixin to reduce the size of the main service file.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

from core.utils import (
    now_iso as _now_iso,
    normalize_paper_title as _normalize_paper_title,
)
from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    ImportPapersRequest,
    ImportPapersResponse,
    ImportedPaperResult,
    PaperCandidate,
    ResearchAdvancedStrategy,
    ResearchJob,
    ResearchPaperFigureListResponse,
    ResearchPaperFigurePreview,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskRequest,
    ResearchTaskResponse,
    ResearchTodoActionRequest,
    ResearchTodoActionResponse,
    ResearchTodoItem,
    ResearchWorkspaceState,
)
from tools.research.knowledge_access import ResearchKnowledgeAccess
from domain.research_workspace import build_workspace_from_task
from tools.paper_figure_toolkit import PaperFigureAnalyzeTarget

logger = logging.getLogger(__name__)

ImportProgressCallback = Callable[[int, int, ImportedPaperResult], Awaitable[None] | None]


class PaperOperationsMixin:
    """Mixin providing paper import, figure analysis, and TODO operations.

    Assumes the host class exposes (from ``LiteratureResearchService``):

    * ``report_service``, ``paper_import_service``, ``paper_search_service``
    * ``chart_analysis_agent``, ``literature_scout_agent``, ``paper_curation_skill``
    * ``research_writer_agent``, ``observability_service``
    * ``import_concurrency``, ``_job_tasks``
    * ``get_task(...)``, ``save_task_state(...)``, ``save_job_state(...)``
    * ``_build_status_metadata(...)``, ``_update_job(...)``
    * ``_set_conversation_active_job(...)``, ``append_runtime_event(...)``
    * ``_persist_runtime_conversation_snapshot(...)``
    * ``_resolve_research_session_id(...)``, ``_update_research_memory(...)``
    * ``_refresh_existing_pool(...)`` (from QARoutingMixin)
    * ``record_import_turn(...)``, ``record_qa_turn(...)``, ``record_notice(...)`` (from ConversationMixin)
    """

    async def list_paper_figures(
        self,
        task_id: str,
        paper_id: str,
        *,
        graph_runtime: Any,
    ) -> ResearchPaperFigureListResponse:
        task, paper = self._resolve_imported_paper(task_id=task_id, paper_id=paper_id)
        return await self.chart_analysis_agent.list_paper_figures(
            task_id=task.task_id,
            paper=paper,
            graph_runtime=graph_runtime,
            load_cached_figure_payload=self._load_cached_figure_payload,
            persist_paper_figure_cache=self._persist_paper_figure_cache,
            parse_imported_paper_document=self._parse_imported_paper_document,
        )

    async def analyze_paper_figure(
        self,
        task_id: str,
        paper_id: str,
        request: AnalyzeResearchPaperFigureRequest,
        *,
        graph_runtime: Any,
    ) -> AnalyzeResearchPaperFigureResponse:
        _, paper = self._resolve_imported_paper(task_id=task_id, paper_id=paper_id)
        return await self.chart_analysis_agent.analyze_paper_figure(
            task_id=task_id,
            paper=paper,
            request=request,
            graph_runtime=graph_runtime,
            load_cached_figure_target=self._load_cached_figure_target,
            parse_imported_paper_document=self._parse_imported_paper_document,
        )

    def _load_cached_figure_payload(self, *, paper: PaperCandidate) -> dict[str, Any] | None:
        cache = paper.metadata.get("paper_figure_cache")
        if not isinstance(cache, dict):
            return None
        cached_document_id = str(cache.get("document_id") or "").strip()
        cached_storage_uri = str(cache.get("storage_uri") or "").strip()
        current_document_id = str(paper.metadata.get("document_id") or "").strip()
        current_storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        if not cached_document_id or cached_document_id != current_document_id:
            return None
        if cached_storage_uri and current_storage_uri and cached_storage_uri != current_storage_uri:
            return None
        figures = cache.get("figures")
        if not isinstance(figures, list):
            return None
        return cache

    def _load_cached_figure_target(self, *, paper: PaperCandidate, figure_id: str | None) -> PaperFigureAnalyzeTarget | None:
        resolved_figure_id = str(figure_id or "").strip()
        if not resolved_figure_id:
            return None
        cache = self._load_cached_figure_payload(paper=paper)
        if cache is None:
            return None
        analyze_targets = cache.get("analyze_targets")
        if not isinstance(analyze_targets, dict):
            return None
        payload = analyze_targets.get(resolved_figure_id)
        if not isinstance(payload, dict):
            return None
        try:
            return PaperFigureAnalyzeTarget.model_validate(payload)
        except Exception:
            return None

    def _persist_paper_figure_cache(
        self,
        *,
        task_id: str,
        paper_id: str,
        document_id: str,
        storage_uri: str,
        figures: list[ResearchPaperFigurePreview],
        targets: list[PaperFigureAnalyzeTarget],
        warnings: list[str],
    ) -> None:
        papers = self.report_service.load_papers(task_id)
        updated_papers: list[PaperCandidate] = []
        for item in papers:
            if item.paper_id != paper_id:
                updated_papers.append(item)
                continue
            metadata = {
                **item.metadata,
                "paper_figure_cache": {
                    "document_id": document_id,
                    "storage_uri": storage_uri,
                    "figures": [figure.model_dump(mode="json") for figure in figures],
                    "analyze_targets": {
                        target.figure_id: target.model_dump(mode="json")
                        for target in targets
                    },
                    "warnings": list(warnings),
                },
            }
            updated_papers.append(item.model_copy(update={"metadata": metadata}))
        self.report_service.save_papers(task_id, updated_papers)

    def _resolve_imported_paper(self, *, task_id: str, paper_id: str) -> tuple[ResearchTask, PaperCandidate]:
        task_response = self.get_task(task_id)
        paper = next((item for item in task_response.papers if item.paper_id == paper_id), None)
        if paper is None:
            raise KeyError(f"Paper not found in task {task_id}: {paper_id}")
        document_id = str(paper.metadata.get("document_id") or "").strip()
        storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        if paper.ingest_status != "ingested" or not document_id or not storage_uri:
            raise ValueError("Paper must be imported before chart analysis is available.")
        return task_response.task, paper

    async def _parse_imported_paper_document(self, *, paper: PaperCandidate, graph_runtime: Any):
        storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        document_id = str(paper.metadata.get("document_id") or "").strip()
        if not storage_uri or not document_id:
            raise ValueError("Imported paper is missing storage metadata.")
        return await ResearchKnowledgeAccess.from_runtime(graph_runtime).parse_document(
            file_path=storage_uri,
            document_id=document_id,
            metadata={
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "source": "paper_figure_analysis",
            },
            skill_name="paper_chart_analysis",
        )

    def persist_runtime_state(
        self,
        *,
        task_response: ResearchTaskResponse | None,
        workspace: ResearchWorkspaceState,
        conversation_id: str | None = None,
        advanced_strategy: ResearchAdvancedStrategy | None = None,
    ) -> ResearchTaskResponse | None:
        updated_response = task_response
        if task_response is not None:
            updated_task = task_response.task.model_copy(
                update={
                    "workspace": workspace,
                    "updated_at": _now_iso(),
                }
            )
            self.save_task_state(updated_task, conversation_id=conversation_id)
            updated_report = task_response.report
            if updated_report is None and updated_task.report_id:
                updated_report = self.report_service.load_report(updated_task.task_id, updated_task.report_id)
            if updated_report is not None:
                updated_report = updated_report.model_copy(update={"workspace": workspace})
                self.report_service.save_report(updated_report)
            updated_response = task_response.model_copy(
                update={
                    "task": updated_task,
                    "report": updated_report,
                }
            )
        self._persist_runtime_conversation_snapshot(
            conversation_id=conversation_id,
            task_response=updated_response,
            workspace=workspace,
            advanced_strategy=advanced_strategy,
        )
        return updated_response

    def get_job(self, job_id: str) -> ResearchJob:
        job = self.report_service.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def start_import_job(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        now = _now_iso()
        correlation_id = f"job_{uuid4().hex}"
        job = ResearchJob(
            job_id=f"job_{uuid4().hex}",
            kind="paper_import",
            status="queued",
            created_at=now,
            updated_at=now,
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            progress_message="导入任务已创建，等待后台执行。",
            metadata={
                "paper_ids": request.paper_ids,
                "include_graph": request.include_graph,
                "include_embeddings": request.include_embeddings,
                "fast_mode": request.fast_mode,
                "question": request.question,
                "top_k": request.top_k,
                "reasoning_style": request.reasoning_style,
                "correlation_id": correlation_id,
            },
            status_metadata=self._build_status_metadata(
                lifecycle_status="queued",
                correlation_id=correlation_id,
            ),
        )
        self.save_job_state(
            job,
            event_type="tool_called",
            payload={"tool_name": "paper_import_job", "job_kind": job.kind},
        )
        self.observability_service.record_metric(
            metric_type="job_created",
            payload={"job_id": job.job_id, "job_kind": job.kind, "task_id": request.task_id},
        )
        self._set_conversation_active_job(request.conversation_id, job.job_id)
        task = asyncio.create_task(self._run_import_job(job.job_id, request, graph_runtime=graph_runtime))
        self._job_tasks[job.job_id] = task
        task.add_done_callback(lambda _: self._job_tasks.pop(job.job_id, None))
        return job

    async def start_todo_import_job(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        now = _now_iso()
        correlation_id = f"job_{uuid4().hex}"
        job = ResearchJob(
            job_id=f"job_{uuid4().hex}",
            kind="todo_import",
            status="queued",
            created_at=now,
            updated_at=now,
            task_id=task_id,
            conversation_id=request.conversation_id,
            progress_message="TODO 补充导入任务已创建，等待后台执行。",
            metadata={
                "todo_id": todo_id,
                "max_papers": request.max_papers,
                "include_graph": request.include_graph,
                "include_embeddings": request.include_embeddings,
                "correlation_id": correlation_id,
            },
            status_metadata=self._build_status_metadata(
                lifecycle_status="queued",
                correlation_id=correlation_id,
            ),
        )
        self.save_job_state(
            job,
            event_type="tool_called",
            payload={"tool_name": "todo_import_job", "job_kind": job.kind},
        )
        self.observability_service.record_metric(
            metric_type="job_created",
            payload={"job_id": job.job_id, "job_kind": job.kind, "task_id": task_id},
        )
        self._set_conversation_active_job(request.conversation_id, job.job_id)
        task = asyncio.create_task(
            self._run_todo_import_job(job.job_id, task_id, todo_id, request, graph_runtime=graph_runtime)
        )
        self._job_tasks[job.job_id] = task
        task.add_done_callback(lambda _: self._job_tasks.pop(job.job_id, None))
        return job

    def update_todo_status(self, task_id: str, todo_id: str, status: str) -> ResearchTaskResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        now = datetime.now(UTC).isoformat()
        updated_todo = todo.model_copy(
            update={
                "status": status,
                "metadata": {
                    **todo.metadata,
                    "last_status_change_at": now,
                },
            }
        )
        updated_task = self._replace_task_todo(task, updated_todo, updated_at=now)
        self.save_task_state(updated_task)
        return self.get_task(task_id)

    async def rerun_todo_search(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime: Any,
    ) -> ResearchTodoActionResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        existing_report = self.report_service.load_report(task.task_id, task.report_id)
        query, discovered_papers, merged_papers, warnings = await self._search_follow_up_papers(
            task=task,
            todo=todo,
            max_papers=request.max_papers,
            graph_runtime=graph_runtime,
        )
        now = datetime.now(UTC).isoformat()
        updated_todo = todo.model_copy(
            update={
                "metadata": {
                    **todo.metadata,
                    "last_action_at": now,
                    "last_action_type": "search",
                    "last_search_query": query,
                    "last_search_found": len(discovered_papers),
                },
            }
        )
        updated_report = self._rebuild_task_report(
            task=task,
            papers=merged_papers,
            existing_report=existing_report,
            warnings=warnings,
            action_title="重新检索",
            action_lines=[
                f"TODO：{updated_todo.content}",
                f"查询：{query}",
                f"新增/刷新候选论文：{len(discovered_papers)} 篇",
            ],
            generated_at=now,
        )
        updated_task = self._replace_task_todo(
            task,
            updated_todo,
            updated_at=now,
            paper_count=len(merged_papers),
            report_id=updated_report.report_id,
        )
        self.save_task_state(updated_task)
        self.report_service.save_papers(task.task_id, merged_papers)
        self.report_service.save_report(updated_report)
        return ResearchTodoActionResponse(
            task=updated_task,
            todo=updated_todo,
            papers=merged_papers,
            report=updated_report,
            warnings=warnings,
        )

    async def import_from_todo(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchTodoActionResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        existing_report = self.report_service.load_report(task.task_id, task.report_id)
        current_papers = self.report_service.load_papers(task.task_id)
        warnings: list[str] = []
        search_query: str | None = None

        candidate_papers = self._select_todo_import_candidates(
            task=task,
            todo=todo,
            papers=current_papers,
            limit=request.max_papers,
        )
        if not candidate_papers:
            search_query, discovered_papers, merged_papers, warnings = await self._search_follow_up_papers(
                task=task,
                todo=todo,
                max_papers=request.max_papers,
                graph_runtime=graph_runtime,
            )
            current_papers = merged_papers
            candidate_papers = self._select_todo_import_candidates(
                task=task,
                todo=todo,
                papers=current_papers,
                limit=request.max_papers,
            )
            if current_papers:
                self.report_service.save_papers(task.task_id, current_papers)

        if not candidate_papers:
            raise ValueError(f"No candidate papers with PDF available for TODO import: {todo_id}")

        import_result = await self.import_papers(
            ImportPapersRequest(
                task_id=task_id,
                paper_ids=[paper.paper_id for paper in candidate_papers],
                include_graph=request.include_graph,
                include_embeddings=request.include_embeddings,
                skill_name=request.skill_name,
            ),
            graph_runtime=graph_runtime,
        )

        refreshed_state = self.get_task(task_id)
        now = datetime.now(UTC).isoformat()
        updated_todo = self._find_todo(refreshed_state.task, todo_id).model_copy(
            update={
                "metadata": {
                    **self._find_todo(refreshed_state.task, todo_id).metadata,
                    "last_action_at": now,
                    "last_action_type": "import",
                    "last_import_count": import_result.imported_count,
                    "last_import_failed_count": import_result.failed_count,
                    "last_import_paper_ids": [paper.paper_id for paper in candidate_papers],
                    **({"last_search_query": search_query} if search_query else {}),
                },
            }
        )
        updated_report = self._rebuild_task_report(
            task=refreshed_state.task,
            papers=refreshed_state.papers,
            existing_report=refreshed_state.report or existing_report,
            warnings=warnings,
            action_title="补充导入",
            action_lines=[
                f"TODO：{updated_todo.content}",
                *([f"补充检索查询：{search_query}"] if search_query else []),
                f"尝试导入：{len(candidate_papers)} 篇",
                f"成功导入：{import_result.imported_count} 篇",
                f"跳过：{import_result.skipped_count} 篇",
                f"失败：{import_result.failed_count} 篇",
            ],
            generated_at=now,
        )
        updated_task = self._replace_task_todo(
            refreshed_state.task,
            updated_todo,
            updated_at=now,
            paper_count=len(refreshed_state.papers),
            report_id=updated_report.report_id,
        )
        self.save_task_state(updated_task)
        self.report_service.save_report(updated_report)
        return ResearchTodoActionResponse(
            task=updated_task,
            todo=updated_todo,
            papers=refreshed_state.papers,
            report=updated_report,
            warnings=warnings,
            import_result=import_result,
        )

    async def import_papers(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
        progress_callback: ImportProgressCallback | None = None,
    ) -> ImportPapersResponse:
        candidate_papers, persisted_papers = self._resolve_import_candidates(request)
        persisted_by_id = {paper.paper_id: paper for paper in persisted_papers}
        results: list[ImportedPaperResult] = []
        session_id, _ = self._resolve_research_session_id(
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )

        if candidate_papers:
            semaphore = asyncio.Semaphore(self.import_concurrency)
            progress_lock = asyncio.Lock()
            completed_count = 0

            async def process_candidate(index: int, paper: PaperCandidate) -> tuple[int, ImportedPaperResult, PaperCandidate]:
                async with semaphore:
                    result, updated_paper = await self._import_single_paper(
                        paper,
                        request=request,
                        graph_runtime=graph_runtime,
                        session_id=session_id,
                    )
                if progress_callback is not None:
                    async with progress_lock:
                        nonlocal completed_count
                        completed_count += 1
                        try:
                            maybe_awaitable = progress_callback(completed_count, len(candidate_papers), result)
                            if maybe_awaitable is not None:
                                await maybe_awaitable
                        except Exception:
                            pass
                return index, result, updated_paper

            completed_results = await asyncio.gather(
                *(process_candidate(index, paper) for index, paper in enumerate(candidate_papers))
            )
            for _, result, updated_paper in sorted(completed_results, key=lambda item: item[0]):
                results.append(result)
                persisted_by_id[updated_paper.paper_id] = updated_paper

        if request.task_id and persisted_papers:
            task = self.report_service.load_task(request.task_id)
            if task is not None:
                imported_document_ids = list(task.imported_document_ids)
                for result in results:
                    if result.status in {"imported", "skipped"} and result.document_id and result.document_id not in imported_document_ids:
                        imported_document_ids.append(result.document_id)
                current_report = self.report_service.load_report(task.task_id, task.report_id)
                updated_todo_items = self._resolve_todos_after_import(
                    task=task,
                    papers_by_id=persisted_by_id,
                )
                task_for_workspace = task.model_copy(
                    update={
                        "imported_document_ids": imported_document_ids,
                        "todo_items": updated_todo_items,
                    }
                )
                updated_task = task.model_copy(
                    update={
                        "updated_at": datetime.now(UTC).isoformat(),
                        "imported_document_ids": imported_document_ids,
                        "todo_items": updated_todo_items,
                        "workspace": build_workspace_from_task(
                            task=task_for_workspace,
                            report=current_report,
                            papers=list(persisted_by_id.values()),
                            stage="qa" if imported_document_ids else "ingest",
                            stop_reason="Paper import finished; the workspace is ready for grounded collection QA.",
                            metadata={
                                "imported_count": sum(1 for result in results if result.status == "imported"),
                                "skipped_count": sum(1 for result in results if result.status == "skipped"),
                                "failed_count": sum(1 for result in results if result.status == "failed"),
                                "fast_import_mode": request.fast_mode,
                                "pending_graph_backfill_ids": [
                                    result.document_id
                                    for result in results
                                    if result.graph_pending and result.document_id
                                ][:12],
                            },
                        ),
                    }
                )
                self.save_task_state(updated_task, conversation_id=request.conversation_id)
                if current_report is not None:
                    self.report_service.save_report(
                        current_report.model_copy(update={"workspace": updated_task.workspace})
                    )
            self.report_service.save_papers(request.task_id, list(persisted_by_id.values()))

        imported_count = sum(1 for result in results if result.status == "imported")
        skipped_count = sum(1 for result in results if result.status == "skipped")
        failed_count = sum(1 for result in results if result.status == "failed")
        pending_graph_backfill_ids = [
            result.document_id
            for result in results
            if result.graph_pending and result.document_id
        ]
        updated_task = self.report_service.load_task(request.task_id) if request.task_id else None
        updated_papers = list(persisted_by_id.values()) if persisted_by_id else []
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            papers=updated_papers,
            document_ids=updated_task.imported_document_ids if updated_task else [],
            selected_paper_ids=request.paper_ids,
            task_intent="research_import",
            metadata_update={
                "imported_count": imported_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "last_imported_document_ids": [
                    result.document_id
                    for result in results
                    if result.status in {"imported", "skipped"} and result.document_id
                ][:12],
                "pending_graph_backfill_ids": pending_graph_backfill_ids[:12],
            },
        )
        return ImportPapersResponse(
            results=results,
            imported_count=imported_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
        )

    async def _import_single_paper(
        self,
        paper: PaperCandidate,
        *,
        request: ImportPapersRequest,
        graph_runtime: Any,
        session_id: str | None,
    ) -> tuple[ImportedPaperResult, PaperCandidate]:
        try:
            existing_document_id = str(paper.metadata.get("document_id") or "").strip()
            existing_storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
            if paper.ingest_status == "ingested" and existing_document_id and existing_storage_uri:
                return (
                    ImportedPaperResult(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        status="skipped",
                        document_id=existing_document_id,
                        storage_uri=existing_storage_uri,
                        parsed=True,
                        indexed=True,
                        metadata={"reason": "already_ingested"},
                    ),
                    paper,
                )

            artifact = await self.paper_import_service.download_paper(paper)
            knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
            parsed_document = await knowledge_access.parse_document(
                file_path=artifact.storage_uri,
                document_id=artifact.document_id,
                session_id=session_id,
                metadata={
                    "research_paper_id": paper.paper_id,
                    "research_title": paper.title,
                    "research_source": paper.source,
                    "research_pdf_url": paper.pdf_url,
                },
                skill_name=request.skill_name,
            )
            index_result = await knowledge_access.index_document(
                parsed_document=parsed_document,
                charts=[],
                include_graph=(request.include_graph and not request.fast_mode),
                include_embeddings=request.include_embeddings,
                session_id=session_id,
                metadata={
                    "research_paper_id": paper.paper_id,
                    "research_title": paper.title,
                    "research_source": paper.source,
                    "fast_mode": request.fast_mode,
                },
                skill_name=request.skill_name,
            )
            graph_pending = bool(request.include_graph and request.fast_mode)
            import_status = "imported" if index_result.status == "succeeded" else "failed"
            zotero_sync = await self._sync_imported_paper_to_zotero(
                paper=paper,
                request=request,
                graph_runtime=graph_runtime,
            )
            result_metadata = {
                "index_status": index_result.status,
                "filename": artifact.filename,
                "index_mode": "fast_embeddings_first" if request.fast_mode else "full_sync",
                "graph_backfill_pending": graph_pending,
            }
            if zotero_sync is not None:
                result_metadata["zotero_sync"] = zotero_sync

            updated_paper_metadata = {
                **paper.metadata,
                "document_id": parsed_document.id,
                "storage_uri": artifact.storage_uri,
                "filename": artifact.filename,
                "index_status": index_result.status,
                "index_mode": "fast_embeddings_first" if request.fast_mode else "full_sync",
                "graph_backfill_pending": graph_pending,
            }
            if zotero_sync is not None:
                updated_paper_metadata["zotero_sync"] = zotero_sync
            return (
                ImportedPaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    status=import_status,
                    document_id=parsed_document.id,
                    storage_uri=artifact.storage_uri,
                    parsed=parsed_document.status == "parsed",
                    indexed=index_result.status == "succeeded",
                    graph_pending=graph_pending,
                    error_message=None if import_status == "imported" else "Indexing failed",
                    metadata=result_metadata,
                ),
                paper.model_copy(
                    update={
                        "ingest_status": "ingested" if import_status == "imported" else "selected",
                        "metadata": updated_paper_metadata,
                    }
                ),
            )
        except Exception as exc:
            return (
                ImportedPaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    status="failed",
                    error_message=str(exc),
                ),
                paper.model_copy(
                    update={
                        "ingest_status": "unavailable" if "No PDF URL available" in str(exc) else paper.ingest_status,
                        "metadata": {
                            **paper.metadata,
                            "last_import_error": str(exc),
                        },
                    }
                ),
            )

    async def _sync_imported_paper_to_zotero(
        self,
        *,
        paper: PaperCandidate,
        request: ImportPapersRequest,
        graph_runtime: Any,
    ) -> dict[str, Any] | None:
        del request
        research_function_service = getattr(graph_runtime, "research_function_service", None)
        if research_function_service is None or not hasattr(research_function_service, "sync_paper_to_zotero"):
            return None
        try:
            return await research_function_service.sync_paper_to_zotero(paper)
        except Exception as exc:
            logger.warning("Failed to sync imported paper to Zotero: %s", exc)
            return {
                "status": "failed",
                "action": "none",
                "zotero_item_key": None,
                "matched_by": None,
                "collection_name": None,
                "attachment_count": 0,
                "warnings": [f"Zotero sync failed after workspace import: {exc.__class__.__name__}"],
            }

    def _resolve_import_candidates(self, request: ImportPapersRequest) -> tuple[list[PaperCandidate], list[PaperCandidate]]:
        persisted_papers: list[PaperCandidate] = []
        if request.task_id:
            persisted_papers = self.report_service.load_papers(request.task_id)
        if request.papers:
            if request.paper_ids:
                allowed_ids = set(request.paper_ids)
                candidates = [paper for paper in request.papers if paper.paper_id in allowed_ids]
            else:
                candidates = request.papers
        elif persisted_papers:
            if request.paper_ids:
                allowed_ids = set(request.paper_ids)
                candidates = [paper for paper in persisted_papers if paper.paper_id in allowed_ids]
            else:
                candidates = persisted_papers
        else:
            candidates = []
        return candidates, persisted_papers

    async def _search_follow_up_papers(
        self,
        *,
        task: ResearchTask,
        todo: ResearchTodoItem,
        max_papers: int,
        graph_runtime: Any,
    ) -> tuple[str, list[PaperCandidate], list[PaperCandidate], list[str]]:
        query = self._build_todo_query(task, todo)
        agent_request = self._build_discovery_only_request(
            message=query,
            days_back=task.days_back,
            max_papers=max_papers,
            sources=list(task.sources),
            conversation_id=None,
            task_id=task.task_id,
            selected_paper_ids=list(task.workspace.must_read_paper_ids),
            selected_document_ids=list(task.imported_document_ids),
            trigger="todo_search",
            metadata_update={
                "todo_id": todo.todo_id,
                "todo_query": query,
                "response_contract": "todo_search",
            },
        )
        response = await self.run_agent(agent_request, graph_runtime=graph_runtime)
        refreshed_state = self.get_task(task.task_id)
        discovered_ids = {
            str(item).strip()
            for item in refreshed_state.task.metadata.get("last_search_discovered_paper_ids", [])
            if str(item).strip()
        }
        discovered_papers = [
            paper
            for paper in refreshed_state.papers
            if paper.paper_id in discovered_ids
        ]
        return query, discovered_papers, refreshed_state.papers, list(response.warnings)


    async def _run_import_job(
        self,
        job_id: str,
        request: ImportPapersRequest,
        *,
        graph_runtime,
    ) -> None:
        paper_total = max(len(request.paper_ids) or len(request.papers), 0)
        should_run_qa = bool(request.task_id and (request.question or "").strip())
        progress_total = max(paper_total + (1 if should_run_qa else 0), 1)
        self._update_job(
            job_id,
            status="running",
            progress_message="后台正在下载论文并执行 parse/index。",
            progress_current=0,
            progress_total=progress_total,
        )
        job = self.get_job(job_id)
        self.append_runtime_event(
            conversation_id=request.conversation_id,
            event_type="tool_called",
            task_id=request.task_id,
            correlation_id=job.status_metadata.correlation_id,
            payload={"tool_name": "paper_import", "paper_count": paper_total},
        )
        try:
            async def update_progress(current: int, total: int, result: ImportedPaperResult) -> None:
                self._update_job(
                    job_id,
                    progress_message=f"后台正在导入论文 {current}/{total}：{result.title} · {result.status}",
                    progress_current=current,
                )

            response = await self.import_papers(
                request,
                graph_runtime=graph_runtime,
                progress_callback=update_progress,
            )
            if request.fast_mode and request.include_graph:
                self._update_job(
                    job_id,
                    progress_message="主导入已完成，后台正在补齐图谱索引。",
                    progress_current=max(paper_total, 1),
                )
                await self._run_graph_backfill_for_import_results(
                    response=response,
                    request=request,
                    graph_runtime=graph_runtime,
                )
            task_response = self.get_task(request.task_id) if request.task_id else None
            ask_response = None
            qa_error_message: str | None = None
            output: dict[str, Any] = {
                "import_result": response.model_dump(mode="json"),
                "task_result": task_response.model_dump(mode="json") if task_response else None,
            }

            if should_run_qa and task_response and task_response.task.imported_document_ids:
                self._update_job(
                    job_id,
                    progress_message="论文导入完成，正在执行研究集合问答。",
                    progress_current=max(paper_total, 1),
                )
                self.append_runtime_event(
                    conversation_id=request.conversation_id,
                    event_type="tool_called",
                    task_id=request.task_id,
                    correlation_id=job.status_metadata.correlation_id,
                    payload={"tool_name": "collection_qa", "question": (request.question or "").strip()},
                )
                try:
                    ask_response = await self.ask_task_collection(
                        request.task_id,
                        ResearchTaskAskRequest(
                            question=(request.question or "").strip(),
                            top_k=request.top_k,
                            paper_ids=request.paper_ids,
                            skill_name=request.skill_name,
                            reasoning_style=request.reasoning_style,
                            conversation_id=request.conversation_id,
                        ),
                        graph_runtime=graph_runtime,
                    )
                    task_response = self.get_task(request.task_id)
                    output["task_result"] = task_response.model_dump(mode="json") if task_response else None
                    output["ask_result"] = ask_response.model_dump(mode="json")
                except Exception as exc:
                    qa_error_message = str(exc)
                    output["qa_error_message"] = qa_error_message

            notice = (
                f"后台导入完成：imported={response.imported_count} · skipped={response.skipped_count} · failed={response.failed_count}"
            )
            if request.fast_mode and request.include_graph:
                notice = f"{notice} · graph_backfill=completed"
            if ask_response is not None:
                notice = f"{notice} · qa=completed"
            elif qa_error_message:
                notice = f"{notice} · qa=failed"
            self._update_job(
                job_id,
                status="completed" if qa_error_message is None else "failed",
                progress_message=notice if qa_error_message is None else f"{notice}：{qa_error_message}",
                progress_current=progress_total if qa_error_message is None else max(paper_total, 1),
                error_message=qa_error_message,
                output=output,
            )
            self.observability_service.record_metric(
                metric_type="job_finished",
                payload={
                    "job_id": job_id,
                    "job_kind": "paper_import",
                    "status": "completed" if qa_error_message is None else "failed",
                },
            )
            if request.conversation_id:
                self.record_import_turn(
                    request.conversation_id,
                    task_response=task_response,
                    import_response=response,
                    selected_paper_ids=request.paper_ids,
                    notice=notice,
                )
                if ask_response is not None and task_response is not None:
                    self.record_qa_turn(
                        request.conversation_id,
                        task_response=task_response,
                        ask_response=ask_response,
                    )
                elif qa_error_message:
                    self.record_notice(
                        request.conversation_id,
                        task_response=task_response,
                        notice=f"研究集合问答失败：{qa_error_message}",
                        kind="error",
                        active_job_id=None,
                        last_error=qa_error_message,
                    )
        except Exception as exc:
            message = f"后台导入失败：{exc}"
            self._update_job(job_id, status="failed", progress_message=message, error_message=str(exc))
            self.observability_service.archive_failure(
                failure_type="paper_import_job_failed",
                payload={"job_id": job_id, "task_id": request.task_id, "error_message": str(exc)},
            )
            if request.conversation_id:
                task_response = self.get_task(request.task_id) if request.task_id else None
                self.record_notice(
                    request.conversation_id,
                    task_response=task_response,
                    notice=message,
                    kind="error",
                    active_job_id=None,
                    last_error=str(exc),
                )
            raise
        finally:
            self._set_conversation_active_job(request.conversation_id, None)

    async def _run_todo_import_job(
        self,
        job_id: str,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> None:
        self._update_job(
            job_id,
            status="running",
            progress_message="后台正在从 TODO 补充导入论文。",
            progress_current=0,
            progress_total=request.max_papers,
        )
        job = self.get_job(job_id)
        self.append_runtime_event(
            conversation_id=request.conversation_id,
            event_type="tool_called",
            task_id=task_id,
            correlation_id=job.status_metadata.correlation_id,
            payload={"tool_name": "todo_import", "todo_id": todo_id, "max_papers": request.max_papers},
        )
        try:
            response = await self.import_from_todo(task_id, todo_id, request, graph_runtime=graph_runtime)
            notice = "后台 TODO 补充导入已完成。"
            if response.import_result is not None:
                notice = (
                    f"后台 TODO 补充导入完成：imported={response.import_result.imported_count} · "
                    f"skipped={response.import_result.skipped_count} · failed={response.import_result.failed_count}"
                )
            self._update_job(
                job_id,
                status="completed",
                progress_message=notice,
                progress_current=response.import_result.imported_count if response.import_result else 0,
                output={
                    "imported_count": response.import_result.imported_count if response.import_result else 0,
                    "skipped_count": response.import_result.skipped_count if response.import_result else 0,
                    "failed_count": response.import_result.failed_count if response.import_result else 0,
                },
            )
            self.observability_service.record_metric(
                metric_type="job_finished",
                payload={"job_id": job_id, "job_kind": "todo_import", "status": "completed"},
            )
            if request.conversation_id:
                self.record_import_turn(
                    request.conversation_id,
                    task_response=ResearchTaskResponse(
                        task=response.task,
                        papers=response.papers,
                        report=response.report,
                        warnings=response.warnings,
                    ),
                    import_response=response.import_result or ImportPapersResponse(),
                    notice=notice,
                )
        except Exception as exc:
            message = f"后台 TODO 补充导入失败：{exc}"
            self._update_job(job_id, status="failed", progress_message=message, error_message=str(exc))
            self.observability_service.archive_failure(
                failure_type="todo_import_job_failed",
                payload={"job_id": job_id, "task_id": task_id, "error_message": str(exc)},
            )
            if request.conversation_id:
                self.record_notice(
                    request.conversation_id,
                    task_response=self.get_task(task_id),
                    notice=message,
                    kind="error",
                    active_job_id=None,
                    last_error=str(exc),
                )
            raise
        finally:
            self._set_conversation_active_job(request.conversation_id, None)

    async def _run_graph_backfill_for_import_results(
        self,
        *,
        response: ImportPapersResponse,
        request: ImportPapersRequest,
        graph_runtime: Any,
    ) -> None:
        if not request.task_id:
            return
        papers = self.report_service.load_papers(request.task_id)
        papers_by_document_id = {
            str(paper.metadata.get("document_id") or ""): paper
            for paper in papers
            if str(paper.metadata.get("document_id") or "")
        }
        for result in response.results:
            if not result.graph_pending or not result.document_id or not result.storage_uri:
                continue
            paper = papers_by_document_id.get(result.document_id)
            if paper is None:
                continue
            try:
                knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
                parsed_document = await knowledge_access.parse_document(
                    file_path=result.storage_uri,
                    document_id=result.document_id,
                    session_id=request.conversation_id,
                    metadata={
                        "research_paper_id": paper.paper_id,
                        "research_title": paper.title,
                        "research_source": paper.source,
                        "graph_backfill": True,
                    },
                    skill_name=request.skill_name,
                )
                backfill_result = await knowledge_access.graph_backfill_document(
                    parsed_document=parsed_document,
                    charts=[],
                    session_id=request.conversation_id,
                    metadata={
                        "research_paper_id": paper.paper_id,
                        "research_title": paper.title,
                        "research_source": paper.source,
                    },
                )
                result.graph_pending = False
                result.metadata["graph_backfill_status"] = backfill_result.status
                result.metadata["graph_backfill_pending"] = False
                paper.metadata["graph_backfill_pending"] = False
                paper.metadata["graph_backfill_status"] = backfill_result.status
            except Exception as exc:  # noqa: BLE001
                result.metadata["graph_backfill_status"] = "failed"
                result.metadata["graph_backfill_error"] = f"{exc.__class__.__name__}: {exc}"
        self.report_service.save_papers(request.task_id, papers)

    def _format_evidence_citation(self, document_id: str | None, page_number: int | None, source_type: str) -> str:
        parts = [part for part in [document_id or "unknown-doc", f"p.{page_number}" if page_number else None, source_type] if part]
        return " · ".join(parts)

    def _load_task_and_todo(self, task_id: str, todo_id: str) -> tuple[ResearchTask, ResearchTodoItem]:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task, self._find_todo(task, todo_id)

    def _find_todo(self, task: ResearchTask, todo_id: str) -> ResearchTodoItem:
        for item in task.todo_items:
            if item.todo_id == todo_id:
                return item
        raise KeyError(todo_id)

    def _replace_task_todo(
        self,
        task: ResearchTask,
        updated_todo: ResearchTodoItem,
        *,
        updated_at: str,
        **task_updates,
    ) -> ResearchTask:
        next_items = [
            updated_todo if item.todo_id == updated_todo.todo_id else item
            for item in task.todo_items
        ]
        return task.model_copy(
            update={
                "updated_at": updated_at,
                "todo_items": next_items,
                **task_updates,
            }
        )

    def _resolve_todos_after_import(
        self,
        *,
        task: ResearchTask,
        papers_by_id: dict[str, PaperCandidate],
    ) -> list[ResearchTodoItem]:
        now = datetime.now(UTC).isoformat()
        resolved_todos: list[ResearchTodoItem] = []
        for item in task.todo_items:
            todo_paper_ids = [
                paper_id
                for paper_id in item.metadata.get("paper_ids", [])
                if isinstance(paper_id, str) and paper_id
            ] or list(task.workspace.ingest_candidate_ids)
            todo_completed = bool(todo_paper_ids) and all(
                (
                    paper := papers_by_id.get(paper_id)
                ) is not None
                and paper.ingest_status == "ingested"
                and str(paper.metadata.get("document_id") or "").strip()
                for paper_id in todo_paper_ids
            )
            if item.status == "open" and item.metadata.get("todo_kind") == "ingest_priority" and todo_completed:
                resolved_todos.append(
                    item.model_copy(
                        update={
                            "status": "done",
                            "metadata": {
                                **item.metadata,
                                "last_status_change_at": now,
                                "auto_completed_by": "import_papers",
                                "completed_paper_ids": todo_paper_ids[:12],
                            },
                        }
                    )
                )
            else:
                resolved_todos.append(item)
        return resolved_todos

    def _build_todo_query(self, task: ResearchTask, todo: ResearchTodoItem) -> str:
        focus = (todo.question or todo.content).strip()
        if not focus:
            return task.topic
        normalized_topic = task.topic.lower()
        normalized_focus = focus.lower()
        if normalized_topic in normalized_focus:
            return focus
        return f"{task.topic} {focus}"

    def _select_todo_import_candidates(
        self,
        *,
        task: ResearchTask,
        todo: ResearchTodoItem,
        papers: list[PaperCandidate],
        limit: int,
    ) -> list[PaperCandidate]:
        available = [
            paper
            for paper in papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        ]
        if not available:
            return []
        ranked = self.paper_search_service.paper_ranker.rank(
            topic=self._build_todo_query(task, todo),
            papers=available,
            max_papers=max(limit, len(available)),
        )
        return ranked[:limit]

    def _merge_papers(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        deduped: dict[str, PaperCandidate] = {}
        key_aliases: dict[str, str] = {}
        for paper in [*existing_papers, *incoming_papers]:
            candidate_keys = self._paper_identity_keys(paper)
            key = next((key_aliases[item] for item in candidate_keys if item in key_aliases), None)
            if key is None:
                key = candidate_keys[0]
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
                for item in candidate_keys:
                    key_aliases[item] = key
                continue
            merged = existing.model_copy(
                update={
                    "authors": existing.authors or paper.authors,
                    "abstract": existing.abstract or paper.abstract,
                    "year": existing.year or paper.year,
                    "venue": existing.venue or paper.venue,
                    "pdf_url": existing.pdf_url or paper.pdf_url,
                    "url": existing.url or paper.url,
                    "citations": max(existing.citations or 0, paper.citations or 0) or None,
                    "published_at": existing.published_at or paper.published_at,
                    "relevance_score": max(existing.relevance_score or 0, paper.relevance_score or 0) or None,
                    "summary": existing.summary or paper.summary,
                    "ingest_status": existing.ingest_status
                    if existing.ingest_status != "not_selected"
                    else paper.ingest_status,
                    "metadata": {**paper.metadata, **existing.metadata},
                }
            )
            deduped[key] = merged
            for item in candidate_keys:
                key_aliases[item] = key
        merged_papers = list(deduped.values())
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged_papers,
            max_papers=max(len(merged_papers), 1),
        )

    def _paper_identity_keys(self, paper: PaperCandidate) -> list[str]:
        keys: list[str] = []
        if paper.doi:
            keys.append(f"doi:{paper.doi.lower()}")
        if paper.arxiv_id:
            keys.append(f"arxiv:{paper.arxiv_id.lower()}")
        keys.append(f"title:{_normalize_paper_title(paper.title)}")
        return keys

    def _rebuild_task_report(
        self,
        *,
        task: ResearchTask,
        papers: list[PaperCandidate],
        existing_report: ResearchReport | None,
        warnings: list[str],
        action_title: str,
        action_lines: list[str],
        generated_at: str,
    ) -> ResearchReport:
        base_report = self.paper_search_service.survey_writer.generate(
            topic=task.topic,
            task_id=task.task_id,
            papers=papers,
            warnings=warnings,
        )
        markdown = base_report.markdown.rstrip()
        if existing_report:
            qa_section = self._extract_markdown_section(existing_report.markdown, "## 研究集合问答补充")
            todo_section = self._extract_markdown_section(existing_report.markdown, "## TODO 执行记录")
            if qa_section:
                markdown = f"{markdown}\n\n{qa_section.strip()}"
            if todo_section:
                markdown = f"{markdown}\n\n{todo_section.strip()}"
        markdown = self._append_todo_action_entry(
            markdown=markdown,
            executed_at=generated_at,
            action_title=action_title,
            action_lines=action_lines,
        )
        carry_highlights = []
        carry_gaps = []
        metadata = {}
        if existing_report:
            carry_highlights = [
                item for item in existing_report.highlights
                if item.startswith("问答补充：") or item.startswith("TODO执行：")
            ]
            carry_gaps = list(existing_report.gaps)
            metadata.update(existing_report.metadata)
        action_highlight = f"TODO执行：{action_title} -> {action_lines[-1]}" if action_lines else f"TODO执行：{action_title}"
        metadata.update(base_report.metadata)
        metadata.update(
            {
                "last_todo_action_at": generated_at,
                "last_todo_action": action_title,
                "todo_action_count": int(metadata.get("todo_action_count") or 0) + 1,
            }
        )
        return base_report.model_copy(
            update={
                "report_id": existing_report.report_id if existing_report else base_report.report_id,
                "generated_at": generated_at,
                "markdown": markdown,
                "highlights": self._merge_text_entries(
                    [*base_report.highlights, action_highlight],
                    carry_highlights,
                    limit=12,
                ),
                "gaps": self._merge_text_entries(base_report.gaps, carry_gaps, limit=12),
                "metadata": metadata,
            }
        )

    def _extract_markdown_section(self, markdown: str, heading: str) -> str | None:
        lines = markdown.splitlines()
        start_index: int | None = None
        for index, line in enumerate(lines):
            if line.strip() == heading:
                start_index = index
                break
        if start_index is None:
            return None
        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            if lines[index].startswith("## ") and lines[index].strip() != heading:
                end_index = index
                break
        return "\n".join(lines[start_index:end_index]).strip()

    def _append_todo_action_entry(
        self,
        *,
        markdown: str,
        executed_at: str,
        action_title: str,
        action_lines: list[str],
    ) -> str:
        section_heading = "## TODO 执行记录"
        entry_lines = [f"### {executed_at} · {action_title}", *[f"- {line}" for line in action_lines]]
        if section_heading in markdown:
            return f"{markdown.rstrip()}\n\n" + "\n".join(entry_lines)
        prefix = markdown.rstrip()
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{section_heading}\n\n" + "\n".join(entry_lines)

    def _merge_text_entries(self, primary: list[str], secondary: list[str], *, limit: int) -> list[str]:
        merged: list[str] = []
        for item in [*primary, *secondary]:
            if item and item not in merged:
                merged.append(item)
            if len(merged) >= limit:
                break
        return merged
