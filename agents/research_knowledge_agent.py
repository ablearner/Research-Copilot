from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import replace

from domain.schemas.research import ImportPapersRequest
from tools.research.workflows.research_document import DocumentWorkflow
from domain.schemas.retrieval import RetrievalHit, merge_retrieval_hits, retrieval_hit_score  # noqa: F401
from tools.research.collection_qa_capability import (
    ResearchCollectionKnowledgeCapability,
    _tokenize_research_text,
    _is_insufficient_research_answer,
)

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

tokenize_research_text = _tokenize_research_text


is_insufficient_research_answer = _is_insufficient_research_answer


class ResearchKnowledgeAgent:
    """Primary agent for grounded evidence over imported papers and research state."""

    name = "ResearchKnowledgeAgent"

    def __init__(
        self,
        *,
        llm_adapter: Any | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.document_workflow = DocumentWorkflow()
        self._collection_qa = ResearchCollectionKnowledgeCapability(llm_adapter=llm_adapter)

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _dedupe_text(self, values: list[str], *, limit: int) -> list[str]:
        deduped = [v.strip() for v in values if v and v.strip()]
        return list(dict.fromkeys(deduped))[:limit]

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
        *,
        task_type: str = "compress_context",
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchToolResult

        if task_type == "import_papers":
            return await self._run_import_papers(context=context, decision=decision)
        if task_type == "sync_to_zotero":
            return await self._run_sync_to_zotero(context=context, decision=decision)
        if task_type == "compress_context":
            return await self._run_compress_context(context=context, decision=decision)
        if task_type == "understand_document":
            return await self.document_workflow.execute(context, decision)
        return ResearchToolResult(
            status="skipped",
            observation=f"ResearchKnowledgeAgent does not support task_type={task_type}",
            metadata={"reason": "unsupported_task_type"},
        )

    async def _run_import_papers(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import build_paper_import_input, build_paper_import_output

        task_response = context.task_response
        attempt_delta = ResearchStateDelta(import_attempted=True)
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for paper import",
                metadata={"reason": "missing_task"},
                state_delta=attempt_delta,
            )
        import_input = build_paper_import_input(context=context, decision=decision)
        paper_ids = import_input.resolved_paper_ids(task_response.papers)
        if not paper_ids:
            return ResearchToolResult(
                status="skipped",
                observation="no importable paper with an available PDF was found",
                metadata={"reason": "no_import_candidates"},
                state_delta=attempt_delta,
            )
        prepared = await context.research_service.paper_service.prepare_paper_import(
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
        import_result = prepared.response
        refreshed = prepared.task_response or task_response
        request = context.request
        rebuild_params = {
            "graph_runtime": context.graph_runtime,
            "conversation_id": request.conversation_id,
            "task": refreshed.task,
            "report": refreshed.report,
            "papers": refreshed.papers,
            "document_ids": refreshed.task.imported_document_ids,
            "selected_paper_ids": paper_ids,
            "skill_name": request.skill_name,
            "reasoning_style": request.reasoning_style,
            "metadata": request.metadata,
        }
        memory_ops: list[MemoryOp] = [MemoryOp(op_type="update_research_memory", params=prepared.memory_update_params)]
        if request.conversation_id:
            memory_ops.append(MemoryOp(
                op_type="record_import_turn",
                params={
                    "conversation_id": request.conversation_id,
                    "task_response": refreshed,
                    "import_response": import_result,
                    "selected_paper_ids": paper_ids,
                },
            ))
        delta = ResearchStateDelta(
            task=prepared.task_response.task if prepared.task_response else None,
            report=prepared.task_response.report if prepared.task_response else None,
            papers=prepared.papers,
            save_task_conversation_id=request.conversation_id,
            task_response=refreshed,
            import_result=import_result,
            import_attempted=True,
            rebuild_execution_context=True,
            rebuild_execution_context_params=rebuild_params,
            memory_ops=memory_ops or None,
        )
        output = build_paper_import_output(paper_ids=paper_ids, import_result=import_result)
        return ResearchToolResult(
            status="succeeded" if import_result.failed_count == 0 else "failed" if import_result.imported_count == 0 and import_result.skipped_count == 0 else "succeeded",
            observation=(
                f"paper import finished; imported={import_result.imported_count}; "
                f"skipped={import_result.skipped_count}; failed={import_result.failed_count}"
            ),
            metadata=output.to_metadata(),
            state_delta=delta,
        )

    async def _run_sync_to_zotero(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import resolve_active_message

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
        synced_count = sum(1 for item in results if str(item.get("status") or "") in {"imported", "reused"})
        failed_count = len(results) - synced_count
        return ResearchToolResult(
            status="succeeded" if failed_count == 0 else "failed" if synced_count == 0 else "succeeded",
            state_delta=ResearchStateDelta(zotero_sync_results=results),
            observation=f"zotero sync finished; synced={synced_count}; failed={failed_count}",
            metadata={
                "paper_ids": paper_ids,
                "synced_count": synced_count,
                "failed_count": failed_count,
                "results": results,
                "collection_name": collection_name,
            },
        )

    async def _run_compress_context(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.agent_protocol.mixins import persist_workspace_results
        from runtime.research.unified_action_adapters import build_context_compression_input, build_context_compression_output

        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return ResearchToolResult(status="skipped", observation="no execution context is available for compression", metadata={"reason": "missing_execution_context"})
        compression_input = build_context_compression_input(context=context, decision=decision)
        selected_paper_ids = compression_input.resolved_selected_paper_ids()
        compressed = context.research_service.research_context_manager.compress_papers(
            papers=list(context.papers),
            selected_paper_ids=selected_paper_ids,
            paper_reading_tool=context.research_service.paper_reading_tool,
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
        execution_context = replace(
            execution_context,
            research_context=updated_context,
            context_slices=context.research_service.build_context_slices(
                updated_context, selected_paper_ids=selected_paper_ids,
            ),
        )
        compression_summary = {
            "paper_count": len({summary.paper_id for summary in compressed}),
            "summary_count": len(compressed),
            "levels": sorted({summary.level for summary in compressed}),
            "compressed_paper_ids": list(dict.fromkeys(summary.paper_id for summary in compressed)),
        }
        ws_result = persist_workspace_results(replace(context, execution_context=execution_context), compression_summary=compression_summary, persist=False)
        output = build_context_compression_output(
            compression_summary=compression_summary,
        )
        memory_ops: list[MemoryOp] = []
        if ws_result is not None and ws_result.memory_save_context_params is not None:
            memory_ops.append(MemoryOp(
                op_type="save_context",
                params=ws_result.memory_save_context_params,
            ))
        delta = ResearchStateDelta(
            task=ws_result.updated_task if ws_result else None,
            report=ws_result.updated_report if ws_result else None,
            save_task_conversation_id=context.request.conversation_id,
            save_task_event_type=ws_result.save_event_type if ws_result else None,
            save_task_event_payload=ws_result.save_event_payload if ws_result else None,
            task_response=ws_result.updated_task_response if ws_result else None,
            compressed_context_summary=compression_summary,
            execution_context=ws_result.updated_execution_context if ws_result else execution_context,
            memory_ops=memory_ops or None,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"context compressed; papers={compression_summary['paper_count']}; summaries={compression_summary['summary_count']}"
            ),
            metadata=output.to_metadata(),
            state_delta=delta,
        )

    # ------------------------------------------------------------------
    # Collection QA knowledge methods (used by ResearchCollectionQACapability)
    # ------------------------------------------------------------------


    def decide(self, state: Any) -> tuple[str, str]:
        return self._collection_qa.decide(state)

    async def plan_collection_queries(self, state: Any) -> list[str]:
        return await self._collection_qa.plan_collection_queries(
            state, max_queries=3, include_cross_lingual=False
        )

    def plan(self, state: Any) -> list[str]:
        return self._collection_qa.plan_collection_queries_heuristic(
            state, max_queries=3, include_cross_lingual=False
        )

    async def retrieve_collection_evidence(self, *, graph_runtime: Any, state: Any, query: str) -> list[RetrievalHit]:
        return await self._collection_qa.retrieve_collection_evidence(
            graph_runtime=graph_runtime, state=state, query=query
        )

    async def retrieve_graph_summary(self, *, graph_runtime: Any, state: Any) -> list[RetrievalHit]:
        return await self._collection_qa.retrieve_graph_summary(graph_runtime=graph_runtime, state=state)

    async def retrieve(self, *, graph_runtime: Any, state: Any, query: str | None = None) -> list[RetrievalHit]:
        if query is None:
            return await self.retrieve_graph_summary(graph_runtime=graph_runtime, state=state)
        return await self.retrieve_collection_evidence(graph_runtime=graph_runtime, state=state, query=query)

    def build_collection_manifest(self, state: Any) -> list[RetrievalHit]:
        return self._collection_qa.build_collection_manifest(state, knowledge_agent=self.name)

    def build_hits(self, state: Any) -> list[RetrievalHit]:
        return self.build_collection_manifest(state)

    async def propose_collection_queries(self, state: Any) -> list[str]:
        return await self._collection_qa.propose_collection_queries(state)

    def propose_queries(self, state: Any) -> list[str]:
        return self._collection_qa.propose_collection_queries_heuristic(state)
