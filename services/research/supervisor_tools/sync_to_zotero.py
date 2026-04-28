"""Sync to Zotero supervisor tool."""

from __future__ import annotations

from typing import Any

from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import resolve_active_message


class SyncToZoteroTool:
    name = "sync_to_zotero"

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
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
