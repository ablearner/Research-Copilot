from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from domain.schemas.research import ImportPapersRequest, PaperCandidate, normalize_reasoning_style
from domain.schemas.retrieval import RetrievalHit
from tools.research.knowledge_access import ResearchKnowledgeAccess

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "论文",
    "研究",
    "paper",
    "papers",
    "study",
    "studies",
}


def tokenize_research_text(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        normalized = token.lower().strip()
        if len(normalized) <= 1 or normalized in _STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


# Retrieval helpers – canonical location: domain.schemas.retrieval
# Re-exported here for backward compatibility.
from domain.schemas.retrieval import merge_retrieval_hits, retrieval_hit_score  # noqa: F401


def is_insufficient_research_answer(*, answer: str, confidence: float, evidence_count: int) -> bool:
    lowered = answer.lower()
    uncertainty_signals = ("证据不足", "无法确认", "不能确认", "信息不足", "insufficient evidence", "not enough evidence")
    return confidence < 0.45 or evidence_count < 2 or any(signal in lowered for signal in uncertainty_signals)


class ResearchKnowledgeAgent:
    """Primary agent for grounded evidence over imported papers and research state."""

    name = "ResearchKnowledgeAgent"

    def __init__(
        self,
        *,
        llm_adapter: Any | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter

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
        return ResearchToolResult(
            status="skipped",
            observation=f"ResearchKnowledgeAgent does not support task_type={task_type}",
            metadata={"reason": "unsupported_task_type"},
        )

    async def _run_import_papers(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import build_paper_import_input, build_paper_import_output

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
        refreshed = context.research_service.get_task(task_response.task.task_id)
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
        memory_ops: list[MemoryOp] = []
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
            task_response=refreshed,
            import_result=import_result,
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
        from runtime.research.agent_protocol.base import ResearchToolResult
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
        compression_summary = {
            "paper_count": len({summary.paper_id for summary in compressed}),
            "summary_count": len(compressed),
            "levels": sorted({summary.level for summary in compressed}),
            "compressed_paper_ids": list(dict.fromkeys(summary.paper_id for summary in compressed)),
        }
        ws_result = persist_workspace_results(context, compression_summary=compression_summary, persist=False)
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
        if not state.queries:
            return "plan_queries", "A collection-aware query plan is needed before evidence gathering."
        if state.document_ids and state.pending_query() is not None:
            return "retrieve_collection_evidence", "There are pending collection retrieval queries that have not run yet."
        if state.document_ids and not state.summary_checked:
            return "retrieve_graph_summary", "Graph summary evidence can still strengthen collection-level coverage."
        if not state.manifest_built:
            return "build_collection_manifest", "The assistant should synthesize paper pool and report metadata into collection evidence."
        if state.qa is None and self._should_refine_before_answer(state):
            return "refine_query", "Evidence is still thin, so the knowledge agent should broaden the collection query."
        if state.qa is None:
            return "answer_question", "Enough collection evidence is available to produce a grounded answer."
        if self._should_refine_after_answer(state):
            return "refine_query", "The draft answer is still under-supported, so one more retrieval pass is justified."
        return "finish", "The collection QA loop has completed evidence gathering, synthesis, and grounded answering."

    async def plan_collection_queries(self, state: Any) -> list[str]:
        planned = self._heuristic_plan_collection_queries(state)
        if not self._should_use_plan_and_execute(state) or self.llm_adapter is None:
            return planned
        reasoning_plan = await self._plan_queries(
            objective=f"Grounded research collection QA for: {state.question}",
            seed_queries=planned,
            context={
                "research_topic": state.task.topic,
                "report_highlights": state.report.highlights[:3] if state.report else [],
                "paper_titles": [paper.title for paper in state.papers[:5]],
                "memory_hints": self._memory_hints(state),
            },
            max_queries=3,
        )
        return self._normalize_queries([*reasoning_plan["queries"], *planned], limit=3)

    def _heuristic_plan_collection_queries(self, state: Any) -> list[str]:
        candidates = [state.question]
        topic = state.task.topic.strip()
        if topic and topic not in state.question:
            candidates.append(f"{topic} {state.question}")
        if state.report and state.report.highlights:
            candidates.append(f"{state.question} {state.report.highlights[0]}")
        memory_hints = self._memory_hints(state)
        memory_focus = str(memory_hints.get("current_task_intent") or "").strip()
        if memory_focus and memory_focus not in state.question:
            candidates.append(f"{state.question} {memory_focus}")
        planned: list[str] = []
        for item in candidates:
            normalized = " ".join(item.strip().split())
            if normalized and normalized not in planned:
                planned.append(normalized)
        return planned[:3]

    def plan(self, state: Any) -> list[str]:
        return self._heuristic_plan_collection_queries(state)

    async def retrieve_collection_evidence(self, *, graph_runtime: Any, state: Any, query: str) -> list[RetrievalHit]:
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        execution_context = getattr(state, "execution_context", None)
        scope_filters = self._scope_filters(state)
        result = await knowledge_access.retrieve(
            question=query,
            document_ids=state.document_ids,
            top_k=state.top_k,
            filters={
                "research_task_id": state.task.task_id,
                "research_topic": state.task.topic,
                "qa_mode": "research_collection",
                **scope_filters,
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=state.task.task_id,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
        )
        return result.retrieval_result.hits

    async def retrieve_graph_summary(self, *, graph_runtime: Any, state: Any) -> list[RetrievalHit]:
        if not state.document_ids:
            return []
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        execution_context = getattr(state, "execution_context", None)
        scope_filters = self._scope_filters(state)
        summary_output = await knowledge_access.query_graph_summary(
            question=state.question,
            document_ids=state.document_ids,
            top_k=max(3, min(state.top_k, 6)),
            filters={
                "research_task_id": state.task.task_id,
                "research_topic": state.task.topic,
                "qa_mode": "research_collection",
                **scope_filters,
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=state.task.task_id,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
        )
        return list(getattr(summary_output, "hits", []) or [])

    async def retrieve(self, *, graph_runtime: Any, state: Any, query: str | None = None) -> list[RetrievalHit]:
        if query is None:
            return await self.retrieve_graph_summary(graph_runtime=graph_runtime, state=state)
        return await self.retrieve_collection_evidence(graph_runtime=graph_runtime, state=state, query=query)

    def build_collection_manifest(self, state: Any) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        top_papers = state.papers[: max(3, min(len(state.papers), state.top_k))]
        for index, paper in enumerate(top_papers, start=1):
            hits.append(self._paper_manifest_hit(index=index, paper=paper))

        if state.report and self._should_include_report_summary_evidence(state):
            for index, highlight in enumerate(state.report.highlights[:3], start=1):
                hits.append(
                    RetrievalHit(
                        id=f"manifest:highlight:{state.report.report_id}:{index}",
                        source_type="text_block",
                        source_id=f"report_highlight_{index}",
                        content=f"研究报告亮点 #{index}: {highlight}",
                        merged_score=max(0.75 - index * 0.05, 0.45),
                        metadata={
                            "provider": "research_collection_manifest",
                            "manifest_kind": "report_highlight",
                            "source": "llm_generated_research_report",
                            "evidence_tier": "report_summary_fallback",
                            "summary_only": True,
                            "llm_generated_summary": True,
                            "report_id": state.report.report_id,
                            "knowledge_agent": self.name,
                        },
                    )
                )
            for index, gap in enumerate(state.report.gaps[:2], start=1):
                hits.append(
                    RetrievalHit(
                        id=f"manifest:gap:{state.report.report_id}:{index}",
                        source_type="text_block",
                        source_id=f"report_gap_{index}",
                        content=f"研究报告证据缺口 #{index}: {gap}",
                        merged_score=max(0.6 - index * 0.05, 0.3),
                        metadata={
                            "provider": "research_collection_manifest",
                            "manifest_kind": "report_gap",
                            "source": "llm_generated_research_report",
                            "evidence_tier": "report_summary_fallback",
                            "summary_only": True,
                            "llm_generated_summary": True,
                            "report_id": state.report.report_id,
                            "knowledge_agent": self.name,
                        },
                    )
                )
        return hits

    def build_hits(self, state: Any) -> list[RetrievalHit]:
        return self.build_collection_manifest(state)

    async def propose_collection_queries(self, state: Any) -> list[str]:
        refined = self._heuristic_propose_collection_queries(state)
        if not self._should_use_plan_and_execute(state) or self.llm_adapter is None:
            return refined
        reasoning_plan = await self._plan_queries(
            objective=f"Broaden evidence coverage for research QA: {state.question}",
            seed_queries=refined or list(state.queries),
            context={
                "research_topic": state.task.topic,
                "existing_queries": list(state.queries),
                "report_gaps": state.report.gaps[:2] if state.report else [],
                "paper_titles": [paper.title for paper in state.papers[:5]],
                "memory_hints": self._memory_hints(state),
            },
            max_queries=2,
        )
        return self._normalize_queries([*reasoning_plan["queries"], *refined], limit=2)

    def _heuristic_propose_collection_queries(self, state: Any) -> list[str]:
        focus_terms: list[str] = []
        for paper in state.papers[:5]:
            focus_terms.extend(tokenize_research_text(f"{paper.title} {paper.summary or paper.abstract}")[:6])
        if state.report:
            focus_terms.extend(tokenize_research_text(" ".join(state.report.highlights[:3]))[:6])
        memory_hints = self._memory_hints(state)
        for value in memory_hints.values():
            if isinstance(value, str):
                focus_terms.extend(tokenize_research_text(value)[:4])
        dedup_terms: list[str] = []
        for term in focus_terms:
            if term not in dedup_terms:
                dedup_terms.append(term)

        candidates = [
            f"{state.task.topic} {state.question}",
            f"{state.question} evidence summary",
            f"{state.task.topic} {' '.join(dedup_terms[:4])}".strip(),
        ]
        refined: list[str] = []
        existing = set(state.queries)
        for query in candidates:
            normalized = " ".join(query.strip().split())
            if not normalized or normalized in existing or normalized in refined:
                continue
            refined.append(normalized)
        return refined[:2]

    def propose_queries(self, state: Any) -> list[str]:
        return self._heuristic_propose_collection_queries(state)

    def _should_use_plan_and_execute(self, state: Any) -> bool:
        if self.llm_adapter is None:
            return False
        return normalize_reasoning_style(self._reasoning_style(state)) != "react"

    async def _plan_queries(
        self,
        *,
        objective: str,
        seed_queries: list[str],
        context: dict[str, Any] | None = None,
        max_queries: int = 4,
    ) -> dict[str, Any]:
        """Internal Plan-and-Execute: decompose objective into queries."""
        import json
        from adapters.llm.base import LLMAdapterError, is_expected_provider_error
        heuristic = {
            "plan_steps": [
                "Identify the core research objective.",
                "Break it into search facets for better coverage.",
                "Emit a compact set of high-yield queries.",
            ],
            "queries": self._normalize_queries(seed_queries or [objective], limit=max_queries),
            "reasoning_summary": "Kept strongest seed queries as a compact next-step query set.",
        }
        if self.llm_adapter is None:
            return heuristic
        payload = {
            "objective": objective,
            "seed_queries": seed_queries,
            "max_queries": max(1, min(max_queries, 6)),
            "context": context or {},
        }
        try:
            result = await self.llm_adapter.generate_structured(
                prompt=(
                    "You are a Plan-and-Execute query planner. "
                    "First plan the subproblems privately, then return only structured output. "
                    "Produce a short list of plan_steps, a deduplicated query set, and a concise reasoning_summary."
                ),
                input_data=payload,
                response_model=None,
            )
            if isinstance(result, str):
                result = json.loads(result)
            if not isinstance(result, dict):
                return heuristic
            queries = self._normalize_queries(
                [*(result.get("queries") or []), *seed_queries],
                limit=max_queries,
            )
            if not queries:
                return heuristic
            return {
                "plan_steps": list(result.get("plan_steps") or heuristic["plan_steps"]),
                "queries": queries,
                "reasoning_summary": str(result.get("reasoning_summary") or heuristic["reasoning_summary"]),
            }
        except (LLMAdapterError, OSError, ValueError, Exception) as exc:
            if is_expected_provider_error(exc):
                import logging
                logging.getLogger(__name__).warning("Plan-and-Execute planning failed; using heuristic: %s", exc)
            return heuristic

    def _reasoning_style(self, state: Any) -> str | None:
        request = getattr(state, "request", None)
        if request is not None:
            style = getattr(request, "reasoning_style", None)
            if style:
                return style
        execution_context = getattr(state, "execution_context", None)
        if execution_context is None:
            return None
        preference_context = getattr(execution_context, "preference_context", None) or {}
        if not isinstance(preference_context, dict):
            return None
        return preference_context.get("reasoning_style")

    def _memory_hints(self, state: Any) -> dict[str, Any]:
        execution_context = getattr(state, "execution_context", None)
        if execution_context is None:
            return {}
        memory_hints = getattr(execution_context, "memory_hints", None) or {}
        return dict(memory_hints) if isinstance(memory_hints, dict) else {}

    def _normalize_queries(self, queries: list[str], *, limit: int) -> list[str]:
        normalized_queries: list[str] = []
        for query in queries:
            normalized = " ".join(str(query).strip().split())
            if normalized and normalized not in normalized_queries:
                normalized_queries.append(normalized)
        return normalized_queries[:limit]

    def _scope_filters(self, state: Any) -> dict[str, Any]:
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        return {
            "qa_scope_mode": str(metadata.get("qa_scope_mode") or "all_imported"),
            "selected_paper_ids": list(metadata.get("selected_paper_ids") or []),
            "selected_document_ids": list(metadata.get("selected_document_ids") or state.document_ids or []),
        }

    def _should_include_report_summary_evidence(self, state: Any) -> bool:
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        policy = str(metadata.get("report_summary_evidence_policy") or "fallback").strip().lower()
        if policy in {"always", "include", "true"}:
            return True
        if policy in {"never", "exclude", "false"}:
            return False
        return not bool(getattr(state, "retrieval_hits", []) or getattr(state, "summary_hits", []))

    def _paper_manifest_hit(self, *, index: int, paper: PaperCandidate) -> RetrievalHit:
        snippet_parts = [
            f"候选论文 #{index}: {paper.title}",
            f"source={paper.source}",
        ]
        if paper.authors:
            snippet_parts.append(f"authors={', '.join(paper.authors[:4])}")
        if paper.year:
            snippet_parts.append(f"year={paper.year}")
        if paper.venue:
            snippet_parts.append(f"venue={paper.venue}")
        if paper.summary:
            snippet_parts.append(f"summary={paper.summary}")
        elif paper.abstract:
            snippet_parts.append(f"abstract={paper.abstract[:360]}")
        if paper.metadata.get("must_read"):
            snippet_parts.append("must_read=true")
        if paper.metadata.get("selected_for_ingest") or paper.pdf_url:
            snippet_parts.append("ingest_ready=true")
        return RetrievalHit(
            id=f"manifest:paper:{paper.paper_id}",
            source_type="text_block",
            source_id=paper.paper_id,
            document_id=str(paper.metadata.get("document_id") or "") or None,
            content=" | ".join(snippet_parts),
            merged_score=float(paper.relevance_score or max(0.95 - index * 0.08, 0.35)),
            metadata={
                "provider": "research_collection_manifest",
                "manifest_kind": "paper_card",
                "source": "paper_metadata_manifest",
                "evidence_tier": "paper_metadata",
                "summary_only": True,
                "paper_source": paper.source,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "knowledge_agent": self.name,
            },
        )

    def _should_refine_before_answer(self, state: Any) -> bool:
        if state.refinement_used or not state.document_ids:
            return False
        return len(state.all_hits()) < 2

    def _should_refine_after_answer(self, state: Any) -> bool:
        if state.refinement_used or not state.document_ids or state.qa is None:
            return False
        confidence = state.qa.confidence if state.qa.confidence is not None else 0.0
        return is_insufficient_research_answer(
            answer=state.qa.answer,
            confidence=confidence,
            evidence_count=len(state.evidence_bundle.evidences),
        )
