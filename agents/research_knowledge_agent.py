from __future__ import annotations

import re
from typing import Any

from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from agents.research_qa_agent import normalize_reasoning_style
from services.research.research_knowledge_access import ResearchKnowledgeAccess

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


def retrieval_hit_score(hit: RetrievalHit) -> float:
    return float(hit.merged_score or hit.vector_score or hit.graph_score or 0.0)


def merge_retrieval_hits(*groups: list[RetrievalHit]) -> list[RetrievalHit]:
    merged: dict[tuple[str, str, str | None, str | None], RetrievalHit] = {}
    for group in groups:
        for hit in group:
            key = (
                hit.id,
                hit.source_id,
                hit.document_id,
                (hit.content or "")[:240] or None,
            )
            existing = merged.get(key)
            if existing is None or retrieval_hit_score(hit) >= retrieval_hit_score(existing):
                merged[key] = hit
    return sorted(merged.values(), key=retrieval_hit_score, reverse=True)


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
