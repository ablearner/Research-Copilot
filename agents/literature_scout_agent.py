from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from domain.schemas.research import PaperCandidate, ResearchTopicPlan, normalize_reasoning_style
from tools.research.paper_search import format_search_warning

if TYPE_CHECKING:
    from tools.research.paper_search import PaperSearchService
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
    "研究",
    "相关",
    "recent",
    "latest",
    "paper",
    "papers",
    "最近",
    "最新",
    "这三个⽉",
    "这三个月",
    "三个月",
    "什么",
    "有关",
    "关于",
    "文章",
    "论文",
    "文献",
    "哪些",
    "吗",
    "领域",
    "新发表",
}
_SOURCE_SEARCH_TIMEOUT_SECONDS = 35.0
_MAX_SOURCE_SEARCH_TIMEOUT_SECONDS = 90.0


class SourceSearchError(Exception):
    def __init__(self, *, source: str, query: str, cause: Exception) -> None:
        super().__init__(str(cause) or cause.__class__.__name__)
        self.source = source
        self.query = query
        self.cause = cause


def _normalize_query(value: str) -> str:
    return " ".join(value.strip().split())


def _dedupe_queries(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for value in values:
        normalized = _normalize_query(value)
        canonical = _canonical_query_key(normalized)
        if normalized and canonical not in seen_keys:
            deduped.append(normalized)
            seen_keys.add(canonical)
    return deduped


def _canonical_query_key(value: str) -> str:
    normalized = _normalize_query(value).lower()
    if not normalized:
        return ""
    normalized = re.sub(r"[^\w\u4e00-\u9fff\-\s]+", " ", normalized)
    normalized = normalized.replace("vision-and-language navigation", "vln")
    normalized = normalized.replace("vision and language navigation", "vln")
    normalized = normalized.replace("vision language navigation", "vln")
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(normalized):
        cleaned = token.strip().lower()
        if not cleaned or cleaned in _STOPWORDS or len(cleaned) <= 1:
            continue
        tokens.append(cleaned)
    if not tokens:
        return normalized
    ordered_unique = list(dict.fromkeys(tokens))
    return " ".join(ordered_unique)


def _paper_keywords(papers: list[PaperCandidate], *, limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for paper in papers[:8]:
        for token in _TOKEN_PATTERN.findall(f"{paper.title} {paper.abstract}"):
            normalized = token.lower().strip()
            if len(normalized) <= 2 or normalized in _STOPWORDS:
                continue
            counts[normalized] += 1
    return [token for token, _count in counts.most_common(limit)]


def _dynamic_source_search_timeout_seconds(tool: Any, *, fallback_seconds: float = _SOURCE_SEARCH_TIMEOUT_SECONDS) -> float:
    configured_timeout = getattr(tool, "timeout_seconds", None)
    if isinstance(configured_timeout, (int, float)) and configured_timeout > 0:
        return min(max(float(fallback_seconds), float(configured_timeout) + 10.0), _MAX_SOURCE_SEARCH_TIMEOUT_SECONDS)
    return float(fallback_seconds)


class LiteratureScoutAgent:
    """Primary agent for literature discovery.

    This agent owns topic planning, source scouting, and query refinement. The
    old planner/source/refinement nodes are intentionally collapsed here so the
    research runtime is not modeled as a long chain of tiny agents.
    """

    name = "LiteratureScoutAgent"

    def __init__(
        self,
        paper_search_service: PaperSearchService | None = None,
        *,
        llm_adapter: Any | None = None,
        curation_tool: Any | None = None,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.llm_adapter = llm_adapter
        self.curation_tool = curation_tool

    # ------------------------------------------------------------------
    # SpecialistAgent protocol — owns the full discovery pipeline
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from types import SimpleNamespace

        from core.utils import now_iso as _now_iso
        from domain.schemas.research import ResearchTaskResponse
        from domain.research_workspace import build_workspace_state
        from runtime.research.agent_protocol.base import (
            ResearchStateDelta,
            ResearchToolResult,
            _update_runtime_progress,
        )
        from runtime.research.unified_action_adapters import (
            build_expert_context,
            build_literature_search_input,
            build_literature_search_output,
        )

        request = context.request
        search_input = build_literature_search_input(context=context, decision=decision)
        expert_context = build_expert_context(context=context, decision=decision)
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

        curation_tool = (
            self.curation_tool
            or getattr(context.research_service, "paper_curation_tool", None)
        )
        exec_ctx = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=active_task,
            report=task_response.report if task_response is not None else None,
            papers=list(task_response.papers if task_response is not None else []),
            document_ids=list(active_task.imported_document_ids),
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )

        # --- Discovery pipeline: agent owns the full orchestration ---
        state = SimpleNamespace(
            topic=search_request.topic,
            days_back=search_request.days_back,
            max_papers=search_request.max_papers,
            sources=list(search_request.sources),
            task_id=active_task.task_id,
            execution_context=exec_ctx,
            supervisor_instruction=expert_context.get("supervisor_instruction"),
            skill_context=expert_context.get("skill_context"),
            expert_context=expert_context,
            max_rounds=2,
            round_index=0,
            queried_pairs=set(),
            search_completed=False,
            curation_completed=False,
            raw_papers=[],
            trace=[],
            warnings=[],
            curated_papers=[],
            must_read_ids=[],
            ingest_candidate_ids=[],
            report=None,
            todo_items=[],
            refinement_used=False,
        )

        _update_runtime_progress(context, stage="search_literature", node="search_literature:planning", status="running", summary="Planning literature discovery queries.")
        plan = await self._discover_plan(state)
        state.initial_plan = plan
        state.active_queries = list(plan.queries)

        _update_runtime_progress(context, stage="search_literature", node="search_literature:source_search", status="running", summary="Searching literature sources.")
        raw_papers, search_warnings = await self.search(state)
        state.warnings = list(search_warnings)

        _update_runtime_progress(context, stage="search_literature", node="search_literature:curation", status="running", summary="Curating candidate papers.")
        curated_papers, must_read_ids, ingest_candidate_ids = curation_tool.curate(
            topic=search_request.topic,
            raw_papers=raw_papers,
            max_papers=search_request.max_papers,
        )
        state.curated_papers = curated_papers
        state.must_read_ids = must_read_ids
        state.ingest_candidate_ids = ingest_candidate_ids

        # Discovery produces evidence only. The supervisor schedules writing as
        # a separate action, and an existing report remains unchanged until then.
        sm = {
            "last_search_query": search_request.topic,
            "last_search_discovered_paper_ids": [p.paper_id for p in curated_papers],
            "last_search_discovered_count": len(curated_papers),
            "search_plan": plan.model_dump(mode="json"),
        }
        final_papers = self._refresh_paper_pool(
            existing_papers=list(task_response.papers),
            incoming_papers=curated_papers,
            ranking_topic=search_request.topic,
        )
        final_report = task_response.report
        discovery_workspace = build_workspace_state(
            objective=active_task.topic,
            stage="discover",
            papers=final_papers,
            imported_document_ids=list(active_task.imported_document_ids),
            report=final_report,
            plan=plan,
            todo_items=list(active_task.todo_items),
            must_read_ids=must_read_ids,
            ingest_candidate_ids=ingest_candidate_ids,
            extra_findings=[f"Discovered {len(curated_papers)} candidate papers for {search_request.topic}."],
            metadata={**active_task.workspace.metadata, **sm},
        )
        workspace = discovery_workspace
        if context.task is not None:
            previous = active_task.workspace
            workspace = previous.model_copy(update={
                "current_stage": "discover",
                "research_questions": list(dict.fromkeys([*previous.research_questions, *discovery_workspace.research_questions])),
                "key_findings": list(dict.fromkeys([*previous.key_findings, *discovery_workspace.key_findings])),
                "must_read_paper_ids": list(dict.fromkeys([*previous.must_read_paper_ids, *must_read_ids])),
                "ingest_candidate_ids": list(dict.fromkeys([*previous.ingest_candidate_ids, *ingest_candidate_ids])),
                "status_summary": discovery_workspace.status_summary,
                "metadata": discovery_workspace.metadata,
            })
        final_task = active_task.model_copy(update={
            "status": "completed",
            "paper_count": len(final_papers),
            "workspace": workspace,
            "updated_at": _now_iso(),
            "metadata": {**active_task.metadata, **sm},
        })
        response = ResearchTaskResponse(
            task=final_task, papers=final_papers, report=final_report,
            warnings=list(search_warnings),
        )

        # --- P1: Build state delta (Runtime will apply + persist) ---
        delta = ResearchStateDelta(
            task=final_task,
            papers=final_papers,
            report=final_report,
            task_response=response,
            save_task_conversation_id=request.conversation_id,
            rebuild_execution_context=True,
            rebuild_execution_context_params={
                "graph_runtime": context.graph_runtime,
                "conversation_id": request.conversation_id,
                "task": response.task,
                "report": response.report,
                "papers": list(response.papers),
                "document_ids": list(response.task.imported_document_ids),
                "selected_paper_ids": request.selected_paper_ids,
                "skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style,
                "metadata": request.metadata,
            },
            record_task_turn=bool(request.conversation_id),
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:completed",
            status="completed",
            summary="Literature discovery completed.",
            extra={"paper_count": len(response.papers), "query_count": len(plan.queries)},
        )
        output = build_literature_search_output(task_response=response)
        action_label = "updated task" if context.task is not None else "created task"
        return ResearchToolResult(
            status="succeeded",
            observation=f"{action_label} {response.task.task_id}; papers={len(response.papers)}; report={bool(response.report)}",
            metadata=output.to_metadata(),
            state_delta=delta,
        )

    # ------------------------------------------------------------------
    # Discovery pipeline steps (timeout + heuristic fallback)
    # ------------------------------------------------------------------

    async def _discover_plan(self, state: Any) -> ResearchTopicPlan:
        from runtime.research.agent_protocol.base import (
            _DISCOVERY_PLAN_TIMEOUT_SECONDS,
            _llm_stage_timeout_seconds,
            _should_fallback_llm_stage,
        )

        timeout = _llm_stage_timeout_seconds(
            self.llm_adapter,
            fallback_seconds=_DISCOVERY_PLAN_TIMEOUT_SECONDS,
            slack_seconds=10.0,
        )
        try:
            return await asyncio.wait_for(self.plan(state), timeout=timeout)
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Discovery planning timed out or hit transport failure; falling back to heuristic planner: %s",
                exc,
            )
            return self._require_search_service().topic_planner.plan(
                topic=state.topic,
                days_back=state.days_back,
                max_papers=state.max_papers,
                sources=state.sources,
            )

    def _refresh_paper_pool(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        paper_search_service = self._require_search_service()
        merged = paper_search_service._dedupe([*existing_papers, *incoming_papers])
        return paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged,
            max_papers=max(len(merged), 1),
        )

    # ------------------------------------------------------------------
    # Topic planning + source search
    # ------------------------------------------------------------------

    async def plan(self, state: Any) -> ResearchTopicPlan:
        paper_search_service = self._require_search_service()
        planner = paper_search_service.topic_planner
        if hasattr(planner, "plan_async"):
            plan_kwargs = {
                "topic": state.topic,
                "days_back": state.days_back,
                "max_papers": state.max_papers,
                "sources": state.sources,
                "supervisor_instruction": getattr(state, "supervisor_instruction", None),
            }
            if getattr(state, "skill_context", None):
                plan_kwargs["skill_context"] = getattr(state, "skill_context", None)
            try:
                base_plan = await planner.plan_async(**plan_kwargs)
            except TypeError:
                plan_kwargs.pop("skill_context", None)
                base_plan = await planner.plan_async(**plan_kwargs)
        else:
            base_plan = planner.plan(
                topic=state.topic,
                days_back=state.days_back,
                max_papers=state.max_papers,
                sources=state.sources,
            )
        plan_metadata = {
            **base_plan.metadata,
            "planner": self.name,
            "manager_agent": "ResearchSupervisorAgent",
            "agent_architecture": "main_agents_plus_tools",
        }
        if self._should_use_plan_and_execute(state) and self.llm_adapter is not None:
            reasoning_plan = await self._plan_queries(
                objective=f"Literature discovery for {state.topic}",
                seed_queries=base_plan.queries,
                context={
                    "topic": state.topic,
                    "days_back": state.days_back,
                    "max_papers": state.max_papers,
                    "sources": list(state.sources),
                    "memory_hints": self._memory_hints(state),
                    "expert_context": getattr(state, "expert_context", {}),
                },
                max_queries=4,
            )
            return base_plan.model_copy(
                update={
                    "queries": _dedupe_queries([*reasoning_plan["queries"], *base_plan.queries])[:4],
                    "metadata": {
                        **plan_metadata,
                        "reasoning_style": "plan_and_execute",
                        "reasoning_summary": reasoning_plan["reasoning_summary"],
                        "plan_steps": " | ".join(reasoning_plan["plan_steps"][:3]),
                    },
                }
            )
        return base_plan.model_copy(
            update={
                "metadata": plan_metadata,
            }
        )

    async def search(self, state: Any) -> tuple[list[PaperCandidate], list[str]]:
        paper_search_service = self._require_search_service()
        task_specs: list[asyncio.Task[tuple[str, str, list[PaperCandidate]]]] = []
        warnings: list[str] = []
        per_query_limit = max(state.max_papers, 12)
        seen_source_query_keys: set[tuple[str, str]] = set()
        for source in state.sources:
            tool = paper_search_service._get_tool(source)
            if tool is None:
                warnings.append(f"{source} 当前尚未接入检索工具，已跳过。")
                continue
            for query in self._queries_for_source(source=source, queries=state.active_queries):
                canonical_query = _canonical_query_key(query)
                if not canonical_query:
                    continue
                cache_key = (source, canonical_query)
                if cache_key in state.queried_pairs:
                    continue
                if cache_key in seen_source_query_keys:
                    continue
                seen_source_query_keys.add(cache_key)
                async def _run_source_search(
                    *,
                    selected_source: str = source,
                    selected_query: str = query,
                    selected_tool: Any = tool,
                ) -> tuple[str, str, list[PaperCandidate]]:
                    source_timeout_seconds = _dynamic_source_search_timeout_seconds(selected_tool)
                    try:
                        papers = await asyncio.wait_for(
                            selected_tool.search(
                                query=selected_query,
                                max_results=per_query_limit,
                                days_back=state.days_back,
                            ),
                            timeout=source_timeout_seconds,
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise SourceSearchError(source=selected_source, query=selected_query, cause=exc) from exc
                    return selected_source, selected_query, papers

                task_specs.append(asyncio.create_task(_run_source_search()))

        found_papers: list[PaperCandidate] = []
        for task in asyncio.as_completed(task_specs):
            try:
                source, query, papers = await task
            except Exception as exc:  # pragma: no cover - provider failures depend on environment
                if isinstance(exc, SourceSearchError):
                    warnings.append(format_search_warning(source=exc.source, query=exc.query, exc=exc.cause))
                else:
                    warnings.append(format_search_warning(source="unknown", query="", exc=exc))
                continue
            state.queried_pairs.add((source, query))
            for paper in papers:
                found_papers.append(
                    paper.model_copy(
                        update={
                            "metadata": {
                                **paper.metadata,
                                "search_query": query[:180],
                                "search_query_key": canonical_query,
                                "search_round": state.round_index + 1,
                                "scout_agent": self.name,
                            }
                        }
                    )
                )
        return found_papers, warnings

    def _queries_for_source(self, *, source: str, queries: list[str]) -> list[str]:
        paper_search_service = self._require_search_service()
        planner = getattr(paper_search_service, "topic_planner", None)
        source_query_selector = getattr(planner, "queries_for_source", None)
        if callable(source_query_selector):
            return _dedupe_queries(source_query_selector(source=source, queries=queries))

        cleaned = [_normalize_query(query) for query in queries if _normalize_query(query)]
        if not cleaned:
            return []

        english_queries = [query for query in cleaned if re.search(r"[A-Za-z]", query) and not re.search(r"[\u4e00-\u9fff]", query)]
        if source in {"arxiv", "semantic_scholar"}:
            if not english_queries:
                # Extract English tokens from mixed-language queries as a fallback
                for query in cleaned:
                    en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query)
                    en_only = " ".join(t for t in en_tokens if len(t) > 1)
                    if en_only.strip():
                        english_queries.append(en_only.strip())
            if not english_queries:
                return []
            # Semantic Scholar's public endpoint is aggressively rate limited;
            # prefer one strong English query over fanning out weak variants.
            limit = 1 if source == "semantic_scholar" else 2
            return _dedupe_queries(english_queries[:limit])

        if source == "openalex" and english_queries:
            return _dedupe_queries(english_queries[:3])

        return _dedupe_queries(cleaned[:4])

    async def propose_queries(self, state: Any) -> list[str]:
        heuristic_queries = self._heuristic_propose_queries(state)
        if not self._should_use_plan_and_execute(state) or self.llm_adapter is None:
            return heuristic_queries
        reasoning_plan = await self._plan_queries(
            objective=f"Broaden literature coverage for {state.topic}",
            seed_queries=heuristic_queries or list(state.active_queries),
            context={
                "topic": state.topic,
                "searched_queries": state.searched_queries(),
                "paper_titles": [paper.title for paper in state.curated_papers[:6]],
                "must_read_ids": list(state.must_read_ids[:4]),
                "warnings": list(state.warnings[-3:]),
                "memory_hints": self._memory_hints(state),
                "expert_context": getattr(state, "expert_context", {}),
            },
            max_queries=3,
        )
        return _dedupe_queries([*reasoning_plan["queries"], *heuristic_queries])[:3]

    def _heuristic_propose_queries(self, state: Any) -> list[str]:
        keywords = _paper_keywords(state.curated_papers)
        memory_hints = self._memory_hints(state)
        memory_focus = " ".join(
            str(value).strip()
            for value in memory_hints.values()
            if isinstance(value, str) and str(value).strip()
        ).strip()
        query_candidates = [
            f"{state.topic} survey review",
            f"{state.topic} benchmark evaluation",
        ]
        if memory_focus:
            query_candidates.append(f"{state.topic} {memory_focus}")
        if keywords:
            query_candidates.append(f"{state.topic} {' '.join(keywords[:4])}")
            query_candidates.append(" ".join(keywords[:6]))
        deduped_queries: list[str] = []
        searched_queries = set(state.searched_queries())
        for query in query_candidates:
            normalized = _normalize_query(query)
            if not normalized or normalized in searched_queries or normalized in deduped_queries:
                continue
            deduped_queries.append(normalized)
        return deduped_queries[:3]

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
            "queries": self._normalize_plan_queries(seed_queries or [objective], limit=max_queries),
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
                    "Produce a short list of plan_steps, a deduplicated query set, and a concise reasoning_summary. "
                    "If context.expert_context contains skill_context, apply relevant search-scope and evidence-boundary guidance. "
                    "IMPORTANT: All queries MUST be in English (translate non-English topics into English academic keywords), "
                    "because the target sources (arXiv, OpenAlex, Semantic Scholar) only index English-language papers."
                ),
                input_data=payload,
                response_model=None,
            )
            if isinstance(result, str):
                result = json.loads(result)
            if not isinstance(result, dict):
                return heuristic
            queries = self._normalize_plan_queries(
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

    def _normalize_plan_queries(self, queries: list[str], *, limit: int) -> list[str]:
        deduped: list[str] = []
        for query in queries:
            normalized = " ".join(str(query).strip().split())
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[:max(1, min(limit, 6))]

    def _reasoning_style(self, state: Any) -> str | None:
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

    def _require_search_service(self) -> PaperSearchService:
        if self.paper_search_service is None:
            raise RuntimeError("LiteratureScoutAgent requires PaperSearchService for this action")
        return self.paper_search_service
