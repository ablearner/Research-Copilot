from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.literature_scout_agent import LiteratureScoutAgent
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.research import (
    PaperCandidate,
    PaperSource,
    ResearchReport,
    ResearchTask,
    ResearchTodoItem,
    ResearchTopicPlan,
    ResearchWorkspaceState,
)
from reasoning.strategies import ReasoningStrategySet
from services.research.paper_search_service import PaperSearchService
from services.research.research_context import ResearchExecutionContext
from services.research.research_workspace import build_workspace_state
from skills.research import PaperCurationSkill

PRIMARY_RESEARCH_AGENTS = (
    "ResearchSupervisorAgent",
    "LiteratureScoutAgent",
    "ResearchWriterAgent",
)
PRIMARY_RESEARCH_AGENTS_LABEL = ",".join(PRIMARY_RESEARCH_AGENTS)
PRIMARY_RESEARCH_SKILLS = (
    "PaperCurationSkill",
    "TopicPlanningSkill",
    "ResearchQueryRewriteSkill",
    "PaperRankingSkill",
    "SurveyWritingSkill",
)
PRIMARY_RESEARCH_SKILLS_LABEL = ",".join(PRIMARY_RESEARCH_SKILLS)
RESEARCH_RUNTIME_ARCHITECTURE = "main_agents_plus_skills"
RESEARCH_DECISION_MODEL = "llm_dynamic_single_manager"
AutonomousResearchDecision = tuple[str, str]
_MANAGER_DECISION_TIMEOUT_SECONDS = 35.0
_MAX_MANAGER_DECISION_TIMEOUT_SECONDS = 180.0


def _manager_decision_timeout_seconds(llm_adapter: Any | None) -> float:
    configured_timeout = getattr(llm_adapter, "timeout_seconds", None)
    provider_binding = getattr(llm_adapter, "provider_binding", None)
    if configured_timeout is None and provider_binding is not None:
        configured_timeout = getattr(provider_binding, "timeout_seconds", None)
    if isinstance(configured_timeout, (int, float)) and configured_timeout > 0:
        return min(
            max(_MANAGER_DECISION_TIMEOUT_SECONDS, float(configured_timeout) + 10.0),
            _MAX_MANAGER_DECISION_TIMEOUT_SECONDS,
        )
    return _MANAGER_DECISION_TIMEOUT_SECONDS


class AutonomousResearchLLMDecision(BaseModel):
    next_step: Literal[
        "plan_search",
        "search_sources",
        "curate_papers",
        "refine_search",
        "write_report",
        "plan_todos",
        "finish",
    ]
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass(slots=True)
class AutonomousResearchTraceStep:
    step_index: int
    agent: str
    decision: str
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AutonomousResearchState:
    topic: str
    days_back: int
    max_papers: int
    sources: list[PaperSource]
    task_id: str | None = None
    execution_context: ResearchExecutionContext | None = None
    max_rounds: int = 2
    round_index: int = 0
    initial_plan: ResearchTopicPlan | None = None
    active_queries: list[str] = field(default_factory=list)
    search_completed: bool = False
    curation_completed: bool = False
    refinement_used: bool = False
    queried_pairs: set[tuple[str, str]] = field(default_factory=set)
    raw_papers: list[PaperCandidate] = field(default_factory=list)
    curated_papers: list[PaperCandidate] = field(default_factory=list)
    must_read_ids: list[str] = field(default_factory=list)
    ingest_candidate_ids: list[str] = field(default_factory=list)
    report: ResearchReport | None = None
    todo_items: list[ResearchTodoItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[AutonomousResearchTraceStep] = field(default_factory=list)
    stagnant_decision_count: int = 0
    last_progress_signature: str = ""

    def searched_queries(self) -> list[str]:
        ordered: list[str] = []
        for _source, query in self.queried_pairs:
            if query not in ordered:
                ordered.append(query)
        return ordered

    def progress_signature(self) -> str:
        return "|".join(
            [
                str(self.initial_plan is not None),
                str(self.search_completed),
                str(self.curation_completed),
                str(len(self.raw_papers)),
                str(len(self.curated_papers)),
                str(len(self.must_read_ids)),
                str(len(self.ingest_candidate_ids)),
                str(self.report is not None),
                str(len(self.todo_items)),
                str(self.round_index),
                str(self.refinement_used),
            ]
        )


@dataclass(slots=True)
class AutonomousResearchBundle:
    plan: ResearchTopicPlan
    papers: list[PaperCandidate]
    report: ResearchReport
    workspace: ResearchWorkspaceState
    warnings: list[str]
    todo_items: list[ResearchTodoItem]
    trace: list[AutonomousResearchTraceStep]
    must_read_ids: list[str]
    ingest_candidate_ids: list[str]


class AutonomousResearchGraphState(TypedDict, total=False):
    runtime_state: AutonomousResearchState
    current_decision: AutonomousResearchDecision | None
    current_step_index: int
    exhausted: bool


class AutonomousResearchRuntime:
    """Manager-agent loop for autonomous literature discovery and synthesis."""

    def __init__(
        self,
        *,
        paper_search_service: PaperSearchService,
        plan_and_solve_reasoning_agent: Any | None = None,
        max_steps: int = 11,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.plan_and_solve_reasoning_agent = plan_and_solve_reasoning_agent
        self.reasoning_strategies = ReasoningStrategySet(
            query_planning=plan_and_solve_reasoning_agent,
        )
        self.scout_agent = LiteratureScoutAgent(
            paper_search_service,
            reasoning_strategies=self.reasoning_strategies,
        )
        self.curation_skill = PaperCurationSkill(paper_search_service)
        llm_adapter = self.reasoning_strategies.llm_adapter
        self.writer_agent = ResearchWriterAgent(
            paper_search_service,
            llm_adapter=llm_adapter,
            reasoning_strategies=self.reasoning_strategies,
        )
        self.llm_adapter = llm_adapter
        self.max_steps = max_steps
        self.graph = self._build_graph().compile()

    async def run(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[PaperSource],
        task_id: str | None = None,
        execution_context: ResearchExecutionContext | None = None,
    ) -> AutonomousResearchBundle:
        state = AutonomousResearchState(
            topic=topic,
            days_back=days_back,
            max_papers=max_papers,
            sources=sources,
            task_id=task_id,
            execution_context=execution_context,
        )
        graph_state = await self._run_manager_loop(
            {
                "runtime_state": state,
                "current_step_index": 0,
                "exhausted": False,
            }
        )
        state = graph_state["runtime_state"]
        if graph_state.get("exhausted"):
            state.warnings.append(f"autonomous research runtime reached max_steps={self.max_steps} before finish")

        if state.initial_plan is None:
            state.initial_plan = await self.scout_agent.plan(state)
        if not state.curated_papers and state.raw_papers:
            curated, must_read_ids, ingest_candidate_ids = self.curation_skill.curate(
                topic=state.topic,
                raw_papers=state.raw_papers,
                max_papers=state.max_papers,
            )
            state.curated_papers = curated
            state.must_read_ids = must_read_ids
            state.ingest_candidate_ids = ingest_candidate_ids
        if state.report is None:
            state.report = self.writer_agent.synthesize(state)
        if not state.todo_items:
            state.todo_items = self.writer_agent.plan_todos(state)
        stop_reason = (
            f"autonomous research runtime reached max_steps={self.max_steps} before finish"
            if graph_state.get("exhausted")
            else "ResearchSupervisorAgent finished discovery, curation, synthesis, and TODO planning."
        )
        workspace = build_workspace_state(
            objective=topic,
            stage="complete",
            papers=state.curated_papers,
            imported_document_ids=[],
            report=state.report,
            plan=state.initial_plan,
            todo_items=state.todo_items,
            must_read_ids=state.must_read_ids,
            ingest_candidate_ids=state.ingest_candidate_ids,
            stop_reason=stop_reason,
            metadata={
                "decision_model": RESEARCH_DECISION_MODEL,
                "autonomy_rounds": state.round_index + 1,
                "trace_steps": len(state.trace),
            },
        )

        plan = state.initial_plan.model_copy(
            update={
                "queries": state.searched_queries() or state.active_queries or state.initial_plan.queries,
                "metadata": {
                    **state.initial_plan.metadata,
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": RESEARCH_RUNTIME_ARCHITECTURE,
                    "primary_agents": PRIMARY_RESEARCH_AGENTS_LABEL,
                    "primary_skills": PRIMARY_RESEARCH_SKILLS_LABEL,
                    "decision_model": RESEARCH_DECISION_MODEL,
                    "autonomy_rounds": str(state.round_index + 1),
                    "trace_steps": str(len(state.trace)),
                    "memory_enabled": state.execution_context.memory_enabled if state.execution_context else False,
                    "session_id": state.execution_context.session_id if state.execution_context else None,
                },
            }
        )
        report = state.report.model_copy(
            update={
                "workspace": workspace,
                "metadata": {
                    **state.report.metadata,
                    "agent_architecture": RESEARCH_RUNTIME_ARCHITECTURE,
                    "primary_agents": PRIMARY_RESEARCH_AGENTS_LABEL,
                    "primary_skills": PRIMARY_RESEARCH_SKILLS_LABEL,
                    "decision_model": RESEARCH_DECISION_MODEL,
                    "memory_enabled": state.execution_context.memory_enabled if state.execution_context else False,
                    "session_id": state.execution_context.session_id if state.execution_context else None,
                }
            }
        )
        return AutonomousResearchBundle(
            plan=plan,
            papers=state.curated_papers,
            report=report,
            workspace=workspace,
            warnings=state.warnings,
            todo_items=state.todo_items,
            trace=state.trace,
            must_read_ids=state.must_read_ids,
            ingest_candidate_ids=state.ingest_candidate_ids,
        )

    async def _run_manager_loop(
        self,
        initial_state: AutonomousResearchGraphState,
    ) -> AutonomousResearchGraphState:
        state: AutonomousResearchGraphState = dict(initial_state)
        while True:
            current_step_index = int(state.get("current_step_index", 0) or 0)
            if current_step_index >= self.max_steps:
                state["exhausted"] = True
                return state
            step_index = current_step_index + 1
            decision = await self._manager_decision(state)
            state["current_step_index"] = step_index
            state["current_decision"] = decision
            state = {**state, **self._on_manager_decision(state, step_index, decision)}
            route = self._route_decision(decision)
            if route == "finish":
                finish_update = await self.finish_node(state)
                return {**state, **finish_update}
            action_update = await self._run_routed_action(state, route)
            state = {**state, **action_update}

    async def _run_routed_action(
        self,
        state: AutonomousResearchGraphState,
        route: str,
    ) -> AutonomousResearchGraphState:
        if route == "plan_search":
            return await self.plan_search_node(state)
        if route == "search_sources":
            return await self.search_sources_node(state)
        if route == "curate_papers":
            return await self.curate_papers_node(state)
        if route == "refine_search":
            return await self.refine_search_node(state)
        if route == "write_report":
            return await self.write_report_node(state)
        if route == "plan_todos":
            return await self.plan_todos_node(state)
        raise ValueError(f"unsupported autonomous research route: {route}")

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AutonomousResearchGraphState)
        graph.add_node("manager_node", self._graph_manager_node)
        graph.add_node("plan_search_node", self.plan_search_node)
        graph.add_node("search_sources_node", self.search_sources_node)
        graph.add_node("curate_papers_node", self.curate_papers_node)
        graph.add_node("refine_search_node", self.refine_search_node)
        graph.add_node("write_report_node", self.write_report_node)
        graph.add_node("plan_todos_node", self.plan_todos_node)
        graph.add_node("finish_node", self.finish_node)

        graph.add_edge("__start__", "manager_node")
        graph.add_conditional_edges(
            "manager_node",
            self._graph_route_after_manager,
            {
                "plan_search": "plan_search_node",
                "search_sources": "search_sources_node",
                "curate_papers": "curate_papers_node",
                "refine_search": "refine_search_node",
                "write_report": "write_report_node",
                "plan_todos": "plan_todos_node",
                "finish": "finish_node",
            },
        )
        for node_name in (
            "plan_search_node",
            "search_sources_node",
            "curate_papers_node",
            "refine_search_node",
            "write_report_node",
            "plan_todos_node",
        ):
            graph.add_edge(node_name, "manager_node")
        graph.add_edge("finish_node", END)
        return graph

    async def _graph_manager_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        current_step_index = int(state.get("current_step_index", 0) or 0)
        if current_step_index >= self.max_steps:
            return {"exhausted": True}
        step_index = current_step_index + 1
        decision = await self._manager_decision(state)
        update = self._on_manager_decision(state, step_index, decision)
        return {
            "current_step_index": step_index,
            "current_decision": decision,
            **update,
        }

    def _graph_route_after_manager(self, state: AutonomousResearchGraphState) -> str:
        if state.get("exhausted"):
            return "finish"
        decision = state.get("current_decision")
        if decision is None:
            return "finish"
        return self._route_decision(decision)

    async def _manager_decision(self, state: AutonomousResearchGraphState) -> AutonomousResearchDecision:
        return await self._decide_next_step(state["runtime_state"])

    def _route_decision(self, decision: AutonomousResearchDecision) -> str:
        return decision[0]

    def _on_manager_decision(
        self,
        state: AutonomousResearchGraphState,
        step_index: int,
        decision: AutonomousResearchDecision,
    ) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        decision_name, rationale = decision
        signature = runtime_state.progress_signature()
        if signature == runtime_state.last_progress_signature:
            runtime_state.stagnant_decision_count += 1
        else:
            runtime_state.stagnant_decision_count = 0
            runtime_state.last_progress_signature = signature
        runtime_state.trace.append(
            AutonomousResearchTraceStep(
                step_index=step_index,
                agent="ResearchSupervisorAgent",
                decision=decision_name,
                rationale=rationale,
                metadata={
                    "round_index": runtime_state.round_index,
                    "stagnant_decision_count": runtime_state.stagnant_decision_count,
                    "primary_agents": PRIMARY_RESEARCH_AGENTS_LABEL,
                    "primary_skills": PRIMARY_RESEARCH_SKILLS_LABEL,
                },
            )
        )
        return {
            "runtime_state": runtime_state,
        }

    async def plan_search_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.initial_plan = await self.scout_agent.plan(runtime_state)
        runtime_state.active_queries = list(runtime_state.initial_plan.queries)
        runtime_state.search_completed = False
        return {"runtime_state": runtime_state}

    async def search_sources_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        found_papers, warnings = await self.scout_agent.search(runtime_state)
        runtime_state.raw_papers.extend(found_papers)
        runtime_state.warnings.extend(warning for warning in warnings if warning not in runtime_state.warnings)
        runtime_state.search_completed = True
        runtime_state.curation_completed = False
        return {"runtime_state": runtime_state}

    async def curate_papers_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        curated, must_read_ids, ingest_candidate_ids = self.curation_skill.curate(
            topic=runtime_state.topic,
            raw_papers=runtime_state.raw_papers,
            max_papers=runtime_state.max_papers,
        )
        runtime_state.curated_papers = curated
        runtime_state.must_read_ids = must_read_ids
        runtime_state.ingest_candidate_ids = ingest_candidate_ids
        runtime_state.curation_completed = True
        return {"runtime_state": runtime_state}

    async def refine_search_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        step_index = state.get("current_step_index", 0)
        refined_queries = await self.scout_agent.propose_queries(runtime_state)
        runtime_state.refinement_used = True
        if not refined_queries:
            runtime_state.trace.append(
                AutonomousResearchTraceStep(
                    step_index=step_index,
                    agent="LiteratureScoutAgent",
                    decision="skip_refinement",
                    rationale="No novel query variations remained after considering searched queries.",
                )
            )
            return {"runtime_state": runtime_state}
        runtime_state.round_index += 1
        runtime_state.active_queries = refined_queries
        runtime_state.search_completed = False
        runtime_state.curation_completed = False
        runtime_state.trace.append(
            AutonomousResearchTraceStep(
                step_index=step_index,
                agent="LiteratureScoutAgent",
                decision="refine_queries",
                rationale="Coverage or PDF availability was weak, so the scout expanded the search space.",
                metadata={"queries": refined_queries},
            )
        )
        return {"runtime_state": runtime_state}

    async def write_report_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        await self._generate_language_summaries_if_needed(runtime_state)
        runtime_state.report = await self.writer_agent.synthesize_async(runtime_state)
        return {"runtime_state": runtime_state}

    async def plan_todos_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.todo_items = await self.writer_agent.plan_todos_async(runtime_state)
        return {"runtime_state": runtime_state}

    # ------------------------------------------------------------------
    # Language-aware summary generation
    # ------------------------------------------------------------------

    _SUMMARIZE_ZH_PROMPT = (
        "你是一个学术论文摘要翻译助手。请将以下英文论文摘要翻译并精简为一段简洁的中文总结（不超过150字）。"
        "保留核心方法和贡献，不要添加评价。专有名词可保留英文。\n\n"
        "论文标题：{title}\n摘要：{abstract}"
    )
    _SUMMARIZE_EN_PROMPT = (
        "You are an academic paper summary assistant. Rewrite the following paper abstract into a concise English summary "
        "(within 90 words). Preserve the core method and contribution, avoid evaluation language, and keep paper titles and "
        "technical terms unchanged.\n\nTitle: {title}\nAbstract: {abstract}"
    )

    @staticmethod
    def _topic_contains_chinese(topic: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in topic)

    def _target_answer_language(self, state: AutonomousResearchState) -> str:
        preference_context = getattr(getattr(state, "execution_context", None), "preference_context", None) or {}
        language = str(preference_context.get("answer_language") or "").strip()
        if language:
            return language
        return "zh-CN" if self._topic_contains_chinese(state.topic) else "en-US"

    async def _generate_language_summaries_if_needed(self, state: AutonomousResearchState) -> None:
        llm_adapter = self.reasoning_strategies.llm_adapter
        if llm_adapter is None:
            return
        target_language = self._target_answer_language(state)
        prefer_chinese = target_language.startswith("zh")
        papers_needing_summary = [
            paper for paper in state.curated_papers
            if (
                (
                    prefer_chinese
                    and (
                        (not paper.summary or not self._topic_contains_chinese(paper.summary[:20]))
                        and paper.abstract
                        and not self._topic_contains_chinese(paper.abstract[:50])
                    )
                )
                or (
                    not prefer_chinese
                    and (
                        (not paper.summary or self._topic_contains_chinese(paper.summary[:20]))
                        and paper.abstract
                    )
                )
            )
        ]
        if not papers_needing_summary:
            return

        logger = logging.getLogger(__name__)

        async def _summarize_one(paper: PaperCandidate) -> tuple[str, str | None]:
            try:
                from pydantic import BaseModel, Field

                class LocalizedSummary(BaseModel):
                    summary: str = Field(
                        description="Localized abstract summary matching the requested answer language"
                    )

                result = await llm_adapter.generate_structured(
                    prompt=self._SUMMARIZE_ZH_PROMPT if prefer_chinese else self._SUMMARIZE_EN_PROMPT,
                    input_data={"title": paper.title, "abstract": paper.abstract[:800]},
                    response_model=LocalizedSummary,
                )
                return paper.paper_id, result.summary
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to generate localized summary for %s: %s", paper.paper_id, exc)
                return paper.paper_id, None

        results = await asyncio.gather(*[_summarize_one(paper) for paper in papers_needing_summary[:10]])
        summary_map = {paper_id: summary for paper_id, summary in results if summary}
        if summary_map:
            state.curated_papers = [
                paper.model_copy(update={"summary": summary_map[paper.paper_id]})
                if paper.paper_id in summary_map
                else paper
                for paper in state.curated_papers
            ]

    async def finish_node(self, state: AutonomousResearchGraphState) -> AutonomousResearchGraphState:
        return {"runtime_state": state["runtime_state"]}

    def refresh_existing_pool(
        self,
        *,
        task: ResearchTask,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        merged: dict[str, PaperCandidate] = {}
        key_aliases: dict[str, str] = {}
        for paper in [*existing_papers, *incoming_papers]:
            candidate_keys = self._paper_identity_keys(paper)
            key = next((key_aliases[item] for item in candidate_keys if item in key_aliases), None)
            if key is None:
                key = candidate_keys[0]
            existing = merged.get(key)
            if existing is None:
                merged[key] = paper
                for item in candidate_keys:
                    key_aliases[item] = key
                continue
            merged[key] = existing.model_copy(
                update={
                    "authors": existing.authors or paper.authors,
                    "abstract": existing.abstract or paper.abstract,
                    "year": existing.year or paper.year,
                    "venue": existing.venue or paper.venue,
                    "pdf_url": existing.pdf_url or paper.pdf_url,
                    "url": existing.url or paper.url,
                    "citations": max(existing.citations or 0, paper.citations or 0) or None,
                    "published_at": existing.published_at or paper.published_at,
                    "relevance_score": max(existing.relevance_score or 0, paper.relevance_score or 0)
                    or None,
                    "summary": existing.summary or paper.summary,
                    "ingest_status": existing.ingest_status
                    if existing.ingest_status != "not_selected"
                    else paper.ingest_status,
                    "metadata": {**paper.metadata, **existing.metadata},
                }
            )
            for item in candidate_keys:
                key_aliases[item] = key
        papers = list(merged.values())
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=papers,
            max_papers=max(len(papers), 1),
        )

    async def _decide_next_step(self, state: AutonomousResearchState) -> tuple[str, str]:
        if (
            state.initial_plan is not None
            and state.search_completed
            and state.curation_completed
            and state.report is not None
            and state.todo_items
        ):
            return "finish", "Core research artifacts are complete; stop the loop instead of overthinking."

        if state.stagnant_decision_count >= 2:
            return "finish", "The research state stopped improving across consecutive manager decisions, so the loop should stop."

        if self.llm_adapter is None:
            if state.initial_plan is None:
                return "plan_search", "No LLM planner available; bootstrapping by creating the initial search plan."
            if not state.search_completed:
                return "search_sources", "No LLM planner available; execute pending search queries."
            if not state.curation_completed:
                return "curate_papers", "No LLM planner available; curate the retrieved paper pool."
            if self._should_refine_search(state):
                return "refine_search", "No LLM planner available; expand the query set because coverage or PDF availability is still weak."
            if state.report is None:
                return "write_report", "No LLM planner available; synthesize the current evidence into a report."
            if not state.todo_items:
                return "plan_todos", "No LLM planner available; generate follow-up todos."
            return "finish", "No LLM planner available and the essential research loop is complete."

        try:
            manager_timeout_seconds = _manager_decision_timeout_seconds(self.llm_adapter)
            decision = await asyncio.wait_for(
                self.llm_adapter.generate_structured(
                    prompt=(
                        "You are the autonomous manager of a multi-agent literature discovery system. "
                        "Choose exactly one next step for the current research loop. Prefer adaptive agent decisions over fixed pipelines. "
                        "Use finish only when the workspace is already sufficiently complete."
                    ),
                    input_data={
                        "topic": state.topic,
                        "days_back": state.days_back,
                        "max_papers": state.max_papers,
                        "sources": state.sources,
                        "round_index": state.round_index,
                        "max_rounds": state.max_rounds,
                        "initial_plan_ready": state.initial_plan is not None,
                        "active_queries": state.active_queries,
                        "searched_queries": state.searched_queries(),
                        "search_completed": state.search_completed,
                        "curation_completed": state.curation_completed,
                        "refinement_used": state.refinement_used,
                        "raw_paper_count": len(state.raw_papers),
                        "curated_paper_count": len(state.curated_papers),
                        "must_read_count": len(state.must_read_ids),
                        "ingest_candidate_count": len(state.ingest_candidate_ids),
                        "has_report": state.report is not None,
                        "todo_count": len(state.todo_items),
                        "warnings": state.warnings[-6:],
                        "recent_trace": [
                            {
                                "step_index": step.step_index,
                                "agent": step.agent,
                                "decision": step.decision,
                                "rationale": step.rationale,
                                "metadata": step.metadata,
                            }
                            for step in state.trace[-8:]
                        ],
                        "allowed_steps": [
                            "plan_search",
                            "search_sources",
                            "curate_papers",
                            "refine_search",
                            "write_report",
                            "plan_todos",
                            "finish",
                        ],
                    },
                    response_model=AutonomousResearchLLMDecision,
                ),
                timeout=manager_timeout_seconds,
            )
        except Exception:
            if state.initial_plan is None:
                return "plan_search", "Manager LLM timed out or failed; bootstrap the initial search plan heuristically."
            if not state.search_completed:
                return "search_sources", "Manager LLM timed out or failed; execute the pending source search heuristically."
            if not state.curation_completed:
                return "curate_papers", "Manager LLM timed out or failed; curate the current paper pool heuristically."
            if self._should_refine_search(state):
                return "refine_search", "Manager LLM timed out or failed; broaden coverage with one refinement round."
            if state.report is None:
                return "write_report", "Manager LLM timed out or failed; synthesize the current evidence into a report."
            if not state.todo_items:
                return "plan_todos", "Manager LLM timed out or failed; generate follow-up todos heuristically."
            return "finish", "Manager LLM timed out or failed after the core research loop completed."
        if (
            decision.next_step in {"search_sources", "refine_search"}
            and state.initial_plan is not None
            and state.search_completed
            and state.curation_completed
            and len(state.curated_papers) >= min(state.max_papers, 6)
            and state.report is not None
        ):
            return "finish", "The current paper pool and report are already sufficient, so another search/refine step would likely be redundant."
        return decision.next_step, (decision.rationale or f"LLM selected {decision.next_step} as the next best research step.")

    def _should_refine_search(self, state: AutonomousResearchState) -> bool:
        if state.refinement_used or state.round_index + 1 >= state.max_rounds:
            return False
        curated_count = len(state.curated_papers)
        if curated_count < min(state.max_papers, 6):
            return True
        if curated_count == 0:
            return False
        open_access_ratio = sum(1 for paper in state.curated_papers if paper.pdf_url) / max(curated_count, 1)
        return open_access_ratio < 0.35

    def _paper_identity_keys(self, paper: PaperCandidate) -> list[str]:
        keys: list[str] = []
        if paper.doi:
            keys.append(f"doi:{paper.doi.lower()}")
        if paper.arxiv_id:
            keys.append(f"arxiv:{paper.arxiv_id.lower()}")
        normalized_title = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", paper.title.lower())).strip()
        keys.append(f"title:{normalized_title}")
        return keys


# Compatibility names for older imports. Runtime code keeps the major agents and
# demotes curation to a skill.
ResearchPlannerAgent = LiteratureScoutAgent
SourceScoutAgent = LiteratureScoutAgent
SearchRefinementAgent = LiteratureScoutAgent
ResearchSynthesisAgent = ResearchWriterAgent
ResearchTodoPlannerAgent = ResearchWriterAgent
