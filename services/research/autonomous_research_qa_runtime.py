from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.research_knowledge_agent import (
    ResearchKnowledgeAgent,
    is_insufficient_research_answer,
    merge_retrieval_hits,
)
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskRequest,
)
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from reasoning.strategies import ReasoningStrategySet
from services.research.research_context import ResearchExecutionContext

PRIMARY_QA_AGENTS = (
    "ResearchSupervisorAgent",
    "ResearchKnowledgeAgent",
    "ResearchWriterAgent",
)
PRIMARY_QA_AGENTS_LABEL = ",".join(PRIMARY_QA_AGENTS)
QA_RUNTIME_ARCHITECTURE = "main_agents_only"
AutonomousResearchQADecision = tuple[str, str]


class AutonomousResearchQALLMDecision(BaseModel):
    next_step: Literal[
        "plan_queries",
        "retrieve_collection_evidence",
        "retrieve_graph_summary",
        "build_collection_manifest",
        "refine_query",
        "answer_question",
        "finish",
    ]
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_VAGUE_COLLECTION_EFFECT_MARKERS = (
    "效果",
    "表现",
    "结果",
    "性能",
    "好不好",
    "怎么样",
    "如何",
    "performance",
    "result",
    "effect",
)


def _is_vague_collection_question(question: str) -> bool:
    compact = "".join(question.strip().split()).lower()
    if not compact:
        return False
    if compact in {"怎么样", "效果怎么样", "效果如何", "表现如何", "结果怎么样", "好不好", "howisit", "howaboutit"}:
        return True
    return len(compact) <= 16 and any(marker in compact for marker in _VAGUE_COLLECTION_EFFECT_MARKERS)


def _collection_effect_question(*, original_question: str, topic: str, paper_count: int) -> str:
    topic_text = topic.strip() or "当前研究主题"
    paper_text = f"{paper_count} 篇" if paper_count > 0 else "当前"
    return (
        f"请基于当前研究集合中关于“{topic_text}”的{paper_text}论文，综合评价这些方法或系统的效果、"
        "实验表现、优势和局限；请按论文分别说明，并给出整体判断，不要只回答单篇论文。"
        f"原始追问：{original_question.strip()}"
    )


@dataclass(slots=True)
class AutonomousResearchQATraceStep:
    step_index: int
    agent: str
    decision: str
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AutonomousResearchQAState:
    task: ResearchTask
    request: ResearchTaskAskRequest
    report: ResearchReport | None
    papers: list[PaperCandidate]
    document_ids: list[str]
    execution_context: ResearchExecutionContext | None = None
    skill_context: dict[str, Any] | None = None
    queries: list[str] = field(default_factory=list)
    completed_queries: set[str] = field(default_factory=set)
    refinement_used: bool = False
    summary_checked: bool = False
    manifest_built: bool = False
    retrieval_hits: list[RetrievalHit] = field(default_factory=list)
    summary_hits: list[RetrievalHit] = field(default_factory=list)
    manifest_hits: list[RetrievalHit] = field(default_factory=list)
    evidence_bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    retrieval_result: HybridRetrievalResult | None = None
    qa: QAResponse | None = None
    warnings: list[str] = field(default_factory=list)
    trace: list[AutonomousResearchQATraceStep] = field(default_factory=list)

    @property
    def question(self) -> str:
        metadata_question = str((self.request.metadata or {}).get("resolved_question") or "").strip()
        if metadata_question:
            return metadata_question
        original_question = self.original_question
        if _is_vague_collection_question(original_question):
            return _collection_effect_question(
                original_question=original_question,
                topic=self.task.topic,
                paper_count=len(self.papers),
            )
        return original_question

    @property
    def original_question(self) -> str:
        return self.request.question

    @property
    def top_k(self) -> int:
        return self.request.top_k

    def pending_query(self) -> str | None:
        for query in self.queries:
            if query not in self.completed_queries:
                return query
        return None

    def all_hits(self) -> list[RetrievalHit]:
        return merge_retrieval_hits(self.retrieval_hits, self.summary_hits, self.manifest_hits)

    def trace_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "step_index": step.step_index,
                "agent": step.agent,
                "decision": step.decision,
                "rationale": step.rationale,
                "metadata": step.metadata,
            }
            for step in self.trace
        ]


@dataclass(slots=True)
class AutonomousResearchQAResult:
    qa: QAResponse
    retrieval_result: HybridRetrievalResult
    evidence_bundle: EvidenceBundle
    trace: list[AutonomousResearchQATraceStep]
    warnings: list[str]


class AutonomousResearchQAGraphState(TypedDict, total=False):
    runtime_state: AutonomousResearchQAState
    graph_runtime: Any
    current_decision: AutonomousResearchQADecision | None
    current_step_index: int
    exhausted: bool


class AutonomousResearchCollectionQARuntime:
    """Manager-agent loop for grounded QA over a research task collection."""

    def __init__(
        self,
        *,
        plan_and_solve_reasoning_agent: Any | None = None,
        max_steps: int = 13,
    ) -> None:
        self.reasoning_strategies = ReasoningStrategySet(
            query_planning=plan_and_solve_reasoning_agent,
        )
        self.knowledge_agent = ResearchKnowledgeAgent(
            reasoning_strategies=self.reasoning_strategies,
        )
        self.writer_agent = ResearchWriterAgent(reasoning_strategies=self.reasoning_strategies)
        self.llm_adapter = self.reasoning_strategies.llm_adapter
        self.max_steps = max_steps
        self.graph = self._build_graph().compile()

    async def run(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext | None = None,
    ) -> AutonomousResearchQAResult:
        state = AutonomousResearchQAState(
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            skill_context=self._resolve_skill_context(graph_runtime, request),
        )
        graph_state = await self._run_manager_loop(
            {
                "runtime_state": state,
                "graph_runtime": graph_runtime,
                "current_step_index": 0,
                "exhausted": False,
            }
        )
        state = graph_state["runtime_state"]
        if graph_state.get("exhausted"):
            state.warnings.append(f"autonomous research qa runtime reached max_steps={self.max_steps} before finish")

        if not state.manifest_built:
            state.manifest_hits = self.knowledge_agent.build_collection_manifest(state)
            state.manifest_built = True
        if state.qa is None:
            state.qa = await self.writer_agent.answer_collection_question(
                graph_runtime=graph_runtime,
                state=state,
                primary_agents=list(PRIMARY_QA_AGENTS),
            )
            state.retrieval_result = state.qa.retrieval_result
            state.evidence_bundle = state.qa.evidence_bundle

        qa = state.qa.model_copy(
            update={
                "metadata": {
                    **state.qa.metadata,
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": QA_RUNTIME_ARCHITECTURE,
                    "primary_agents": list(PRIMARY_QA_AGENTS),
                    "autonomy_trace_steps": len(state.trace),
                    "autonomy_trace": state.trace_payload(),
                    "warnings": state.warnings,
                    "selected_skill": state.skill_context.get("name") if isinstance(state.skill_context, dict) else None,
                    "memory_enabled": state.execution_context.memory_enabled if state.execution_context else False,
                    "session_id": state.execution_context.session_id if state.execution_context else None,
                    "collection_hit_mix": {
                        "retrieval_hits": len(state.retrieval_hits),
                        "graph_summary_hits": len(state.summary_hits),
                        "manifest_hits": len(state.manifest_hits),
                        "evidence_count": len(state.evidence_bundle.evidences),
                    },
                }
            }
        )
        return AutonomousResearchQAResult(
            qa=qa,
            retrieval_result=qa.retrieval_result or HybridRetrievalResult(query=RetrievalQuery(query=state.question)),
            evidence_bundle=qa.evidence_bundle,
            trace=state.trace,
            warnings=state.warnings,
        )

    async def _run_manager_loop(
        self,
        initial_state: AutonomousResearchQAGraphState,
    ) -> AutonomousResearchQAGraphState:
        state: AutonomousResearchQAGraphState = dict(initial_state)
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
        state: AutonomousResearchQAGraphState,
        route: str,
    ) -> AutonomousResearchQAGraphState:
        if route == "plan_queries":
            return await self.plan_queries_node(state)
        if route == "retrieve_collection_evidence":
            return await self.retrieve_collection_evidence_node(state)
        if route == "retrieve_graph_summary":
            return await self.retrieve_graph_summary_node(state)
        if route == "build_collection_manifest":
            return await self.build_collection_manifest_node(state)
        if route == "refine_query":
            return await self.refine_query_node(state)
        if route == "answer_question":
            return await self.answer_question_node(state)
        raise ValueError(f"unsupported autonomous research qa route: {route}")

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AutonomousResearchQAGraphState)
        graph.add_node("manager_node", self._graph_manager_node)
        graph.add_node("plan_queries_node", self.plan_queries_node)
        graph.add_node("retrieve_collection_evidence_node", self.retrieve_collection_evidence_node)
        graph.add_node("retrieve_graph_summary_node", self.retrieve_graph_summary_node)
        graph.add_node("build_collection_manifest_node", self.build_collection_manifest_node)
        graph.add_node("refine_query_node", self.refine_query_node)
        graph.add_node("answer_question_node", self.answer_question_node)
        graph.add_node("finish_node", self.finish_node)

        graph.add_edge("__start__", "manager_node")
        graph.add_conditional_edges(
            "manager_node",
            self._graph_route_after_manager,
            {
                "plan_queries": "plan_queries_node",
                "retrieve_collection_evidence": "retrieve_collection_evidence_node",
                "retrieve_graph_summary": "retrieve_graph_summary_node",
                "build_collection_manifest": "build_collection_manifest_node",
                "refine_query": "refine_query_node",
                "answer_question": "answer_question_node",
                "finish": "finish_node",
            },
        )
        for node_name in (
            "plan_queries_node",
            "retrieve_collection_evidence_node",
            "retrieve_graph_summary_node",
            "build_collection_manifest_node",
            "refine_query_node",
            "answer_question_node",
        ):
            graph.add_edge(node_name, "manager_node")
        graph.add_edge("finish_node", END)
        return graph

    async def _graph_manager_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
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

    def _graph_route_after_manager(self, state: AutonomousResearchQAGraphState) -> str:
        if state.get("exhausted"):
            return "finish"
        decision = state.get("current_decision")
        if decision is None:
            return "finish"
        return self._route_decision(decision)

    async def _manager_decision(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQADecision:
        return await self._decide_next_step(state["runtime_state"])

    def _route_decision(self, decision: AutonomousResearchQADecision) -> str:
        return decision[0]

    def _on_manager_decision(
        self,
        state: AutonomousResearchQAGraphState,
        step_index: int,
        decision: AutonomousResearchQADecision,
    ) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        decision_name, rationale = decision
        runtime_state.trace.append(
            AutonomousResearchQATraceStep(
                step_index=step_index,
                agent="ResearchSupervisorAgent",
                decision=decision_name,
                rationale=rationale,
                metadata={
                    "completed_queries": len(runtime_state.completed_queries),
                    "primary_agents": PRIMARY_QA_AGENTS_LABEL,
                },
            )
        )
        return {
            "runtime_state": runtime_state,
        }

    async def plan_queries_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.queries = await self.knowledge_agent.plan_collection_queries(runtime_state)
        return {"runtime_state": runtime_state}

    async def retrieve_collection_evidence_node(
        self, state: AutonomousResearchQAGraphState
    ) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        query = runtime_state.pending_query()
        if query is None:
            return {"runtime_state": runtime_state}
        try:
            hits = await self.knowledge_agent.retrieve_collection_evidence(
                graph_runtime=state["graph_runtime"],
                state=runtime_state,
                query=query,
            )
            runtime_state.retrieval_hits = merge_retrieval_hits(runtime_state.retrieval_hits, hits)
        except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
            runtime_state.warnings.append(f"collection_retrieval:{query} failed: {exc}")
        runtime_state.completed_queries.add(query)
        return {"runtime_state": runtime_state}

    async def retrieve_graph_summary_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.summary_checked = True
        try:
            hits = await self.knowledge_agent.retrieve_graph_summary(
                graph_runtime=state["graph_runtime"],
                state=runtime_state,
            )
            runtime_state.summary_hits = merge_retrieval_hits(runtime_state.summary_hits, hits)
        except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
            runtime_state.warnings.append(f"graph_summary:{runtime_state.question} failed: {exc}")
        return {"runtime_state": runtime_state}

    async def build_collection_manifest_node(
        self, state: AutonomousResearchQAGraphState
    ) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.manifest_hits = self.knowledge_agent.build_collection_manifest(runtime_state)
        runtime_state.manifest_built = True
        return {"runtime_state": runtime_state}

    async def refine_query_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        step_index = state.get("current_step_index", 0)
        refined_queries = await self.knowledge_agent.propose_collection_queries(runtime_state)
        runtime_state.refinement_used = True
        if not refined_queries:
            runtime_state.trace.append(
                AutonomousResearchQATraceStep(
                    step_index=step_index,
                    agent="ResearchKnowledgeAgent",
                    decision="skip_refinement",
                    rationale="No novel collection query variants remained after previous retrieval attempts.",
                )
            )
            return {"runtime_state": runtime_state}
        runtime_state.queries.extend(refined_queries)
        runtime_state.summary_checked = False
        runtime_state.qa = None
        runtime_state.retrieval_result = None
        runtime_state.evidence_bundle = EvidenceBundle()
        runtime_state.trace.append(
            AutonomousResearchQATraceStep(
                step_index=step_index,
                agent="ResearchKnowledgeAgent",
                decision="refine_queries",
                rationale="The collection answer needed broader evidence coverage, so the knowledge agent expanded the query set.",
                metadata={"queries": refined_queries},
            )
        )
        return {"runtime_state": runtime_state}

    async def answer_question_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
        runtime_state = state["runtime_state"]
        runtime_state.qa = await self.writer_agent.answer_collection_question(
            graph_runtime=state["graph_runtime"],
            state=runtime_state,
            primary_agents=list(PRIMARY_QA_AGENTS),
        )
        runtime_state.retrieval_result = runtime_state.qa.retrieval_result
        runtime_state.evidence_bundle = runtime_state.qa.evidence_bundle
        return {"runtime_state": runtime_state}

    async def finish_node(self, state: AutonomousResearchQAGraphState) -> AutonomousResearchQAGraphState:
        return {
            "runtime_state": state["runtime_state"],
            "graph_runtime": state["graph_runtime"],
        }

    async def _decide_next_step(self, state: AutonomousResearchQAState) -> tuple[str, str]:
        if self.llm_adapter is None:
            if not state.queries:
                return "plan_queries", "No LLM planner available; create an initial collection query set."
            if state.document_ids and state.pending_query() is not None:
                return "retrieve_collection_evidence", "No LLM planner available; execute pending collection retrieval."
            if state.document_ids and not state.summary_checked:
                return "retrieve_graph_summary", "No LLM planner available; gather graph summary evidence."
            if not state.manifest_built:
                return "build_collection_manifest", "No LLM planner available; build collection manifest evidence."
            if state.qa is None:
                return "answer_question", "No LLM planner available; answer from current evidence."
            return "finish", "No LLM planner available and the QA loop has produced an answer."

        decision = await self.llm_adapter.generate_structured(
            prompt=(
                "You are the autonomous manager of a multi-agent research collection QA system. "
                "Choose exactly one next step. Prefer adaptive evidence gathering and answering decisions over fixed pipelines. "
                "Use finish only when the answer is sufficiently grounded or no higher-value step remains."
            ),
            input_data={
                "question": state.question,
                "original_question": state.original_question,
                "paper_count": len(state.papers),
                "document_count": len(state.document_ids),
                "queries": state.queries,
                "completed_queries": list(state.completed_queries),
                "pending_query": state.pending_query(),
                "summary_checked": state.summary_checked,
                "manifest_built": state.manifest_built,
                "refinement_used": state.refinement_used,
                "retrieval_hit_count": len(state.retrieval_hits),
                "summary_hit_count": len(state.summary_hits),
                "manifest_hit_count": len(state.manifest_hits),
                "evidence_count": len(state.evidence_bundle.evidences),
                "has_answer": state.qa is not None,
                "answer_confidence": state.qa.confidence if state.qa is not None else None,
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
                    "plan_queries",
                    "retrieve_collection_evidence",
                    "retrieve_graph_summary",
                    "build_collection_manifest",
                    "refine_query",
                    "answer_question",
                    "finish",
                ],
            },
            response_model=AutonomousResearchQALLMDecision,
        )
        return decision.next_step, (decision.rationale or f"LLM selected {decision.next_step} as the next best QA step.")

    def _should_refine_before_answer(self, state: AutonomousResearchQAState) -> bool:
        if state.refinement_used or not state.document_ids:
            return False
        return len(state.all_hits()) < 2

    def _should_refine_after_answer(self, state: AutonomousResearchQAState) -> bool:
        if state.refinement_used or not state.document_ids or state.qa is None:
            return False
        confidence = state.qa.confidence if state.qa.confidence is not None else 0.0
        return is_insufficient_research_answer(
            answer=state.qa.answer,
            confidence=confidence,
            evidence_count=len(state.evidence_bundle.evidences),
        )

    def _resolve_skill_context(self, graph_runtime, request: ResearchTaskAskRequest) -> dict[str, Any] | None:
        resolver = getattr(graph_runtime, "resolve_skill_context", None)
        if not callable(resolver):
            return None
        return resolver(
            task_type="ask_document",
            preferred_skill_name=request.skill_name or "research_report",
        )


# Compatibility names for older imports. The QA runtime now keeps only the
# major agents for collection evidence and grounded answering.
ResearchCollectionPlannerAgent = ResearchKnowledgeAgent
ResearchCollectionRetrievalAgent = ResearchKnowledgeAgent
ResearchCollectionGraphSummaryAgent = ResearchKnowledgeAgent
ResearchCollectionManifestAgent = ResearchKnowledgeAgent
ResearchCollectionRefinementAgent = ResearchKnowledgeAgent
ResearchCollectionLeadAgent = ResearchKnowledgeAgent
ResearchCollectionAnswerAgent = ResearchWriterAgent
