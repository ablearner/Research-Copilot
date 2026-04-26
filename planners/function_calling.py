from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, format_llm_error, is_expected_provider_error
from adapters.llm.langchain_binding import LangChainProviderBinding, ensure_provider_binding

logger = logging.getLogger(__name__)


class ReactTraceStep(BaseModel):
    step: int
    agent: str = "planner"
    action: str
    observation: str | None = None


class RouterPlan(BaseModel):
    route: Literal["ask", "parse", "index", "chart_understand", "finalize"] = "finalize"
    next_action: str
    chart_graph_extraction_needed: bool = False
    requires_document_understanding: bool = False
    requires_graph_extraction: bool = False
    reasoning_summary: str = ""
    react_trace: list[ReactTraceStep] = Field(default_factory=list)


class RouterPlanner:
    def __init__(self, llm_adapter: BaseLLMAdapter | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def plan(self, state: dict[str, Any]) -> RouterPlan:
        task_type = state.get("task_type")
        if task_type == "ask":
            return RouterPlan(
                route="ask",
                next_action="retrieval_planner",
                reasoning_summary="Router selected the QA multi-agent path: plan retrieval, collect evidence, answer, then validate.",
                react_trace=[
                    ReactTraceStep(step=1, agent="router_planner", action="inspect_task", observation="task_type=ask"),
                    ReactTraceStep(step=2, agent="router_planner", action="route", observation="ask -> retrieval_planner"),
                ],
            )
        if task_type == "parse":
            return RouterPlan(
                route="parse",
                next_action="document_understanding",
                requires_document_understanding=True,
                reasoning_summary="Router selected the document understanding path for parsing.",
                react_trace=[
                    ReactTraceStep(step=1, agent="router_planner", action="inspect_task", observation="task_type=parse"),
                    ReactTraceStep(step=2, agent="router_planner", action="route", observation="parse -> document_understanding"),
                ],
            )
        if task_type == "index":
            return RouterPlan(
                route="index",
                next_action="document_understanding",
                requires_document_understanding=True,
                requires_graph_extraction=True,
                reasoning_summary="Router selected the indexing path: parse, extract graph, index embeddings, then index graph.",
                react_trace=[
                    ReactTraceStep(step=1, agent="router_planner", action="inspect_task", observation="task_type=index"),
                    ReactTraceStep(step=2, agent="router_planner", action="route", observation="index -> document_understanding"),
                ],
            )
        if task_type == "chart_understand":
            return RouterPlan(
                route="chart_understand",
                next_action="chart_vision",
                chart_graph_extraction_needed=bool(state.get("metadata", {}).get("extract_chart_graph", False)),
                reasoning_summary="Router selected the chart understanding path and will optionally extract graph structure after vision analysis.",
                react_trace=[
                    ReactTraceStep(step=1, agent="router_planner", action="inspect_task", observation="task_type=chart_understand"),
                    ReactTraceStep(step=2, agent="router_planner", action="route", observation="chart_understand -> chart_vision"),
                ],
            )
        return RouterPlan(
            route="finalize",
            next_action="finalize",
            reasoning_summary=f"Unsupported task_type={task_type}.",
            react_trace=[ReactTraceStep(step=1, agent="router_planner", action="route", observation="unsupported -> finalize")],
        )


class RetrievalPlan(BaseModel):
    modes: list[Literal["vector", "graph", "summary"]] = Field(default_factory=list)
    query: str
    retrieval_focus: str | None = None
    reasoning_summary: str
    react_trace: list[ReactTraceStep] = Field(default_factory=list)
    max_steps: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="before")
    @classmethod
    def normalize_wrapped_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "retrieval_plan" in value and isinstance(value["retrieval_plan"], dict):
            nested = dict(value["retrieval_plan"])
            for key in ("query", "reasoning_summary", "retrieval_focus", "react_trace", "max_steps", "modes"):
                if key not in nested and key in value:
                    nested[key] = value[key]
            if "query" not in nested and isinstance(value.get("question"), str):
                nested["query"] = value["question"]
            if "reasoning_summary" not in nested and isinstance(value.get("reasoning_summary"), str):
                nested["reasoning_summary"] = value["reasoning_summary"]
            return nested
        return value


class RetrievalPlanner:
    def __init__(self, llm_adapter: BaseLLMAdapter | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.binding = self._build_binding(llm_adapter)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Retrieval Planner Agent in a LangGraph GraphRAG system. "
                        "Return a concise structured retrieval plan. "
                        "Use at most {max_steps} reasoning steps internally. "
                        "Select from modes: vector, graph, summary. "
                        "Keep react_trace short and structured. "
                        "Do not expose full chain of thought."
                    ),
                ),
                ("human", "{payload_json}"),
            ]
        )

    async def plan(
        self,
        *,
        question: str,
        state: dict[str, Any],
        max_steps: int = 3,
    ) -> RetrievalPlan:
        heuristic_plan = self._fallback_plan(question=question, state=state, max_steps=max_steps)
        if self.binding is None:
            return heuristic_plan

        payload = {
            "question": question,
            "task_type": state.get("task_type"),
            "document_ids": state.get("document_ids", []),
            "retrieval_attempt": state.get("retrieval_attempt", 0),
            "session_memory": state.get("session_memory") or {},
            "validation_result": state.get("validation_result") or {},
            "top_k": state.get("top_k", 10),
            "filters": state.get("filters", {}),
            "heuristic_plan": heuristic_plan.model_dump(mode="json"),
        }
        try:
            messages = await self.prompt.ainvoke(
                {
                    "payload_json": json.dumps(payload, ensure_ascii=False, indent=2),
                    "max_steps": max(1, min(max_steps, 5)),
                }
            )
            response = await self.binding.ainvoke_structured(
                messages=messages.to_messages(),
                response_model=RetrievalPlan,
                metadata={"agent": "retrieval_planner", "task": "plan_retrieval"},
            )
            return RetrievalPlan.model_validate(response)
        except (LLMAdapterError, OSError, ValueError) as exc:
            if is_expected_provider_error(exc):
                logger.warning(
                    "Structured retrieval planning failed; using heuristic plan. cause=%s",
                    format_llm_error(exc),
                )
            else:
                logger.warning("Structured retrieval planning failed; using heuristic plan", exc_info=exc)
            return heuristic_plan

    def _fallback_plan(
        self,
        *,
        question: str,
        state: dict[str, Any],
        max_steps: int,
    ) -> RetrievalPlan:
        session_memory = state.get("session_memory") or {}
        validation_result = state.get("validation_result") or {}
        modes: list[Literal["vector", "graph", "summary"]] = ["vector", "graph"]
        if state.get("retrieval_attempt", 0) == 0:
            modes.append("summary")
        retrieval_focus = validation_result.get("retrieval_focus") or session_memory.get("task_intent")
        if session_memory.get("current_document_id") and session_memory.get("current_document_id") in (
            state.get("document_ids") or []
        ):
            query = f"{question}\nFocus on current document context."
        else:
            query = question
        if retrieval_focus:
            query = f"{query}\nRetrieval focus: {retrieval_focus}"
        return RetrievalPlan(
            modes=modes,
            query=query,
            retrieval_focus=retrieval_focus,
            reasoning_summary="Retrieval planner selected a GraphRAG retrieval bundle centered on vector and graph evidence, with graph summary retrieval on the first pass.",
            react_trace=[
                ReactTraceStep(step=1, agent="retrieval_planner", action="inspect_question", observation=question[:120]),
                ReactTraceStep(step=2, agent="retrieval_planner", action="check_memory", observation=str(bool(session_memory))),
                ReactTraceStep(step=3, agent="retrieval_planner", action="select_modes", observation=",".join(modes)),
            ],
            max_steps=max(1, min(max_steps, 5)),
        )

    def _build_binding(self, llm_adapter: BaseLLMAdapter | None) -> LangChainProviderBinding | None:
        if llm_adapter is None:
            return None
        try:
            return ensure_provider_binding(llm_adapter)
        except TypeError:
            return None


class ValidationDecision(BaseModel):
    decision: Literal["finalize", "retry_retrieval"] = "finalize"
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    critique_summary: str = ""
    retrieval_focus: str | None = None
    reasoning_summary: str = ""
    react_trace: list[ReactTraceStep] = Field(default_factory=list)


class AnswerValidationPlanner:
    def __init__(self, llm_adapter: BaseLLMAdapter | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.binding = self._build_binding(llm_adapter)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Validation and Critique Agent in a LangGraph QA system. "
                        "Judge whether the answer is sufficiently supported by evidence. "
                        "Return only structured output. "
                        "Use lightweight critique and never reveal private chain of thought."
                    ),
                ),
                ("human", "{payload_json}"),
            ]
        )

    async def validate(
        self,
        *,
        question: str,
        answer: Any,
        evidence_count: int,
        retrieval_attempt: int,
    ) -> ValidationDecision:
        heuristic_decision = self._fallback_validate(
            question=question,
            answer=answer,
            evidence_count=evidence_count,
            retrieval_attempt=retrieval_attempt,
        )
        if self.binding is None:
            return heuristic_decision

        answer_text = getattr(answer, "answer", "") or ""
        answer_confidence = getattr(answer, "confidence", None)
        answer_payload = {
            "answer": answer_text[:4000],
            "confidence": answer_confidence,
            "question": getattr(answer, "question", question),
        }
        payload = {
            "question": question,
            "answer": answer_payload,
            "evidence_count": evidence_count,
            "retrieval_attempt": retrieval_attempt,
            "heuristic_decision": heuristic_decision.model_dump(mode="json"),
        }
        try:
            messages = await self.prompt.ainvoke({"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)})
            response = await self.binding.ainvoke_structured(
                messages=messages.to_messages(),
                response_model=ValidationDecision,
                metadata={"agent": "validation", "task": "validate_answer"},
            )
            return ValidationDecision.model_validate(response)
        except (LLMAdapterError, OSError, ValueError) as exc:
            if is_expected_provider_error(exc):
                logger.warning(
                    "Structured validation failed; using heuristic validation. cause=%s",
                    format_llm_error(exc),
                )
            else:
                logger.warning("Structured validation failed; using heuristic validation", exc_info=exc)
            return heuristic_decision

    def _fallback_validate(
        self,
        *,
        question: str,
        answer: Any,
        evidence_count: int,
        retrieval_attempt: int,
    ) -> ValidationDecision:
        answer_text = getattr(answer, "answer", "") or ""
        confidence = getattr(answer, "confidence", None)
        warnings: list[str] = []
        if evidence_count == 0 or "证据不足" in answer_text:
            warnings.append("Evidence is insufficient for a grounded answer.")
            if retrieval_attempt < 1:
                return ValidationDecision(
                    decision="retry_retrieval",
                    confidence=min(confidence if confidence is not None else 0.2, 0.2),
                    warnings=warnings,
                    critique_summary="Validation requested one additional retrieval round due to weak support.",
                    retrieval_focus=question,
                    reasoning_summary="Validation found weak grounding and requested one more retrieval pass.",
                    react_trace=[
                        ReactTraceStep(step=1, agent="validation", action="check_evidence", observation=f"evidence_count={evidence_count}"),
                        ReactTraceStep(step=2, agent="validation", action="decide", observation="retry_retrieval"),
                    ],
                )
        return ValidationDecision(
            decision="finalize",
            confidence=confidence,
            warnings=warnings,
            critique_summary="Validation accepted the current answer.",
            reasoning_summary="Validation accepted the answer because the current evidence and answer quality are sufficient.",
            react_trace=[
                ReactTraceStep(step=1, agent="validation", action="check_evidence", observation=f"evidence_count={evidence_count}"),
                ReactTraceStep(step=2, agent="validation", action="decide", observation="finalize"),
            ],
        )

    def _build_binding(self, llm_adapter: BaseLLMAdapter | None) -> LangChainProviderBinding | None:
        if llm_adapter is None:
            return None
        try:
            return ensure_provider_binding(llm_adapter)
        except TypeError:
            return None


# Compatibility aliases for older internal names.
RouterPlannerAgent = RouterPlanner
RetrievalPlannerAgent = RetrievalPlanner
ValidationAgent = AnswerValidationPlanner
