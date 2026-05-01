"""Research QA specialist and RAG-layer ReAct QA worker."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult
from domain.schemas.unified_runtime import UnifiedAgentResult, UnifiedAgentTask
from services.research.research_external_tool_gateway import ResearchExternalToolGateway
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import ToolCallTrace, ToolExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReAct data models (moved from reasoning/react.py)
# ---------------------------------------------------------------------------

class ReActDecision(BaseModel):
    thought: str
    action: Literal["tool", "finish"]
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ReActStep(BaseModel):
    step_index: int
    thought: str
    action: Literal["tool", "finish"]
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] = Field(default_factory=dict)


class ReActFinalDraft(BaseModel):
    answer: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ResearchQAAgentError(RuntimeError):
    """Raised when the RAG ReAct QA worker reasoning flow fails."""


# ---------------------------------------------------------------------------
# Reasoning style helpers (moved from reasoning/style.py)
# ---------------------------------------------------------------------------

DEFAULT_AGENT_REASONING_STYLE = "react"


def normalize_reasoning_style(style: str | None) -> str:
    normalized = (style or DEFAULT_AGENT_REASONING_STYLE).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return DEFAULT_AGENT_REASONING_STYLE
    aliases = {
        "chain_of_thought": "react",
        "chainofthought": "react",
        "cot": "react",
        "planandsolve": "plan_and_execute",
        "plan_and_solve": "plan_and_execute",
        "plan_and_execute": "plan_and_execute",
        "react": "react",
        "auto": "auto",
    }
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Task-level ResearchQAAgent
# ---------------------------------------------------------------------------

class ResearchQAAgent:
    """Supervisor-delegated QA specialist.

    The supervisor owns task understanding and route selection. This specialist
    executes the selected QA route through the QA executor/toolset and returns a
    unified worker result.
    """

    name = "ResearchQAAgent"
    _STREAMING_DEFAULT_CONFIDENCE: float = 0.7

    async def execute(self, task: UnifiedAgentTask, runtime_context: Any) -> UnifiedAgentResult:
        from services.research.qa.executor import ResearchQAExecutor
        from services.research.unified_action_adapters import (
            build_collection_qa_input,
            build_collection_qa_output,
            resolve_active_message,
        )

        supervisor_context = runtime_context.metadata.get("supervisor_tool_context")
        decision = runtime_context.metadata.get("supervisor_decision")
        if supervisor_context is None or decision is None:
            return self._unified_result(
                task=task,
                status="failed",
                observation="missing supervisor runtime context for ResearchQAAgent",
                metadata={"reason": "missing_supervisor_runtime_context"},
            )
        task_response = supervisor_context.task_response
        if task_response is None:
            return self._unified_result(
                task=task,
                status="skipped",
                observation="no research task is available for collection QA",
                metadata={"reason": "missing_task"},
            )

        request = supervisor_context.request
        active_message = resolve_active_message(decision)
        qa_input = build_collection_qa_input(
            context=supervisor_context,
            task_id=task_response.task.task_id,
            active_message=active_message,
        )
        qa_result = await ResearchQAExecutor(supervisor_context.research_service).execute(
            task_response.task.task_id,
            qa_input.to_research_task_ask_request(),
            graph_runtime=supervisor_context.graph_runtime,
        )
        supervisor_context.qa_result = qa_result
        supervisor_context.task_response = supervisor_context.research_service.get_task(task_response.task.task_id)
        supervisor_context.execution_context = supervisor_context.research_service.build_execution_context(
            graph_runtime=supervisor_context.graph_runtime,
            conversation_id=request.conversation_id,
            task=supervisor_context.task_response.task if supervisor_context.task_response else None,
            report=supervisor_context.task_response.report if supervisor_context.task_response else None,
            papers=supervisor_context.task_response.papers if supervisor_context.task_response else None,
            document_ids=list(qa_result.document_ids),
            selected_paper_ids=list(qa_result.paper_ids),
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else request.metadata,
        )
        if request.conversation_id:
            supervisor_context.research_service.record_qa_turn(
                request.conversation_id,
                task_response=supervisor_context.task_response,
                ask_response=qa_result,
            )
        if (
            supervisor_context.execution_context is not None
            and supervisor_context.execution_context.session_id
            and qa_result.paper_ids
        ):
            supervisor_context.research_service.memory_gateway.set_active_papers(
                supervisor_context.execution_context.session_id,
                list(qa_result.paper_ids),
            )

        output = build_collection_qa_output(qa_result=qa_result)
        metadata = output.to_metadata()
        metadata.update(
            {
                "executed_by": self.name,
                "qa_execution_path": "research_qa_agent",
            }
        )
        qa_metadata = qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else {}
        quality_check = qa_metadata.get("answer_quality_check")
        if isinstance(quality_check, dict) and quality_check.get("needs_recovery"):
            recovery_route = str(quality_check.get("suggested_recovery_qa_route") or "").strip()
            if recovery_route:
                metadata.update(
                    {
                        "reason": "qa_route_replan_requested",
                        "observation_envelope": {
                            "progress_made": False,
                            "confidence": qa_result.qa.confidence,
                            "suggested_next_actions": ["answer_question"],
                            "state_delta": {
                                "preferred_qa_route": recovery_route,
                                "qa_recovery_reason": quality_check.get("suggested_recovery_rationale"),
                            },
                        },
                    }
                )
                return self._unified_result(
                    task=task,
                    status="skipped",
                    observation=(
                        "research QA produced an under-supported answer; supervisor replan requested "
                        f"with qa_route={recovery_route}"
                    ),
                    metadata=metadata,
                )
        return self._unified_result(
            task=task,
            status="succeeded",
            observation=(
                f"answered research collection question; evidence={len(qa_result.qa.evidence_bundle.evidences)}; "
                f"confidence={qa_result.qa.confidence if qa_result.qa.confidence is not None else 'empty'}"
            ),
            metadata=metadata,
        )

    def _unified_result(
        self,
        *,
        task: UnifiedAgentTask,
        status: str,
        observation: str,
        metadata: dict[str, Any],
    ) -> UnifiedAgentResult:
        return UnifiedAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
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
                "execution_adapter": "research_qa_agent",
                "delegate_type": self.__class__.__name__,
            },
        )


# ---------------------------------------------------------------------------
# RAG-layer ReAct QA worker
# ---------------------------------------------------------------------------


class RagReActQAWorker:
    """Tool-layer ReAct worker used by RAG answer synthesis."""

    name = "RagReActQAWorker"

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        decision_prompt_path: str | Path = "prompts/reasoning/react_decide_next_step.txt",
        synthesis_prompt_path: str | Path = "prompts/reasoning/react_synthesize_answer.txt",
        mcp_client_registry: Any | None = None,
        external_tool_gateway: ResearchExternalToolGateway | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.decision_prompt_path = Path(decision_prompt_path)
        self.synthesis_prompt_path = Path(synthesis_prompt_path)
        self.external_tool_gateway = external_tool_gateway or (
            ResearchExternalToolGateway(registry=mcp_client_registry)
            if mcp_client_registry is not None
            else None
        )

    # ------------------------------------------------------------------
    # Public entry: full ReAct QA loop
    # ------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        available_tool_names: list[str] | None = None,
        max_steps: int = 4,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        initial_retrieval_result: HybridRetrievalResult | None = None,
        initial_evidence_bundle: EvidenceBundle | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> QAResponse:
        tool_specs = self.tool_registry.filter_tools(
            enabled_only=True,
            names=available_tool_names,
        )
        tool_names = [tool.name for tool in tool_specs]
        tool_desc = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
                "tags": tool.tags,
            }
            for tool in tool_specs
        ]

        ext_names, ext_descs = await self._discover_external_tools()
        local_name_set = set(tool_names)
        for en, ed in zip(ext_names, ext_descs):
            if en not in local_name_set:
                tool_names.append(en)
                tool_desc.append(ed)

        # Fast path: evidence already available → directly call answer tool
        if initial_evidence_bundle is not None and initial_evidence_bundle.evidences:
            direct_steps: list[ReActStep] = []
            if "answer_with_evidence" in tool_names:
                tool_input = self._normalize_answer_tool_input(
                    question=question,
                    tool_input={},
                    metadata=metadata,
                    session_context=session_context,
                    task_context=task_context,
                    preference_context=preference_context,
                    retrieval_result=initial_retrieval_result,
                    evidence_bundle=initial_evidence_bundle,
                )
                tool_result = await self.tool_executor.execute_tool_call(
                    tool_name="answer_with_evidence",
                    tool_input=tool_input,
                )
                direct_steps.append(
                    ReActStep(
                        step_index=1,
                        thought="Use the evidence already accumulated in the graph state to answer directly.",
                        action="tool",
                        tool_name="answer_with_evidence",
                        tool_input=tool_input,
                        observation=self._tool_result_observation(tool_result),
                    )
                )
                if tool_result.status == "succeeded":
                    try:
                        qa = QAResponse.model_validate(tool_result.output)
                        return qa.model_copy(
                            update={
                                "metadata": {
                                    **qa.metadata,
                                    **(metadata or {}),
                                    "answered_by": self.name,
                                    "reasoning_style": "react",
                                    "react_steps": [step.model_dump(mode="json") for step in direct_steps],
                                    "used_tools": [
                                        step.tool_name
                                        for step in direct_steps
                                        if step.action == "tool" and step.tool_name
                                    ],
                                }
                            }
                        )
                    except Exception:
                        logger.warning("Failed to validate answer_with_evidence output as QAResponse", exc_info=True)

            final = await self._synthesize_answer(
                question=question,
                steps=direct_steps,
                retrieval_result=initial_retrieval_result,
                evidence_bundle=initial_evidence_bundle or EvidenceBundle(),
                session_context=session_context,
                task_context=task_context,
                preference_context=preference_context,
                on_token=on_token,
            )
            return QAResponse(
                answer=final.answer,
                question=question,
                evidence_bundle=initial_evidence_bundle or EvidenceBundle(),
                retrieval_result=initial_retrieval_result,
                confidence=final.confidence,
                metadata={
                    **(metadata or {}),
                    "answered_by": self.name,
                    "reasoning_style": "react",
                    "warnings": final.warnings,
                    "react_steps": [step.model_dump(mode="json") for step in direct_steps],
                    "used_tools": [
                        step.tool_name
                        for step in direct_steps
                        if step.action == "tool" and step.tool_name
                    ],
                },
            )

        # Full ReAct loop
        step_count = max(1, min(max_steps, 12))
        steps: list[ReActStep] = []
        final_answer: str | None = None
        final_confidence: float | None = None
        seeded_retrieval_result = initial_retrieval_result
        seeded_evidence_bundle = initial_evidence_bundle or EvidenceBundle()

        for index in range(1, step_count + 1):
            try:
                decision = await self._decide_next_step(
                    question=question,
                    tool_descriptions=tool_desc,
                    steps=steps,
                    session_context=session_context,
                    task_context=task_context,
                    preference_context=preference_context,
                )
            except ResearchQAAgentError:
                decision = self._fallback_decide_next_step(
                    question=question,
                    steps=steps,
                    session_context=session_context,
                    task_context=task_context,
                    preference_context=preference_context,
                    initial_retrieval_result=seeded_retrieval_result,
                    initial_evidence_bundle=seeded_evidence_bundle,
                )

            if decision.action == "finish":
                final_answer = decision.final_answer or "证据不足"
                final_confidence = decision.confidence
                steps.append(
                    ReActStep(
                        step_index=index,
                        thought=decision.thought,
                        action="finish",
                        observation={"status": "finished_by_model"},
                    )
                )
                break

            tool_name = decision.tool_name or ""
            if tool_name not in tool_names:
                steps.append(
                    ReActStep(
                        step_index=index,
                        thought=decision.thought,
                        action="tool",
                        tool_name=tool_name,
                        tool_input=decision.tool_input,
                        observation={
                            "status": "tool_not_allowed",
                            "error": f"Tool not allowed: {tool_name}",
                        },
                    )
                )
                continue

            if tool_name == "answer_with_evidence":
                decision.tool_input = self._normalize_answer_tool_input(
                    question=question,
                    tool_input=decision.tool_input,
                    metadata=metadata,
                    session_context=session_context,
                    task_context=task_context,
                    preference_context=preference_context,
                    retrieval_result=seeded_retrieval_result,
                    evidence_bundle=seeded_evidence_bundle,
                )
            elif tool_name == "hybrid_retrieve":
                decision.tool_input = self._normalize_retrieve_tool_input(
                    question=question,
                    tool_input=decision.tool_input,
                    session_context=session_context,
                    task_context=task_context,
                )

            tool_result = await self._execute_tool(
                tool_name=tool_name,
                tool_input=decision.tool_input,
                external_tool_names=ext_names,
            )
            steps.append(
                ReActStep(
                    step_index=index,
                    thought=decision.thought,
                    action="tool",
                    tool_name=tool_name,
                    tool_input=decision.tool_input,
                    observation=self._tool_result_observation(tool_result),
                )
            )

            if tool_result.status == "succeeded" and tool_name == "hybrid_retrieve":
                try:
                    retrieval_output = tool_result.output if isinstance(tool_result.output, dict) else {}
                    if isinstance(retrieval_output.get("retrieval_result"), dict):
                        seeded_retrieval_result = HybridRetrievalResult.model_validate(
                            retrieval_output["retrieval_result"]
                        )
                    if isinstance(retrieval_output.get("evidence_bundle"), dict):
                        seeded_evidence_bundle = EvidenceBundle.model_validate(
                            retrieval_output["evidence_bundle"]
                        )
                except Exception:
                    logger.warning("Failed to capture hybrid retrieval output for ReAct context", exc_info=True)

            if tool_result.status == "succeeded" and tool_name == "answer_with_evidence":
                try:
                    qa = QAResponse.model_validate(tool_result.output)
                    return qa.model_copy(
                        update={
                            "metadata": {
                                **qa.metadata,
                                **(metadata or {}),
                                "answered_by": self.name,
                                "reasoning_style": "react",
                                "react_steps": [step.model_dump(mode="json") for step in steps],
                                "used_tools": [
                                    step.tool_name
                                    for step in steps
                                    if step.action == "tool" and step.tool_name
                                ],
                            }
                        }
                    )
                except Exception:
                    logger.warning("Failed to validate answer_with_evidence output as QAResponse", exc_info=True)

        retrieval_result, evidence_bundle = self._extract_retrieval_context(steps)
        retrieval_result = retrieval_result or seeded_retrieval_result
        if not evidence_bundle.evidences and seeded_evidence_bundle.evidences:
            evidence_bundle = seeded_evidence_bundle

        if final_answer is None:
            final = await self._synthesize_answer(
                question=question,
                steps=steps,
                retrieval_result=retrieval_result,
                evidence_bundle=evidence_bundle,
                session_context=session_context,
                task_context=task_context,
                preference_context=preference_context,
                on_token=on_token,
            )
            final_answer = final.answer
            final_confidence = final.confidence
            warnings = final.warnings
        else:
            warnings = []

        if self._looks_insufficient(final_answer):
            final_confidence = min(final_confidence if final_confidence is not None else 0.2, 0.2)

        return QAResponse(
            answer=final_answer,
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=final_confidence,
            metadata={
                **(metadata or {}),
                "answered_by": self.name,
                "reasoning_style": "react",
                "warnings": warnings,
                "react_steps": [step.model_dump(mode="json") for step in steps],
                "used_tools": [
                    step.tool_name
                    for step in steps
                    if step.action == "tool" and step.tool_name
                ],
            },
        )

    # Keep backward-compatible alias
    reason = answer

    # ------------------------------------------------------------------
    # External MCP tool discovery
    # ------------------------------------------------------------------

    async def _discover_external_tools(self) -> tuple[list[str], list[dict[str, Any]]]:
        if self.external_tool_gateway is None or not self.external_tool_gateway.is_configured():
            return [], []
        try:
            ext_tools = await self.external_tool_gateway.discover_tools()
        except Exception:
            logger.warning("Failed to discover external MCP tools", exc_info=True)
            return [], []
        names = [t.name for t in ext_tools if t.enabled]
        descs = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "tags": t.tags,
                "source": "external",
            }
            for t in ext_tools
            if t.enabled
        ]
        return names, descs

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        external_tool_names: list[str],
    ) -> ToolExecutionResult:
        if tool_name in external_tool_names and self.external_tool_gateway is not None:
            ext_result = await self.external_tool_gateway.call_tool(
                tool_name=tool_name,
                arguments=tool_input,
            )
            status = "succeeded" if ext_result.status == "succeeded" else "failed"
            trace = ToolCallTrace(
                call_id=ext_result.call_id,
                tool_name=ext_result.tool_name,
                input=tool_input,
                output=ext_result.output,
                status=status,
                tool_category="external_mcp",
                latency_ms=ext_result.latency_ms,
                error_message=ext_result.error_message,
            )
            return ToolExecutionResult(
                call_id=ext_result.call_id,
                tool_name=ext_result.tool_name,
                status=status,
                output=ext_result.output,
                error_message=ext_result.error_message,
                validation_passed=ext_result.status != "invalid_input",
                trace=trace,
            )
        return await self.tool_executor.execute_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
        )

    # ------------------------------------------------------------------
    # ReAct decision logic
    # ------------------------------------------------------------------

    async def _decide_next_step(
        self,
        *,
        question: str,
        tool_descriptions: list[dict[str, Any]],
        steps: list[ReActStep],
        session_context: dict[str, Any] | None,
        task_context: dict[str, Any] | None,
        preference_context: dict[str, Any] | None,
    ) -> ReActDecision:
        prompt = self._load_prompt(self.decision_prompt_path, "ReAct decision")
        try:
            return await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data={
                    "question": question,
                    "available_tools": tool_descriptions,
                    "previous_steps": [step.model_dump(mode="json") for step in steps],
                    "memory_context": {
                        "session_context": session_context or {},
                        "task_context": task_context or {},
                        "preference_context": preference_context or {},
                    },
                },
                response_model=ReActDecision,
            )
        except (LLMAdapterError, OSError, ValueError) as exc:
            classified = getattr(exc, "classified", None)
            if classified is not None:
                logger.warning(
                    "ReAct decision failed (reason=%s, compress=%s, fallback=%s)",
                    classified.reason.value,
                    classified.should_compress,
                    classified.should_fallback,
                )
            else:
                logger.exception("Failed to decide next ReAct step")
            raise ResearchQAAgentError("Failed to decide next ReAct step") from exc

    def _fallback_decide_next_step(
        self,
        *,
        question: str,
        steps: list[ReActStep],
        session_context: dict[str, Any] | None,
        task_context: dict[str, Any] | None,
        preference_context: dict[str, Any] | None,
        initial_retrieval_result: HybridRetrievalResult | None,
        initial_evidence_bundle: EvidenceBundle | None,
    ) -> ReActDecision:
        has_evidence = bool(initial_evidence_bundle and initial_evidence_bundle.evidences)
        if not steps:
            if has_evidence:
                return ReActDecision(
                    thought="Existing graph evidence is already available in the current LangGraph state, so answer directly with grounded evidence.",
                    action="tool",
                    tool_name="answer_with_evidence",
                    tool_input=self._normalize_answer_tool_input(
                        question=question,
                        tool_input={},
                        metadata={"decision_source": "react_fallback"},
                        session_context=session_context,
                        task_context=task_context,
                        preference_context=preference_context,
                        retrieval_result=initial_retrieval_result,
                        evidence_bundle=initial_evidence_bundle or EvidenceBundle(),
                    ),
                    confidence=0.6,
                )
            return ReActDecision(
                thought="No seeded evidence is available yet, so retrieve evidence before answering.",
                action="tool",
                tool_name="hybrid_retrieve",
                tool_input=self._normalize_retrieve_tool_input(
                    question=question,
                    tool_input={},
                    session_context=session_context,
                    task_context=task_context,
                ),
                confidence=0.4,
            )

        last_step = steps[-1]
        if last_step.tool_name == "hybrid_retrieve" and last_step.observation.get("status") == "succeeded":
            return ReActDecision(
                thought="Retrieval has succeeded, so produce a grounded answer from the retrieved evidence.",
                action="tool",
                tool_name="answer_with_evidence",
                tool_input=self._normalize_answer_tool_input(
                    question=question,
                    tool_input={},
                    metadata={"decision_source": "react_fallback"},
                    session_context=session_context,
                    task_context=task_context,
                    preference_context=preference_context,
                    retrieval_result=initial_retrieval_result,
                    evidence_bundle=initial_evidence_bundle or EvidenceBundle(),
                ),
                confidence=0.65,
            )

        if last_step.tool_name == "answer_with_evidence" and last_step.observation.get("status") == "succeeded":
            answer_text = "证据不足"
            output = last_step.observation.get("output")
            if isinstance(output, dict):
                answer_text = str(output.get("answer") or answer_text)
            return ReActDecision(
                thought="A grounded answer is already available from the answer tool, so finish.",
                action="finish",
                final_answer=answer_text,
                confidence=0.7,
            )

        return ReActDecision(
            thought="Tool planning degraded, so finish conservatively with the currently available evidence.",
            action="finish",
            final_answer="证据不足",
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Answer synthesis
    # ------------------------------------------------------------------

    async def _synthesize_answer(
        self,
        *,
        question: str,
        steps: list[ReActStep],
        retrieval_result: HybridRetrievalResult | None,
        evidence_bundle: EvidenceBundle,
        session_context: dict[str, Any] | None,
        task_context: dict[str, Any] | None,
        preference_context: dict[str, Any] | None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> ReActFinalDraft:
        if not evidence_bundle.evidences:
            return ReActFinalDraft(answer="证据不足", confidence=0.0, warnings=["empty_evidence_bundle"])

        prompt = self._load_prompt(self.synthesis_prompt_path, "ReAct synthesis")
        input_data = {
            "question": question,
            "steps": [step.model_dump(mode="json") for step in steps],
            "retrieval_result": retrieval_result.model_dump(mode="json") if retrieval_result else None,
            "evidence_bundle": evidence_bundle.model_dump(mode="json"),
            "memory_context": {
                "session_context": session_context or {},
                "task_context": task_context or {},
                "preference_context": preference_context or {},
            },
        }
        if on_token is not None:
            try:
                full_text = await self.llm_adapter.generate_streaming(
                    prompt=prompt,
                    input_data=input_data,
                    on_token=on_token,
                )
                return ReActFinalDraft(answer=full_text, confidence=self._STREAMING_DEFAULT_CONFIDENCE)
            except Exception:
                logger.warning("ResearchQAAgent streaming synthesis failed, falling back to structured", exc_info=True)
        try:
            return await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data=input_data,
                response_model=ReActFinalDraft,
            )
        except (LLMAdapterError, OSError, ValueError) as exc:
            logger.exception("Failed to synthesize ReAct final answer")
            raise ResearchQAAgentError("Failed to synthesize ReAct final answer") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_retrieve_tool_input(
        self,
        *,
        question: str,
        tool_input: dict[str, Any],
        session_context: dict[str, Any] | None,
        task_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(tool_input)
        normalized.setdefault("question", question)
        if not isinstance(normalized.get("document_ids"), list):
            normalized["document_ids"] = []
        normalized.setdefault("filters", {})
        if session_context:
            session_id = session_context.get("session_id")
            if session_id and not normalized.get("session_id"):
                normalized["session_id"] = session_id
        if task_context and task_context.get("task_id") and not normalized.get("task_id"):
            normalized["task_id"] = task_context.get("task_id")
        return normalized

    def _normalize_answer_tool_input(
        self,
        *,
        question: str,
        tool_input: dict[str, Any],
        metadata: dict[str, Any] | None,
        session_context: dict[str, Any] | None,
        task_context: dict[str, Any] | None,
        preference_context: dict[str, Any] | None,
        retrieval_result: HybridRetrievalResult | None,
        evidence_bundle: EvidenceBundle,
    ) -> dict[str, Any]:
        normalized = dict(tool_input)
        normalized.setdefault("question", question)
        if retrieval_result is not None and "retrieval_result" not in normalized:
            normalized["retrieval_result"] = retrieval_result.model_dump(mode="json")
        if "evidence_bundle" not in normalized:
            normalized["evidence_bundle"] = evidence_bundle.model_dump(mode="json")
        normalized.setdefault("metadata", metadata or {})
        normalized.setdefault("session_context", session_context or {})
        normalized.setdefault("task_context", task_context or {})
        normalized.setdefault("preference_context", preference_context or {})
        return normalized

    def _extract_retrieval_context(
        self,
        steps: list[ReActStep],
    ) -> tuple[HybridRetrievalResult | None, EvidenceBundle]:
        for step in reversed(steps):
            if step.action != "tool" or step.observation.get("status") != "succeeded":
                continue
            output = step.observation.get("output")
            if step.tool_name == "hybrid_retrieve" and isinstance(output, dict):
                try:
                    retrieval_result = output.get("retrieval_result")
                    evidence_bundle = output.get("evidence_bundle")
                    return (
                        HybridRetrievalResult.model_validate(retrieval_result)
                        if isinstance(retrieval_result, dict)
                        else None,
                        EvidenceBundle.model_validate(evidence_bundle)
                        if isinstance(evidence_bundle, dict)
                        else EvidenceBundle(),
                    )
                except Exception:
                    logger.warning("Failed to parse hybrid_retrieve output", exc_info=True)
        return None, EvidenceBundle()

    def _tool_result_observation(self, tool_result: ToolExecutionResult) -> dict[str, Any]:
        return {
            "status": tool_result.status,
            "call_id": tool_result.call_id,
            "tool_name": tool_result.tool_name,
            "output": tool_result.output,
            "error_message": tool_result.error_message,
            "latency_ms": tool_result.trace.latency_ms,
        }

    def _load_prompt(self, path: Path, purpose: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed to load ReAct prompt", extra={"path": str(path), "purpose": purpose})
            raise ResearchQAAgentError(f"Failed to load ReAct prompt: {path}") from exc

    def _looks_insufficient(self, answer: str) -> bool:
        normalized = answer.strip().lower()
        return "证据不足" in normalized or "insufficient evidence" in normalized
