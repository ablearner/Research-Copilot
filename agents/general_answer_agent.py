from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

logger = logging.getLogger(__name__)


class GeneralAnswerResult(BaseModel):
    answer: str
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    answer_type: str = "general"
    warnings: list[str] = Field(default_factory=list)


class GeneralAnswerAgent:
    """Worker agent for general non-research questions."""

    name = "GeneralAnswerAgent"
    _STREAMING_DEFAULT_CONFIDENCE: float = 0.7

    def __init__(
        self,
        *,
        llm_adapter: BaseLLMAdapter | None = None,
        llm_timeout_seconds: float = 30.0,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.llm_timeout_seconds = llm_timeout_seconds

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchToolResult
        from runtime.research.unified_action_adapters import resolve_active_message

        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        on_token = None
        if context.progress_callback is not None:
            async def on_token(text: str) -> None:
                await context.progress_callback({"type": "token", "text": text})
        result = await self.answer(
            question=question,
            conversation_context={
                "mode": context.request.mode,
                "task_id": context.request.task_id,
                "has_task": context.task is not None,
                "selected_paper_ids": [] if payload.get("ignore_research_context") else list(context.request.selected_paper_ids),
                "ignore_research_context": bool(payload.get("ignore_research_context")),
            },
            on_token=on_token,
        )
        warnings = list(result.warnings)
        provider_fallback = result.answer_type in {"fallback", "provider_timeout", "provider_error"}
        should_reroute = (not provider_fallback) and (
            "route_mismatch" in warnings or (
                result.answer_type == "reroute_hint"
            ) or (
                result.confidence < 0.45 and (
                    context.request.task_id is not None
                    or bool(context.request.selected_paper_ids)
                    or bool(context.request.selected_document_ids)
                    or bool(context.request.chart_image_path)
                    or bool(context.request.document_file_path)
                )
            )
        )
        if should_reroute:
            return ResearchToolResult(
                status="skipped",
                observation="general answer agent detected a likely route mismatch and requested supervisor rerouting",
                metadata={
                    **result.model_dump(mode="json"),
                    "reason": "route_mismatch",
                    "suggested_action": self._suggested_action(context=context),
                },
            )
        context.general_answer = result.answer
        context.general_answer_metadata = result.model_dump(mode="json")
        return ResearchToolResult(
            status="succeeded",
            observation="general question answered directly without research workspace tools",
            metadata=context.general_answer_metadata,
        )

    def _suggested_action(self, *, context: ResearchAgentToolContext) -> str:
        if context.request.chart_image_path:
            return "supervisor_understand_chart"
        if context.request.document_file_path:
            return "understand_document"
        if context.request.task_id or context.request.selected_paper_ids or context.request.selected_document_ids:
            return "answer_question"
        return "search_literature"

    # ------------------------------------------------------------------
    # Core LLM answer method
    # ------------------------------------------------------------------

    async def answer(
        self,
        *,
        question: str,
        conversation_context: dict[str, Any] | None = None,
        on_token: Callable[[str], Awaitable[None]] | None = None,
    ) -> GeneralAnswerResult:
        if self.llm_adapter is None:
            return GeneralAnswerResult(
                answer=(
                    "这是一个通用问题，但当前没有可用的通用回答模型。"
                    "配置 LLM 后，GeneralAnswerAgent 就可以直接回答这类问题。"
                ),
                confidence=0.5,
                answer_type="fallback",
                warnings=["missing_llm_adapter"],
            )
        prompt = (
            "你是 GeneralAnswerAgent，负责回答不需要论文检索、RAG、本地文档、图表解析或研究工作区操作的通用问题。\n"
            "请直接回答用户问题，语言尽量跟随用户输入。\n"
            "如果问题明显更适合科研检索、论文问答、文档理解或图表理解，请在 warnings 中加入 route_mismatch，"
            "并把 answer_type 设为 reroute_hint，同时给出一个简短帮助性回答。\n"
            "不要伪造论文证据、网页检索结果或本地已导入内容。"
        )
        input_data = {
            "question": question,
            "conversation_context": conversation_context or {},
        }
        if on_token is not None:
            try:
                full_text = await asyncio.wait_for(
                    self.llm_adapter.generate_streaming(
                        prompt=prompt,
                        input_data=input_data,
                        on_token=on_token,
                    ),
                    timeout=max(1.0, float(self.llm_timeout_seconds)),
                )
                return GeneralAnswerResult(answer=full_text, confidence=self._STREAMING_DEFAULT_CONFIDENCE, answer_type="streaming")
            except TimeoutError:
                logger.warning("GeneralAnswerAgent streaming LLM call timed out")
                return GeneralAnswerResult(
                    answer="通用回答模型暂时没有响应，请稍后重试。",
                    confidence=0.5,
                    answer_type="provider_timeout",
                    warnings=["llm_timeout"],
                )
            except Exception:
                logger.warning("GeneralAnswerAgent streaming failed, falling back to structured", exc_info=True)
        try:
            return await asyncio.wait_for(
                self.llm_adapter.generate_structured(
                    prompt=prompt,
                    input_data=input_data,
                    response_model=GeneralAnswerResult,
                ),
                timeout=max(1.0, float(self.llm_timeout_seconds)),
            )
        except TimeoutError:
            logger.warning("GeneralAnswerAgent LLM call timed out")
            return GeneralAnswerResult(
                answer="通用回答模型暂时没有响应，请稍后重试。",
                confidence=0.5,
                answer_type="provider_timeout",
                warnings=["llm_timeout"],
            )
        except LLMAdapterError:
            logger.warning("GeneralAnswerAgent LLM provider failed", exc_info=True)
            return GeneralAnswerResult(
                answer="通用回答模型暂时不可用，请稍后重试。",
                confidence=0.5,
                answer_type="provider_error",
                warnings=["llm_provider_error"],
            )
        except Exception as exc:
            raise LLMAdapterError(f"GeneralAnswerAgent failed: {exc}") from exc
