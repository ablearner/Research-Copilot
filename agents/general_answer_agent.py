from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError
from domain.schemas.unified_runtime import UnifiedAgentResult, UnifiedAgentTask
from services.research.research_specialist_capabilities import (
    GeneralAnswerCapability,
    build_specialist_unified_result,
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
        execution_capability: GeneralAnswerCapability | None = None,
        llm_timeout_seconds: float = 30.0,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.execution_capability = execution_capability or GeneralAnswerCapability()
        self.llm_timeout_seconds = llm_timeout_seconds

    async def execute(self, task: UnifiedAgentTask, runtime_context: Any) -> UnifiedAgentResult:
        supervisor_context = runtime_context.metadata.get("supervisor_tool_context")
        decision = runtime_context.metadata.get("supervisor_decision")
        if supervisor_context is None or decision is None:
            return build_specialist_unified_result(
                task=task,
                agent_name=self.name,
                status="failed",
                observation="missing supervisor runtime context for GeneralAnswerAgent",
                metadata={"reason": "missing_supervisor_runtime_context"},
                execution_adapter="general_answer_agent",
                delegate_type=self.__class__.__name__,
            )
        result = await self.execution_capability.run(
            context=supervisor_context,
            decision=decision,
            general_answer_agent=self,
        )
        metadata = {
            **dict(result.metadata),
            "executed_by": self.name,
            "specialist_execution_path": "general_answer_agent",
        }
        return build_specialist_unified_result(
            task=task,
            agent_name=self.name,
            status=result.status,
            observation=result.observation,
            metadata=metadata,
            execution_adapter="general_answer_agent",
            delegate_type=self.__class__.__name__,
        )

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
