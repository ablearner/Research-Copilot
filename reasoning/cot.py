from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, format_llm_error, is_expected_provider_error
from adapters.llm.langchain_binding import LangChainProviderBinding, ensure_provider_binding
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult

logger = logging.getLogger(__name__)


class CoTReasoningDraft(BaseModel):
    answer: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    reasoning_summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class CoTReasoningAgent:
    """Hidden chain-of-thought answer synthesis without exposing private reasoning."""

    def __init__(self, llm_adapter: BaseLLMAdapter | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.binding = self._build_binding(llm_adapter)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an evidence-grounded chain-of-thought answer agent. "
                        "Reason privately and step by step, but do not reveal private chain of thought. "
                        "Return only structured output with the final answer, a short reasoning_summary, and optional warnings. "
                        "Use only the supplied evidence. If the evidence is weak, answer conservatively."
                    ),
                ),
                ("human", "{payload_json}"),
            ]
        )

    async def reason(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        memory_hints: dict[str, Any] | None = None,
        skill_context: dict[str, Any] | None = None,
    ) -> QAResponse:
        answer_metadata = {
            **(metadata or {}),
            "answered_by": self.__class__.__name__,
            "reasoning_style": "cot",
            "evidence_count": len(evidence_bundle.evidences),
        }
        if not evidence_bundle.evidences:
            return QAResponse(
                answer="证据不足",
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=0.0,
                metadata={**answer_metadata, "reason": "empty_evidence_bundle"},
            )

        heuristic = self._fallback_response(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=answer_metadata,
            reason="fallback_without_llm",
        )
        if self.binding is None:
            return heuristic

        payload = {
            "question": question,
            "evidences": [
                {
                    "document_id": evidence.document_id,
                    "page_number": evidence.page_number,
                    "source_type": evidence.source_type,
                    "score": evidence.score,
                    "snippet": (evidence.snippet or "")[:400],
                }
                for evidence in evidence_bundle.evidences[:8]
            ],
            "retrieval_hits": [
                {
                    "document_id": hit.document_id,
                    "source_type": hit.source_type,
                    "source_id": hit.source_id,
                    "score": hit.merged_score if hit.merged_score is not None else hit.vector_score or hit.graph_score,
                    "content": (hit.content or "")[:400],
                }
                for hit in (retrieval_result.hits[:8] if retrieval_result else [])
            ],
            "session_context": session_context or {},
            "task_context": task_context or {},
            "preference_context": preference_context or {},
            "memory_hints": memory_hints or {},
            "skill_context": skill_context or {},
        }
        try:
            messages = await self.prompt.ainvoke(
                {"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)}
            )
            draft = await self.binding.ainvoke_structured(
                messages=messages.to_messages(),
                response_model=CoTReasoningDraft,
                metadata={"agent": "cot_reasoning_agent", "task": "grounded_answer"},
            )
            answer = (draft.answer or "").strip() or heuristic.answer
            return QAResponse(
                answer=answer,
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=draft.confidence if draft.confidence is not None else heuristic.confidence,
                metadata={
                    **answer_metadata,
                    "reasoning_summary": draft.reasoning_summary,
                    "warnings": draft.warnings,
                },
            )
        except (LLMAdapterError, OSError, ValueError, Exception) as exc:
            if is_expected_provider_error(exc):
                logger.warning(
                    "CoT reasoning failed; using heuristic fallback. cause=%s",
                    format_llm_error(exc),
                )
            else:
                logger.warning("CoT reasoning failed; using heuristic fallback", exc_info=exc)
            return heuristic.model_copy(
                update={
                    "metadata": {
                        **heuristic.metadata,
                        "fallback": True,
                        "fallback_reason": exc.__class__.__name__,
                    }
                }
            )

    def _fallback_response(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None,
        metadata: dict[str, Any],
        reason: str,
    ) -> QAResponse:
        snippets = [
            (evidence.snippet or "").strip()
            for evidence in evidence_bundle.evidences[:3]
            if (evidence.snippet or "").strip()
        ]
        if not snippets:
            return QAResponse(
                answer="证据不足",
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=0.0,
                metadata={**metadata, "reason": reason},
            )
        answer_lines = ["根据当前证据，可归纳出以下要点："]
        for index, snippet in enumerate(snippets, start=1):
            answer_lines.append(f"{index}. {snippet[:220]}")
        answer_lines.append("以上结论仅基于当前证据整理，如需更强结论应补充更直接的论文证据。")
        return QAResponse(
            answer="\n".join(answer_lines),
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.34,
            metadata={
                **metadata,
                "reason": reason,
                "reasoning_summary": "The CoT fallback compressed the strongest evidence snippets into a concise grounded answer.",
            },
        )

    def _build_binding(self, llm_adapter: BaseLLMAdapter | None) -> LangChainProviderBinding | None:
        if llm_adapter is None:
            return None
        try:
            return ensure_provider_binding(llm_adapter)
        except TypeError:
            return None
