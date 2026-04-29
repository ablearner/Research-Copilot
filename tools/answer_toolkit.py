import logging
from pathlib import Path
import re
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, format_llm_error, is_expected_provider_error
from adapters.local_runtime import LocalLLMAdapter
from chains.answer_chain import AnswerChain
from core.prompt_resolver import PromptResolver
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult
from reasoning.cot import CoTReasoningAgent
from reasoning.strategies import ReasoningStrategySet
from reasoning.style import normalize_reasoning_style

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_GENERIC_DOCUMENT_QUESTION_MARKERS = (
    "summarize",
    "summary",
    "main idea",
    "what is this paper about",
    "what is this document about",
    "主要讲了什么",
    "主要内容是什么",
    "概括一下",
    "总结一下",
)
_GENERIC_DOCUMENT_ENTITY_HINTS = (
    "document",
    "paper",
    "resume",
    "文档",
    "论文",
    "文章",
    "简历",
    "资料",
)
_GENERIC_DOCUMENT_SUMMARY_HINTS = (
    "about",
    "summary",
    "main idea",
    "讲了什么",
    "说了什么",
    "主要讲什么",
)

_CHART_QUESTION_HINTS = ("图", "chart", "graph", "figure", "plot", "diagram", "axis", "柱状图", "折线图", "散点图", "饼图")
_CHART_SNIPPET_HINTS = ("chart", "graph", "figure", "plot", "diagram", "axis", "图表", "图示", "坐标轴", "柱状图", "折线图", "散点图", "饼图")


class AnswerAgentError(RuntimeError):
    """Raised when evidence-grounded answer generation fails."""


class AnswerInput(BaseModel):
    question: str
    evidence_bundle: EvidenceBundle
    retrieval_result: HybridRetrievalResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_context: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] = Field(default_factory=dict)
    preference_context: dict[str, Any] = Field(default_factory=dict)
    retrieval_cache_summary: str | None = None
    memory_hints: dict[str, Any] = Field(default_factory=dict)


def looks_insufficient(answer: str) -> bool:
    normalized = answer.strip().lower()
    return "证据不足" in normalized or "insufficient evidence" in normalized


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _contains_latin(text: str) -> bool:
    return any(("a" <= char.lower() <= "z") for char in text)


def _extract_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = {
        token
        for token in _TOKEN_PATTERN.findall(lowered)
        if len(token) >= 3 and token not in _EN_STOPWORDS
    }
    for match in _CJK_PATTERN.findall(text):
        sequence = match.strip()
        if len(sequence) < 2:
            continue
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def is_generic_document_question(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    # Heuristic fallback only: this should help alignment checks, not act as
    # the top-level router for the whole system.
    if any(marker in normalized for marker in _GENERIC_DOCUMENT_QUESTION_MARKERS):
        return True
    if any(marker in question for marker in _GENERIC_DOCUMENT_QUESTION_MARKERS if _contains_cjk(marker)):
        return True
    if any(marker in question for marker in _GENERIC_DOCUMENT_ENTITY_HINTS if _contains_cjk(marker)) and any(
        marker in question for marker in _GENERIC_DOCUMENT_SUMMARY_HINTS if _contains_cjk(marker)
    ):
        return True
    return any(marker in normalized for marker in _GENERIC_DOCUMENT_ENTITY_HINTS if not _contains_cjk(marker)) and any(
        marker in normalized for marker in _GENERIC_DOCUMENT_SUMMARY_HINTS if not _contains_cjk(marker)
    )


def is_chart_question(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    # Tool-local heuristic only: chart routing at the supervisor/service layer
    # should rely on richer context first.
    return any(marker in normalized for marker in _CHART_QUESTION_HINTS) or any(
        marker in question for marker in _CHART_QUESTION_HINTS if _contains_cjk(marker)
    )


def has_chart_grounding(evidence_bundle: EvidenceBundle) -> bool:
    chart_like_sources = {"chart", "page_image"}
    if any(evidence.source_type in chart_like_sources for evidence in evidence_bundle.evidences):
        return True

    for evidence in evidence_bundle.evidences[:5]:
        snippet = (evidence.snippet or "").strip()
        if not snippet:
            continue
        lowered = snippet.lower()
        if any(marker in lowered for marker in _CHART_SNIPPET_HINTS[:6]):
            return True
        if any(marker in snippet for marker in _CHART_SNIPPET_HINTS[6:]):
            return True
    return False


def assess_question_evidence_alignment(
    *,
    question: str,
    evidence_bundle: EvidenceBundle,
    retrieval_result: HybridRetrievalResult | None,
) -> dict[str, Any]:
    # Keep this function lightweight and conservative. It should detect obvious
    # alignment issues, not fully classify user intent.
    if is_generic_document_question(question):
        return {
            "aligned": True,
            "generic_document_question": True,
            "question_term_count": 0,
            "overlap_count": 0,
            "top_score": None,
        }

    question_terms = _extract_terms(question)
    evidence_terms: set[str] = set()
    for evidence in evidence_bundle.evidences[:5]:
        snippet = (evidence.snippet or "").strip()
        if snippet:
            evidence_terms.update(_extract_terms(snippet[:500]))

    overlap = question_terms & evidence_terms
    top_score = max(
        (
            score
            for score in [
                *[
                    evidence.score
                    for evidence in evidence_bundle.evidences[:5]
                    if isinstance(evidence.score, (int, float))
                ],
                *[
                    (hit.merged_score if hit.merged_score is not None else hit.vector_score or hit.graph_score)
                    for hit in (retrieval_result.hits[:5] if retrieval_result else [])
                    if (
                        hit.merged_score is not None
                        or hit.vector_score is not None
                        or hit.graph_score is not None
                    )
                ],
            ]
            if isinstance(score, (int, float))
        ),
        default=None,
    )
    same_script = (_contains_cjk(question) and any(_contains_cjk(e.snippet or "") for e in evidence_bundle.evidences[:5])) or (
        _contains_latin(question) and any(_contains_latin(e.snippet or "") for e in evidence_bundle.evidences[:5])
    )
    aligned = bool(overlap)
    if not aligned and top_score is not None and top_score >= 0.72:
        aligned = True
    if not aligned and not same_script and top_score is not None and top_score >= 0.8:
        aligned = True

    return {
        "aligned": aligned,
        "generic_document_question": False,
        "question_term_count": len(question_terms),
        "evidence_term_count": len(evidence_terms),
        "overlap_count": len(overlap),
        "same_script": same_script,
        "top_score": top_score,
    }


def normalize_answer_response(
    *,
    response: QAResponse,
    question: str,
    evidence_bundle: EvidenceBundle,
    retrieval_result: HybridRetrievalResult | None,
    metadata: dict[str, Any],
) -> QAResponse:
    return response.model_copy(
        update={
            "question": question,
            "evidence_bundle": evidence_bundle,
            "retrieval_result": retrieval_result,
            "confidence": response.confidence,
            "metadata": {
                **response.metadata,
                **metadata,
                "answered_by": "AnswerAgent",
                "evidence_count": len(evidence_bundle.evidences),
            },
        }
    )


def insufficient_evidence_response(
    *,
    question: str,
    evidence_bundle: EvidenceBundle,
    retrieval_result: HybridRetrievalResult | None,
    metadata: dict[str, Any],
    reason: str = "empty_evidence_bundle",
) -> QAResponse:
    return QAResponse(
        answer="证据不足",
        question=question,
        evidence_bundle=evidence_bundle,
        retrieval_result=retrieval_result,
        confidence=0.0,
        metadata={
            **metadata,
            "answered_by": "AnswerAgent",
            "reason": reason,
        },
    )


def fallback_answer_response(
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
        return insufficient_evidence_response(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata={
                **metadata,
                "reason": reason,
                "fallback": True,
                "fallback_mode": "local_extract",
            },
            reason="fallback_without_snippets",
        )

    answer_lines = ["当前模型回答失败，下面仅列出可能相关的证据片段供参考："]
    for index, snippet in enumerate(snippets, start=1):
        answer_lines.append(f"{index}. {snippet[:240]}")
    answer_lines.append("这些片段不一定能直接回答当前问题；如果问题与文档无关，请忽略这些证据。")
    return QAResponse(
        answer="\n".join(answer_lines),
        question=question,
        evidence_bundle=evidence_bundle,
        retrieval_result=retrieval_result,
        confidence=0.22,
        metadata={
            **metadata,
            "answered_by": "AnswerAgentFallback",
            "fallback": True,
            "fallback_mode": "local_extract",
            "fallback_reason": reason,
            "evidence_count": len(evidence_bundle.evidences),
        },
    )


def missing_chart_grounding_response(
    *,
    question: str,
    evidence_bundle: EvidenceBundle,
    retrieval_result: HybridRetrievalResult | None,
    metadata: dict[str, Any],
) -> QAResponse:
    language = "zh" if _contains_cjk(question) else "en"
    answer = (
        "当前检索到的证据主要是文档文本，还没有图表本身的视觉或图表级证据，因此我不能可靠回答这张图具体在讲什么。请改用图表理解或图表融合问答链路。"
        if language == "zh"
        else "The current evidence is mostly document text and does not include chart-specific visual evidence, so I cannot reliably answer what this graph is mainly about. Please use chart understanding or fused chart+document QA for that figure."
    )
    return QAResponse(
        answer=answer,
        question=question,
        evidence_bundle=evidence_bundle,
        retrieval_result=retrieval_result,
        confidence=0.08,
        metadata={
            **metadata,
            "answered_by": "AnswerAgentGuardrail",
            "reason": "missing_chart_grounding",
            "question_type": "chart",
            "suggested_route": "ask_fused_or_chart_understand",
            "evidence_count": len(evidence_bundle.evidences),
        },
    )


async def local_fallback_answer_response(
    *,
    question: str,
    evidence_bundle: EvidenceBundle,
    retrieval_result: HybridRetrievalResult | None,
    metadata: dict[str, Any],
    reason: str,
) -> QAResponse:
    alignment = assess_question_evidence_alignment(
        question=question,
        evidence_bundle=evidence_bundle,
        retrieval_result=retrieval_result,
    )
    if not alignment["aligned"]:
        return insufficient_evidence_response(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata={
                **metadata,
                "fallback": True,
                "fallback_reason": reason,
                "alignment": alignment,
            },
            reason="weak_question_evidence_alignment",
        )
    try:
        response = await LocalLLMAdapter().generate_structured(
            prompt="Generate a concise answer using only the supplied evidence.",
            input_data={
                "question": question,
                "evidence_bundle": evidence_bundle.model_dump(mode="json"),
            },
            response_model=QAResponse,
        )
        return response.model_copy(
            update={
                "question": question,
                "evidence_bundle": evidence_bundle,
                "retrieval_result": retrieval_result,
                "metadata": {
                    **response.metadata,
                    **metadata,
                    "answered_by": "AnswerAgentFallback",
                    "fallback": True,
                    "fallback_mode": "local_extract",
                    "fallback_reason": reason,
                    "evidence_count": len(evidence_bundle.evidences),
                },
            }
        )
    except Exception:
        logger.exception("Local fallback answer generation failed")
        return fallback_answer_response(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=metadata,
            reason=reason,
        )


class AnswerAgent:
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        prompt_path: str | Path = "prompts/document/answer_question_with_hybrid_rag.txt",
        prompt_resolver: PromptResolver | None = None,
        cot_reasoning_agent: CoTReasoningAgent | None = None,
        reasoning_strategies: ReasoningStrategySet | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter or LocalLLMAdapter()
        self.prompt_path = Path(prompt_path)
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.reasoning_strategies = reasoning_strategies or ReasoningStrategySet(
            answer_synthesis=cot_reasoning_agent,
        )
        self.cot_reasoning_agent = (
            cot_reasoning_agent
            or self.reasoning_strategies.cot_reasoning_agent
        )
        self.chain = AnswerChain(
            llm=self.llm_adapter,
            prompt_path=self.prompt_path,
            prompt_resolver=self.prompt_resolver,
        )
        self.answer_tool = StructuredTool.from_function(
            coroutine=self.answer,
            name="answer_with_evidence",
            description="Generate a structured answer grounded in evidence.",
            args_schema=AnswerInput,
        )
        self.answer_chain = RunnableLambda(lambda payload: payload)

    async def answer(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        retrieval_cache_summary: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> QAResponse:
        resolved_output_style = dict(preference_context or {})
        reasoning_style = normalize_reasoning_style(
            resolved_output_style.get("reasoning_style") or (metadata or {}).get("reasoning_style")
        )
        answer_metadata = {
            **(metadata or {}),
            "session_context": session_context or {},
            "task_context": task_context or {},
            "preference_context": preference_context or {},
            "retrieval_cache_summary": retrieval_cache_summary,
            "memory_hints": memory_hints or {},
            "output_style": resolved_output_style,
        }
        cot_strategy = self.reasoning_strategies.answer_synthesis or self.cot_reasoning_agent
        use_cot = cot_strategy is not None and (
            reasoning_style == "cot"
            or ((metadata or {}).get("qa_mode") == "research_collection" and reasoning_style != "react")
        )
        if not evidence_bundle.evidences:
            return insufficient_evidence_response(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=answer_metadata,
                reason="empty_evidence_bundle",
            )
        if is_chart_question(question) and not has_chart_grounding(evidence_bundle):
            return missing_chart_grounding_response(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=answer_metadata,
            )
        if use_cot:
            return await cot_strategy.reason(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=answer_metadata,
                session_context=session_context,
                task_context=task_context,
                preference_context=preference_context,
                memory_hints=memory_hints,
            )

        try:
            response = await self.chain.ainvoke(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=metadata,
                session_context=session_context,
                task_context=task_context,
                preference_context=preference_context,
                retrieval_cache_summary=retrieval_cache_summary,
                memory_hints=memory_hints,
            )
            return normalize_answer_response(
                response=response,
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=answer_metadata,
            )
        except (LLMAdapterError, OSError, ValueError) as exc:
            if is_expected_provider_error(exc):
                logger.warning(
                    "Failed to generate evidence-grounded answer; using fallback answer. cause=%s",
                    format_llm_error(exc),
                )
            else:
                logger.exception("Failed to generate evidence-grounded answer")
            return await local_fallback_answer_response(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                metadata=answer_metadata,
                reason=exc.__class__.__name__,
            )

    async def answer_with_evidence(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        retrieval_cache_summary: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> QAResponse:
        return await self.answer(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=metadata,
            session_context=session_context,
            task_context=task_context,
            preference_context=preference_context,
            retrieval_cache_summary=retrieval_cache_summary,
            memory_hints=memory_hints,
        )


AnswerTools = AnswerAgent
AnswerToolsError = AnswerAgentError
