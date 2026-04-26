from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adapters.llm.langchain_binding import ensure_provider_binding
from core.prompt_resolver import PromptResolver
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult


class AnswerChain:
    def __init__(
        self,
        llm: Any,
        prompt_path: str | Path = "prompts/document/answer_question_with_hybrid_rag.txt",
        prompt_resolver: PromptResolver | None = None,
    ) -> None:
        self.llm = ensure_provider_binding(llm)
        self.prompt_path = Path(prompt_path)
        self.prompt_resolver = prompt_resolver or PromptResolver()

    async def ainvoke(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        retrieval_cache_summary: str | None = None,
        memory_hints: dict[str, Any] | None = None,
        skill_context: dict[str, Any] | None = None,
    ) -> QAResponse:
        prompt_text = self._resolve_prompt(skill_context)
        skill_style = skill_context.get("output_style") if isinstance(skill_context, dict) else {}
        if not isinstance(skill_style, dict):
            skill_style = {}
        resolved_output_style = {**skill_style, **(preference_context or {})}
        answer_metadata = {
            **(metadata or {}),
            "session_context": session_context or {},
            "task_context": task_context or {},
            "preference_context": preference_context or {},
            "retrieval_cache_summary": retrieval_cache_summary,
            "memory_hints": memory_hints or {},
            "skill_context": skill_context or {},
            "output_style": resolved_output_style,
        }
        compact_evidence_bundle = self._compact_evidence_bundle(evidence_bundle)
        compact_retrieval_result = self._compact_retrieval_result(retrieval_result)
        payload = {
            "question": question,
            "evidence_bundle": compact_evidence_bundle,
            "retrieval_result": compact_retrieval_result,
            "rules": {
                "answer_only_from_evidence": True,
                "return_insufficient_when_unsupported": True,
                "do_not_use_external_knowledge": True,
                "cite_text_chart_and_graph_evidence_when_relevant": True,
            },
            "memory_context": {
                "session_context": session_context or {},
                "task_context": task_context or {},
                "preference_context": preference_context or {},
                "retrieval_cache_summary": retrieval_cache_summary,
                "memory_hints": memory_hints or {},
            },
            "skill_context": skill_context or {},
            "output_style": resolved_output_style,
            "metadata": answer_metadata,
        }
        binding_metadata = {
            **answer_metadata,
            "_input_data_override": payload,
        }
        response = await self.llm.ainvoke_structured(
            messages=[
                SystemMessage(content=prompt_text),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            response_model=QAResponse,
            metadata=binding_metadata,
        )
        return QAResponse.model_validate(response)

    def _compact_evidence_bundle(self, evidence_bundle: EvidenceBundle) -> dict[str, Any]:
        evidences: list[dict[str, Any]] = []
        for evidence in evidence_bundle.evidences[:5]:
            evidences.append(
                {
                    "id": evidence.id,
                    "document_id": evidence.document_id,
                    "page_id": evidence.page_id,
                    "page_number": evidence.page_number,
                    "source_type": evidence.source_type,
                    "source_id": evidence.source_id,
                    "snippet": (evidence.snippet or "")[:800],
                    "score": evidence.score,
                    "metadata": {
                        key: value
                        for key, value in evidence.metadata.items()
                        if key
                        in {
                            "title",
                            "caption",
                            "chart_type",
                            "section",
                            "image_path",
                            "source",
                            "anchor_source",
                            "anchor_selection",
                            "provider",
                            "manifest_kind",
                            "evidence_tier",
                            "summary_only",
                            "llm_generated_summary",
                            "paper_id",
                        }
                    },
                }
            )
        return {
            "summary": (evidence_bundle.summary or "")[:1200] or None,
            "evidences": evidences,
            "metadata": {
                "evidence_count": len(evidence_bundle.evidences),
                **{
                    key: value
                    for key, value in evidence_bundle.metadata.items()
                    if key in {"document_ids", "query", "source"}
                },
            },
        }

    def _compact_retrieval_result(self, retrieval_result: HybridRetrievalResult | None) -> dict[str, Any] | None:
        if retrieval_result is None:
            return None
        return {
            "query": {
                "query": retrieval_result.query.query[:1000],
                "document_ids": retrieval_result.query.document_ids[:10],
                "mode": retrieval_result.query.mode,
                "top_k": retrieval_result.query.top_k,
                "graph_query_mode": retrieval_result.query.graph_query_mode,
            },
            "hits": [
                {
                    "id": hit.id,
                    "source_type": hit.source_type,
                    "source_id": hit.source_id,
                    "document_id": hit.document_id,
                    "content": (hit.content or "")[:500],
                    "vector_score": hit.vector_score,
                    "graph_score": hit.graph_score,
                    "merged_score": hit.merged_score,
                    "metadata": {
                        key: value
                        for key, value in hit.metadata.items()
                        if key
                        in {
                            "source",
                            "provider",
                            "manifest_kind",
                            "evidence_tier",
                            "summary_only",
                            "llm_generated_summary",
                            "paper_id",
                            "title",
                            "chart_type",
                            "caption",
                            "anchor_source",
                            "anchor_selection",
                        }
                    },
                }
                for hit in retrieval_result.hits[:5]
            ],
            "metadata": {
                "hit_count": len(retrieval_result.hits),
                **{
                    key: value
                    for key, value in retrieval_result.metadata.items()
                    if key in {"source", "vector_hit_count", "graph_hit_count", "graph_summary_hit_count"}
                },
            },
        }

    def _resolve_prompt(self, skill_context: dict[str, Any] | None) -> str:
        prompt_set = skill_context.get("prompt_set") if isinstance(skill_context, dict) else {}
        explicit = prompt_set.get("answer_prompt_path") if isinstance(prompt_set, dict) else None
        skill_name = skill_context.get("name") if isinstance(skill_context, dict) else None
        resolved_prompt_path = self.prompt_resolver.resolve_prompt_path(
            prompt_key="answer_prompt_path",
            skill_name=skill_name if isinstance(skill_name, str) else None,
            explicit_prompt_path=explicit if isinstance(explicit, str) else str(self.prompt_path),
        )
        return resolved_prompt_path.read_text(encoding="utf-8")
