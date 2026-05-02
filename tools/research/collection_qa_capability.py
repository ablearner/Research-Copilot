from __future__ import annotations

import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask, ResearchTaskAskRequest, normalize_reasoning_style
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery, merge_retrieval_hits
from retrieval.evidence_builder import build_evidence_bundle
from tools.research.knowledge_access import ResearchKnowledgeAccess

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
    "论文",
    "研究",
    "paper",
    "papers",
    "study",
    "studies",
}


def _tokenize_research_text(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        normalized = token.lower().strip()
        if len(normalized) <= 1 or normalized in _STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


class ResearchCollectionKnowledgeCapability:
    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def plan_collection_queries(self, state: Any) -> list[str]:
        planned = self._heuristic_plan_collection_queries(state)
        if not self._should_use_plan_and_execute(state) or self.llm_adapter is None:
            return planned
        reasoning_plan = await self._plan_queries(
            objective=f"Grounded research collection QA for: {state.question}",
            seed_queries=planned,
            context={
                "research_topic": state.task.topic,
                "report_highlights": state.report.highlights[:3] if state.report else [],
                "paper_titles": [paper.title for paper in state.papers[:5]],
                "memory_hints": self._memory_hints(state),
            },
            max_queries=5,
        )
        return self._normalize_queries([*reasoning_plan["queries"], *planned], limit=5)

    async def retrieve_collection_evidence(self, *, graph_runtime: Any, state: Any, query: str) -> list[RetrievalHit]:
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        execution_context = getattr(state, "execution_context", None)
        result = await knowledge_access.retrieve(
            question=query,
            document_ids=state.document_ids,
            top_k=state.top_k,
            filters={
                "research_task_id": state.task.task_id,
                "research_topic": state.task.topic,
                "qa_mode": "research_collection",
                **self._scope_filters(state),
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=state.task.task_id,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
        )
        return result.retrieval_result.hits

    async def retrieve_graph_summary(self, *, graph_runtime: Any, state: Any) -> list[RetrievalHit]:
        if not state.document_ids:
            return []
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        execution_context = getattr(state, "execution_context", None)
        summary_output = await knowledge_access.query_graph_summary(
            question=state.question,
            document_ids=state.document_ids,
            top_k=max(3, min(state.top_k, 6)),
            filters={
                "research_task_id": state.task.task_id,
                "research_topic": state.task.topic,
                "qa_mode": "research_collection",
                **self._scope_filters(state),
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=state.task.task_id,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
        )
        return list(getattr(summary_output, "hits", []) or [])

    def build_collection_manifest(self, state: Any) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        top_papers = state.papers[: max(3, min(len(state.papers), state.top_k))]
        for index, paper in enumerate(top_papers, start=1):
            hits.append(self._paper_manifest_hit(index=index, paper=paper))
        if state.report and self._should_include_report_summary_evidence(state):
            for index, highlight in enumerate(state.report.highlights[:3], start=1):
                hits.append(
                    RetrievalHit(
                        id=f"manifest:highlight:{state.report.report_id}:{index}",
                        source_type="text_block",
                        source_id=f"report_highlight_{index}",
                        content=f"研究报告亮点 #{index}: {highlight}",
                        merged_score=max(0.75 - index * 0.05, 0.45),
                        metadata={
                            "provider": "research_collection_manifest",
                            "manifest_kind": "report_highlight",
                            "source": "llm_generated_research_report",
                            "evidence_tier": "report_summary_fallback",
                            "summary_only": True,
                            "llm_generated_summary": True,
                            "report_id": state.report.report_id,
                            "knowledge_capability": self.__class__.__name__,
                        },
                    )
                )
            for index, gap in enumerate(state.report.gaps[:2], start=1):
                hits.append(
                    RetrievalHit(
                        id=f"manifest:gap:{state.report.report_id}:{index}",
                        source_type="text_block",
                        source_id=f"report_gap_{index}",
                        content=f"研究报告证据缺口 #{index}: {gap}",
                        merged_score=max(0.6 - index * 0.05, 0.3),
                        metadata={
                            "provider": "research_collection_manifest",
                            "manifest_kind": "report_gap",
                            "source": "llm_generated_research_report",
                            "evidence_tier": "report_summary_fallback",
                            "summary_only": True,
                            "llm_generated_summary": True,
                            "report_id": state.report.report_id,
                            "knowledge_capability": self.__class__.__name__,
                        },
                    )
                )
        return hits

    def _heuristic_plan_collection_queries(self, state: Any) -> list[str]:
        candidates = [state.question]
        topic = state.task.topic.strip()
        if topic and topic not in state.question:
            candidates.append(f"{topic} {state.question}")
        if state.report and state.report.highlights:
            candidates.append(f"{state.question} {state.report.highlights[0]}")
        memory_hints = self._memory_hints(state)
        memory_focus = str(memory_hints.get("current_task_intent") or "").strip()
        if memory_focus and memory_focus not in state.question:
            candidates.append(f"{state.question} {memory_focus}")
        candidates.extend(self._cross_lingual_queries(state))
        planned: list[str] = []
        for item in candidates:
            normalized = " ".join(item.strip().split())
            if normalized and normalized not in planned:
                planned.append(normalized)
        return planned[:5]

    def _cross_lingual_queries(self, state: Any) -> list[str]:
        question = state.question
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in question)
        if not has_cjk:
            return []
        papers = getattr(state, "papers", []) or []
        if not papers:
            return []
        en_titles = [
            paper.title for paper in papers[:3]
            if paper.title and not any("\u4e00" <= ch <= "\u9fff" for ch in paper.title)
        ]
        if not en_titles:
            return []
        keyword_map = {
            "实验": "experiment results",
            "方法": "method approach",
            "结果": "results evaluation",
            "实现": "implementation method",
            "模型": "model architecture",
            "数据集": "dataset benchmark",
            "评估": "evaluation metrics",
            "对比": "comparison baseline",
            "框架": "framework architecture",
            "训练": "training procedure",
            "性能": "performance",
            "消融": "ablation study",
            "贡献": "contribution",
            "局限": "limitation",
            "创新": "novelty contribution",
        }
        en_keywords = []
        for zh_key, en_val in keyword_map.items():
            if zh_key in question:
                en_keywords.append(en_val)
        queries: list[str] = []
        for title in en_titles[:2]:
            if en_keywords:
                queries.append(f"{title} {' '.join(en_keywords[:3])}")
            else:
                queries.append(title)
        return queries

    def _scope_filters(self, state: Any) -> dict[str, Any]:
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        return {
            "qa_scope_mode": str(metadata.get("qa_scope_mode") or "all_imported"),
            "selected_paper_ids": list(metadata.get("selected_paper_ids") or []),
            "selected_document_ids": list(metadata.get("selected_document_ids") or state.document_ids or []),
        }

    def _should_include_report_summary_evidence(self, state: Any) -> bool:
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        policy = str(metadata.get("report_summary_evidence_policy") or "fallback").strip().lower()
        if policy in {"always", "include", "true"}:
            return True
        if policy in {"never", "exclude", "false"}:
            return False
        return not bool(getattr(state, "retrieval_hits", []) or getattr(state, "summary_hits", []))

    def _paper_manifest_hit(self, *, index: int, paper: PaperCandidate) -> RetrievalHit:
        snippet_parts = [f"候选论文 #{index}: {paper.title}", f"source={paper.source}"]
        if paper.authors:
            snippet_parts.append(f"authors={', '.join(paper.authors[:4])}")
        if paper.year:
            snippet_parts.append(f"year={paper.year}")
        if paper.venue:
            snippet_parts.append(f"venue={paper.venue}")
        if paper.summary:
            snippet_parts.append(f"summary={paper.summary}")
        elif paper.abstract:
            snippet_parts.append(f"abstract={paper.abstract[:360]}")
        if paper.metadata.get("must_read"):
            snippet_parts.append("must_read=true")
        if paper.metadata.get("selected_for_ingest") or paper.pdf_url:
            snippet_parts.append("ingest_ready=true")
        return RetrievalHit(
            id=f"manifest:paper:{paper.paper_id}",
            source_type="text_block",
            source_id=paper.paper_id,
            document_id=str(paper.metadata.get("document_id") or "") or None,
            content=" | ".join(snippet_parts),
            merged_score=float(paper.relevance_score or max(0.95 - index * 0.08, 0.35)),
            metadata={
                "provider": "research_collection_manifest",
                "manifest_kind": "paper_card",
                "source": "paper_metadata_manifest",
                "evidence_tier": "paper_metadata",
                "summary_only": True,
                "paper_source": paper.source,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "knowledge_capability": self.__class__.__name__,
            },
        )

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
        import json

        from adapters.llm.base import LLMAdapterError, is_expected_provider_error

        heuristic = {
            "plan_steps": [
                "Identify the core research objective.",
                "Break it into search facets for better coverage.",
                "Emit a compact set of high-yield queries.",
            ],
            "queries": self._normalize_queries(seed_queries or [objective], limit=max_queries),
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
                    "Produce a short list of plan_steps, a deduplicated query set, and a concise reasoning_summary."
                ),
                input_data=payload,
                response_model=None,
            )
            if isinstance(result, str):
                result = json.loads(result)
            if not isinstance(result, dict):
                return heuristic
            queries = self._normalize_queries([*(result.get("queries") or []), *seed_queries], limit=max_queries)
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

    def _reasoning_style(self, state: Any) -> str | None:
        request = getattr(state, "request", None)
        if request is not None:
            style = getattr(request, "reasoning_style", None)
            if style:
                return style
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

    def _normalize_queries(self, queries: list[str], *, limit: int) -> list[str]:
        normalized_queries: list[str] = []
        for query in queries:
            normalized = " ".join(str(query).strip().split())
            if normalized and normalized not in normalized_queries:
                normalized_queries.append(normalized)
        return normalized_queries[:limit]


class ResearchCollectionAnswerCapability:
    def __init__(self, *, llm_adapter: Any | None = None, paper_analysis_skill: Any | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.paper_analysis_skill = paper_analysis_skill

    async def answer_collection_question(
        self,
        *,
        graph_runtime: Any,
        state: Any,
        primary_agents: list[str],
    ) -> QAResponse:
        all_hits = merge_retrieval_hits(state.retrieval_hits, state.summary_hits, state.manifest_hits)
        evidence_bundle = build_evidence_bundle(all_hits[: max(state.top_k * 2, 12)])
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(
                query=state.question,
                document_ids=state.document_ids,
                mode="hybrid",
                top_k=state.top_k,
                filters={
                    "research_task_id": state.task.task_id,
                    "research_topic": state.task.topic,
                    "qa_mode": "research_collection",
                },
            ),
            hits=all_hits[: max(state.top_k * 2, 12)],
            evidence_bundle=evidence_bundle,
            metadata={
                "autonomy_mode": "lead_agent_loop",
                "agent_architecture": "main_agents_only",
                "primary_agents": primary_agents,
                "collection_hit_mix": {
                    "retrieval_hits": len(state.retrieval_hits),
                    "graph_summary_hits": len(state.summary_hits),
                    "manifest_hits": len(state.manifest_hits),
                },
            },
        )
        original_question = str(getattr(state, "original_question", state.question))
        resolved_question = state.question
        answer_metadata = {
            "research_task_id": state.task.task_id,
            "research_topic": state.task.topic,
            "qa_mode": "research_collection",
            "autonomy_mode": "lead_agent_loop",
            "agent_architecture": "supervisor_to_research_qa_agent",
            "primary_agents": primary_agents,
            "answer_capability": self.__class__.__name__,
            "original_question": original_question,
            "resolved_question": resolved_question,
        }
        execution_context = getattr(state, "execution_context", None)
        session_context = getattr(execution_context, "session_context", None) or {}
        task_context = {
            **(getattr(execution_context, "task_context", None) or {}),
            "task_id": state.task.task_id,
            "research_topic": state.task.topic,
            "paper_count": len(state.papers),
            "report_id": state.report.report_id if state.report else None,
            "selected_paper_ids": list(getattr(state.request, "paper_ids", []) or []),
            "selected_paper_titles": [paper.title for paper in state.papers[:8]],
            "qa_scope_mode": str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported"),
            "question_scope_document_count": len(state.document_ids),
        }
        _request_meta = getattr(state.request, "metadata", {}) or {}
        supervisor_instruction = str(_request_meta.get("supervisor_instruction") or "").strip() or None
        preference_context = {
            **(getattr(execution_context, "preference_context", None) or {}),
            "reasoning_style": state.request.reasoning_style or "cot",
            "skill_name": state.request.skill_name,
            "min_length": getattr(state.request, "min_length", 400),
            "return_citations": getattr(state.request, "return_citations", True),
            "answer_language": self._preferred_answer_language(state.question),
            "follow_user_language": True,
            "preserve_paper_title_language": True,
        }
        if supervisor_instruction:
            preference_context["supervisor_instruction"] = supervisor_instruction
        memory_hints = getattr(execution_context, "memory_hints", None) or {}
        selected_paper_analysis = await self._analyze_selected_papers(state)
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        qa = await knowledge_access.answer_with_evidence(
            question=state.question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=answer_metadata,
            session_context=session_context,
            task_context=task_context,
            preference_context=preference_context,
            memory_hints=memory_hints,
            available_tool_names=["answer_with_evidence"],
        )
        citations = self._build_citations(state=state, retrieval_result=retrieval_result)
        extended_analysis = await self._build_extended_analysis_async(
            state=state,
            citations=citations,
            evidence_bundle=evidence_bundle,
        )
        scope_mode = str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported")
        use_selected_paper_analysis_answer = (
            selected_paper_analysis is not None
            and (scope_mode == "metadata_only" or not (qa.answer or "").strip() or len(evidence_bundle.evidences) == 0)
        )
        if selected_paper_analysis is not None and selected_paper_analysis.key_points:
            analysis_lines = ["", "## 选中论文分析要点"]
            analysis_lines.extend(f"- {point}" for point in selected_paper_analysis.key_points[:4])
            extended_analysis = f"{extended_analysis.rstrip()}\n" + "\n".join(analysis_lines)
        structured_answer = self._compose_structured_answer(
            state=state,
            raw_answer=(
                selected_paper_analysis.answer
                if use_selected_paper_analysis_answer and selected_paper_analysis is not None
                else qa.answer
            ),
            citations=citations,
            extended_analysis=extended_analysis,
            evidence_count=len(evidence_bundle.evidences),
        )
        related_sections = self._related_sections(citations)
        return qa.model_copy(
            update={
                "question": original_question,
                "answer": structured_answer,
                "evidence_bundle": evidence_bundle,
                "retrieval_result": retrieval_result,
                "metadata": {
                    **qa.metadata,
                    "answer_format": "direct_research_collection",
                    "original_question": original_question,
                    "resolved_question": resolved_question,
                    "citations": citations,
                    "related_sections": related_sections,
                    "extended_analysis": extended_analysis,
                    "paper_scope": {
                        "paper_ids": [paper.paper_id for paper in state.papers],
                        "paper_titles": [paper.title for paper in state.papers],
                        "document_ids": list(state.document_ids),
                        "scope_mode": str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported"),
                    },
                    "selection_warnings": list(getattr(state.request, "metadata", {}).get("selection_warnings") or []),
                    "selection_summary": getattr(state.request, "metadata", {}).get("selection_summary"),
                    "scope_statistics": {
                        "paper_count": len(state.papers),
                        "document_count": len(state.document_ids),
                        "evidence_count": len(evidence_bundle.evidences),
                    },
                    **(
                        {
                            "selected_paper_analysis": selected_paper_analysis.model_dump(mode="json"),
                            "recommended_paper_ids": list(selected_paper_analysis.recommended_paper_ids),
                        }
                        if selected_paper_analysis is not None
                        else {}
                    ),
                },
            }
        )

    async def _analyze_selected_papers(self, state: Any):
        if self.paper_analysis_skill is None:
            return None
        paper_ids = list(getattr(state.request, "paper_ids", []) or [])
        if not paper_ids or not getattr(state, "papers", None):
            return None
        return await self.paper_analysis_skill.analyze_async(
            question=str(getattr(state, "original_question", state.question)),
            papers=list(state.papers),
            task_topic=getattr(state.task, "topic", ""),
            report_highlights=list(getattr(getattr(state, "report", None), "highlights", [])[:4]),
        )

    def _preferred_answer_language(self, question: str) -> str:
        return "zh-CN" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en-US"

    def _build_citations(self, *, state: Any, retrieval_result: HybridRetrievalResult) -> list[dict[str, Any]]:
        paper_labels = {paper.paper_id: f"P{index}" for index, paper in enumerate(state.papers, start=1)}
        paper_by_id = {paper.paper_id: paper for paper in state.papers}
        document_to_paper_id = {
            str(paper.metadata.get("document_id")): paper.paper_id
            for paper in state.papers
            if str(paper.metadata.get("document_id") or "").strip()
        }
        fallback_doc_labels: dict[str, str] = {}
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None, str | None]] = set()
        for hit in retrieval_result.hits[: max(state.top_k, 8)]:
            paper_id = self._resolve_paper_id(
                hit=hit,
                document_to_paper_id=document_to_paper_id,
                paper_by_id=paper_by_id,
            )
            document_id = hit.document_id
            if paper_id:
                label = paper_labels.setdefault(paper_id, f"P{len(paper_labels) + 1}")
                paper = paper_by_id.get(paper_id)
                title = paper.title if paper is not None else str(hit.metadata.get("title") or paper_id)
            else:
                fallback_key = document_id or hit.source_id or hit.id
                if fallback_key not in fallback_doc_labels:
                    fallback_doc_labels[fallback_key] = f"D{len(fallback_doc_labels) + 1}"
                label = fallback_doc_labels[fallback_key]
                title = str(hit.metadata.get("title") or document_id or hit.source_id)
            page_number = hit.metadata.get("page_number") if isinstance(hit.metadata.get("page_number"), int) else None
            section = next(
                (
                    str(hit.metadata.get(key)).strip()
                    for key in ("section", "heading", "caption", "title")
                    if str(hit.metadata.get(key) or "").strip()
                ),
                None,
            )
            marker = (paper_id, document_id, hit.source_id)
            if marker in seen:
                continue
            seen.add(marker)
            citations.append(
                {
                    "label": label,
                    "paper_id": paper_id,
                    "title": title,
                    "document_id": document_id,
                    "page_number": page_number,
                    "section": section,
                    "source_type": hit.source_type,
                    "source_id": hit.source_id,
                    "snippet": (hit.content or "")[:280],
                    "score": hit.merged_score if hit.merged_score is not None else hit.vector_score or hit.graph_score,
                }
            )
        return citations[:8]

    def _compose_structured_answer(
        self,
        *,
        state: Any,
        raw_answer: str,
        citations: list[dict[str, Any]],
        extended_analysis: str,
        evidence_count: int,
    ) -> str:
        del citations, extended_analysis, evidence_count
        language = self._preferred_answer_language(state.question)
        direct_answer = (raw_answer or "").strip()
        if not direct_answer:
            return "证据不足" if language == "zh-CN" else "Insufficient evidence."
        if language == "zh-CN" and direct_answer.strip().lower() == "insufficient evidence.":
            return "证据不足"
        if language != "zh-CN" and direct_answer.strip() == "证据不足":
            return "Insufficient evidence."
        return direct_answer

    async def _build_extended_analysis_async(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_bundle: EvidenceBundle,
    ) -> str:
        if self.llm_adapter is None:
            return self._build_extended_analysis(citations=citations, evidence_bundle=evidence_bundle)
        try:
            from pydantic import BaseModel, Field

            class _AnalysisResponse(BaseModel):
                analysis: str = Field(description="中文扩展分析（2-4句话）")

            citation_counter = Counter(citation["label"] for citation in citations if citation.get("label"))
            result = await self.llm_adapter.generate_structured(
                prompt=(
                    "你是一个学术证据分析助手。请根据以下证据信息生成一段简洁的中文扩展分析。\n\n"
                    "证据条数：{evidence_count}\n"
                    "引用条目数：{citation_count}\n"
                    "主要支持来源：{top_sources}\n"
                    "问题：{question}\n\n"
                    "要求：分析证据覆盖度、可靠性和局限性（2-4句话）"
                ),
                input_data={
                    "evidence_count": str(len(evidence_bundle.evidences)),
                    "citation_count": str(len(citations)),
                    "top_sources": ", ".join(
                        f"[{label}]×{count}" for label, count in citation_counter.most_common(3)
                    ),
                    "question": state.question,
                },
                response_model=_AnalysisResponse,
            )
            return result.analysis
        except Exception:
            return self._build_extended_analysis(citations=citations, evidence_bundle=evidence_bundle)

    def _build_extended_analysis(self, *, citations: list[dict[str, Any]], evidence_bundle: EvidenceBundle) -> str:
        citation_counter = Counter(citation["label"] for citation in citations if citation.get("label"))
        most_supported = ", ".join(f"[{label}]×{count}" for label, count in citation_counter.most_common(3)) or "当前没有形成稳定的论文支持分布"
        return (
            f"从证据覆盖度看，本轮回答主要依赖 {len(evidence_bundle.evidences)} 条证据与 "
            f"{len(citations)} 个可回溯引用条目，最主要的支持来源为 {most_supported}。"
        )

    def _related_sections(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        related: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None]] = set()
        for citation in citations:
            marker = (citation.get("paper_id"), citation.get("source_id"))
            if marker in seen:
                continue
            seen.add(marker)
            related.append(
                {
                    "paper_id": citation.get("paper_id"),
                    "section_id": citation.get("source_id"),
                    "heading": citation.get("section"),
                    "relevance_score": citation.get("score"),
                }
            )
        return related

    def _resolve_paper_id(
        self,
        *,
        hit: Any,
        document_to_paper_id: dict[str, str],
        paper_by_id: dict[str, Any],
    ) -> str | None:
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        if isinstance(metadata.get("paper_id"), str) and metadata.get("paper_id") in paper_by_id:
            return metadata["paper_id"]
        if isinstance(metadata.get("research_paper_id"), str) and metadata.get("research_paper_id") in paper_by_id:
            return metadata["research_paper_id"]
        if hit.source_id in paper_by_id:
            return hit.source_id
        if hit.document_id and hit.document_id in document_to_paper_id:
            return document_to_paper_id[hit.document_id]
        return None


class ResearchCollectionQACapability:
    """Collection-level research QA capability below the QA specialist."""

    def __init__(
        self,
        *,
        llm_adapter: Any | None = None,
        paper_analysis_skill: Any | None = None,
    ) -> None:
        self.knowledge = ResearchCollectionKnowledgeCapability(llm_adapter=llm_adapter)
        self.answer = ResearchCollectionAnswerCapability(
            llm_adapter=llm_adapter,
            paper_analysis_skill=paper_analysis_skill,
        )

    async def run_collection_qa(
        self,
        *,
        graph_runtime: Any,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: Any,
        resolved_question: str,
        original_question: str,
        primary_agents: list[str],
    ) -> QAResponse:
        request_with_resolution = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "resolved_question": resolved_question,
                }
            }
        )
        runtime_state = SimpleNamespace(
            task=task,
            request=request_with_resolution,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            queries=[],
            completed_queries=set(),
            refinement_used=False,
            summary_checked=False,
            manifest_built=False,
            retrieval_hits=[],
            summary_hits=[],
            manifest_hits=[],
            evidence_bundle=EvidenceBundle(),
            retrieval_result=None,
            qa=None,
            warnings=[],
            trace=[],
            question=resolved_question,
            original_question=original_question,
            top_k=request.top_k,
        )
        runtime_state.queries = await self.knowledge.plan_collection_queries(runtime_state)
        for query in list(runtime_state.queries):
            try:
                hits = await self.knowledge.retrieve_collection_evidence(
                    graph_runtime=graph_runtime,
                    state=runtime_state,
                    query=query,
                )
                runtime_state.retrieval_hits = merge_retrieval_hits(runtime_state.retrieval_hits, hits)
            except Exception as exc:
                runtime_state.warnings.append(f"collection_retrieval:{query} failed: {exc}")
            runtime_state.completed_queries.add(query)
        runtime_state.summary_checked = True
        try:
            summary_hits = await self.knowledge.retrieve_graph_summary(
                graph_runtime=graph_runtime,
                state=runtime_state,
            )
            runtime_state.summary_hits = merge_retrieval_hits(runtime_state.summary_hits, summary_hits)
        except Exception as exc:
            runtime_state.warnings.append(f"graph_summary:{runtime_state.question} failed: {exc}")
        runtime_state.manifest_hits = self.knowledge.build_collection_manifest(runtime_state)
        runtime_state.manifest_built = True
        qa = await self.answer.answer_collection_question(
            graph_runtime=graph_runtime,
            state=runtime_state,
            primary_agents=primary_agents,
        )
        return qa.model_copy(
            update={
                "metadata": {
                    **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": "supervisor_to_research_qa_agent",
                    "primary_agents": list(primary_agents),
                    "primary_tools": [
                        "ResearchCollectionKnowledgeCapability.plan_collection_queries",
                        "ResearchCollectionKnowledgeCapability.retrieve_collection_evidence",
                        "ResearchCollectionKnowledgeCapability.retrieve_graph_summary",
                        "ResearchCollectionAnswerCapability.answer_collection_question",
                    ],
                    "supervisor_execution_mode": "single_supervisor_action",
                    "supervisor_agent_architecture": "supervisor_direct_execution",
                    "qa_execution_path": "research_collection_qa_capability",
                    "qa_warnings": list(runtime_state.warnings),
                    "planned_queries": list(runtime_state.queries),
                    "completed_queries": list(runtime_state.completed_queries),
                    "memory_enabled": execution_context.memory_enabled,
                    "session_id": execution_context.session_id,
                }
            }
        )
