from __future__ import annotations

from skills.base import (
    SkillMemoryPolicy,
    SkillOutputStyle,
    SkillPromptSet,
    SkillRetrievalPolicy,
    SkillSpec,
)


def build_core_research_skill_profiles() -> list[SkillSpec]:
    return [
        SkillSpec(
            name="literature_review",
            description="Produce a literature review grounded in collection-level evidence.",
            applicable_tasks=["ask_document", "function_call"],
            prompt_set=SkillPromptSet(
                answer_prompt_path="prompts/document/answer_question_with_hybrid_rag_research_report.txt",
                rewrite_prompt_path="prompts/research/rewrite_literature_query.txt",
            ),
            preferred_tools=["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
            retrieval_policy=SkillRetrievalPolicy(
                mode="hybrid",
                top_k=12,
                graph_query_mode="summary",
                enable_graph_summary=True,
            ),
            memory_policy=SkillMemoryPolicy(),
            output_style=SkillOutputStyle(language="zh-CN", detail_level="detailed", tone="scholarly"),
            metadata={
                "research_skill": True,
                "output_contract": "background, method clusters, evidence-backed conclusions, evidence gaps",
            },
        ),
        SkillSpec(
            name="paper_compare",
            description="Compare multiple papers along explicit dimensions with evidence references.",
            applicable_tasks=["ask_document", "function_call"],
            prompt_set=SkillPromptSet(
                answer_prompt_path="prompts/answer_agent/answer_with_evidence.md",
                rewrite_prompt_path="prompts/research/rewrite_literature_query.txt",
            ),
            preferred_tools=["hybrid_retrieve", "answer_with_evidence"],
            retrieval_policy=SkillRetrievalPolicy(mode="hybrid", top_k=10, graph_query_mode="auto"),
            output_style=SkillOutputStyle(language="zh-CN", detail_level="detailed", tone="comparative"),
            metadata={
                "research_skill": True,
                "comparison_ready": True,
                "output_contract": "comparison dimensions, per-paper evidence, tradeoffs, recommendation",
            },
        ),
        SkillSpec(
            name="chart_interpretation",
            description="Interpret charts and figures with document context and evidence-backed takeaways.",
            applicable_tasks=["understand_chart", "function_call", "ask_document"],
            prompt_set=SkillPromptSet(
                answer_prompt_path="prompts/answer_agent/answer_with_evidence.md",
            ),
            preferred_tools=["understand_chart", "hybrid_retrieve", "answer_with_evidence"],
            retrieval_policy=SkillRetrievalPolicy(mode="hybrid", top_k=8, graph_query_mode="auto"),
            output_style=SkillOutputStyle(language="zh-CN", detail_level="normal", tone="analytical"),
            metadata={
                "research_skill": True,
                "visual_reasoning": True,
                "output_contract": "chart reading, trend summary, caveats, evidence linkage",
            },
        ),
        SkillSpec(
            name="research_gap_discovery",
            description="Identify unresolved evidence gaps and next research actions from current collection state.",
            applicable_tasks=["ask_document", "function_call"],
            prompt_set=SkillPromptSet(
                answer_prompt_path="prompts/answer_agent/answer_with_evidence.md",
                rewrite_prompt_path="prompts/research/rewrite_literature_query.txt",
            ),
            preferred_tools=["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
            retrieval_policy=SkillRetrievalPolicy(
                mode="hybrid",
                top_k=10,
                graph_query_mode="summary",
                enable_graph_summary=True,
            ),
            output_style=SkillOutputStyle(language="zh-CN", detail_level="detailed", tone="diagnostic"),
            metadata={
                "research_skill": True,
                "output_contract": "known findings, missing evidence, recommended search directions",
            },
        ),
        SkillSpec(
            name="paper_deep_analysis",
            description="Perform evidence-aware deep analysis for selected papers or imported documents.",
            applicable_tasks=["ask_document", "function_call"],
            prompt_set=SkillPromptSet(
                answer_prompt_path="prompts/answer_agent/answer_with_evidence.md",
            ),
            preferred_tools=["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
            retrieval_policy=SkillRetrievalPolicy(
                mode="hybrid",
                top_k=12,
                graph_query_mode="entity",
                enable_graph_summary=True,
            ),
            output_style=SkillOutputStyle(language="zh-CN", detail_level="detailed", tone="factual"),
            metadata={
                "research_skill": True,
                "deep_analysis": True,
                "output_contract": "claims, evidence, limitations, actionable follow-up",
            },
        ),
    ]
