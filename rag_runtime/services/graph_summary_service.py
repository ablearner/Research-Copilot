import hashlib
import logging
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.evidence import Evidence
from domain.schemas.graph_rag import GraphCommunity, GraphCommunityBuildResult, GraphCommunitySummary, GraphSummaryBuildResult

logger = logging.getLogger(__name__)

_COMMUNITY_SUMMARY_PROMPT = (
    "你是一个知识图谱社区摘要生成助手。请根据以下图社区信息生成一段简洁的中文摘要。\n\n"
    "主题：{topic}\n"
    "节点数：{node_count}\n"
    "边数：{edge_count}\n"
    "关键三元组：\n{triples}\n\n"
    "要求：\n"
    "- 用中文总结该社区的核心内容（1-3句话）\n"
    "- 保留关键实体名称（可保留英文）\n"
    "- 指出实体间的主要关系"
)


class _CommunitySummaryResponse(BaseModel):
    summary: str = Field(description="社区摘要（中文）")


class GraphSummaryService:
    """Generate graph community summaries using LLM with template fallback.
    
    When llm_adapter is provided, uses LLM for intelligent summarization.
    Falls back to triple-concatenation when LLM is unavailable.
    """

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    def summarize_communities(
        self,
        community_result: GraphCommunityBuildResult,
        max_facts: int = 8,
    ) -> GraphSummaryBuildResult:
        summaries = [
            self.summarize_community(community, max_facts=max_facts)
            for community in community_result.communities
        ]
        logger.info(
            "Built graph community summaries",
            extra={"document_id": community_result.document_id, "summary_count": len(summaries)},
        )
        return GraphSummaryBuildResult(
            document_id=community_result.document_id,
            summaries=summaries,
            metadata={"community_count": len(community_result.communities), "max_facts": max_facts},
        )

    async def summarize_communities_async(
        self,
        community_result: GraphCommunityBuildResult,
        max_facts: int = 8,
    ) -> GraphSummaryBuildResult:
        """Async version — uses LLM if available."""
        import asyncio
        tasks = [
            self.summarize_community_async(community, max_facts=max_facts)
            for community in community_result.communities
        ]
        summaries = await asyncio.gather(*tasks)
        logger.info(
            "Built graph community summaries (async)",
            extra={"document_id": community_result.document_id, "summary_count": len(summaries)},
        )
        return GraphSummaryBuildResult(
            document_id=community_result.document_id,
            summaries=list(summaries),
            metadata={"community_count": len(community_result.communities), "max_facts": max_facts},
        )

    def summarize_community(self, community: GraphCommunity, max_facts: int = 8) -> GraphCommunitySummary:
        """Synchronous summarize — uses heuristic template."""
        summary = self._heuristic_summary(community, max_facts=max_facts)
        return GraphCommunitySummary(
            id=self._summary_id(community.id),
            community_id=community.id,
            document_id=community.document_id,
            topic=community.topic,
            summary=summary,
            node_ids=community.node_ids,
            edge_ids=community.edge_ids,
            source_references=self._dedupe_evidence(community.source_references),
            confidence=community.confidence,
            metadata={"community_size": len(community.node_ids), "edge_count": len(community.edge_ids)},
        )

    async def summarize_community_async(self, community: GraphCommunity, max_facts: int = 8) -> GraphCommunitySummary:
        """Async summarize — uses LLM if available."""
        if self.llm_adapter is not None:
            try:
                summary = await self._llm_summary(community, max_facts=max_facts)
            except Exception:  # noqa: BLE001
                summary = self._heuristic_summary(community, max_facts=max_facts)
        else:
            summary = self._heuristic_summary(community, max_facts=max_facts)
        return GraphCommunitySummary(
            id=self._summary_id(community.id),
            community_id=community.id,
            document_id=community.document_id,
            topic=community.topic,
            summary=summary,
            node_ids=community.node_ids,
            edge_ids=community.edge_ids,
            source_references=self._dedupe_evidence(community.source_references),
            confidence=community.confidence,
            metadata={"community_size": len(community.node_ids), "edge_count": len(community.edge_ids)},
        )

    async def _llm_summary(self, community: GraphCommunity, max_facts: int = 8) -> str:
        triples_text = "\n".join(
            f"  {self._node_name(t.subject)} --[{t.predicate.type}]--> {self._node_name(t.object)}"
            for t in community.triples[:max_facts]
        )
        result = await self.llm_adapter.generate_structured(
            prompt=_COMMUNITY_SUMMARY_PROMPT,
            input_data={
                "topic": community.topic,
                "node_count": str(len(community.node_ids)),
                "edge_count": str(len(community.edge_ids)),
                "triples": triples_text or "(无三元组)",
            },
            response_model=_CommunitySummaryResponse,
        )
        return result.summary

    def _heuristic_summary(self, community: GraphCommunity, max_facts: int = 8) -> str:
        facts: list[str] = []
        for triple in community.triples[:max_facts]:
            subject = self._node_name(triple.subject)
            obj = self._node_name(triple.object)
            facts.append(f"{subject} {triple.predicate.type} {obj}")
        if not facts:
            facts = [self._node_name(node) for node in community.nodes[:max_facts]]
        return f"Topic {community.topic}: " + "; ".join(fact for fact in facts if fact)

    def _node_name(self, node) -> str:
        return str(node.properties.get("name") or node.properties.get("title") or node.id)

    def _summary_id(self, community_id: str) -> str:
        digest = hashlib.sha256(community_id.encode("utf-8")).hexdigest()[:16]
        return f"gsummary_{digest}"

    def _dedupe_evidence(self, evidences: list[Evidence]) -> list[Evidence]:
        deduped = {}
        for evidence in evidences:
            deduped[evidence.id] = evidence
        return list(deduped.values())
