import logging
from typing import Any

from domain.schemas.research import PaperCandidate
from tools.research.external_tool_gateway import ResearchExternalToolGateway

logger = logging.getLogger(__name__)


class ZoteroSearchTool:
    def __init__(self, *, graph_runtime: Any | None = None) -> None:
        self.external_tool_gateway = ResearchExternalToolGateway(graph_runtime=graph_runtime)

    async def search(self, *, query: str, max_results: int, days_back: int) -> list:
        del days_back
        if not self.external_tool_gateway.is_configured():
            return []
        result = await self.external_tool_gateway.call_tool(
            tool_name="zotero_search_items",
            arguments={
                "query": query,
                "limit": max_results,
                "include_attachments": True,
            },
        )
        if result.status != "succeeded" or not isinstance(result.output, dict):
            logger.info(
                "Zotero search tool result | query=%s | status=%s | hits=0",
                query[:180],
                result.status,
            )
            return []
        items = result.output.get("items")
        if not isinstance(items, list):
            logger.info(
                "Zotero search tool result | query=%s | status=%s | invalid_items=1 | hits=0",
                query[:180],
                result.status,
            )
            return []
        papers = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_key = str(item.get("key") or "").strip()
            title = str(item.get("title") or "").strip()
            if not item_key or not title:
                continue
            attachments = item.get("attachments")
            pdf_url = None
            local_path = None
            if isinstance(attachments, list):
                for attachment in attachments:
                    if not isinstance(attachment, dict):
                        continue
                    content_type = str(attachment.get("content_type") or "").lower()
                    if "pdf" not in content_type and not str(attachment.get("title") or "").lower().endswith("pdf"):
                        continue
                    pdf_url = str(attachment.get("url") or "").strip() or None
                    local_path = str(attachment.get("local_path") or "").strip() or None
                    break
            papers.append(
                PaperCandidate(
                    paper_id=f"zotero:{item_key}",
                    title=title,
                    authors=list(item.get("creators") or []),
                    abstract=str(item.get("abstract") or ""),
                    year=int(item["year"]) if isinstance(item.get("year"), str) and str(item.get("year")).isdigit() else None,
                    source="zotero",
                    doi=str(item.get("doi") or "").strip() or None,
                    pdf_url=pdf_url,
                    url=str(item.get("url") or "").strip() or None,
                    metadata={
                        "zotero_item_key": item_key,
                        "zotero_collections": list(item.get("collections") or []),
                        "zotero_local_path": local_path,
                    },
                )
            )
        logger.info(
            "Zotero search tool result | query=%s | items=%s | papers=%s | titles=%s",
            query[:180],
            len(items),
            len(papers),
            " | ".join(paper.title for paper in papers[:5]),
        )
        return papers
