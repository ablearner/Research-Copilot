import pytest

from domain.schemas.research import PaperCandidate
from tools.research.paper_search import PaperSearchService


class RecordingToolStub:
    def __init__(self, *, source: str) -> None:
        self.source = source
        self.queries: list[str] = []

    async def search(self, *, query: str, max_results: int, days_back: int):
        self.queries.append(query)
        return [
            PaperCandidate(
                paper_id=f"{self.source}:{query}",
                title=f"{query} survey paper",
                abstract=f"This paper studies {query}.",
                source=self.source,  # type: ignore[arg-type]
                year=2026,
                published_at="2026-03-01T00:00:00+00:00",
            )
        ]


@pytest.mark.asyncio
async def test_paper_search_service_rewrites_chinese_topic_to_english_provider_queries() -> None:
    arxiv = RecordingToolStub(source="arxiv")
    openalex = RecordingToolStub(source="openalex")
    semantic_scholar = RecordingToolStub(source="semantic_scholar")
    service = PaperSearchService(
        arxiv_tool=arxiv,
        openalex_tool=openalex,
        semantic_scholar_tool=semantic_scholar,
    )

    await service.search(
        topic="最近 6 个月大模型方向有哪些值得关注的论文？",
        days_back=180,
        max_papers=6,
        sources=["arxiv", "openalex", "semantic_scholar"],
    )

    assert arxiv.queries == ["large language model", "foundation model"]
    assert openalex.queries == ["large language model", "foundation model", "LLM"]
    assert semantic_scholar.queries == ["large language model"]
    all_queries = [*arxiv.queries, *openalex.queries, *semantic_scholar.queries]
    assert all(not any("\u4e00" <= char <= "\u9fff" for char in query) for query in all_queries)


@pytest.mark.asyncio
async def test_paper_search_service_preserves_local_language_queries_for_zotero_source() -> None:
    arxiv = RecordingToolStub(source="arxiv")
    openalex = RecordingToolStub(source="openalex")
    zotero = RecordingToolStub(source="zotero")
    service = PaperSearchService(
        arxiv_tool=arxiv,
        openalex_tool=openalex,
        zotero_tool=zotero,
    )

    await service.search(
        topic="最近 6 个月大模型方向有哪些值得关注的论文？",
        days_back=180,
        max_papers=6,
        sources=["zotero", "arxiv"],
    )

    assert zotero.queries
    assert any(any("\u4e00" <= char <= "\u9fff" for char in query) for query in zotero.queries)
    assert arxiv.queries == ["large language model", "foundation model"]
