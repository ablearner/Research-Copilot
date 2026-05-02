import httpx
import pytest

from tools.research.paper_search import format_search_warning
from tools.research.semantic_scholar_search_tool import SemanticScholarSearchTool


class SequencedAsyncClient:
    def __init__(self, responses: list[httpx.Response], calls: list[tuple[str, dict | None]]) -> None:
        self.responses = responses
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, params: dict | None = None):
        self.calls.append((url, params))
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", url, params=params)
        return response


@pytest.mark.asyncio
async def test_semantic_scholar_search_tool_retries_after_429(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []
    sleep_calls: list[float] = []
    responses = [
        httpx.Response(status_code=429, headers={"Retry-After": "1"}),
        httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "paperId": "abc123",
                        "title": "UAV Path Planning with Retrieval-Augmented Research Agents",
                        "abstract": "A study of literature agents for UAV path planning.",
                        "year": 2026,
                        "venue": "Semantic Scholar Corpus",
                        "publicationDate": "2026-04-10T00:00:00+00:00",
                        "authors": [{"name": "Alice"}],
                        "externalIds": {"DOI": "10.1000/test"},
                        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                        "url": "https://www.semanticscholar.org/paper/abc123",
                        "citationCount": 12,
                        "isOpenAccess": True,
                    }
                ]
            },
        ),
    ]

    monkeypatch.setattr(
        "tools.research.semantic_scholar_search_tool.httpx.AsyncClient",
        lambda **kwargs: SequencedAsyncClient(responses, calls),
    )

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("tools.research.semantic_scholar_search_tool.asyncio.sleep", fake_sleep)

    tool = SemanticScholarSearchTool(base_url="https://api.semanticscholar.org/graph/v1")
    papers = await tool.search(query="UAV path planning", max_results=5, days_back=365)

    assert len(calls) == 2
    assert sleep_calls == [1.0]
    assert len(papers) == 1
    assert papers[0].paper_id == "semantic_scholar:abc123"


def test_format_search_warning_hides_raw_semantic_scholar_429_details() -> None:
    response = httpx.Response(status_code=429)
    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search")
    exc = httpx.HTTPStatusError(
        "Client error '429' for url 'https://api.semanticscholar.org/graph/v1/paper/search'",
        request=request,
        response=response,
    )

    warning = format_search_warning(
        source="semantic_scholar",
        query="UAV path planning",
        exc=exc,
    )

    assert "Semantic Scholar 限流" in warning
    assert "HTTP 429" in warning
    assert "Client error" not in warning
    assert "developer.mozilla.org" not in warning
