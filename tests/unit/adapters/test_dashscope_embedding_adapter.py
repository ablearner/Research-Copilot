import httpx
import pytest

from adapters.embedding.dashscope_adapter import DashScopeEmbeddingAdapter


class FakeAsyncClient:
    def __init__(self, responses: list[Exception | httpx.Response]) -> None:
        self._responses = list(responses)
        self.is_closed = False

    async def post(self, url: str, json: dict) -> httpx.Response:
        del url, json
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.is_closed = True


@pytest.mark.asyncio
async def test_dashscope_embedding_adapter_retries_with_fresh_client_on_connect_error(monkeypatch) -> None:
    adapter = DashScopeEmbeddingAdapter(api_key="test-key", max_retries=1, retry_delay_seconds=0)
    request = httpx.Request("POST", "https://example.com/v1/embeddings")
    first_client = FakeAsyncClient([httpx.ConnectError("connection dropped", request=request)])
    second_client = FakeAsyncClient(
        [httpx.Response(200, request=request, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})]
    )
    created_clients: list[FakeAsyncClient] = []

    async def fake_get_client():
        if adapter._client is None:
            adapter._client = first_client if not created_clients else second_client
            created_clients.append(adapter._client)
        return adapter._client

    monkeypatch.setattr(adapter, "_get_client", fake_get_client)

    vectors = await adapter.embed_texts(["hello"])

    assert [vector.values for vector in vectors] == [[0.1, 0.2, 0.3]]
    assert created_clients == [first_client, second_client]
    assert first_client.is_closed is True
    assert adapter._client is second_client

    await adapter.close()

    assert second_client.is_closed is True
