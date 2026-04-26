import json

import httpx
import pytest

from services.research.zotero_local_mcp import ZoteroLocalGateway


class AsyncClientStub:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.requests: list[tuple[str, str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        return self._responses.pop(0)

    async def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return self._responses.pop(0)


class AsyncClientFactoryStub:
    def __init__(self, response_groups: list[list[httpx.Response]]) -> None:
        self.response_groups = response_groups
        self.clients: list[AsyncClientStub] = []

    def __call__(self, **kwargs):
        del kwargs
        group = self.response_groups.pop(0)
        client = AsyncClientStub(group)
        self.clients.append(client)
        return client


@pytest.fixture(autouse=True)
def bypass_base_url_resolution(monkeypatch):
    async def _noop_resolve(self):
        return self.base_url

    monkeypatch.setattr(ZoteroLocalGateway, "_resolve_base_url", _noop_resolve)


@pytest.mark.asyncio
async def test_zotero_local_gateway_search_items_with_attachments(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/items/top"),
            json=[
                {
                    "data": {
                        "key": "ITEM1",
                        "itemType": "journalArticle",
                        "title": "Agentic RAG",
                        "creators": [{"firstName": "Ada", "lastName": "Lovelace"}],
                        "abstractNote": "A paper about grounded generation.",
                        "date": "2025-12-01",
                        "url": "https://example.com/paper",
                        "DOI": "10.1000/example",
                        "collections": ["C1"],
                    }
                }
            ],
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/items/ITEM1/children"),
            json=[
                {
                    "data": {
                        "key": "ATT1",
                        "itemType": "attachment",
                        "title": "Agentic RAG PDF",
                        "contentType": "application/pdf",
                        "url": "https://example.com/paper.pdf",
                    },
                    "links": {"enclosure": {"href": "file:///tmp/agentic-rag.pdf"}},
                }
            ],
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    output = await gateway.search_items(query="agentic", limit=5, include_attachments=True)

    assert len(output.items) == 1
    assert output.items[0].title == "Agentic RAG"
    assert output.items[0].creators == ["Ada Lovelace"]
    assert output.items[0].attachments[0].local_path == "file:///tmp/agentic-rag.pdf"
    assert client.requests[0][2]["params"] == {"q": "agentic", "limit": "5", "qmode": "titleCreatorYear"}


@pytest.mark.asyncio
async def test_zotero_local_gateway_search_items_for_selected_collection(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/connector/getSelectedCollection"),
            json={"editable": True, "key": "COLL1", "name": "Research-Copilot"},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/collections/COLL1/items/top"),
            json=[
                {
                    "data": {
                        "key": "ITEM1",
                        "itemType": "journalArticle",
                        "title": "Scoped Search",
                    }
                }
            ],
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    output = await gateway.search_items(query="scoped", current_collection_only=True, include_attachments=False)

    assert len(output.items) == 1
    assert output.items[0].title == "Scoped Search"
    assert client.requests[1][1] == "http://127.0.0.1:23119/api/users/0/collections/COLL1/items/top"


@pytest.mark.asyncio
async def test_zotero_local_gateway_search_items_for_named_collection(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/collections"),
            json=[
                {"data": {"key": "COLL1", "name": "Research-Copilot"}},
                {"data": {"key": "COLL2", "name": "Archive"}},
            ],
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/collections/COLL1/items/top"),
            json=[],
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    await gateway.search_items(query="copilot", collection_name="research-copilot", include_attachments=False)

    assert client.requests[1][1] == "http://127.0.0.1:23119/api/users/0/collections/COLL1/items/top"


@pytest.mark.asyncio
async def test_zotero_local_gateway_search_items_raises_for_missing_selected_collection(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/connector/getSelectedCollection"),
            json={"editable": True, "name": "My Library"},
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    with pytest.raises(ValueError, match="No Zotero collection is currently selected"):
        await gateway.search_items(query="scoped", current_collection_only=True, include_attachments=False)


@pytest.mark.asyncio
async def test_zotero_local_gateway_import_paper_downloads_pdf(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/connector/getSelectedCollection"),
            json={"editable": True, "libraryID": 1, "libraryName": "My Library", "name": "Reading List"},
        ),
        httpx.Response(
            201,
            request=httpx.Request("POST", "http://127.0.0.1:23119/connector/saveItems"),
            json={"successful": {"0": {"key": "ITEM1"}}},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://example.com/paper.pdf"),
            content=b"%PDF-1.4 stub",
            headers={"content-type": "application/pdf"},
        ),
        httpx.Response(
            201,
            request=httpx.Request("POST", "http://127.0.0.1:23119/connector/saveAttachment"),
            json={"ok": True},
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    output = await gateway.import_paper(
        title="Agentic RAG",
        authors=["Ada Lovelace"],
        year=2025,
        url="https://example.com/paper",
        pdf_url="https://example.com/paper.pdf",
        collection_name="Reading List",
    )

    assert output.status == "imported"
    assert output.imported_item_key == "ITEM1"
    assert output.attachment_title == "Agentic RAG PDF"
    assert output.selected_collection is not None
    assert output.selected_collection.collection_name == "Reading List"

    save_items_payload = client.requests[1][2]["json"]
    assert save_items_payload["items"][0]["title"] == "Agentic RAG"
    assert save_items_payload["items"][0]["creators"][0]["lastName"] == "Lovelace"

    attachment_headers = client.requests[3][2]["headers"]
    metadata = json.loads(attachment_headers["X-Metadata"])
    assert metadata["title"] == "Agentic RAG PDF"
    assert metadata["url"] == "https://example.com/paper.pdf"


@pytest.mark.asyncio
async def test_zotero_local_gateway_attach_pdf_to_existing_item(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://example.com/paper.pdf"),
            content=b"%PDF-1.4 stub",
            headers={"content-type": "application/pdf"},
        ),
        httpx.Response(
            201,
            request=httpx.Request("POST", "http://127.0.0.1:23119/connector/saveAttachment"),
            json={"ok": True},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "http://127.0.0.1:23119/api/users/0/items/ITEM1/children"),
            json=[
                {
                    "data": {
                        "key": "ATT1",
                        "itemType": "attachment",
                        "title": "Agentic RAG PDF",
                        "contentType": "application/pdf",
                        "url": "https://example.com/paper",
                    }
                }
            ],
        ),
    ]
    client = AsyncClientStub(responses)
    monkeypatch.setattr(gateway, "_build_async_client", lambda **kwargs: client)

    output = await gateway.attach_pdf_to_item(
        item_key="ITEM1",
        pdf_url="https://example.com/paper.pdf",
        title="Agentic RAG PDF",
        source_url="https://example.com/paper",
    )

    assert output.status == "attached"
    assert output.item_key == "ITEM1"
    assert output.attachment_count == 1
    metadata = json.loads(client.requests[1][2]["headers"]["X-Metadata"])
    assert metadata["parentItemKey"] == "ITEM1"


@pytest.mark.asyncio
async def test_zotero_local_gateway_falls_back_to_windows_host_from_wsl(monkeypatch) -> None:
    gateway = ZoteroLocalGateway(base_url="http://127.0.0.1:23119", user_id="0")
    monkeypatch.setattr(gateway, "_is_wsl", lambda: True)
    monkeypatch.setattr(gateway, "_windows_host_gateway", lambda: "172.23.160.1")

    factory = AsyncClientFactoryStub(
        [
            [
                httpx.Response(
                    200,
                    request=httpx.Request("GET", "http://172.23.160.1:23119/connector/ping"),
                    text="ok",
                )
            ],
            [
                httpx.Response(
                    200,
                    request=httpx.Request("GET", "http://172.23.160.1:23119/connector/ping"),
                    text="ok",
                )
            ],
        ]
    )

    monkeypatch.setattr(gateway, "_build_async_client", factory)

    async def fake_probe() -> str:
        gateway.base_url = "http://172.23.160.1:23119"
        return gateway.base_url

    monkeypatch.setattr(gateway, "_resolve_base_url", fake_probe)

    output = await gateway.ping()

    assert output.status == "ok"
    assert output.base_url == "http://172.23.160.1:23119"
    assert factory.clients[-1].requests[0][1] == "http://172.23.160.1:23119/connector/ping"
