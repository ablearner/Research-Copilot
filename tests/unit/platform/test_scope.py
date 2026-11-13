import pytest
from domain.schemas.retrieval import RetrievalQuery, HybridRetrievalResult
from services.platform.agent_bridge import ScopedRetriever


@pytest.mark.asyncio
async def test_retrieval_rechecks_scope_after_provider_returns():
    deleted = False

    class Provider:
        async def retrieve(self, query):
            nonlocal deleted
            deleted = True
            return HybridRetrievalResult(query=query)

    async def validate():
        if deleted:
            raise PermissionError("Document deleted during retrieval")

    retriever = ScopedRetriever(Provider(), ["document"])
    retriever.validate = validate
    with pytest.raises(PermissionError):
        await retriever.retrieve(RetrievalQuery(query="evidence", document_ids=["document"]))


def test_attempts_do_not_share_mutable_index_or_graph(tmp_path):
    from core.config import Settings
    from services.platform.agent_bridge import scoped_settings

    settings = Settings(_env_file=None)
    first = scoped_settings(settings, tmp_path / "task" / "1", "workspace")
    second = scoped_settings(settings, tmp_path / "task" / "2", "workspace")
    assert first.milvus_collection_name != second.milvus_collection_name
    assert first.graph_store_provider == "memory"


@pytest.mark.asyncio
async def test_provider_cannot_return_other_tenant_evidence():
    from domain.schemas.retrieval import RetrievalHit

    class Provider:
        async def retrieve(self, query):
            return HybridRetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        id="other",
                        source_type="text_block",
                        source_id="other",
                        document_id="foreign-document",
                        content="other user's unique secret",
                    )
                ],
            )

    with pytest.raises(PermissionError):
        await ScopedRetriever(Provider(), ["my-document"]).retrieve(RetrievalQuery(query="secret"))


@pytest.mark.asyncio
async def test_authorized_hit_cannot_hide_foreign_graph_source():
    from domain.schemas.graph import GraphNode
    from domain.schemas.evidence import Evidence
    from domain.schemas.retrieval import RetrievalHit

    class Provider:
        async def retrieve(self, query):
            return HybridRetrievalResult(
                query=query,
                hits=[
                    RetrievalHit(
                        id="hit",
                        source_type="graph_node",
                        source_id="node",
                        document_id="mine",
                        graph_nodes=[
                            GraphNode(
                                id="node",
                                label="foreign secret",
                                source_reference=Evidence(
                                    id="foreign-source",
                                    document_id="foreign",
                                    source_type="graph_node",
                                    source_id="node",
                                ),
                            )
                        ],
                    )
                ],
            )

    with pytest.raises(PermissionError):
        await ScopedRetriever(Provider(), ["mine"]).retrieve(RetrievalQuery(query="secret"))


def test_business_bridge_can_construct_attempt_scoped_graph(tmp_path):
    from core.config import Settings
    from services.platform.agent_bridge import AgentBridge
    from adapters.local_runtime import InMemoryGraphStore

    settings = Settings(
        _env_file=None,
        runtime_backend="business",
        llm_provider="openai",
        llm_model="text-model",
        openai_api_key="test-key",
        embedding_provider="openai",
        vector_store_provider="milvus",
        graph_text_timeout_seconds=60,
        reranker_allow_download=False,
        reranker_unavailable_policy="heuristic",
    )
    bridge = AgentBridge(settings, tmp_path, "workspace", [])
    assert isinstance(bridge.runtime.graph_index_service.graph_store, InMemoryGraphStore)
    assert bridge.runtime.graph_extraction_tools.text_graph_timeout_seconds == 60


@pytest.mark.asyncio
async def test_bridge_materializes_only_authorized_document_in_attempt(tmp_path):
    from types import SimpleNamespace
    from services.platform.agent_bridge import AgentBridge

    bridge = AgentBridge.__new__(AgentBridge)
    bridge.allowed = ["allowed"]
    bridge.settings = SimpleNamespace(upload_dir=str(tmp_path / "uploads"))
    bridge.pdf_paths = {}
    with pytest.raises(PermissionError):
        await bridge.materialize_document("foreign", b"%PDF-test")
    path = await bridge.materialize_document("allowed", b"%PDF-test")
    assert path.read_bytes() == b"%PDF-test"
    assert path.parent == tmp_path / "uploads"
    assert bridge.pdf_paths["allowed"] == str(path)
