import asyncio

import pytest

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError
from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode, GraphTriple
from tools.graph_extraction_toolkit import GraphExtractionAgent


@pytest.mark.asyncio
async def test_graph_extraction_agent_extracts_and_merges(mock_llm, sample_text_block) -> None:
    agent = GraphExtractionAgent(mock_llm)
    result = await agent.extract_from_text_blocks("doc1", [sample_text_block])
    merged = agent.merge_graph_candidates("doc1", [result])

    assert result.triples
    assert merged.nodes
    assert merged.status == "succeeded"


class FailingLLM(BaseLLMAdapter):
    async def _generate_structured(self, prompt, input_data, response_model):
        raise NotImplementedError

    async def _analyze_image_structured(self, prompt, image_path, response_model):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt, file_path, response_model):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt, input_data, response_model):
        raise LLMAdapterError("LLM adapter operation failed: extract_graph_triples (PermissionDeniedError)")


def _build_chunk_graph_result(document_id: str, block_id: str, snippet: str) -> GraphExtractionResult:
    evidence = Evidence(
        id=f"ev_{block_id}",
        document_id=document_id,
        page_id=f"{block_id}_page",
        page_number=1,
        source_type="text_block",
        source_id=block_id,
        snippet=snippet,
    )
    subject = GraphNode(
        id=f"{block_id}_subject",
        label="Metric",
        properties={"name": snippet[:40]},
        source_reference=evidence,
    )
    obj = GraphNode(
        id=f"{block_id}_object",
        label="TimePeriod",
        properties={"name": "2025"},
        source_reference=evidence,
    )
    predicate = GraphEdge(
        id=f"{block_id}_edge",
        type="OCCURS_IN",
        source_node_id=subject.id,
        target_node_id=obj.id,
        properties={"confidence": 0.8},
        source_reference=evidence,
    )
    return GraphExtractionResult(
        document_id=document_id,
        nodes=[subject, obj],
        edges=[predicate],
        triples=[GraphTriple(subject=subject, predicate=predicate, object=obj)],
    )


@pytest.mark.asyncio
async def test_graph_extraction_agent_marks_expected_provider_error(sample_text_block) -> None:
    agent = GraphExtractionAgent(FailingLLM())

    result = await agent.extract_from_text_blocks("doc1", [sample_text_block])

    assert result.status == "succeeded"
    assert result.metadata["fallback"] is True
    assert result.metadata["fallback_mode"] == "local_extract"
    assert result.metadata["expected_provider_error"] is True
    assert "PermissionDeniedError" in result.metadata["error_detail"]
    assert result.nodes


@pytest.mark.asyncio
async def test_graph_extraction_agent_falls_back_on_timeout(sample_text_block) -> None:
    agent = GraphExtractionAgent(text_graph_timeout_seconds=0.01)

    async def slow_chain(*args, **kwargs):
        await asyncio.sleep(0.05)
        raise AssertionError("wait_for should time out before this point")

    agent.chain.ainvoke_from_text_blocks = slow_chain

    result = await agent.extract_from_text_blocks("doc1", [sample_text_block])

    assert result.status == "succeeded"
    assert result.metadata["fallback"] is True
    assert result.metadata["fallback_mode"] == "local_extract"
    assert result.metadata["expected_provider_error"] is True
    assert "timed out" in result.metadata["error_detail"].lower()
    assert result.nodes


@pytest.mark.asyncio
async def test_graph_extraction_agent_merges_partial_chunk_results(sample_text_block) -> None:
    second_block = sample_text_block.model_copy(
        update={"id": "tb2", "page_id": "p2", "page_number": 2, "text": "Costs decreased in 2025."}
    )
    agent = GraphExtractionAgent(text_graph_chunk_size=1, text_graph_chunk_chars=10_000)

    async def chunked_chain(document_id, text_blocks, page_summaries):
        del page_summaries
        block = text_blocks[0]
        if block.id == "tb2":
            raise OSError("socket closed")
        return _build_chunk_graph_result(document_id, block.id, block.text)

    agent.chain.ainvoke_from_text_blocks = chunked_chain

    result = await agent.extract_from_text_blocks("doc1", [sample_text_block, second_block])

    assert result.status == "partial"
    assert result.triples
    assert result.metadata["chunked"] is True
    assert result.metadata["chunk_count"] == 2
    assert result.metadata["failed_chunk_count"] == 1
    assert result.metadata["degraded_chunk_indexes"] == [2]


@pytest.mark.asyncio
async def test_graph_extraction_agent_preserves_successful_chunks_when_one_chunk_times_out(sample_text_block) -> None:
    second_block = sample_text_block.model_copy(
        update={"id": "tb2", "page_id": "p2", "page_number": 2, "text": "Operating margin improved in 2025."}
    )
    agent = GraphExtractionAgent(
        text_graph_timeout_seconds=0.01,
        text_graph_chunk_size=1,
        text_graph_chunk_chars=10_000,
    )

    async def chunked_chain(document_id, text_blocks, page_summaries):
        del page_summaries
        block = text_blocks[0]
        if block.id == "tb2":
            await asyncio.sleep(0.05)
        return _build_chunk_graph_result(document_id, block.id, block.text)

    agent.chain.ainvoke_from_text_blocks = chunked_chain

    result = await agent.extract_from_text_blocks("doc1", [sample_text_block, second_block])

    assert result.status == "succeeded"
    assert result.metadata["chunked"] is True
    assert result.metadata["fallback"] is True
    assert result.metadata["fallback_chunk_count"] == 1
    assert result.metadata["degraded_chunk_indexes"] == [2]
    assert any(node.source_reference.source_id == "tb2" for node in result.nodes)
