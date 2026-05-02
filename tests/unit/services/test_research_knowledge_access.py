from __future__ import annotations

import pytest

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalQuery
from tools.research.knowledge_access import ResearchKnowledgeAccess
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import HybridRetrieveToolInput, HybridRetrieveToolOutput, ToolSpec


def _retrieval_result(question: str, document_ids: list[str]) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        query=RetrievalQuery(query=question, document_ids=document_ids, top_k=3),
        hits=[],
        evidence_bundle=EvidenceBundle(),
        metadata={},
    )


class FallbackRetrievalTools:
    def __init__(self) -> None:
        self.called = False

    async def retrieve(self, **kwargs):
        self.called = True
        result = _retrieval_result(kwargs["question"], kwargs["document_ids"])
        return HybridRetrieveToolOutput(
            question=kwargs["question"],
            document_ids=kwargs["document_ids"],
            evidence_bundle=result.evidence_bundle,
            retrieval_result=result,
            metadata={"source": "fallback"},
        )


@pytest.mark.asyncio
async def test_retrieve_prefers_registered_tool_executor() -> None:
    registry = ToolRegistry()
    fallback = FallbackRetrievalTools()

    async def handle_hybrid_retrieve(**kwargs):
        result = _retrieval_result(kwargs["question"], kwargs["document_ids"])
        return HybridRetrieveToolOutput(
            question=kwargs["question"],
            document_ids=kwargs["document_ids"],
            evidence_bundle=result.evidence_bundle,
            retrieval_result=result,
            metadata={"source": "tool_executor"},
        )

    registry.register(
        ToolSpec(
            name="hybrid_retrieve",
            description="Retrieve evidence.",
            input_schema=HybridRetrieveToolInput,
            output_schema=HybridRetrieveToolOutput,
            handler=handle_hybrid_retrieve,
        )
    )
    runtime = type(
        "Runtime",
        (),
        {
            "tool_registry": registry,
            "tool_executor": ToolExecutor(registry),
            "retrieval_tools": fallback,
        },
    )()

    output = await ResearchKnowledgeAccess.from_runtime(runtime).retrieve(
        question="agentic rag",
        document_ids=["doc-1"],
        top_k=3,
    )

    assert output.metadata["source"] == "tool_executor"
    assert fallback.called is False
    assert len(runtime.tool_executor.get_traces()) == 1


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_runtime_retrieval_tools() -> None:
    fallback = FallbackRetrievalTools()
    runtime = type("Runtime", (), {"retrieval_tools": fallback})()

    output = await ResearchKnowledgeAccess.from_runtime(runtime).retrieve(
        question="agentic rag",
        document_ids=["doc-1"],
        top_k=3,
    )

    assert output.metadata["source"] == "fallback"
    assert fallback.called is True
    assert runtime.knowledge_access is ResearchKnowledgeAccess.from_runtime(runtime)
