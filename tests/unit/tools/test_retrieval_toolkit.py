"""Tests for tools.retrieval_toolkit data models."""

from tools.retrieval_toolkit import RetrievalAgentResult, RetrievalInput
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalQuery


class TestRetrievalInput:
    def test_minimal(self):
        inp = RetrievalInput(question="What is attention?")
        assert inp.question == "What is attention?"
        assert inp.doc_id is None
        assert inp.document_ids == []

    def test_with_doc_ids(self):
        inp = RetrievalInput(question="Q", document_ids=["d1", "d2"])
        assert len(inp.document_ids) == 2


class TestRetrievalAgentResult:
    def test_create(self):
        bundle = EvidenceBundle(evidences=[])
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query="What is attention?"),
        )
        result = RetrievalAgentResult(
            question="What is attention?",
            evidence_bundle=bundle,
            retrieval_result=retrieval_result,
        )
        assert result.question == "What is attention?"
        assert result.document_ids == []
