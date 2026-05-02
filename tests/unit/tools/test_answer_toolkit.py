"""Tests for tools.answer_toolkit utility functions and AnswerAgent basics."""


from domain.schemas.evidence import Evidence, EvidenceBundle
from tools.answer_toolkit import (
    assess_question_evidence_alignment,
    has_chart_grounding,
    insufficient_evidence_response,
    is_chart_question,
    is_generic_document_question,
    looks_insufficient,
)


def _make_evidence(text: str, source_type: str = "text_block") -> Evidence:
    return Evidence(
        id="ev1",
        document_id="doc1",
        snippet=text,
        source_type=source_type,
        score=0.8,
    )


def _make_bundle(*items: Evidence) -> EvidenceBundle:
    return EvidenceBundle(evidences=list(items))


class TestLooksInsufficient:
    def test_chinese_marker(self):
        assert looks_insufficient("证据不足，无法回答") is True

    def test_english_marker(self):
        assert looks_insufficient("  Insufficient evidence to answer.") is True

    def test_normal_answer(self):
        assert looks_insufficient("Transformer uses self-attention mechanism.") is False

    def test_empty(self):
        assert looks_insufficient("") is False


class TestIsGenericDocumentQuestion:
    def test_summary_question(self):
        assert is_generic_document_question("summarize this document") is True

    def test_specific_question(self):
        assert is_generic_document_question("what is the learning rate used?") is False

    def test_empty(self):
        assert is_generic_document_question("") is False


class TestIsChartQuestion:
    def test_chart_keyword(self):
        assert is_chart_question("what does this chart show?") is True

    def test_figure_keyword(self):
        assert is_chart_question("explain figure 3") is True

    def test_no_chart(self):
        assert is_chart_question("what is attention?") is False


class TestHasChartGrounding:
    def test_chart_evidence(self):
        bundle = _make_bundle(_make_evidence("chart data", source_type="chart"))
        assert has_chart_grounding(bundle) is True

    def test_no_chart_evidence(self):
        bundle = _make_bundle(_make_evidence("text data", source_type="text_block"))
        assert has_chart_grounding(bundle) is False

    def test_empty_bundle(self):
        bundle = _make_bundle()
        assert has_chart_grounding(bundle) is False


class TestAssessQuestionEvidenceAlignment:
    def test_aligned(self):
        bundle = _make_bundle(_make_evidence("Attention is a mechanism for weighting"))
        result = assess_question_evidence_alignment(
            question="What is attention?",
            evidence_bundle=bundle,
            retrieval_result=None,
        )
        assert "aligned" in result
        assert isinstance(result["aligned"], bool)

    def test_empty_evidence(self):
        bundle = _make_bundle()
        result = assess_question_evidence_alignment(
            question="What is attention?",
            evidence_bundle=bundle,
            retrieval_result=None,
        )
        assert result["aligned"] is False


class TestInsufficientEvidenceResponse:
    def test_returns_qa_response(self):
        bundle = _make_bundle()
        response = insufficient_evidence_response(
            question="test question",
            evidence_bundle=bundle,
            retrieval_result=None,
            metadata={},
        )
        assert response.question == "test question"
        assert response.confidence == 0.0
        assert looks_insufficient(response.answer)
