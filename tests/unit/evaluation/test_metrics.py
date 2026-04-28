"""Unit tests for evaluation metric functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    groundedness_score,
    informative_tokens,
    keyword_recall,
    normalize_text,
    percentile,
    polarity_accuracy,
    reference_token_f1,
    retrieval_recall_at_k,
    route_accuracy,
    tool_call_success_rate,
)
from evaluation.schemas import EvaluationCase


# ── keyword_recall ──


class TestKeywordRecall:
    def test_full_recall(self):
        recall, matched = keyword_recall(
            ["attention", "transformer"],
            "The attention mechanism in transformers is powerful",
        )
        assert recall == 1.0
        assert set(matched) == {"attention", "transformer"}

    def test_partial_recall(self):
        recall, matched = keyword_recall(
            ["attention", "lstm", "cnn"],
            "Attention mechanisms outperform traditional approaches",
        )
        assert recall == pytest.approx(1 / 3)
        assert matched == ["attention"]

    def test_zero_recall(self):
        recall, matched = keyword_recall(
            ["quantum", "entanglement"],
            "The paper discusses neural network optimization",
        )
        assert recall == 0.0
        assert matched == []

    def test_empty_keywords(self):
        recall, matched = keyword_recall([], "any text")
        assert recall is None
        assert matched == []

    def test_none_answer(self):
        recall, matched = keyword_recall(["keyword"], None)
        assert recall == 0.0

    def test_case_insensitive(self):
        recall, matched = keyword_recall(["Transformer"], "the transformer model")
        assert recall == 1.0


# ── groundedness_score ──


class TestGroundednessScore:
    def test_fully_grounded(self):
        score = groundedness_score(
            answer="Transformers use self-attention",
            evidence_texts=["Self-attention is the core of transformer architecture"],
            grounding_keywords=["self-attention", "transformer"],
        )
        assert score is not None and score > 0.5

    def test_ungrounded(self):
        score = groundedness_score(
            answer="Quantum computing uses qubits",
            evidence_texts=["Neural networks process data in layers"],
            grounding_keywords=["quantum", "qubits"],
        )
        assert score == 0.0

    def test_empty_answer(self):
        score = groundedness_score(
            answer="", evidence_texts=["some evidence"]
        )
        assert score is None

    def test_empty_evidence(self):
        score = groundedness_score(
            answer="some answer", evidence_texts=[]
        )
        assert score is None

    def test_no_keywords_uses_tokens(self):
        score = groundedness_score(
            answer="The paper presents results on image classification",
            evidence_texts=["The paper presents experimental results on image classification benchmarks"],
        )
        assert score is not None and score > 0.5


# ── route_accuracy ──


class TestRouteAccuracy:
    def test_correct_route(self):
        assert route_accuracy("ask_document", "ask_document") is True

    def test_wrong_route(self):
        assert route_accuracy("ask_document", "chart_understand") is False

    def test_none_expected(self):
        assert route_accuracy(None, "ask_document") is None

    def test_case_insensitive(self):
        assert route_accuracy("Ask_Document", "ask_document") is True


# ── reference_token_f1 ──


class TestReferenceTokenF1:
    def test_perfect_match(self):
        p, r, f1 = reference_token_f1(
            "attention mechanism", "attention mechanism"
        )
        assert f1 == 1.0

    def test_partial_overlap(self):
        p, r, f1 = reference_token_f1(
            "attention is important",
            "attention and convolution are both important",
        )
        assert p is not None and 0 < p < 1
        assert r is not None and 0 < r <= 1
        assert f1 is not None and 0 < f1 < 1

    def test_no_overlap(self):
        p, r, f1 = reference_token_f1("quantum physics", "neural networks")
        assert f1 == 0.0

    def test_none_inputs(self):
        assert reference_token_f1(None, "text") == (None, None, None)
        assert reference_token_f1("text", None) == (None, None, None)


# ── polarity_accuracy ──


class TestPolarityAccuracy:
    def test_both_yes(self):
        assert polarity_accuracy("Yes, it does", "Yes") is True

    def test_mismatch(self):
        assert polarity_accuracy("Yes", "No") is False

    def test_none_polarity(self):
        assert polarity_accuracy("hello", "world") is None


# ── tool_call_success_rate ──


class TestToolCallSuccessRate:
    def test_all_succeeded(self):
        traces = [
            {"tool_name": "search", "status": "succeeded"},
            {"tool_name": "analyze", "status": "succeeded"},
        ]
        rate, success, total = tool_call_success_rate(traces)
        assert rate == 1.0
        assert success == 2
        assert total == 2

    def test_partial_failure(self):
        traces = [
            {"tool_name": "search", "status": "succeeded"},
            {"tool_name": "analyze", "status": "failed"},
        ]
        rate, _, _ = tool_call_success_rate(traces)
        assert rate == 0.5

    def test_empty_traces(self):
        rate, _, _ = tool_call_success_rate([])
        assert rate is None

    def test_filter_by_name(self):
        traces = [
            {"tool_name": "search", "status": "succeeded"},
            {"tool_name": "analyze", "status": "failed"},
        ]
        rate, success, total = tool_call_success_rate(traces, expected_tool_names=["search"])
        assert rate == 1.0
        assert total == 1


# ── percentile ──


class TestPercentile:
    def test_single_value(self):
        assert percentile([5.0], 0.5) == 5.0

    def test_p50(self):
        result = percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert result is not None

    def test_empty(self):
        assert percentile([], 0.5) is None


# ── informative_tokens ──


class TestInformativeTokens:
    def test_removes_stopwords(self):
        tokens = informative_tokens("the attention mechanism is important")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_unique(self):
        tokens = informative_tokens("attention attention attention")
        assert tokens == ["attention"]


# ── Schema validation ──


class TestEvaluationCaseSchema:
    def test_search_literature_case(self):
        case = EvaluationCase(
            id="test_search",
            kind="search_literature",
            question="Search for papers on transformers",
            expected_keywords=["transformer"],
        )
        assert case.kind == "search_literature"

    def test_write_review_case(self):
        case = EvaluationCase(
            id="test_review",
            kind="write_review",
            question="Write a review on attention",
            require_evidence=False,
        )
        assert not case.needs_evidence

    def test_import_and_qa_needs_evidence(self):
        case = EvaluationCase(
            id="test_import",
            kind="import_and_qa",
            question="What is attention?",
        )
        assert case.needs_evidence is True

    def test_missing_question_raises(self):
        with pytest.raises(Exception):
            EvaluationCase(id="bad", kind="search_literature")


# ── core_cases.json validity ──


class TestCoreCasesDataset:
    @pytest.fixture
    def cases_path(self):
        return Path(__file__).resolve().parents[3] / "evaluation" / "datasets" / "core_cases.json"

    def test_file_exists(self, cases_path: Path):
        assert cases_path.exists()

    def test_valid_json(self, cases_path: Path):
        data = json.loads(cases_path.read_text())
        assert "cases" in data
        assert len(data["cases"]) >= 20

    def test_all_cases_parse(self, cases_path: Path):
        data = json.loads(cases_path.read_text())
        for raw in data["cases"]:
            case = EvaluationCase.model_validate(raw)
            assert case.question

    def test_covers_all_research_kinds(self, cases_path: Path):
        data = json.loads(cases_path.read_text())
        kinds = {c["kind"] for c in data["cases"]}
        assert "search_literature" in kinds
        assert "import_and_qa" in kinds
        assert "write_review" in kinds
        assert "multi_turn_session" in kinds
