import pytest

from tools.answer_toolkit import AnswerAgent
from adapters.llm.base import BaseLLMAdapter
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle


class CoTReasonerStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def reason(self, **kwargs):
        self.calls.append(kwargs)
        return QAResponse(
            answer="这是基于研究证据的 CoT 综合答案。",
            question=kwargs["question"],
            evidence_bundle=kwargs["evidence_bundle"],
            retrieval_result=kwargs.get("retrieval_result"),
            confidence=0.83,
            metadata={"reasoning_mode": "cot_stub"},
        )


@pytest.mark.asyncio
async def test_answer_agent_does_not_call_llm_for_empty_evidence(mock_llm) -> None:
    response = await AnswerAgent(mock_llm).answer("What happened?", EvidenceBundle())

    assert response.answer == "证据不足"
    assert response.confidence == 0
    assert mock_llm.calls == []


@pytest.mark.asyncio
async def test_answer_agent_uses_evidence_bundle(mock_llm, sample_evidence) -> None:
    response = await AnswerAgent(mock_llm).answer("What happened?", EvidenceBundle(evidences=[sample_evidence]))

    assert response.confidence == 0.8
    assert response.evidence_bundle.evidences[0].id == sample_evidence.id


class PermissionDeniedError(Exception):
    status_code = 403


class FailingLLMAdapter(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type[QAResponse]):
        raise PermissionDeniedError("free tier exhausted")

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class InsufficientLLMAdapter(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type[QAResponse]):
        return QAResponse(
            answer="证据不足",
            question=input_data["question"],
            evidence_bundle=EvidenceBundle.model_validate(input_data["evidence_bundle"]),
            confidence=0.15,
            metadata={"source": "mock-llm"},
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class CapturingLLMAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_data: dict | None = None

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type[QAResponse]):
        self.last_input_data = input_data
        return QAResponse(
            answer="captured",
            question=input_data["question"],
            evidence_bundle=EvidenceBundle.model_validate(input_data["evidence_bundle"]),
            confidence=0.7,
            metadata={"source": "capturing-llm"},
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_answer_agent_uses_local_fallback_when_provider_denied() -> None:
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb1",
                "snippet": "求职意向：机器人规控算法实习生。",
            },
            {
                "id": "ev2",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb2",
                "snippet": "教育背景：哈尔滨工业大学（深圳）自动化专业。",
            },
        ],
        summary="这是一份聚焦机器人规控算法岗位的中文简历。",
    )

    response = await AnswerAgent(FailingLLMAdapter()).answer("这份文档讲了什么？", evidence_bundle)

    assert response.metadata["fallback"] is True
    assert response.metadata["fallback_mode"] == "local_extract"
    assert "机器人规控算法" in response.answer


@pytest.mark.asyncio
async def test_answer_agent_returns_insufficient_for_unrelated_question_when_provider_denied() -> None:
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb1",
                "snippet": "求职意向：机器人规控算法实习生。",
            },
            {
                "id": "ev2",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb2",
                "snippet": "教育背景：哈尔滨工业大学（深圳）自动化专业。",
            },
        ],
        summary="这是一份聚焦机器人规控算法岗位的中文简历。",
    )

    response = await AnswerAgent(FailingLLMAdapter()).answer("明天天气怎么样？", evidence_bundle)

    assert response.answer == "证据不足"
    assert response.confidence == 0.0
    assert response.metadata["reason"] == "weak_question_evidence_alignment"
    assert response.metadata["fallback"] is True


@pytest.mark.asyncio
async def test_answer_agent_preserves_model_insufficient_answer_when_evidence_exists() -> None:
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb1",
                "snippet": "项目目标：构建文档问答系统。",
            },
            {
                "id": "ev2",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb2",
                "snippet": "回答必须基于检索到的证据。",
            },
        ],
        summary="文档描述了一个面向文档问答的系统。",
    )

    response = await AnswerAgent(InsufficientLLMAdapter()).answer("这份文档讲了什么？", evidence_bundle)

    assert response.answer == "证据不足"
    assert response.confidence == 0.15
    assert response.metadata["source"] == "mock-llm"
    assert response.metadata["answered_by"] == "AnswerAgent"
    assert "fallback" not in response.metadata


@pytest.mark.asyncio
async def test_answer_agent_blocks_chart_question_without_chart_grounding(mock_llm) -> None:
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "source_type": "text_block",
                "source_id": "tb1",
                "snippet": "This paper studies sparse spikes deconvolution over the space of measures.",
            }
        ],
        summary="A research paper about sparse spikes deconvolution.",
    )

    response = await AnswerAgent(mock_llm).answer("What is the main focus of this graph?", evidence_bundle)

    assert response.metadata["answered_by"] == "AnswerAgentGuardrail"
    assert response.metadata["reason"] == "missing_chart_grounding"
    assert response.confidence == pytest.approx(0.08)
    assert "does not include chart-specific visual evidence" in response.answer
    assert mock_llm.calls == []


@pytest.mark.asyncio
async def test_answer_agent_allows_chart_question_with_chart_grounding(mock_llm) -> None:
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "source_type": "chart",
                "source_id": "chart1",
                "snippet": "Figure 1 is a bar chart showing revenue growth in 2025.",
            }
        ],
        summary="A chart about revenue growth.",
    )

    response = await AnswerAgent(mock_llm).answer("What does this chart show?", evidence_bundle)

    assert response.confidence == 0.8
    assert response.metadata["answered_by"] == "AnswerAgent"
    assert mock_llm.calls == ["generate_structured"]


@pytest.mark.asyncio
async def test_answer_agent_preserves_chart_metadata_for_answer_chain() -> None:
    capturing_llm = CapturingLLMAdapter()
    evidence_bundle = EvidenceBundle(
        evidences=[
            {
                "id": "ev1",
                "document_id": "doc1",
                "page_id": "p2",
                "page_number": 2,
                "source_type": "chart",
                "source_id": "chart1",
                "snippet": "Chart summary: A three-stage system pipeline.\nVisual answer: The figure shows encoder, planner, and executor modules connected in sequence.",
                "metadata": {
                    "title": "System overview",
                    "caption": "Figure 2. Overall pipeline.",
                    "chart_type": "unknown",
                    "source": "chart_grounding",
                    "anchor_source": "paper_figure_cache",
                    "anchor_selection": "deterministic_figure_reference",
                },
            }
        ],
        summary="A system diagram with three connected stages.",
    )

    response = await AnswerAgent(capturing_llm).answer("第二篇论文的系统框图是什么结构？", evidence_bundle)

    assert response.answer == "captured"
    compact_bundle = capturing_llm.last_input_data["evidence_bundle"]
    compact_metadata = compact_bundle["evidences"][0]["metadata"]
    assert compact_metadata["chart_type"] == "unknown"
    assert compact_metadata["caption"] == "Figure 2. Overall pipeline."
    assert compact_metadata["anchor_source"] == "paper_figure_cache"
    assert compact_metadata["anchor_selection"] == "deterministic_figure_reference"


@pytest.mark.asyncio
async def test_answer_agent_uses_cot_for_research_collection_when_not_react(mock_llm, sample_evidence) -> None:
    cot_reasoner = CoTReasonerStub()
    response = await AnswerAgent(mock_llm, cot_reasoning_agent=cot_reasoner).answer_with_evidence(
        "哪篇论文最值得优先阅读？",
        EvidenceBundle(evidences=[sample_evidence]),
        metadata={"qa_mode": "research_collection"},
        preference_context={"reasoning_style": "auto"},
        memory_hints={"preferred_sources": ["survey"]},
    )

    assert response.metadata["reasoning_mode"] == "cot_stub"
    assert len(cot_reasoner.calls) == 1
    assert cot_reasoner.calls[0]["memory_hints"] == {"preferred_sources": ["survey"]}
    assert mock_llm.calls == []


@pytest.mark.asyncio
async def test_answer_agent_uses_cot_as_default_reasoning_style(mock_llm, sample_evidence) -> None:
    cot_reasoner = CoTReasonerStub()
    response = await AnswerAgent(mock_llm, cot_reasoning_agent=cot_reasoner).answer_with_evidence(
        "这份材料的核心结论是什么？",
        EvidenceBundle(evidences=[sample_evidence]),
    )

    assert response.metadata["reasoning_mode"] == "cot_stub"
    assert len(cot_reasoner.calls) == 1
    assert mock_llm.calls == []


@pytest.mark.asyncio
async def test_answer_agent_keeps_react_path_outside_cot(mock_llm, sample_evidence) -> None:
    cot_reasoner = CoTReasonerStub()
    response = await AnswerAgent(mock_llm, cot_reasoning_agent=cot_reasoner).answer_with_evidence(
        "哪篇论文最值得优先阅读？",
        EvidenceBundle(evidences=[sample_evidence]),
        metadata={"qa_mode": "research_collection", "reasoning_style": "react"},
        preference_context={"reasoning_style": "react"},
    )

    assert response.metadata["answered_by"] == "AnswerAgent"
    assert cot_reasoner.calls == []
    assert mock_llm.calls == ["generate_structured"]
