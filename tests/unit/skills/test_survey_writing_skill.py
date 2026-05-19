import pytest

from domain.schemas.research import PaperCandidate
from tools.research.survey_writing import SurveyWritingTool


class _MiniLLMStub:
    model = "gpt-5.4-mini"

    def __init__(self) -> None:
        self.last_input_data: dict | None = None
        self.last_prompt: str | None = None

    async def generate_structured(self, prompt: str, input_data: dict, response_model: type):
        self.last_prompt = prompt
        self.last_input_data = input_data
        return response_model.model_validate({"markdown": f"min_length={input_data['min_length']}"})


def test_survey_writing_skill_generates_structured_cited_review() -> None:
    skill = SurveyWritingTool()
    papers = [
        PaperCandidate(
            paper_id="p1",
            title="Agentic Review Planning for Scientific Discovery",
            authors=["Alice", "Bob"],
            abstract="This paper studies planner-guided literature review systems. It compares retrieval planning, evidence grounding, and synthesis quality.",
            year=2026,
            source="arxiv",
            pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
            url="https://arxiv.org/abs/2601.00001",
        ),
        PaperCandidate(
            paper_id="p2",
            title="Multi-Agent Scientific Question Answering with Evidence Control",
            authors=["Carol"],
            abstract="We study scientific QA systems with evidence control, structured citation generation, and document-scoped retrieval.",
            year=2025,
            source="openalex",
            url="https://openalex.org/W123",
        ),
    ]

    report = skill.generate(
        topic="agentic scientific assistants",
        task_id="task-review-1",
        papers=papers,
        style="academic",
        min_length=800,
        include_citations=True,
    )

    assert "## 研究背景" in report.markdown
    assert "## 方法对比" in report.markdown
    assert "## 代表论文逐篇解读" in report.markdown
    assert "## 证据边界与局限" in report.markdown
    assert "## 研究空白与未来方向" in report.markdown
    assert "[P1]" in report.markdown
    assert len(report.markdown) >= 800


def test_survey_writing_skill_supports_english_output() -> None:
    skill = SurveyWritingTool()
    papers = [
        PaperCandidate(
            paper_id="p1",
            title="Agentic Review Planning for Scientific Discovery",
            authors=["Alice"],
            abstract="This paper studies planner-guided literature review systems.",
            year=2026,
            source="arxiv",
        )
    ]

    report = skill.generate(
        topic="agentic scientific assistants",
        task_id="task-review-en-1",
        papers=papers,
        style="academic",
        min_length=300,
        include_citations=True,
        language="en-US",
    )

    assert "# Literature Review:" in report.markdown
    assert "## Representative Papers" in report.markdown
    assert report.metadata["language"] == "en-US"


@pytest.mark.asyncio
async def test_survey_writing_skill_reduces_min_length_for_mini_models() -> None:
    skill = SurveyWritingTool(llm_adapter=_MiniLLMStub())
    papers = [
        PaperCandidate(
            paper_id="p1",
            title="Agentic Review Planning for Scientific Discovery",
            authors=["Alice"],
            abstract="This paper studies planner-guided literature review systems.",
            year=2026,
            source="arxiv",
        )
    ]

    report = await skill.generate_async(
        topic="agentic scientific assistants",
        task_id="task-review-mini-1",
        papers=papers,
        style="academic",
        min_length=800,
        include_citations=True,
    )

    assert "min_length=600" in report.markdown


@pytest.mark.asyncio
async def test_survey_writing_skill_passes_active_skill_context_to_llm() -> None:
    llm_stub = _MiniLLMStub()
    skill = SurveyWritingTool(llm_adapter=llm_stub)
    papers = [
        PaperCandidate(
            paper_id="p1",
            title="Agentic Review Planning for Scientific Discovery",
            authors=["Alice"],
            abstract="This paper studies planner-guided literature review systems.",
            year=2026,
            source="arxiv",
        )
    ]

    await skill.generate_async(
        topic="agentic scientific assistants",
        task_id="task-review-skill-1",
        papers=papers,
        skill_context="## Active Skills\nUse an evidence boundary section.",
        supervisor_instruction="Write a concise survey.",
    )

    assert llm_stub.last_input_data is not None
    assert llm_stub.last_input_data["skill_context"].startswith("## Active Skills")
    assert llm_stub.last_input_data["supervisor_instruction"] == "Write a concise survey."
    assert "skill_context" in (llm_stub.last_prompt or "")
