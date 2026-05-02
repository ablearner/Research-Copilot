import pytest

from domain.schemas.research import PaperCandidate
from tools.research import (
    PaperAnalyzer,
    PaperReader,
    ResearchEvaluator,
    ReviewWriter,
    WritingPolisher,
)


def test_paper_reading_skill_extracts_structured_card() -> None:
    skill = PaperReader()
    paper = PaperCandidate(
        paper_id="paper-1",
        title="Planner-Guided Scientific Review",
        abstract=(
            "This paper proposes a planner-guided review framework. "
            "The method coordinates retrieval, outline generation, and evidence synthesis. "
            "Experiments on benchmark review tasks show stronger citation grounding."
        ),
        source="arxiv",
    )

    card = skill.extract(
        paper=paper,
        metadata={
            "formulas": [{"name": "score", "formula": "s = r + c", "explanation": "ranking score"}],
            "figures": [{"figure_id": "fig-1", "explanation": "system overview"}],
        },
    )

    assert card.paper_id == "paper-1"
    assert card.contribution
    assert card.method
    assert card.experiment
    assert len(card.key_formulas) == 1
    assert len(card.figures) == 1


@pytest.mark.asyncio
async def test_paper_reading_skill_llm_prompt_follows_answer_language_metadata() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.prompt = ""

        async def generate_structured(self, prompt: str, input_data: dict, response_model: type):
            self.prompt = prompt
            return response_model.model_validate(
                {
                    "contribution": "It improves evidence synthesis.",
                    "method": "It uses planner-guided retrieval.",
                    "experiment": "It is evaluated on benchmark review tasks.",
                    "limitation": "Full-text validation is still limited.",
                    "summary": "A planner-guided review assistant.",
                }
            )

    llm = LLMStub()
    skill = PaperReader(llm_adapter=llm)
    paper = PaperCandidate(
        paper_id="paper-1",
        title="Planner-Guided Scientific Review",
        abstract="This paper proposes a planner-guided review framework.",
        source="arxiv",
    )

    card = await skill.extract_async(
        paper=paper,
        metadata={"answer_language": "en-US"},
    )

    assert "Respond in English" in llm.prompt
    assert card.summary == "A planner-guided review assistant."


@pytest.mark.asyncio
async def test_paper_analysis_skill_heuristic_follows_question_language() -> None:
    skill = PaperAnalyzer()
    paper = PaperCandidate(
        paper_id="paper-1",
        title="Planner-Guided Scientific Review",
        abstract=(
            "This paper proposes a planner-guided review framework. "
            "The method coordinates retrieval, outline generation, and evidence synthesis. "
            "Experiments on benchmark review tasks show stronger citation grounding."
        ),
        source="arxiv",
    )

    result = await skill.analyze_async(
        question="What method does this paper use?",
        papers=[paper],
        task_topic="scientific review",
    )

    assert "I analyzed the 1 currently selected papers" in result.answer
    assert "The papers can be understood through their contributions" in result.answer


def test_research_evaluation_skill_flags_low_quality_review() -> None:
    skill = ResearchEvaluator()

    evaluation = skill.evaluate_result(
        task_type="write_review",
        result_status="succeeded",
        payload={
            "report_id": "report-1",
            "report_word_count": 180,
            "report_has_citations": False,
            "report_has_key_sections": False,
        },
        task_instruction="write a grounded review",
        expected_schema={
            "required_fields": ["report_id"],
            "min_report_words": 800,
            "require_citations": True,
            "require_key_sections": True,
        },
    )

    assert evaluation.passed is False
    assert "report_too_short" in evaluation.issues
    assert "missing_citations" in evaluation.issues
    assert evaluation.replan_suggestion == "retry_review_quality"


def test_review_writing_skill_polishes_for_target_journal() -> None:
    skill = ReviewWriter(polish_skill=WritingPolisher())
    papers = [
        PaperCandidate(
            paper_id="paper-1",
            title="Scientific QA with Evidence Control",
            abstract="This paper studies evidence control and structured citation generation.",
            source="openalex",
            url="https://openalex.org/W1",
        )
    ]

    report = skill.generate(
        topic="scientific QA",
        task_id="task-1",
        papers=papers,
        style="academic",
        min_length=600,
        include_citations=True,
        target_journal="ACL",
    )

    assert "ACL" in report.markdown
    assert report.metadata["writer"] == "ReviewWriter"
