from __future__ import annotations

from domain.schemas.research import PaperCandidate, ResearchReport
from tools.research.survey_writing import SurveyWriter
from tools.research.writing_polish import WritingPolisher


class ReviewWriter:
    """Wrapper around survey writing with an optional polishing stage.
    
    Supports both sync and async generation. When LLM adapters are configured
    on the underlying skills, async methods will use LLM-powered generation.
    """

    def __init__(
        self,
        *,
        survey_writer: SurveyWriter | None = None,
        polish_skill: WritingPolisher | None = None,
    ) -> None:
        self.survey_writer = survey_writer or SurveyWriter()
        self.polish_skill = polish_skill or WritingPolisher()

    def generate(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        target_journal: str | None = None,
        language: str = "zh-CN",
    ) -> ResearchReport:
        """Synchronous generate — uses heuristic logic."""
        report = self.survey_writer.generate(
            topic=topic,
            task_id=task_id,
            papers=papers,
            warnings=warnings,
            style=style,
            min_length=min_length,
            include_citations=include_citations,
            language=language,
        )
        polished_markdown = self.polish_skill.polish(
            text=report.markdown,
            tone=style,
            target_journal=target_journal,
        )
        return report.model_copy(
            update={
                "markdown": polished_markdown,
                "metadata": {
                    **report.metadata,
                    "writer": "ReviewWriter",
                    "target_journal": target_journal,
                },
            }
        )

    async def generate_async(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        target_journal: str | None = None,
        language: str = "zh-CN",
        supervisor_instruction: str | None = None,
    ) -> ResearchReport:
        """Async generate — uses LLM-powered survey writing and polishing if available."""
        report = await self.survey_writer.generate_async(
            topic=topic,
            task_id=task_id,
            papers=papers,
            warnings=warnings,
            style=style,
            min_length=min_length,
            include_citations=include_citations,
            language=language,
            supervisor_instruction=supervisor_instruction,
        )
        polished_markdown = await self.polish_skill.polish_async(
            text=report.markdown,
            tone=style,
            target_journal=target_journal,
        )
        return report.model_copy(
            update={
                "markdown": polished_markdown,
                "metadata": {
                    **report.metadata,
                    "writer": "ReviewWriter+LLM",
                    "target_journal": target_journal,
                },
            }
        )
