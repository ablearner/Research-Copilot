from __future__ import annotations

from skills.research import PaperAnalysisSkill

from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import AnalyzePapersFunctionOutput


class PaperAnalysisAgent:
    """Top-level worker agent for selected-paper analysis tasks."""

    name = "PaperAnalysisAgent"

    def __init__(self, *, paper_analysis_skill: PaperAnalysisSkill) -> None:
        self.paper_analysis_skill = paper_analysis_skill

    async def analyze(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        task_topic: str = "",
        report_highlights: list[str] | None = None,
        evidence_hits: list[RetrievalHit] | None = None,
    ) -> AnalyzePapersFunctionOutput:
        return await self.paper_analysis_skill.analyze_async(
            question=question,
            papers=papers,
            task_topic=task_topic,
            report_highlights=report_highlights or [],
            evidence_hits=evidence_hits or [],
        )
