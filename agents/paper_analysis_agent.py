from __future__ import annotations

from services.research.capabilities import PaperAnalyzer

from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from domain.schemas.unified_runtime import UnifiedAgentResult, UnifiedAgentTask
from services.research.research_specialist_capabilities import (
    PaperAnalysisCapability,
    build_specialist_unified_result,
)


class PaperAnalysisAgent:
    """Top-level worker agent for selected-paper analysis tasks."""

    name = "PaperAnalysisAgent"

    def __init__(
        self,
        *,
        paper_analysis_skill: PaperAnalyzer,
        execution_capability: PaperAnalysisCapability | None = None,
    ) -> None:
        self.paper_analysis_skill = paper_analysis_skill
        self.execution_capability = execution_capability or PaperAnalysisCapability()

    async def execute(self, task: UnifiedAgentTask, runtime_context) -> UnifiedAgentResult:
        supervisor_context = runtime_context.metadata.get("supervisor_tool_context")
        decision = runtime_context.metadata.get("supervisor_decision")
        if supervisor_context is None or decision is None:
            return build_specialist_unified_result(
                task=task,
                agent_name=self.name,
                status="failed",
                observation="missing supervisor runtime context for PaperAnalysisAgent",
                metadata={"reason": "missing_supervisor_runtime_context"},
                execution_adapter="paper_analysis_agent",
                delegate_type=self.__class__.__name__,
            )
        result = await self.execution_capability.run(
            context=supervisor_context,
            decision=decision,
            paper_analysis_agent=self,
        )
        metadata = {
            **dict(result.metadata),
            "executed_by": self.name,
            "specialist_execution_path": "paper_analysis_agent",
        }
        return build_specialist_unified_result(
            task=task,
            agent_name=self.name,
            status=result.status,
            observation=result.observation,
            metadata=metadata,
            execution_adapter="paper_analysis_agent",
            delegate_type=self.__class__.__name__,
        )

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
