import pytest

from domain.schemas.research import PaperCandidate, ResearchTask
from services.research.autonomous_research_runtime import (
    AutonomousResearchRuntime,
    _manager_decision_timeout_seconds,
)
from services.research.paper_search_service import PaperSearchService
from services.research.literature_research_service import LiteratureResearchService
from services.research.research_report_service import ResearchReportService


class ArxivToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        lowered = query.lower()
        if "survey" in lowered or "benchmark" in lowered:
            return [
                PaperCandidate(
                    paper_id="arxiv:2401.00001",
                    title="A Survey of UAV Path Planning Benchmarks",
                    authors=["Alice"],
                    abstract="This survey summarizes benchmark tasks for UAV path planning.",
                    year=2026,
                    venue="arXiv",
                    source="arxiv",
                    arxiv_id="2401.00001",
                    pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
                    url="https://arxiv.org/abs/2401.00001",
                    is_open_access=True,
                    published_at="2026-04-01T00:00:00+00:00",
                ),
                PaperCandidate(
                    paper_id="arxiv:2401.00002",
                    title="Open Dataset Evaluation for UAV Navigation",
                    authors=["Bob"],
                    abstract="We compare open datasets and evaluation settings for UAV navigation.",
                    year=2026,
                    venue="arXiv",
                    source="arxiv",
                    arxiv_id="2401.00002",
                    pdf_url="https://arxiv.org/pdf/2401.00002.pdf",
                    url="https://arxiv.org/abs/2401.00002",
                    is_open_access=True,
                    published_at="2026-04-03T00:00:00+00:00",
                ),
            ]
        return [
            PaperCandidate(
                paper_id="arxiv:2401.00000",
                title="Learning-Based UAV Path Planning in Dynamic Scenes",
                authors=["Carol"],
                abstract="We study learning-based UAV path planning for dynamic scenes.",
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id="2401.00000",
                pdf_url=None,
                url="https://arxiv.org/abs/2401.00000",
                is_open_access=False,
                published_at="2026-03-28T00:00:00+00:00",
            )
        ]


class OpenAlexToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        lowered = query.lower()
        if "survey" not in lowered and "benchmark" not in lowered:
            return []
        return [
            PaperCandidate(
                paper_id="https://openalex.org/W240100003",
                title="Robust UAV Route Planning with Open-Source Evaluation",
                authors=["Dave"],
                abstract="This paper studies robust UAV route planning with open-source evaluation assets.",
                year=2026,
                venue="OpenAlex Venue",
                source="openalex",
                pdf_url="https://example.com/uav-route-planning.pdf",
                url="https://openalex.org/W240100003",
                citations=8,
                is_open_access=True,
                published_at="2026-04-04",
            )
        ]


class SemanticScholarToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id="semantic_scholar:abc123",
                title="Semantic Scholar Survey of Multi-Agent UAV Path Planning",
                authors=["Eve"],
                abstract="A survey of multi-agent UAV path planning methods and evaluation practices.",
                year=2026,
                venue="Semantic Scholar Corpus",
                source="semantic_scholar",
                pdf_url="https://example.com/semantic-uav-survey.pdf",
                url="https://www.semanticscholar.org/paper/abc123",
                citations=21,
                is_open_access=True,
                published_at="2026-04-05T00:00:00+00:00",
            )
        ]


class IEEEToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id="ieee:123456",
                title="IEEE Benchmarking Study for Multi-Agent UAV Path Planning",
                authors=["Frank"],
                abstract="We benchmark multi-agent UAV path planning under shared evaluation settings.",
                year=2026,
                venue="IEEE Robotics and Automation Letters",
                source="ieee",
                pdf_url="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=123456",
                url="https://ieeexplore.ieee.org/document/123456",
                citations=7,
                is_open_access=True,
                published_at="2026-04-06T00:00:00+00:00",
            )
        ]


def test_autonomous_manager_timeout_uses_adapter_timeout_with_slack() -> None:
    timeout_seconds = _manager_decision_timeout_seconds(type("Adapter", (), {"timeout_seconds": 90.0})())

    assert timeout_seconds == 100.0


@pytest.mark.asyncio
async def test_autonomous_research_runtime_refines_search_and_generates_todos() -> None:
    runtime = AutonomousResearchRuntime(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        )
    )

    bundle = await runtime.run(
        topic="无人机路径规划",
        days_back=365,
        max_papers=6,
        sources=["arxiv", "openalex"],
        task_id="task_runtime_1",
    )

    assert len(bundle.papers) >= 3
    assert bundle.report.metadata["autonomy_mode"] == "lead_agent_loop"
    assert bundle.report.metadata["agent_architecture"] == "main_agents_plus_skills"
    assert bundle.report.metadata["decision_model"] == "llm_dynamic_single_manager"
    assert "ResearchSupervisorAgent" in bundle.report.metadata["primary_agents"]
    assert "LiteratureScoutAgent" in bundle.report.metadata["primary_agents"]
    assert "PaperCurationSkill" in bundle.report.metadata["primary_skills"]
    assert bundle.plan.metadata["autonomy_mode"] == "lead_agent_loop"
    assert bundle.plan.metadata["agent_architecture"] == "main_agents_plus_skills"
    assert bundle.plan.metadata["decision_model"] == "llm_dynamic_single_manager"
    assert bundle.todo_items
    assert any(step.agent == "LiteratureScoutAgent" for step in bundle.trace)
    assert bundle.must_read_ids
    assert bundle.ingest_candidate_ids
    assert bundle.workspace.current_stage == "complete"
    assert bundle.report.workspace.current_stage == "complete"
    assert bundle.workspace.next_actions


@pytest.mark.asyncio
async def test_literature_research_service_run_task_persists_autonomous_outputs(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=report_service,
        paper_import_service=object(),
    )
    task = ResearchTask(
        task_id="task_runtime_2",
        topic="无人机路径规划",
        status="created",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        days_back=365,
        max_papers=6,
        sources=["arxiv", "openalex"],
    )
    report_service.save_task(task)

    response = await service.run_task("task_runtime_2")

    assert response.task.todo_items
    assert response.task.metadata["autonomy_mode"] == "lead_agent_loop"
    assert response.task.metadata["agent_architecture"] == "main_agents_plus_skills"
    assert "ResearchWriterAgent" in response.task.metadata["primary_agents"]
    assert "PaperCurationSkill" in response.task.metadata["primary_skills"]
    assert response.report is not None
    assert response.report.metadata["autonomy_mode"] == "lead_agent_loop"
    assert response.report.metadata["agent_architecture"] == "main_agents_plus_skills"
    assert response.report.metadata["decision_model"] == "llm_dynamic_single_manager"
    assert response.task.workspace.current_stage == "complete"
    assert response.report.workspace.current_stage == "complete"
    persisted_task = report_service.load_task("task_runtime_2")
    assert persisted_task is not None
    assert persisted_task.todo_items
    assert persisted_task.workspace.current_stage == "complete"


@pytest.mark.asyncio
async def test_autonomous_research_runtime_searches_semantic_scholar_and_ieee_sources() -> None:
    runtime = AutonomousResearchRuntime(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
            semantic_scholar_tool=SemanticScholarToolStub(),
            ieee_tool=IEEEToolStub(),
        )
    )

    bundle = await runtime.run(
        topic="多智能体无人机路径规划",
        days_back=365,
        max_papers=4,
        sources=["semantic_scholar", "ieee"],
        task_id="task_runtime_3",
    )

    sources = {paper.source for paper in bundle.papers}
    assert "semantic_scholar" in sources
    assert "ieee" in sources
    assert not any("尚未接入检索工具" in warning for warning in bundle.warnings)
