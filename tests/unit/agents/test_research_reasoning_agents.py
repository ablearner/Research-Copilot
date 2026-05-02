from types import SimpleNamespace

import pytest

from agents.literature_scout_agent import LiteratureScoutAgent, _dynamic_source_search_timeout_seconds
from agents.research_knowledge_agent import ResearchKnowledgeAgent
from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask, ResearchTaskAskRequest, ResearchTopicPlan
from domain.schemas.retrieval import RetrievalHit
from runtime.research.agent_protocol.base import _llm_stage_timeout_seconds


class PlanAndExecuteLLMStub:
    """Stub LLM that returns structured plan output for Plan-and-Execute tests."""
    def __init__(self, *, queries: list[str]) -> None:
        self.queries = queries
        self.calls: list[dict] = []
        self.timeout_seconds = 30.0

    async def generate_structured(self, *, prompt, input_data, response_model=None):
        self.calls.append({"prompt": prompt, "input_data": input_data})
        return {
            "queries": list(self.queries),
            "reasoning_summary": "stubbed reasoning summary",
            "plan_steps": ["decompose objective", "expand evidence coverage"],
        }


class TopicPlannerStub:
    def plan(self, *, topic: str, days_back: int, max_papers: int, sources: list[str]) -> ResearchTopicPlan:
        return ResearchTopicPlan(
            topic=topic,
            normalized_topic=topic.lower(),
            queries=[f"{topic} baseline"],
            days_back=days_back,
            max_papers=max_papers,
            sources=list(sources),
            metadata={"source": "topic_planner_stub"},
        )


class PaperSearchServiceStub:
    def __init__(self) -> None:
        self.topic_planner = TopicPlannerStub()


def test_dynamic_source_search_timeout_uses_tool_timeout_with_slack() -> None:
    tool = SimpleNamespace(timeout_seconds=20.0)

    timeout_seconds = _dynamic_source_search_timeout_seconds(tool)

    assert timeout_seconds == 35.0


def test_llm_stage_timeout_uses_adapter_timeout_with_slack() -> None:
    adapter = SimpleNamespace(timeout_seconds=90.0)

    timeout_seconds = _llm_stage_timeout_seconds(
        adapter,
        fallback_seconds=12.0,
        slack_seconds=10.0,
    )

    assert timeout_seconds == 100.0


@pytest.mark.asyncio
async def test_literature_scout_agent_uses_plan_and_execute_for_topic_plan() -> None:
    llm_stub = PlanAndExecuteLLMStub(queries=["无人机路径规划 survey benchmark"])
    scout = LiteratureScoutAgent(
        paper_search_service=PaperSearchServiceStub(),
        llm_adapter=llm_stub,
    )
    state = SimpleNamespace(
        topic="无人机路径规划",
        days_back=365,
        max_papers=6,
        sources=["arxiv", "openalex"],
        execution_context=SimpleNamespace(
            preference_context={"reasoning_style": "plan_and_solve"},
            memory_hints={"preferred_sources": ["survey"]},
        ),
    )

    plan = await scout.plan(state)

    assert plan.metadata["reasoning_style"] == "plan_and_execute"
    assert "stubbed reasoning summary" == plan.metadata["reasoning_summary"]
    assert "无人机路径规划 survey benchmark" in plan.queries
    assert len(llm_stub.calls) == 1


@pytest.mark.asyncio
async def test_research_knowledge_agent_uses_plan_and_execute_queries_when_not_react() -> None:
    llm_stub = PlanAndExecuteLLMStub(queries=["无人机路径规划 survey benchmark evidence"])
    agent = ResearchKnowledgeAgent(llm_adapter=llm_stub)
    task = ResearchTask(
        task_id="task_reasoning_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-18T00:00:00+00:00",
        updated_at="2026-04-18T00:00:00+00:00",
        sources=["arxiv"],
    )
    report = ResearchReport(
        report_id="report_reasoning_1",
        task_id=task.task_id,
        topic=task.topic,
        generated_at="2026-04-18T00:00:00+00:00",
        markdown="stub",
        highlights=["survey papers compare evaluation settings"],
    )
    state = SimpleNamespace(
        question="哪篇论文最值得优先阅读？",
        task=task,
        report=report,
        papers=[
            PaperCandidate(
                paper_id="paper_1",
                title="UAV Survey",
                authors=["Alice"],
                abstract="A survey of UAV path planning methods and benchmarks.",
                year=2026,
                venue="arXiv",
                source="arxiv",
            )
        ],
        request=ResearchTaskAskRequest(question="哪篇论文最值得优先阅读？", reasoning_style="plan_and_solve"),
        execution_context=SimpleNamespace(memory_hints={"preferred_sources": ["survey"]}),
        queries=[],
    )

    queries = await agent.plan_collection_queries(state)

    assert "无人机路径规划 survey benchmark evidence" in queries
    assert len(llm_stub.calls) == 1


@pytest.mark.asyncio
async def test_research_knowledge_agent_skips_plan_and_execute_for_react() -> None:
    llm_stub = PlanAndExecuteLLMStub(queries=["should not be used"])
    agent = ResearchKnowledgeAgent(llm_adapter=llm_stub)
    task = ResearchTask(
        task_id="task_reasoning_2",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-18T00:00:00+00:00",
        updated_at="2026-04-18T00:00:00+00:00",
        sources=["arxiv"],
    )
    state = SimpleNamespace(
        question="哪篇论文最值得优先阅读？",
        task=task,
        report=None,
        papers=[],
        request=ResearchTaskAskRequest(question="哪篇论文最值得优先阅读？", reasoning_style="react"),
        execution_context=SimpleNamespace(memory_hints={"preferred_sources": ["survey"]}),
        queries=[],
    )

    queries = await agent.plan_collection_queries(state)

    assert queries
    assert llm_stub.calls == []


def test_research_knowledge_agent_uses_report_summary_only_as_fallback() -> None:
    agent = ResearchKnowledgeAgent()
    task = ResearchTask(
        task_id="task_manifest_fallback",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-18T00:00:00+00:00",
        updated_at="2026-04-18T00:00:00+00:00",
        sources=["arxiv"],
    )
    report = ResearchReport(
        report_id="report_manifest_fallback",
        task_id=task.task_id,
        topic=task.topic,
        generated_at="2026-04-18T00:00:00+00:00",
        markdown="stub",
        highlights=["初始综述认为 UAV Survey 覆盖 benchmark。"],
        gaps=["缺少全文实验细节。"],
    )
    state = SimpleNamespace(
        task=task,
        report=report,
        papers=[
            PaperCandidate(
                paper_id="paper_1",
                title="UAV Survey",
                abstract="A survey of UAV path planning methods and benchmarks.",
                source="arxiv",
            )
        ],
        request=ResearchTaskAskRequest(question="效果怎么样"),
        retrieval_hits=[],
        summary_hits=[],
        top_k=10,
    )

    hits = agent.build_collection_manifest(state)

    manifest_kinds = [hit.metadata.get("manifest_kind") for hit in hits]
    assert "paper_card" in manifest_kinds
    assert "report_highlight" in manifest_kinds
    report_hit = next(hit for hit in hits if hit.metadata.get("manifest_kind") == "report_highlight")
    assert report_hit.metadata["evidence_tier"] == "report_summary_fallback"
    assert report_hit.metadata["summary_only"] is True
    assert report_hit.metadata["llm_generated_summary"] is True


def test_research_knowledge_agent_excludes_report_summary_when_primary_evidence_exists() -> None:
    agent = ResearchKnowledgeAgent()
    task = ResearchTask(
        task_id="task_manifest_primary",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-18T00:00:00+00:00",
        updated_at="2026-04-18T00:00:00+00:00",
        sources=["arxiv"],
    )
    report = ResearchReport(
        report_id="report_manifest_primary",
        task_id=task.task_id,
        topic=task.topic,
        generated_at="2026-04-18T00:00:00+00:00",
        markdown="stub",
        highlights=["这条初始综述不应在已有全文证据时进入 QA 证据。"],
    )
    state = SimpleNamespace(
        task=task,
        report=report,
        papers=[
            PaperCandidate(
                paper_id="paper_1",
                title="UAV Survey",
                abstract="A survey of UAV path planning methods and benchmarks.",
                source="arxiv",
            )
        ],
        request=ResearchTaskAskRequest(question="效果怎么样"),
        retrieval_hits=[
            RetrievalHit(
                id="doc_hit_1",
                source_type="text_block",
                source_id="doc_1:block_1",
                document_id="doc_1",
                content="Full-text evidence compares UAV planning performance.",
                merged_score=0.9,
            )
        ],
        summary_hits=[],
        top_k=10,
    )

    hits = agent.build_collection_manifest(state)

    manifest_kinds = [hit.metadata.get("manifest_kind") for hit in hits]
    assert "paper_card" in manifest_kinds
    assert "report_highlight" not in manifest_kinds
