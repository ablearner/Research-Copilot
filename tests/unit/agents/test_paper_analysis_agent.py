import asyncio
from types import SimpleNamespace

import pytest

import agents.paper_analysis_agent as paper_analysis_module
from agents.paper_analysis_agent import PaperAnalysisAgent
from domain.schemas.research import PaperCandidate
from tools.research import PaperAnalyzer


@pytest.mark.asyncio
async def test_paper_analysis_skips_unqueryable_import_index() -> None:
    class ExplodingKnowledgeAccess:
        async def retrieve(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("retrieve should be skipped for unindexed papers")

        async def query_graph_summary(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("graph summary should be skipped for unindexed papers")

    agent = PaperAnalysisAgent(paper_analysis_skill=PaperAnalyzer())
    paper = PaperCandidate(
        paper_id="paper-1",
        title="Imported But Not Indexed",
        abstract="This paper proposes a GUI agent method.",
        source="arxiv",
        metadata={
            "document_id": "doc-paper-1",
            "index_status": "timeout",
            "index_error": "Indexing timed out after 60s",
        },
    )
    context = SimpleNamespace(
        knowledge_access=ExplodingKnowledgeAccess(),
        graph_runtime=SimpleNamespace(),
        execution_context=None,
        task=SimpleNamespace(task_id="task-1", topic="agents"),
    )

    hits = await agent._collect_analysis_evidence(
        context=context,
        question="详细讲解方法部分",
        papers=[paper],
    )

    assert hits == []


@pytest.mark.asyncio
async def test_paper_analysis_evidence_collection_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowKnowledgeAccess:
        async def retrieve(self, **kwargs):
            await asyncio.sleep(0.5)

        async def query_graph_summary(self, **kwargs):
            await asyncio.sleep(0.5)

    monkeypatch.setattr(paper_analysis_module, "_ANALYSIS_EVIDENCE_TIMEOUT_SECONDS", 0.01)
    agent = PaperAnalysisAgent(paper_analysis_skill=PaperAnalyzer())
    paper = PaperCandidate(
        paper_id="paper-1",
        title="Indexed Paper",
        abstract="This paper proposes a GUI agent method.",
        source="arxiv",
        metadata={"document_id": "doc-paper-1", "index_status": "succeeded"},
    )
    context = SimpleNamespace(
        knowledge_access=SlowKnowledgeAccess(),
        graph_runtime=SimpleNamespace(),
        execution_context=None,
        task=SimpleNamespace(task_id="task-1", topic="agents"),
    )

    hits = await agent._collect_analysis_evidence(
        context=context,
        question="详细讲解方法部分",
        papers=[paper],
    )

    assert hits == []
