from types import SimpleNamespace

import pytest

from agents.research_knowledge_agent import ResearchKnowledgeAgent
from agents.research_writer_agent import ResearchWriterAgent
from tools.research.collection_qa_capability import (
    ResearchCollectionAnswerCapability,
    ResearchCollectionKnowledgeCapability,
)


@pytest.mark.asyncio
async def test_legacy_knowledge_planning_delegates_with_three_query_limit(monkeypatch):
    seen = {}

    async def fake_plan(self, state, *, max_queries=5, include_cross_lingual=True):
        seen.update(max_queries=max_queries, include_cross_lingual=include_cross_lingual)
        return ["one", "two", "three"]

    monkeypatch.setattr(ResearchCollectionKnowledgeCapability, "plan_collection_queries", fake_plan)
    queries = await ResearchKnowledgeAgent().plan_collection_queries(SimpleNamespace())

    assert queries == ["one", "two", "three"]
    assert seen == {"max_queries": 3, "include_cross_lingual": False}


@pytest.mark.asyncio
async def test_legacy_writer_answer_delegates_with_writer_metadata(monkeypatch):
    seen = {}
    expected = object()

    async def fake_answer(self, *, graph_runtime, state, primary_agents, writer_agent=None):
        seen.update(graph_runtime=graph_runtime, state=state, primary_agents=primary_agents, writer_agent=writer_agent)
        return expected

    monkeypatch.setattr(ResearchCollectionAnswerCapability, "answer_collection_question", fake_answer)
    runtime, state = object(), object()
    result = await ResearchWriterAgent().answer_collection_question(
        graph_runtime=runtime, state=state, primary_agents=["ResearchWriterAgent"]
    )

    assert result is expected
    assert seen == {
        "graph_runtime": runtime,
        "state": state,
        "primary_agents": ["ResearchWriterAgent"],
        "writer_agent": "ResearchWriterAgent",
    }


def test_legacy_manifest_delegates_and_keeps_legacy_agent_attribution(monkeypatch):
    seen = {}

    def fake_manifest(self, state, *, knowledge_agent=None):
        seen["knowledge_agent"] = knowledge_agent
        return ["hit"]

    monkeypatch.setattr(ResearchCollectionKnowledgeCapability, "build_collection_manifest", fake_manifest)
    result = ResearchKnowledgeAgent().build_collection_manifest(SimpleNamespace())

    assert result == ["hit"]
    assert seen == {"knowledge_agent": "ResearchKnowledgeAgent"}

@pytest.mark.asyncio
async def test_legacy_query_plan_stays_at_three_without_crosslingual_expansion():
    state = SimpleNamespace(
        question="这篇论文的方法和效果如何？",
        task=SimpleNamespace(topic="path planning"),
        report=None,
        papers=[SimpleNamespace(title="UAV Path Planning Survey")],
        execution_context=SimpleNamespace(memory_hints={"current_task_intent": "prioritize reading"}),
        request=SimpleNamespace(reasoning_style="react"),
    )

    legacy = await ResearchKnowledgeAgent().plan_collection_queries(state)
    canonical = await ResearchCollectionKnowledgeCapability().plan_collection_queries(state)

    assert len(legacy) == 3
    assert all("UAV Path Planning Survey" not in query for query in legacy)
    assert len(canonical) == 5
    assert any("UAV Path Planning Survey" in query for query in canonical)


@pytest.mark.asyncio
async def test_legacy_knowledge_retrieval_uses_capability_scope_filters(monkeypatch):
    from domain.schemas.retrieval import RetrievalHit
    from tools.research.knowledge_access import ResearchKnowledgeAccess

    hit = RetrievalHit(id="hit", source_type="text_block", source_id="block", content="evidence")
    seen = {}

    class Access:
        async def retrieve(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(retrieval_result=SimpleNamespace(hits=[hit]))

    monkeypatch.setattr(ResearchKnowledgeAccess, "from_runtime", lambda runtime: Access())
    state = SimpleNamespace(
        task=SimpleNamespace(task_id="task", topic="planning"),
        request=SimpleNamespace(metadata={"qa_scope_mode": "selected_papers", "selected_paper_ids": ["paper"]}),
        execution_context=SimpleNamespace(session_id="session", memory_hints={"focus": "methods"}),
        document_ids=["doc"],
        top_k=4,
    )

    hits = await ResearchKnowledgeAgent().retrieve_collection_evidence(
        graph_runtime=object(), state=state, query="methods"
    )

    assert hits == [hit]
    assert seen["document_ids"] == ["doc"]
    assert seen["session_id"] == "session"
    assert seen["filters"]["selected_paper_ids"] == ["paper"]
    assert seen["filters"]["qa_scope_mode"] == "selected_papers"


@pytest.mark.asyncio
async def test_legacy_writer_preserves_language_citations_and_selected_analysis(monkeypatch):
    from domain.schemas.api import QAResponse
    from domain.schemas.evidence import EvidenceBundle
    from domain.schemas.retrieval import RetrievalHit
    from tools.research.knowledge_access import ResearchKnowledgeAccess

    seen = {}

    class Access:
        async def answer_with_evidence(self, **kwargs):
            seen.update(kwargs)
            return QAResponse(
                question=kwargs["question"], answer="Insufficient evidence.",
                evidence_bundle=EvidenceBundle(), metadata={},
            )

    class PaperAnalysis:
        async def analyze_async(self, **kwargs):
            return SimpleNamespace(
                answer="该论文的方法值得阅读。", key_points=["方法清楚"],
                recommended_paper_ids=["paper"],
                model_dump=lambda **_: {"answer": "该论文的方法值得阅读。"},
            )

    monkeypatch.setattr(ResearchKnowledgeAccess, "from_runtime", lambda runtime: Access())
    hit = RetrievalHit(
        id="hit", source_type="text_block", source_id="section", document_id="doc",
        content="The method improves planning.", merged_score=0.9,
        metadata={"paper_id": "paper", "section": "Methods"},
    )
    state = SimpleNamespace(
        question="这篇论文的方法如何？", original_question="这篇论文的方法如何？",
        task=SimpleNamespace(task_id="task", topic="planning"), report=None,
        papers=[SimpleNamespace(paper_id="paper", title="UAV Study", metadata={"document_id": "doc"})],
        request=SimpleNamespace(
            paper_ids=["paper"], metadata={"qa_scope_mode": "metadata_only"},
            reasoning_style="cot", skill_name=None,
        ),
        execution_context=None, document_ids=["doc"], top_k=4,
        retrieval_hits=[hit], summary_hits=[], manifest_hits=[],
    )

    response = await ResearchWriterAgent(paper_analysis_tool=PaperAnalysis()).answer_collection_question(
        graph_runtime=object(), state=state, primary_agents=["ResearchWriterAgent"]
    )

    assert response.answer == "该论文的方法值得阅读。"
    assert seen["preference_context"]["answer_language"] == "zh-CN"
    assert seen["metadata"]["writer_agent"] == "ResearchWriterAgent"
    assert response.metadata["citations"][0]["label"] == "P1"
    assert response.metadata["citations"][0]["section"] == "Methods"
    assert response.metadata["selected_paper_analysis"]["answer"] == response.answer
    assert response.metadata["recommended_paper_ids"] == ["paper"]
    assert "当前结论仍需补充更多直接证据" in response.metadata["extended_analysis"]


def test_legacy_decision_delegates_to_knowledge_capability(monkeypatch):
    monkeypatch.setattr(
        ResearchCollectionKnowledgeCapability, "decide",
        lambda self, state: ("finish", "capability decision"),
        raising=False,
    )
    result = ResearchKnowledgeAgent().decide(SimpleNamespace(queries=[]))
    assert result == ("finish", "capability decision")
