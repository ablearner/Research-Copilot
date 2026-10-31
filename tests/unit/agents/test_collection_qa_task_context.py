import json
from types import SimpleNamespace

import pytest

from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.worker_task import WorkerTaskInput
from tools.research.collection_qa_capability import ResearchCollectionAnswerCapability
from tools.research.knowledge_access import ResearchKnowledgeAccess


@pytest.mark.asyncio
@pytest.mark.parametrize("skill", [None, "Current skill"])
@pytest.mark.parametrize("stale_fallback", [False, True])
async def test_answer_boundary_receives_bounded_task_without_stale_prompt_context(monkeypatch, skill, stale_fallback):
    seen = {}

    class Access:
        async def answer_with_evidence(self, **kwargs):
            seen.update(kwargs)
            return QAResponse(question=kwargs["question"], answer="Answer", evidence_bundle=EvidenceBundle())

    monkeypatch.setattr(ResearchKnowledgeAccess, "from_runtime", lambda runtime: Access())
    task = WorkerTaskInput(
        task_id="answer-2", action_name="answer_question", instruction="Explain p1",
        context_slice=ResearchContextSlice(selected_papers=["p1"]),
        dependency_summaries=[{"task_id": "prior", "summary": "Bounded prior finding"}],
    ).model_dump(mode="json")
    expert = {"task_context": task, "supervisor_instruction": "Explain p1"}
    if skill:
        expert["skill_context"] = skill
    state = SimpleNamespace(
        question="Explain p1", task=SimpleNamespace(task_id="workspace", topic="agents"),
        report=None, papers=[], document_ids=[], top_k=4,
        retrieval_hits=[], summary_hits=[], manifest_hits=[],
        request=SimpleNamespace(paper_ids=["p1"], reasoning_style="cot", skill_name=None,
                                metadata={"expert_context": expert, **({"skill_context": "STALE skill fallback"} if stale_fallback else {})}),
        execution_context=SimpleNamespace(
            session_context={"history": "UNBOUNDED unrelated session"},
            task_context={"history": "UNBOUNDED unrelated task"},
            memory_hints={"raw": "UNBOUNDED unrelated memory"},
            preference_context={"citation_style": "numeric", "raw": "UNBOUNDED preferences",
                                "skill_context": "STALE skill", "active_skill_names": ["STALE"],
                                "supervisor_instruction": "STALE instruction",
                                "expert_context": {"skill_context": "STALE nested"}},
        ),
    )
    await ResearchCollectionAnswerCapability().answer_collection_question(
        graph_runtime=object(), state=state, primary_agents=["ResearchQAAgent"],
    )
    assert seen["preference_context"]["expert_context"] == expert
    assert seen["task_context"]["dependency_summaries"] == task["dependency_summaries"]
    assert "UNBOUNDED" not in json.dumps(seen, default=str)
    assert "STALE" not in json.dumps(seen, default=str)
    assert seen["preference_context"]["citation_style"] == "numeric"
    assert seen["preference_context"].get("skill_context") == skill


@pytest.mark.asyncio
@pytest.mark.parametrize("skill", [None, "Current skill"])
async def test_metadata_only_fallback_analysis_receives_current_task_without_workspace_report(monkeypatch, skill):
    from domain.schemas.research import PaperCandidate
    from tools.research.paper_analysis import PaperAnalysisTool

    calls = []

    class LLM:
        async def generate_structured(self, *, prompt, input_data, response_model):
            calls.append(input_data)
            return response_model.model_validate({"answer": "Scoped analysis", "focus": "analysis"})

    class Access:
        async def answer_with_evidence(self, **kwargs):
            return QAResponse(question=kwargs["question"], answer="No evidence", evidence_bundle=EvidenceBundle())

    monkeypatch.setattr(ResearchKnowledgeAccess, "from_runtime", lambda runtime: Access())
    task = WorkerTaskInput(
        task_id="qa", action_name="answer_question", instruction="Explain p1",
        context_slice=ResearchContextSlice(selected_papers=["p1"]),
        dependency_summaries=[{"task_id": "prior", "summary": "Bounded finding"}],
    ).model_dump(mode="json")
    state = SimpleNamespace(
        question="Explain p1", task=SimpleNamespace(task_id="workspace", topic="agents"),
        report=SimpleNamespace(report_id="old", highlights=["UNRELATED p2 conclusion"]),
        papers=[PaperCandidate(paper_id="p1", title="p1", source="arxiv")], document_ids=[], top_k=4,
        retrieval_hits=[], summary_hits=[], manifest_hits=[], execution_context=None,
        request=SimpleNamespace(paper_ids=["p1"], reasoning_style="cot", skill_name=None,
            metadata={"qa_scope_mode": "metadata_only", "expert_context": {
                "task_context": task, "supervisor_instruction": "Explain p1", **({"skill_context": skill} if skill else {}),
            }}),
    )
    response = await ResearchCollectionAnswerCapability(paper_analysis_tool=PaperAnalysisTool(llm_adapter=LLM())).answer_collection_question(
        graph_runtime=object(), state=state, primary_agents=["ResearchQAAgent"],
    )
    assert response.answer == "Scoped analysis"
    assert calls[0]["task_context"] == task
    assert calls[0].get("skill_context") == skill
    assert "UNRELATED" not in json.dumps(calls)
