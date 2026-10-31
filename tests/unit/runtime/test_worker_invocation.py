import asyncio

import pytest

from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import PaperCandidate, ResearchAgentRunRequest, ResearchTask, ResearchTaskResponse
from domain.schemas.research_context import ResearchContext, ResearchExecutionContext
from runtime.research.agent_protocol import ResearchAgentToolContext, ResearchStateDelta, ResearchToolResult
from runtime.research.supervisor_graph_runtime_core import ResearchSupervisorGraphRuntime
from runtime.research.worker_invocation import prepare_worker_invocation
from tools.research.skill_resolver import ResearchSkillSelection


def context():
    class Resource:
        def __deepcopy__(self, memo):
            raise AssertionError("Live resources must not be copied")

        def append_runtime_event(self, **kwargs):
            pass

    resource = Resource()
    return ResearchAgentToolContext(
        request=ResearchAgentRunRequest(message="research", metadata={"nested": [1]}),
        research_service=resource, graph_runtime=resource, knowledge_access=resource,
        unified_agent_registry=resource,
        task_response=ResearchTaskResponse(
            task=ResearchTask(task_id="workspace", topic="agents", created_at="now", updated_at="now"),
            papers=[PaperCandidate(paper_id=pid, title=pid, source="arxiv") for pid in ("p1", "p2")],
        ),
        execution_context=ResearchExecutionContext(research_context=ResearchContext(research_topic="agents")),
        skill_context="stale skill", supervisor_instruction="stale instruction",
        skill_selection=ResearchSkillSelection(active_skill_names=["stale"], skill_context="stale skill"),
        warnings=[], general_answer_metadata={"items": []},
    )


def decision(task_id="t1", skill=None):
    metadata = {"skill_context": skill, "active_skill_names": [task_id]} if skill else {}
    message = AgentMessage(task_id=task_id, agent_from="manager", agent_to="GeneralAnswerAgent",
                           task_type="general_answer", instruction=f"Instruction {task_id}",
                           depends_on=["prior"], metadata=metadata)
    return ResearchSupervisorDecision(action_name="general_answer", thought="", rationale="",
                                      metadata={"active_message": message})


def test_snapshot_isolates_data_but_shares_runtime_resources():
    parent = context()
    original_decision = decision(skill="task skill")
    original_decision.metadata["state_update"] = {"pending_agent_messages": ["unrelated"]}
    worker, task_decision = prepare_worker_invocation(context=parent, decision=original_decision, agent_results=[])
    assert worker is not parent
    for name in ("research_service", "graph_runtime", "knowledge_access", "unified_agent_registry"):
        assert getattr(worker, name) is getattr(parent, name)
    worker.request.metadata["nested"].append(2)
    worker.execution_context.research_context.research_goals.append("private")
    worker.general_answer_metadata["items"].append("private")
    worker.warnings.append("private")
    worker.papers[0].title = "private"
    worker.task.workspace.key_findings.append("private")
    worker.skill_selection.active_skill_names.append("private")
    task_decision.metadata["active_message"].instruction = "private"
    assert parent.request.metadata["nested"] == [1]
    assert parent.execution_context.research_context.research_goals == []
    assert parent.general_answer_metadata == {"items": []}
    assert parent.warnings == []
    assert parent.papers[0].title == "p1"
    assert parent.task.workspace.key_findings == []
    assert [paper.paper_id for paper in worker.papers] == ["p1", "p2"]
    assert parent.skill_selection.active_skill_names == ["stale"]
    assert original_decision.metadata["active_message"].instruction == "Instruction t1"
    assert "state_update" not in task_decision.metadata
    assert original_decision.metadata["state_update"]["pending_agent_messages"] == ["unrelated"]
    assert worker.expert_context()["task_context"]["task_id"] == "t1"


def test_absent_skill_clears_previous_task_skill_and_instruction():
    parent = context()
    worker, _ = prepare_worker_invocation(context=parent, decision=decision(), agent_results=[])
    assert worker.skill_context is None
    assert worker.skill_selection.active_skill_names == []
    assert worker.supervisor_instruction == "Instruction t1"
    assert "skill_context" not in worker.expert_context()
    assert parent.skill_context == "stale skill"
    fast = ResearchSupervisorDecision(action_name="general_answer", thought="", rationale="")
    fast_worker, _ = prepare_worker_invocation(context=parent, decision=fast, agent_results=[])
    assert fast_worker.supervisor_instruction != "stale instruction"
    assert fast_worker.skill_context is None


def test_dependency_artifact_resolution_returns_detached_full_payload():
    parent = context()
    result = AgentResultMessage(task_id="prior", agent_from="a", agent_to="manager",
                                task_type="general_answer", status="succeeded",
                                payload={"answer": "long output " * 1000, "nested": {"items": [1]}})
    worker, _ = prepare_worker_invocation(context=parent, decision=decision(), agent_results=[result])
    ref = worker.worker_input.artifact_refs[0]
    artifact = worker.read_artifact(ref.key)
    assert artifact["answer"] == result.payload["answer"]
    artifact["nested"]["items"].append(2)
    assert worker.read_artifact(ref.key)["nested"]["items"] == [1]
    assert result.payload["nested"]["items"] == [1]
    with pytest.raises(KeyError):
        worker.read_artifact("unknown")


@pytest.mark.asyncio
async def test_overlapping_invocations_do_not_mix_instructions_or_skills():
    parent = context()
    gate = asyncio.Event()
    arrived = []

    async def run(task_id, skill):
        worker, _ = prepare_worker_invocation(context=parent, decision=decision(task_id, skill), agent_results=[])
        arrived.append(task_id)
        if len(arrived) == 2:
            gate.set()
        await gate.wait()
        worker.request.metadata["nested"].append(task_id)
        return worker.supervisor_instruction, worker.skill_context, worker.request.metadata["nested"]

    outputs = await asyncio.gather(run("one", "skill one"), run("two", None))
    assert outputs == [("Instruction one", "skill one", [1, "one"]), ("Instruction two", None, [1, "two"])]
    assert parent.request.metadata["nested"] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["succeeded", "failed"])
async def test_action_node_uses_snapshot_and_commits_only_returned_delta(status):
    parent = context()
    parent.unified_agent_registry = None
    runtime = ResearchSupervisorGraphRuntime.__new__(ResearchSupervisorGraphRuntime)
    runtime._rebuild_execution_context = lambda **kwargs: None
    runtime._persist_state_delta = lambda **kwargs: None
    runtime._workspace_summary = lambda context: ""
    current = decision()
    message = current.metadata["active_message"]
    observed = []

    async def execute(**kwargs):
        worker = kwargs["context"]
        observed.append(worker)
        worker.general_answer = "uncommitted"
        worker.request.metadata["nested"].append("private")
        assert parent.general_answer is None
        return ResearchToolResult(status=status, observation="done", metadata={"answer": "committed"},
                                  state_delta=ResearchStateDelta(general_answer="committed"))

    runtime._execute_agent_run_action = execute
    state = {"context": parent, "current_decision": current, "agent_messages": [message],
             "pending_agent_messages": [message], "agent_results": []}
    update = await runtime._run_action_node(state, "general_answer")
    assert observed[0] is not parent
    assert parent.general_answer == "committed"
    assert parent.request.metadata["nested"] == [1]
    assert parent.supervisor_instruction == "stale instruction"
    assert update["agent_results"][0].context_slice.context_scope == "worker"
    assert update["agent_messages"][0].context_slice.context_scope == "worker"
    assert update["agent_results"][0].payload["summary"] == "committed"
    assert parent.result_artifacts["t1"] == {"answer": "committed"}
    assert observed[0].result_artifacts == {}
    next_decision = decision("t2")
    next_decision.metadata["active_message"].depends_on = ["t1"]
    next_worker, _ = prepare_worker_invocation(
        context=parent, decision=next_decision, agent_results=update["agent_results"],
    )
    assert next_worker.worker_input.dependency_summaries[0]["summary"] == "committed"
    assert next_worker.read_artifact("result:t1") == {"answer": "committed"}
    assert next_worker.result_artifacts == {}
