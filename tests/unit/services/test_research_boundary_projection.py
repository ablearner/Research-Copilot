from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.sub_manager import TaskEvaluation
from runtime.research.unified_runtime import serialize_agent_exchange


def test_agent_exchange_projects_legacy_and_unified_fields_from_internal_messages() -> None:
    message = AgentMessage(
        task_id="task_1", agent_from="ResearchSupervisorAgent", agent_to="ResearchAnalystAgent",
        task_type="answer_question", instruction="answer", depends_on=["task_0"],
    )
    result = AgentResultMessage(
        task_id="task_1", agent_from="ResearchAnalystAgent", agent_to="ResearchSupervisorAgent",
        task_type="answer_question", status="succeeded", instruction="answer",
        metadata={"unified_action_output": {"unified_input_adapter": "collection_qa_input"}},
        evaluation=TaskEvaluation(passed=False, score=0.0, issues=["insufficient evidence"]),
    )

    projected = serialize_agent_exchange([message], [result])

    assert projected["agent_messages"][0]["task_id"] == "task_1"
    assert projected["agent_results"][0]["status"] == "succeeded"
    assert projected["delegation_plan"][0]["status"] == "failed"
    assert projected["unified_delegation_plan"][0]["status"] == "failed"
    assert projected["unified_agent_results"][0]["action_output"]["unified_input_adapter"] == "collection_qa_input"
