from domain.schemas.agent_message import AgentMessage, AgentResultMessage


def test_agent_message_supports_protocol_aliases() -> None:
    message = AgentMessage.model_validate(
        {
            "task_id": "task-1",
            "from": "ManagerAgent",
            "to": "ResearchKnowledgeAgent",
            "task_type": "search",
            "instruction": "search recent papers",
            "depends_on": ["seed-task"],
            "retry_count": 1,
        }
    )

    dumped = message.model_dump(mode="json", by_alias=True)

    assert message.agent_from == "ManagerAgent"
    assert message.agent_to == "ResearchKnowledgeAgent"
    assert message.dependencies == ["seed-task"]
    assert dumped["from"] == "ManagerAgent"
    assert dumped["to"] == "ResearchKnowledgeAgent"
    assert dumped["depends_on"] == ["seed-task"]


def test_agent_result_message_supports_legacy_dependency_field() -> None:
    result = AgentResultMessage.model_validate(
        {
            "task_id": "task-2",
            "agent_from": "ResearchKnowledgeAgent",
            "agent_to": "ManagerAgent",
            "task_type": "review",
            "status": "succeeded",
            "dependencies": ["task-1"],
        }
    )

    assert result.depends_on == ["task-1"]
    assert result.dependencies == ["task-1"]
