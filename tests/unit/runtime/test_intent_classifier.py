from runtime.research.intent_classifier import should_force_finalize


def _base_force_finalize_kwargs(**overrides):
    data = {
        "exhausted": False,
        "stagnant_count": 0,
        "repeated_count": 0,
        "mode": "auto",
        "has_qa_result": False,
        "latest_task_type": None,
        "latest_status": None,
        "latest_next_actions": set(),
        "workflow_constraint": "",
        "has_preference_result": False,
        "advanced_action": None,
        "has_paper_analysis": False,
        "new_topic_detected": False,
        "has_task_response": False,
        "has_report": False,
        "auto_import": False,
        "has_message": True,
        "import_attempted": False,
        "has_import_result": False,
    }
    data.update(overrides)
    return data


def test_answer_question_with_qa_result_is_terminal_even_when_recovery_suggested() -> None:
    assert should_force_finalize(
        **_base_force_finalize_kwargs(
            latest_task_type="answer_question",
            latest_status="skipped",
            has_qa_result=True,
            latest_next_actions={"answer_question"},
        )
    )


def test_failed_answer_question_is_terminal_to_avoid_retry_loop() -> None:
    assert should_force_finalize(
        **_base_force_finalize_kwargs(
            latest_task_type="answer_question",
            latest_status="failed",
            has_qa_result=False,
            latest_next_actions={"answer_question"},
        )
    )
