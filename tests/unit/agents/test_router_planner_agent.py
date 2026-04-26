from planners.function_calling import RetrievalPlan


def test_retrieval_plan_accepts_wrapped_payload() -> None:
    plan = RetrievalPlan.model_validate(
        {
            "retrieval_plan": {
                "modes": ["vector", "graph", "summary"],
                "retrieval_focus": "cross-document analysis",
            },
            "question": "What changed after upload?",
            "reasoning_summary": "Use hybrid retrieval with graph summary support.",
        }
    )

    assert plan.query == "What changed after upload?"
    assert plan.reasoning_summary == "Use hybrid retrieval with graph summary support."
    assert plan.modes == ["vector", "graph", "summary"]
    assert plan.retrieval_focus == "cross-document analysis"
