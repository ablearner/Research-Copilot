from domain.schemas.research import ImportPapersRequest, ResearchAgentRunRequest, ResearchTaskAskRequest
from agents.research_qa_agent import DEFAULT_AGENT_REASONING_STYLE, normalize_reasoning_style


def test_agent_reasoning_style_defaults_to_react() -> None:
    assert DEFAULT_AGENT_REASONING_STYLE == "react"
    assert normalize_reasoning_style(None) == "react"
    assert normalize_reasoning_style("") == "react"
    assert normalize_reasoning_style("chain-of-thought") == "react"
    assert normalize_reasoning_style("cot") == "react"


def test_research_requests_default_to_cot_schema_value() -> None:
    assert ResearchAgentRunRequest(message="调研无人机路径规划").reasoning_style == "cot"
    assert ResearchTaskAskRequest(question="效果怎么样").reasoning_style == "cot"
    assert ImportPapersRequest(task_id="task_1").reasoning_style == "cot"
