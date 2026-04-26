from apps.api.routers.ask import AskDocumentRequest, AskFusedRequest
from domain.schemas.research import ImportPapersRequest, ResearchAgentRunRequest, ResearchTaskAskRequest
from reasoning.style import DEFAULT_AGENT_REASONING_STYLE, normalize_reasoning_style


def test_agent_reasoning_style_defaults_to_cot() -> None:
    assert DEFAULT_AGENT_REASONING_STYLE == "cot"
    assert normalize_reasoning_style(None) == "cot"
    assert normalize_reasoning_style("") == "cot"
    assert normalize_reasoning_style("chain-of-thought") == "cot"


def test_research_requests_default_to_cot_reasoning() -> None:
    assert ResearchAgentRunRequest(message="调研无人机路径规划").reasoning_style == "cot"
    assert ResearchTaskAskRequest(question="效果怎么样").reasoning_style == "cot"
    assert ImportPapersRequest(task_id="task_1").reasoning_style == "cot"


def test_document_ask_requests_default_to_cot_reasoning() -> None:
    assert AskDocumentRequest(question="这篇文档讲了什么？").reasoning_style == "cot"
    assert AskFusedRequest(question="这张图说明什么？", image_path="/tmp/chart.png").reasoning_style == "cot"
