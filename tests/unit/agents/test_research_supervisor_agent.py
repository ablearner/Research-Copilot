import pytest

from adapters.llm.base import BaseLLMAdapter
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.agent_message import AgentResultMessage


class ManagerDecisionLLMStub(BaseLLMAdapter):
    def __init__(self, response_payload: dict) -> None:
        super().__init__()
        self.response_payload = response_payload

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        return response_model.model_validate(self.response_payload)

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_research_supervisor_agent_uses_llm_to_route_search_task() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "search_literature",
                "worker_agent": "LiteratureScoutAgent",
                "instruction": "Search and curate literature for the requested topic.",
                "thought": "Need initial discovery before downstream work.",
                "rationale": "A scout should gather the first evidence set.",
                "phase": "plan",
                "payload": {"goal": "调研无人机路径规划方向的自主文献助手论文", "mode": "research"},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            has_task=False,
        )
    )

    assert decision.action_name == "search_literature"
    assert decision.metadata["worker_agent"] == "LiteratureScoutAgent"
    assert decision.metadata["active_message"].task_type == "search_literature"


@pytest.mark.asyncio
async def test_research_supervisor_agent_uses_llm_to_route_selected_paper_analysis_task() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "analyze_papers",
                "worker_agent": "PaperAnalysisAgent",
                "instruction": "Analyze the papers by method and experiment.",
                "thought": "User explicitly asked for selected-paper analysis.",
                "rationale": "A unified paper analysis worker is the best next worker.",
                "phase": "reflect",
                "payload": {
                    "analysis_focus": "compare",
                    "dimensions": ["method", "experiment", "year"],
                },
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="对比这些论文的方法和实验",
            mode="research",
            has_task=True,
            paper_count=3,
            paper_analysis_requested=True,
            analysis_focus="compare",
            comparison_dimensions=["method", "experiment", "year"],
        )
    )

    assert decision.action_name == "analyze_papers"
    assert decision.metadata["worker_agent"] == "PaperAnalysisAgent"
    assert decision.metadata["active_message"].payload["dimensions"] == ["method", "experiment", "year"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_keeps_qa_mode_on_qa_specialist() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "analyze_papers",
                "worker_agent": "PaperAnalysisAgent",
                "instruction": "Analyze the selected paper instead.",
                "thought": "The question mentions paper details.",
                "rationale": "This should be overridden in explicit QA mode.",
                "phase": "reflect",
                "payload": {"analysis_focus": "explain"},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="这篇论文的方法是什么？",
            mode="qa",
            has_task=True,
            imported_document_count=1,
            paper_analysis_requested=True,
            analysis_focus="explain",
        )
    )

    assert decision.action_name == "answer_question"
    assert decision.metadata["worker_agent"] == "ResearchQAAgent"
    assert decision.metadata["active_message"].task_type == "answer_question"
    assert decision.metadata["active_message"].payload["trigger"] == "qa_mode_guardrail"


@pytest.mark.asyncio
async def test_research_supervisor_agent_can_route_general_question_to_general_answer_agent() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "general_answer",
                "worker_agent": "GeneralAnswerAgent",
                "instruction": "Answer the user's general question directly.",
                "thought": "This does not require research workspace tools.",
                "rationale": "A lightweight general-answer worker is the best fit.",
                "phase": "act",
                "payload": {"goal": "Python 里的生成器是什么？", "mode": "auto"},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="Python 里的生成器是什么？",
            mode="auto",
            has_task=False,
        )
    )

    assert decision.action_name == "general_answer"
    assert decision.metadata["worker_agent"] == "GeneralAnswerAgent"
    assert decision.metadata["active_message"].task_type == "general_answer"


@pytest.mark.asyncio
async def test_research_supervisor_agent_supplies_qa_route_when_llm_omits_it() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "answer_question",
                "worker_agent": "ResearchKnowledgeAgent",
                "instruction": "Answer with the current imported evidence.",
                "thought": "The user is asking a scoped follow-up.",
                "rationale": "Research QA should answer this.",
                "phase": "act",
                "payload": {"goal": "请解释这篇论文的方法细节"},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="请解释这篇论文的方法细节",
            mode="qa",
            route_mode="paper_follow_up",
            has_task=True,
            imported_document_count=1,
            active_paper_ids=["paper-1"],
        )
    )

    assert decision.action_name == "answer_question"
    assert decision.metadata["worker_agent"] == "ResearchQAAgent"
    assert decision.metadata["active_message"].agent_to == "ResearchQAAgent"
    assert decision.action_input["routing_authority"] == "supervisor_llm"
    assert decision.action_input["qa_route"] == "document_drilldown"
    assert decision.metadata["active_message"].payload["qa_route"] == "document_drilldown"


@pytest.mark.asyncio
async def test_research_supervisor_agent_replans_qa_route_from_worker_observation() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "write_review",
                "worker_agent": "ResearchWriterAgent",
                "instruction": "Ignored because guardrail should use observation.",
                "thought": "Ignored.",
                "rationale": "Ignored.",
                "phase": "act",
                "payload": {},
            }
        )
    )
    latest_result = AgentResultMessage(
        task_id="qa-1",
        agent_from="ResearchQAAgent",
        agent_to="ResearchSupervisorAgent",
        task_type="answer_question",
        status="skipped",
        payload={
            "observation_envelope": {
                "progress_made": False,
                "suggested_next_actions": ["answer_question"],
                "state_delta": {
                    "preferred_qa_route": "document_drilldown",
                    "qa_recovery_reason": "collection answer was under-supported",
                },
            }
        },
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="请解释这篇论文的方法细节",
            mode="qa",
            has_task=True,
            imported_document_count=1,
        ),
        agent_results=[latest_result],
    )

    assert decision.action_name == "answer_question"
    assert decision.action_input["qa_route"] == "document_drilldown"
    assert decision.action_input["qa_route_source"] == "worker_observation"


@pytest.mark.asyncio
async def test_research_supervisor_agent_can_route_zotero_sync_with_resolved_candidate_scope() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "sync_to_zotero",
                "worker_agent": "ResearchKnowledgeAgent",
                "instruction": "Sync the requested paper to Zotero.",
                "thought": "The user is asking to import a candidate paper into Zotero.",
                "rationale": "Zotero sync is the correct action for citation-library import.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="导入 p1 到 zotero",
            mode="auto",
            has_task=True,
            paper_count=3,
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
            ],
            user_intent={"resolved_paper_ids": ["paper-1"]},
        )
    )

    assert decision.action_name == "sync_to_zotero"
    assert decision.metadata["worker_agent"] == "ResearchKnowledgeAgent"
    assert decision.metadata["active_message"].task_type == "sync_to_zotero"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-1"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_can_route_workspace_import_with_resolved_candidate_scope() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "import_papers",
                "worker_agent": "ResearchKnowledgeAgent",
                "instruction": "Import the requested paper into the local workspace for grounded QA.",
                "thought": "The user wants to ingest the first candidate into the workspace.",
                "rationale": "Workspace import is the correct action for local evidence ingestion.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="把第一篇导入工作区供后续问答",
            mode="auto",
            has_task=True,
            paper_count=3,
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
            ],
            user_intent={"resolved_paper_ids": ["paper-1"]},
        )
    )

    assert decision.action_name == "import_papers"
    assert decision.metadata["worker_agent"] == "ResearchKnowledgeAgent"
    assert decision.metadata["active_message"].task_type == "import_papers"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-1"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_guardrail_routes_sync_to_zotero_intent_without_prompt_mapping() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "write_review",
                "worker_agent": "ResearchWriterAgent",
                "instruction": "Write a report.",
                "thought": "Fallback LLM output that should be ignored by the guardrail.",
                "rationale": "The guardrail should intercept before this.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="导入第一篇论文到 Zotero",
            mode="auto",
            has_task=True,
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
            ],
            user_intent={"intent": "sync_to_zotero", "resolved_paper_ids": ["paper-1"]},
        )
    )

    assert decision.action_name == "sync_to_zotero"
    assert decision.metadata["decision_source"] == "manager_guardrail"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-1"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_guardrail_routes_workspace_import_intent_without_prompt_mapping() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "write_review",
                "worker_agent": "ResearchWriterAgent",
                "instruction": "Write a report.",
                "thought": "Fallback LLM output that should be ignored by the guardrail.",
                "rationale": "The guardrail should intercept before this.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="把第一篇导入工作区供后续问答",
            mode="auto",
            has_task=True,
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
            ],
            user_intent={"intent": "paper_import", "resolved_paper_ids": ["paper-1"]},
        )
    )

    assert decision.action_name == "import_papers"
    assert decision.metadata["decision_source"] == "manager_guardrail"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-1"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_guardrail_routes_figure_question_to_chart_drilldown_qa() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "general_answer",
                "worker_agent": "GeneralAnswerAgent",
                "instruction": "Fallback LLM output that should be ignored by the guardrail.",
                "thought": "Ignored.",
                "rationale": "Ignored.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="第二篇论文的系统框图",
            mode="auto",
            has_task=True,
            imported_document_count=2,
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "Paper Two"},
            ],
            user_intent={"intent": "figure_qa", "resolved_paper_ids": ["paper-2"]},
        )
    )

    assert decision.action_name == "analyze_paper_figures"
    assert decision.metadata["decision_source"] == "manager_guardrail"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-2"]


@pytest.mark.asyncio
async def test_research_supervisor_agent_guardrail_routes_explicit_chart_input() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "finalize",
                "worker_agent": "ResearchSupervisorAgent",
                "instruction": "Stop and ask for clarification.",
                "thought": "Ignored fallback output.",
                "rationale": "The guardrail should use the provided image.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="请解释这张图表达了什么",
            mode="chart",
            has_chart_input=True,
            user_intent={"intent": "figure_qa", "needs_clarification": True},
        )
    )

    assert decision.action_name == "understand_chart"
    assert decision.metadata["worker_agent"] == "ChartAnalysisAgent"
    assert decision.metadata["active_message"].payload["trigger"] == "chart_input_guardrail"


@pytest.mark.asyncio
async def test_research_supervisor_agent_finalizes_after_explicit_chart_input_is_understood() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "clarify_request",
                "worker_agent": "ResearchSupervisorAgent",
                "instruction": "Ask which paper figure the user means.",
                "thought": "Ignored fallback output.",
                "rationale": "The chart was already processed.",
                "phase": "act",
                "payload": {"clarification_question": "你想看哪篇论文里的图？"},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="请解释这张图表达了什么",
            mode="chart",
            has_chart_input=True,
            chart_understood=True,
            user_intent={"intent": "figure_qa", "needs_clarification": True},
        )
    )

    assert decision.action_name == "finalize"
    assert decision.stop_reason == "Chart understanding completed."
    assert decision.metadata["state_update"]["clarification_request"] is None


@pytest.mark.asyncio
async def test_research_supervisor_agent_guardrail_keeps_explicit_figure_scope_without_leaking_active_papers() -> None:
    manager = ResearchSupervisorAgent(
        llm_adapter=ManagerDecisionLLMStub(
            {
                "action_name": "general_answer",
                "worker_agent": "GeneralAnswerAgent",
                "instruction": "Ignored fallback output.",
                "thought": "Ignored.",
                "rationale": "Ignored.",
                "phase": "act",
                "payload": {},
            }
        )
    )

    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(
            goal="第二篇论文的系统框图",
            mode="auto",
            has_task=True,
            imported_document_count=2,
            active_paper_ids=["paper-legacy-1", "paper-legacy-2"],
            candidate_papers=[
                {"index": 1, "paper_id": "paper-1", "title": "Paper One"},
                {"index": 2, "paper_id": "paper-2", "title": "Paper Two"},
            ],
            user_intent={"intent": "figure_qa", "resolved_paper_ids": ["paper-2"]},
        )
    )

    assert decision.action_name == "analyze_paper_figures"
    assert decision.metadata["active_message"].payload["paper_ids"] == ["paper-2"]
    assert decision.metadata["active_message"].payload["paper_scope_source"] == "user_intent_resolver"


def test_research_supervisor_agent_stops_without_llm_instead_of_using_legacy_rule_fallback() -> None:
    manager = ResearchSupervisorAgent()

    decision = manager.decide_next_action(
        ResearchSupervisorState(
            goal="AI",
            mode="research",
            has_task=False,
        )
    )

    assert decision.action_name == "finalize"
    assert decision.metadata["decision_source"] == "guardrail"
    assert "Rule-based planning has been removed" in (decision.stop_reason or "")
