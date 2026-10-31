import pytest
from langgraph.graph import END, START, StateGraph

from adapters.llm.base import BaseLLMAdapter
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.agent_message import AgentResultMessage
from domain.schemas.sub_manager import TaskEvaluation
from runtime.research.agent_protocol.base import ResearchAgentGraphState


class PlanLLM(BaseLLMAdapter):
    def __init__(self, *outputs):
        super().__init__(max_retries=0)
        self.outputs = list(outputs)
        self.calls = 0
        self.inputs = []

    async def _generate_structured(self, prompt, input_data, response_model):
        self.inputs.append(input_data)
        output = self.outputs[self.calls]
        self.calls += 1
        return response_model.model_validate(output)

    async def _analyze_image_structured(self, **kwargs):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, **kwargs):
        raise NotImplementedError

    async def _extract_graph_triples(self, **kwargs):
        raise NotImplementedError


def proposal():
    return {
        "action_name": "general_answer",  # Deliberately differs from authoritative plan.
        "payload": {"goal": "wrong top-level payload"},
        "plan": [
            {"step_id": 1, "action": "search_literature", "instruction": "Find agent papers",
             "params": {"goal": "agent papers"}},
            {"step_id": 2, "action": "analyze_papers", "depends_on": [1],
             "instruction": "Compare experiments", "params": {"paper_ids": ["p1", "p2"]}},
            {"step_id": 3, "action": "write_review", "depends_on": [2],
             "instruction": "Write the comparison"},
        ],
    }


def result_for(decision, *, passed=True, status="succeeded", task_id=None):
    message = decision.metadata["active_message"]
    return AgentResultMessage(
        task_id=task_id or message.task_id, agent_from=message.agent_to,
        agent_to=message.agent_from, task_type=message.task_type, status=status,
        payload={"paper_count": 2, "answer": "Compared methods and experiments", "report_id": "r1",
                 "report_word_count": 20, "evidence_count": 2, "paper_ids": ["p1", "p2"],
                 "analyzed_paper_ids": ["p1", "p2"], "analysis_focus": "compare"},
        evaluation=TaskEvaluation(passed=passed, score=10 if passed else 0,
                                  issues=[] if passed else ["missing_evidence"]),
    )


def test_graph_retains_plan_between_nodes():
    observed = {}
    graph = StateGraph(ResearchAgentGraphState)
    graph.add_node("seed", lambda state: {"execution_plan": {"plan_id": "p", "steps": []}})
    def observe(state):
        observed.update(state)
        return {}
    graph.add_node("observe", observe)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "observe")
    graph.add_edge("observe", END)
    graph.compile().invoke({"planner_runs": 0})
    assert observed.get("execution_plan", {}).get("plan_id") == "p"


@pytest.mark.asyncio
async def test_plan_first_step_owns_action_params_and_instruction():
    llm = PlanLLM(proposal())
    manager = ResearchSupervisorAgent(llm_adapter=llm)
    decision = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    assert decision.action_name == "search_literature"
    assert decision.action_input["goal"] == "agent papers"
    assert decision.metadata["active_message"].instruction == "Find agent papers"
    plan = decision.metadata["state_update"]["execution_plan"]
    assert plan["steps"][0]["task_id"] == decision.metadata["active_message"].task_id


@pytest.mark.asyncio
async def test_real_graph_executes_three_steps_with_one_planning_call():
    llm = PlanLLM(proposal())
    manager = ResearchSupervisorAgent(llm_adapter=llm)
    calls = []
    async def decide(state):
        decision = await manager.decide_next_action_async(
            ResearchSupervisorState(goal="research", execution_plan=state.get("execution_plan"),
                                    candidate_papers=[{"paper_id": "p1"}, {"paper_id": "p2"}],
                                    planning_llm_calls=state.get("planning_llm_calls", 0)),
            agent_messages=state.get("agent_messages", []),
            agent_results=state.get("agent_results", []),
            planner_runs=state.get("planner_runs", 0),
            replan_count=state.get("replan_count", 0),
        )
        return {**decision.metadata.get("state_update", {}), "current_decision": decision}
    def execute(state):
        decision = state["current_decision"]
        calls.append(decision.action_name)
        return {"agent_results": [*state.get("agent_results", []), result_for(decision)]}
    graph = StateGraph(ResearchAgentGraphState)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda state: END if state["current_decision"].action_name == "finalize" else "execute")
    graph.add_edge("execute", "decide")
    result = await graph.compile().ainvoke({"planner_runs": 0})
    assert calls == ["search_literature", "analyze_papers", "write_review"]
    assert llm.calls == result["planning_llm_calls"] == 1
    assert result["execution_plan"]["status"] == "completed"
    assert all(step["status"] == "done" for step in result["execution_plan"]["steps"])


async def next_decision(manager, previous, results):
    update = previous.metadata["state_update"]
    return await manager.decide_next_action_async(
        ResearchSupervisorState(goal="research", execution_plan=update["execution_plan"],
                                candidate_papers=[{"paper_id": "p1"}, {"paper_id": "p2"}],
                                planning_llm_calls=update["planning_llm_calls"]),
        agent_messages=update["agent_messages"], agent_results=results,
        planner_runs=update["planner_runs"], replan_count=update["replan_count"],
    )


@pytest.mark.asyncio
async def test_replan_retains_completed_steps_and_failure_evidence():
    replacement = {"action_name": "analyze_papers", "plan": [
        {"step_id": 2, "action": "analyze_papers", "depends_on": [1],
         "instruction": "Compare only supported results"},
        {"step_id": 3, "action": "write_review", "depends_on": [2]},
    ]}
    llm = PlanLLM(proposal(), replacement)
    manager = ResearchSupervisorAgent(llm_adapter=llm)
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    first_result = result_for(first)
    second = await next_decision(manager, first, [first_result])
    second_result = result_for(second, passed=False)
    retried = await next_decision(manager, second, [first_result, second_result])
    plan = retried.metadata["state_update"]["execution_plan"]
    assert plan["revision"] == 2
    assert plan["steps"][0]["status"] == "done"
    assert retried.action_name == "analyze_papers"
    assert retried.metadata["active_message"].task_id != second.metadata["active_message"].task_id
    assert "missing_evidence" in str(llm.inputs[-1]["state"]["execution_plan"])
    assert plan["plan_id"] == first.metadata["plan_id"]
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_replan_limit_returns_partial_plan_without_more_model_calls():
    llm = PlanLLM(proposal())
    manager = ResearchSupervisorAgent(llm_adapter=llm, max_plan_replans=0)
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    stopped = await next_decision(manager, first, [result_for(first, passed=False)])
    assert stopped.action_name == "finalize"
    assert "limit" in stopped.stop_reason
    assert stopped.metadata["state_update"]["execution_plan"]["status"] == "blocked"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_unrelated_result_cannot_complete_running_step():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM(proposal()))
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    before = first.metadata["state_update"]["execution_plan"]
    stopped = await next_decision(manager, first, [result_for(first, task_id="unrelated-compression")])
    assert stopped.action_name == "finalize"
    after = stopped.metadata["state_update"]["execution_plan"]
    assert after["steps"][0]["status"] == "running"
    assert before["status"] == "active"  # Snapshot was not mutated.


@pytest.mark.asyncio
async def test_success_without_search_results_fails_acceptance():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM(proposal()), max_plan_replans=0)
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    result = result_for(first).model_copy(update={"evaluation": None, "payload": {"paper_count": 0}})
    stopped = await next_decision(manager, first, [result])
    assert stopped.action_name == "finalize"
    assert "insufficient_paper_count" in stopped.stop_reason


@pytest.mark.asyncio
async def test_worker_pass_does_not_bypass_plan_acceptance():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM(proposal()), max_plan_replans=0)
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    result = result_for(first).model_copy(update={"payload": {"paper_count": 0}})
    stopped = await next_decision(manager, first, [result])
    assert stopped.action_name == "finalize"
    assert "insufficient_paper_count" in stopped.stop_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unknown_action", "bad_dependency", "discovery_constraint", "bad_params", "bad_threshold", "bad_fields", "bad_top_k"])
async def test_invalid_plan_stops_before_dispatch(change):
    payload = proposal()
    state = ResearchSupervisorState(goal="research")
    if change == "unknown_action":
        payload["plan"][1]["action"] = "nonexistent"
    elif change == "bad_dependency":
        payload["plan"][1]["depends_on"] = [99]
    elif change == "discovery_constraint":
        state.workflow_constraint = "discovery_only"
    elif change == "bad_params":
        payload["plan"][1]["params"] = {"paper_ids": "p1"}
    elif change == "bad_threshold":
        payload["plan"][1]["expected_output_schema"] = {"min_paper_count": "many"}
    elif change == "bad_fields":
        payload["plan"][1]["expected_output_schema"] = {"required_fields": "paper_ids"}
    else:
        payload["plan"][1]["params"] = {"top_k": "many"}
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM(payload))
    stopped = await manager.decide_next_action_async(state)
    assert stopped.action_name == "finalize"
    assert "Invalid execution plan" in stopped.stop_reason
    assert "active_message" not in stopped.metadata


@pytest.mark.asyncio
async def test_plan_analysis_requires_all_selected_papers():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM(proposal()), max_plan_replans=0)
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal="research"))
    first_result = result_for(first)
    second = await next_decision(manager, first, [first_result])
    incomplete = result_for(second)
    incomplete.payload["analyzed_paper_ids"] = ["p1"]
    stopped = await next_decision(manager, second, [first_result, incomplete])
    assert stopped.action_name == "finalize"
    assert "missing_paper_ids:p2" in stopped.stop_reason


@pytest.mark.asyncio
async def test_unresolved_plan_scope_does_not_expand_to_whole_collection():
    llm = PlanLLM({"action_name": "analyze_papers", "plan": [
        {"step_id": 1, "action": "analyze_papers", "params": {"paper_ids": ["missing-id"]}},
    ]})
    manager = ResearchSupervisorAgent(llm_adapter=llm)
    decision = await manager.decide_next_action_async(ResearchSupervisorState(
        goal="compare", candidate_papers=[{"paper_id": "real-paper", "title": "Known paper"}],
    ))
    assert decision.action_name == "finalize"
    assert "Unresolved" in decision.stop_reason
    assert "active_message" not in decision.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize('payload,status,scope,accepted', [
    ({'paper_ids': ['p1', 'p2'], 'synced_count': 1, 'failed_count': 1}, 'succeeded', True, False),
    ({'paper_ids': ['p1', 'p2'], 'synced_count': 1, 'failed_count': 0}, 'succeeded', True, False),
    ({'paper_ids': ['p1', 'p2'], 'synced_count': 2, 'failed_count': 1}, 'succeeded', True, False),
    ({'reason': 'missing_task'}, 'skipped', False, False),
    ({'paper_ids': ['p1', 'p2'], 'synced_count': 2, 'failed_count': 0}, 'succeeded', True, True),
    ({'paper_ids': ['p1', 'p2'], 'synced_count': 2, 'failed_count': 0}, 'skipped', True, True),
])
async def test_sync_plan_acceptance_requires_complete_outputs(payload, status, scope, accepted):
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM({'action_name': 'sync_to_zotero', 'plan': [
        {'step_id': 1, 'action': 'sync_to_zotero', 'allow_skip': True,
         'params': {'paper_ids': ['p1', 'p2']} if scope else {},
         'expected_output_schema': {'min_synced_count': 0, 'require_no_failures': False}},
    ]}), max_plan_replans=0)
    first = await manager.decide_next_action_async(ResearchSupervisorState(
        goal='save to Zotero', candidate_papers=[{'paper_id': 'p1'}, {'paper_id': 'p2'}],
    ))
    returned = result_for(first, status=status).model_copy(update={'payload': payload})
    stopped = await next_decision(manager, first, [returned])
    assert stopped.action_name == 'finalize'
    assert (stopped.metadata['state_update']['execution_plan']['status'] == 'completed') is accepted


@pytest.mark.asyncio
@pytest.mark.parametrize('params,schema', [
    ({'qa_route': []}, {}), ({'qa_route': {}}, {}),
    ({}, {'required_paper_ids': 'p1'}), ({}, {'required_paper_ids': [['p1']]}),
    ({}, {'required_paper_ids': ['']}), ({}, {'required_fields': ['']}),
    ({}, {'min_paper_count': True}), ({}, {'min_synced_count': -1}),
    ({}, {'min_unsupported_count': 1}), ({}, {'require_unknown': True}),
    ({}, {'require_no_failures': 'true'}), ({'top_k': True}, {}),
    ({'days_back': []}, {}), ({'import_top_k': -1}, {}),
])
async def test_malformed_plan_criteria_stop_before_dispatch(params, schema):
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM({'action_name': 'search_literature', 'plan': [
        {'step_id': 1, 'action': 'search_literature', 'params': params,
         'expected_output_schema': schema},
    ]}))
    stopped = await manager.decide_next_action_async(ResearchSupervisorState(goal='research'))
    assert stopped.action_name == 'finalize'
    assert 'Invalid execution plan' in stopped.stop_reason
    assert 'active_message' not in stopped.metadata


@pytest.mark.asyncio
async def test_plan_accepts_legacy_descriptive_schema_fields():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM({'action_name': 'search_literature', 'plan': [
        {'step_id': 1, 'action': 'search_literature',
         'expected_output_schema': {'paper_count': 'int', 'task_id': 'str', 'min_paper_count': 2}},
    ]}))
    first = await manager.decide_next_action_async(ResearchSupervisorState(goal='research'))
    assert first.action_name == 'search_literature'
    assert first.metadata['active_message'].expected_output_schema['min_paper_count'] == 2


@pytest.mark.asyncio
async def test_resolved_scope_preserves_stricter_required_papers_criterion():
    manager = ResearchSupervisorAgent(llm_adapter=PlanLLM({'action_name': 'analyze_papers', 'plan': [
        {'step_id': 1, 'action': 'analyze_papers', 'params': {'paper_ids': ['p1']},
         'expected_output_schema': {'required_paper_ids': ['p2']}},
    ]}))
    first = await manager.decide_next_action_async(ResearchSupervisorState(
        goal='compare', candidate_papers=[{'paper_id': 'p1'}, {'paper_id': 'p2'}],
    ))
    assert set(first.metadata['active_message'].expected_output_schema['required_paper_ids']) == {'p1', 'p2'}


@pytest.mark.parametrize('action', [
    'search_literature', 'write_review', 'import_papers', 'sync_to_zotero',
    'answer_question', 'general_answer', 'recommend_from_preferences', 'analyze_papers',
    'compress_context', 'understand_document', 'supervisor_understand_chart', 'analyze_paper_figures',
])
def test_every_planned_action_rejects_skipped_missing_output(action):
    from agents.research_plan_coordinator import ResearchPlanCoordinator
    from domain.schemas.research_plan import PlanStep
    from tools.research.research_evaluation import ResearchEvaluationTool

    coordinator = ResearchPlanCoordinator(ResearchSupervisorAgent(llm_adapter=PlanLLM()))
    schema = coordinator._acceptance_schema(PlanStep(step_id=1, action=action, allow_skip=True))
    evaluation = ResearchEvaluationTool().evaluate_result(
        task_type=action, result_status='skipped', payload={'reason': 'missing_task'},
        task_instruction='', expected_schema=schema,
    )
    assert not evaluation.passed


@pytest.mark.parametrize('action,payload,accepted', [
    ('compress_context', {'summary_count': 0}, False),
    ('compress_context', {'summary_count': 2}, True),
    ('recommend_from_preferences', {'recommendations': []}, False),
    ('recommend_from_preferences', {'recommendations': [{'paper_id': 'p1', 'title': 'Paper'}]}, True),
    ('answer_question', {'scope_mode': 'metadata_only', 'evidence_count': 0}, False),
    ('answer_question', {'scope_mode': 'metadata_only', 'evidence_count': 0, 'paper_ids': ['p1']}, True),
])
def test_plan_reuse_acceptance_requires_meaningful_outputs(action, payload, accepted):
    from agents.research_plan_coordinator import ResearchPlanCoordinator
    from domain.schemas.research_plan import PlanStep
    from tools.research.research_evaluation import ResearchEvaluationTool

    coordinator = ResearchPlanCoordinator(ResearchSupervisorAgent(llm_adapter=PlanLLM()))
    schema = coordinator._acceptance_schema(PlanStep(step_id=1, action=action, allow_skip=True))
    evaluation = ResearchEvaluationTool().evaluate_result(
        task_type=action, result_status='skipped', payload=payload,
        task_instruction='', expected_schema=schema,
    )
    assert evaluation.passed is accepted


def test_plan_recommendations_require_a_list_of_outputs():
    from agents.research_plan_coordinator import ResearchPlanCoordinator
    from domain.schemas.research_plan import PlanStep
    from tools.research.research_evaluation import ResearchEvaluationTool

    coordinator = ResearchPlanCoordinator(ResearchSupervisorAgent(llm_adapter=PlanLLM()))
    schema = coordinator._acceptance_schema(PlanStep(step_id=1, action='recommend_from_preferences'))
    evaluation = ResearchEvaluationTool().evaluate_result(
        task_type='recommend_from_preferences', result_status='succeeded',
        payload={'recommendations': 'nothing found'}, task_instruction='', expected_schema=schema,
    )
    assert not evaluation.passed
