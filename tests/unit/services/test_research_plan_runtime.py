from types import SimpleNamespace

import pytest

from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorDecision
from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import ResearchAgentRunRequest
from domain.schemas.research_plan import PlanStep, ResearchPlan
from domain.schemas.sub_manager import TaskEvaluation
from runtime.research.agent_protocol import ResearchAgentToolContext
from runtime.research.supervisor_graph_runtime_core import ResearchSupervisorGraphRuntime


def runtime_context():
    runtime = ResearchSupervisorGraphRuntime.__new__(ResearchSupervisorGraphRuntime)
    runtime.max_steps = 2
    runtime.manager_agent = ResearchSupervisorAgent()
    runtime._persist_execution_context = lambda context: None
    runtime._workspace_summary = lambda context: ""
    context = ResearchAgentToolContext(request=ResearchAgentRunRequest(message="research"),
                                       research_service=SimpleNamespace(append_runtime_event=lambda **kwargs: None), graph_runtime=None, warnings=[])
    return runtime, context


def running_plan():
    return ResearchPlan(plan_id="p", steps=[
        PlanStep(step_id=1, action="general_answer", status="running", task_id="t1"),
        PlanStep(step_id=2, action="general_answer", depends_on=[1]),
    ]).model_dump(mode="json")


def test_plan_ignores_single_action_stop_and_repeated_action_guard():
    runtime, context = runtime_context()
    context.qa_result = SimpleNamespace()
    state = {"context": context, "execution_plan": running_plan(), "repeated_action_count": 3}
    assert not runtime._should_force_finalize(state)
    assert runtime._should_force_finalize({**state, "stagnant_decision_count": 2})
    assert runtime._should_force_finalize({**state, "exhausted": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [True, False])
async def test_budget_stop_reconciles_last_result_without_fabricating_worker_result(passed):
    runtime, context = runtime_context()
    message = AgentMessage(task_id="t1", agent_from="manager", agent_to="general", task_type="general_answer")
    result = AgentResultMessage(task_id="t1", agent_from="general", agent_to="manager",
                                task_type="general_answer", status="succeeded",
                                evaluation=TaskEvaluation(passed=passed, score=10 if passed else 0))
    state = {"context": context, "execution_plan": running_plan(), "agent_messages": [message],
             "agent_results": [result], "current_step_index": 2,
             "current_decision": ResearchSupervisorDecision(action_name="general_answer", thought="", rationale="", metadata={"active_message": message})}
    state.update(await runtime.supervisor_node(state))
    assert runtime._route_after_supervisor_node(state) == "finalize_node"
    update = await runtime.finalize_node(state)
    assert len(update["agent_results"]) == 1
    plan = update["execution_plan"]
    assert plan["steps"][0]["status"] == ("done" if passed else "failed")
    assert plan["steps"][1]["status"] == "pending"
    assert plan["status"] in {"blocked", "failed"}
    assert "max_steps" in plan["reason"]
    assert context.warnings


def test_projection_includes_pending_steps_and_keeps_stable_ids():
    from domain.schemas.research_context import ResearchContext
    runtime, context = runtime_context()
    runtime.research_service = SimpleNamespace(build_context_slices=lambda *args, **kwargs: {})
    context.execution_context = SimpleNamespace(research_context=ResearchContext())
    plan = running_plan()
    runtime._rebuild_execution_context(context=context, agent_messages=[], agent_results=[],
                                      pending_messages=[], execution_plan=plan)
    projected = context.execution_context.research_context.current_task_plan
    assert [step.status for step in projected] == ["running", "planned"]
    assert projected[0].metadata["task_id"] == "t1"
    assert projected[1].metadata["plan_step_status"] == "pending"
    ids = [step.task_id for step in projected]
    plan["steps"][1].update(status="running", task_id="t2")
    runtime._rebuild_execution_context(context=context, agent_messages=[], agent_results=[],
                                      pending_messages=[], execution_plan=plan)
    assert [step.task_id for step in context.execution_context.research_context.current_task_plan] == ids


def test_response_exposes_plan_and_distinct_call_counts():
    from domain.schemas.research import ResearchAdvancedStrategy, ResearchWorkspaceState
    from runtime.research.result_aggregator import ResearchAgentResultAggregator
    runtime, context = runtime_context()
    runtime._resolve_workspace = lambda **kwargs: ResearchWorkspaceState()
    runtime._resolved_advanced_strategy = lambda *args, **kwargs: ResearchAdvancedStrategy()
    runtime.research_service = SimpleNamespace(persist_runtime_state=lambda **kwargs: None)
    runtime._build_messages = lambda *args, **kwargs: []
    runtime._log_internal_runtime_state = lambda **kwargs: None
    runtime._next_actions = lambda *args, **kwargs: []
    runtime._response_runtime_metadata = lambda: {}
    runtime._active_paper_ids_for_manager = lambda context: []
    runtime.tool_registry = SimpleNamespace(filter_by_toolset=lambda name: [])
    runtime.result_aggregator = ResearchAgentResultAggregator(runtime=runtime)
    plan = running_plan()
    plan.update(status="blocked", reason="step budget exhausted", revision=2, replan_attempts=1)
    result = AgentResultMessage(task_id="t1", agent_from="worker", agent_to="manager", task_type="general_answer", status="succeeded")
    response = runtime._build_response(
        request=context.request, context=context, trace=[], failed=False, exhausted=False,
        agent_messages=[], agent_results=[result, result], planner_runs=5, replan_count=1,
        clarification_request=None, active_plan_id="p", execution_plan=plan, planning_llm_calls=2,
        supervisor_decision_count=6,
    )
    assert response.status == "partial"
    assert response.metadata["execution_plan"] == plan
    assert response.metadata["planning_llm_call_count"] == 2
    assert response.metadata["plan_replan_count"] == 1
    assert response.metadata["worker_call_count"] == 1
    assert response.metadata["supervisor_decision_count"] == 6


@pytest.mark.asyncio
async def test_real_runtime_graph_retains_plan_across_repeated_worker_actions():
    from langgraph.graph import END, START, StateGraph
    from agents.research_supervisor_agent import ResearchSupervisorState
    from runtime.research.agent_protocol import ResearchAgentGraphState
    from runtime.research.plan_executor import create_plan
    runtime, context = runtime_context()
    runtime.max_steps = 4
    calls = []
    plan = create_plan([PlanStep(step_id=i, action="general_answer", instruction=f"part {i}", params={"goal": f"part {i}"})
                        for i in range(1, 4)], allowed_actions={"general_answer"})
    planning_calls = []
    async def generate_structured(**kwargs):
        planning_calls.append(kwargs)
        return kwargs["response_model"].model_validate({
            "action_name": "general_answer", "plan": [step.model_dump() for step in plan.steps],
        })
    runtime.manager_agent.llm_adapter = SimpleNamespace(generate_structured=generate_structured)
    async def decide(state):
        return await runtime.manager_agent.decide_next_action_async(
            ResearchSupervisorState(goal="research", execution_plan=state.get("execution_plan"),
                                    planning_llm_calls=state.get("planning_llm_calls", 0)),
            agent_messages=state.get("agent_messages", []), agent_results=state.get("agent_results", []),
            planner_runs=state.get("planner_runs", 0),
        )
    runtime._decide_next_action = decide
    async def worker(state):
        message = state["current_decision"].metadata["active_message"]
        calls.append(message.task_id)
        result = AgentResultMessage(task_id=message.task_id, agent_from=message.agent_to,
                                    agent_to=message.agent_from, task_type=message.task_type,
                                    status="succeeded", payload={"answer": "A supported answer."}, evaluation=TaskEvaluation(passed=True, score=10))
        return {"agent_results": [*state.get("agent_results", []), result]}
    graph = StateGraph(ResearchAgentGraphState)
    graph.add_node("supervisor", runtime.supervisor_node)
    graph.add_node("worker", worker)
    graph.add_node("finalize", runtime.finalize_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: "finalize" if state.get("exhausted") or state["current_decision"].action_name == "finalize" else "worker")
    graph.add_edge("worker", "supervisor")
    graph.add_edge("finalize", END)
    final = await graph.compile().ainvoke({"context": context, "execution_plan": None, "planning_llm_calls": 0})
    assert len(planning_calls) == 1
    assert final["planning_llm_calls"] == 1
    assert final["supervisor_decision_count"] == 4
    assert len(set(calls)) == 3
    assert final["execution_plan"]["status"] == "completed"
    assert [step["status"] for step in final["execution_plan"]["steps"]] == ["done"] * 3
    assert len(final["agent_results"]) == 3
    assert not context.warnings


@pytest.mark.asyncio
async def test_completion_at_budget_is_successful_and_evaluates_last_worker():
    runtime, context = runtime_context()
    plan = running_plan()
    plan["steps"] = plan["steps"][:1]
    result = AgentResultMessage(task_id="t1", agent_from="worker", agent_to="manager",
                                task_type="general_answer", status="succeeded")
    evaluated_calls = []
    def evaluate(**kwargs):
        evaluated_calls.append(kwargs["results"])
        return [result.model_copy(update={"evaluation": TaskEvaluation(passed=True, score=10)})]
    runtime.manager_agent._evaluate_results = evaluate
    update = await runtime.finalize_node({"context": context, "execution_plan": plan,
                                         "agent_results": [result], "exhausted": True})
    assert evaluated_calls == [[result]]
    assert update["execution_plan"]["status"] == "completed"
    assert not update.get("exhausted", True)
    assert not context.warnings


@pytest.mark.asyncio
async def test_guardrail_decision_is_counted_but_budget_check_is_not():
    runtime, context = runtime_context()
    state = {"context": context, "execution_plan": running_plan(),
             "stagnant_decision_count": 2, "supervisor_decision_count": 3}
    update = await runtime.supervisor_node(state)
    assert update["current_decision"].action_name == "finalize"
    assert update["supervisor_decision_count"] == 4
    budget = await runtime.supervisor_node({**state, "current_step_index": runtime.max_steps})
    assert budget == {"exhausted": True}


def test_projection_uses_dispatched_acceptance_schema():
    from domain.schemas.research_context import ResearchContext
    runtime, context = runtime_context()
    runtime.research_service = SimpleNamespace(build_context_slices=lambda *args, **kwargs: {})
    context.execution_context = SimpleNamespace(research_context=ResearchContext())
    message = AgentMessage(task_id="t1", agent_from="manager", agent_to="worker",
                           task_type="general_answer", expected_output_schema={"required_keys": ["answer"]})
    runtime._rebuild_execution_context(context=context, agent_messages=[message], agent_results=[],
                                      pending_messages=[], execution_plan=running_plan())
    assert context.execution_context.research_context.current_task_plan[0].expected_output_schema == message.expected_output_schema


@pytest.mark.asyncio
async def test_fast_route_decision_increments_runtime_counter_without_planner_run():
    runtime, context = runtime_context()
    async def fast_decision(state):
        return ResearchSupervisorDecision(action_name="general_answer", thought="", rationale="",
                                          metadata={"fast_route": True})
    runtime._decide_next_action = fast_decision
    update = await runtime.supervisor_node({"context": context, "planner_runs": 0,
                                           "supervisor_decision_count": 0})
    assert update["supervisor_decision_count"] == 1
    assert update.get("planner_runs", 0) == 0
