from copy import deepcopy

import pytest

from domain.schemas.agent_message import AgentResultMessage
from domain.schemas.research_plan import PlanStep, ResearchPlan
from runtime.research.plan_executor import (
    advance_plan, create_plan, ready_step, replace_remaining_plan, start_step,
)

ACTIONS = {'search_literature', 'generate_report'}


def step(step_id=1, action='search_literature', **kwargs):
    return PlanStep(step_id=step_id, action=action, **kwargs)


def plan(steps=None):
    return create_plan(steps if steps is not None else [step()], allowed_actions=ACTIONS, plan_id='p')


def result(task_id='t', status='succeeded', passed=None):
    return AgentResultMessage(
        task_id=task_id, agent_from='worker', agent_to='supervisor',
        task_type='search_literature', status=status, payload={'papers': ['artifact']},
        evaluation=None if passed is None else {'passed': passed, 'issues': ['quality'] if not passed else []},
    )


@pytest.mark.parametrize('steps', [
    [], [step(1), step(1)], [step(depends_on=[99])], [step(depends_on=[1])],
    [step(1, depends_on=[2]), step(2, depends_on=[1])],
    [step(action='unknown')], [step(action='finalize')], [step(action='clarify')],
])
def test_invalid_plans_rejected(steps):
    with pytest.raises(ValueError):
        create_plan(steps, allowed_actions=ACTIONS | {'finalize', 'clarify'})


def test_discovery_constraint():
    with pytest.raises(ValueError):
        create_plan([step(action='generate_report')], allowed_actions=ACTIONS, workflow_constraint='discovery_only')


def test_creation_resets_runtime_fields_and_copies_inputs():
    steps = [step(status='done', attempts=7, task_id='injected', params={'nested': []})]
    before = deepcopy(steps)
    created = plan(steps)
    assert created.plan_id == 'p' and created.revision == 1
    assert created.steps[0].status == 'pending'
    assert created.steps[0].task_id is None and created.steps[0].attempts == 0
    created.steps[0].params['nested'].append(1)
    assert steps == before


def test_dispatches_in_dependency_order_and_does_not_mutate():
    original = plan([step(2, depends_on=[1], params={'query': 'follow-up'}), step(1)])
    assert ready_step(original).step_id == 1
    running = start_step(original, 1, 't')
    assert original.steps[1].status == 'pending'
    assert running.steps[1].attempts == 1 and ready_step(running) is None
    advanced = advance_plan(running, [result()])
    assert running.steps[1].status == 'running'
    assert advanced.steps[1].status == 'done'
    assert ready_step(advanced).step_id == 2
    assert advanced.history[-1]['result']['payload']['papers'] == ['artifact']
    finished = advance_plan(start_step(advanced, 2, 't2'), [result('t2')])
    assert finished.status == 'completed' and ready_step(finished) is None


@pytest.mark.parametrize('task_status', ['planned', 'running'])
def test_nonterminal_result_does_not_complete(task_status):
    running = start_step(plan(), 1, 't')
    assert advance_plan(running, [result(status=task_status)]) == running


def test_unrelated_and_duplicate_results_do_not_advance():
    running = start_step(plan([
        step(), step(2, depends_on=[1], params={'query': 'follow-up'}),
    ]), 1, 't')
    assert advance_plan(running, [result('unrelated')]) == running
    accepted = advance_plan(running, [result()])
    assert advance_plan(accepted, [result()]) == accepted
    next_running = start_step(accepted, 2, 't2')
    assert advance_plan(next_running, [result()]) == next_running


@pytest.mark.parametrize('status,passed,allow_skip,expected', [
    ('succeeded', None, False, 'completed'), ('succeeded', True, False, 'completed'),
    ('succeeded', False, False, 'needs_replan'), ('failed', True, False, 'needs_replan'),
    ('skipped', True, True, 'completed'), ('skipped', None, True, 'needs_replan'),
    ('skipped', False, True, 'needs_replan'), ('skipped', True, False, 'needs_replan'),
])
def test_result_acceptance(status, passed, allow_skip, expected):
    running = start_step(plan([step(allow_skip=allow_skip)]), 1, 't')
    advanced = advance_plan(running, [result(status=status, passed=passed)])
    assert advanced.status == expected
    if expected == 'needs_replan':
        assert advanced.reason and advanced.steps[0].status == 'failed'
        assert ready_step(advanced) is None


def test_blocked_plan_cannot_dispatch():
    blocked = ResearchPlan(plan_id='blocked', steps=[step(depends_on=[9])])
    advanced = advance_plan(blocked, [])
    assert advanced.status == 'blocked' and advanced.reason
    assert ready_step(advanced) is None


@pytest.mark.parametrize('step_id,task_id', [(2, 't'), (1, '')])
def test_start_rejects_invalid_dispatch(step_id, task_id):
    with pytest.raises(ValueError):
        start_step(plan(), step_id, task_id)


def test_start_rejects_unready_or_concurrent_dispatch():
    pending = plan([step(), step(2, depends_on=[1])])
    with pytest.raises(ValueError):
        start_step(pending, 2, 't')
    with pytest.raises(ValueError):
        start_step(start_step(pending, 1, 't'), 1, 'again')


def test_replacement_preserves_done_and_archives_removed_steps():
    original = plan([step(params={'q': 'x'}), step(2, action='generate_report', depends_on=[1])])
    original = advance_plan(start_step(original, 1, 't'), [result()])
    original = advance_plan(start_step(original, 2, 't2'), [result('t2', 'failed')])
    before = original.model_dump()
    revised = replace_remaining_plan(original, [step(3, action='generate_report', depends_on=[1])], allowed_actions=ACTIONS)
    assert original.model_dump() == before
    assert revised.plan_id == 'p' and revised.revision == 2 and revised.status == 'active'
    assert revised.steps[0] == original.steps[0]
    assert ready_step(revised).step_id == 3
    assert revised.history[-1]['reason'] == original.reason
    assert revised.history[-1]['steps'][0]['step_id'] == 2


@pytest.mark.parametrize('replacement', [
    [step(1)], [step(3, params={'q': 'x', 'filters': {'b': 2, 'a': 1}})],
    [step(3, depends_on=[2])], [step(3, depends_on=[4]), step(4, depends_on=[3])],
])
def test_replacement_rejects_completed_collisions_repeats_and_bad_dependencies(replacement):
    original = plan([step(params={'filters': {'a': 1, 'b': 2}, 'q': 'x'}), step(2)])
    original = advance_plan(start_step(original, 1, 't'), [result()])
    with pytest.raises(ValueError):
        replace_remaining_plan(original, replacement, allowed_actions=ACTIONS)


def test_dispatch_cannot_reuse_completed_task_id():
    original = plan([step(), step(2, action='generate_report', depends_on=[1])])
    original = advance_plan(start_step(original, 1, 't'), [result()])
    with pytest.raises(ValueError, match='task ID'):
        start_step(original, 2, 't')


def test_dispatch_cannot_reuse_archived_failed_task_id():
    original = advance_plan(start_step(plan(), 1, 't'), [result(status='failed')])
    revised = replace_remaining_plan(original, [step(2)], allowed_actions=ACTIONS)
    with pytest.raises(ValueError, match='task ID'):
        start_step(revised, 2, 't')


def test_replacement_preserves_accepted_skip_and_pending_dependencies():
    original = advance_plan(start_step(plan([step(allow_skip=True)]), 1, 't'),
                            [result(status='skipped', passed=True)])
    revised = replace_remaining_plan(original, [
        step(3, action='generate_report', depends_on=[2]),
        step(2, params={'query': 'new'}, depends_on=[1]),
    ], allowed_actions=ACTIONS)
    assert revised.steps[0].status == 'skipped'
    assert ready_step(revised).step_id == 2


def test_history_and_result_artifacts_are_detached_from_inputs():
    returned_result = result()
    original = start_step(plan(), 1, 't')
    original.history.append({'nested': []})
    updated = advance_plan(original, [returned_result])
    updated.history[0]['nested'].append('changed')
    updated.history[-1]['result']['payload']['papers'].append('changed')
    assert original.history[0]['nested'] == []
    assert returned_result.payload['papers'] == ['artifact']


def test_clarify_request_is_not_an_executable_plan_action():
    with pytest.raises(ValueError):
        create_plan([step(action='clarify_request')], allowed_actions={'clarify_request'})


def test_dispatch_rejects_repeated_accepted_action_with_equivalent_params():
    original = plan([
        step(params={'query': 'x', 'filters': {'a': 1, 'b': 2}}),
        step(2, params={'filters': {'b': 2, 'a': 1}, 'query': 'x'}, depends_on=[1]),
    ])
    accepted = advance_plan(start_step(original, 1, 't'), [result()])
    with pytest.raises(ValueError, match='completed action'):
        start_step(accepted, 2, 't2')
    assert accepted.steps[1].status == 'pending'


def test_dispatch_allows_same_action_with_distinct_params():
    original = plan([
        step(params={'query': 'x'}), step(2, params={'query': 'y'}, depends_on=[1]),
    ])
    accepted = advance_plan(start_step(original, 1, 't'), [result()])
    dispatched = start_step(accepted, 2, 't2')
    assert dispatched.steps[1].status == 'running'


def test_discovery_constraint_allows_separate_review_step():
    created = create_plan(
        [step(1), step(2, action='write_review', depends_on=[1])],
        allowed_actions={'search_literature', 'write_review'}, workflow_constraint='discovery_only',
    )
    searched = advance_plan(start_step(created, 1, 'search'), [result('search')])
    assert ready_step(searched).action == 'write_review'
