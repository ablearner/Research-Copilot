# Consolidated final-review fix

Base: `19820dd` on `refactor/code-structure-20261031`.
Scope: only the Important QA snapshot consistency finding and the Minor canonical web build-directory correction in `.superpowers/sdd/final-review.md`.

## Changes

- `ResearchQAAgent.run_action` still loads the current response after QA execution so paper caches and response warnings remain current. It now overlays `QAExecutionResult.updated_task` and `updated_report` onto that response. The resulting proposed response is used both for the delta's visible context and its post-commit hydration inputs.
- This preserves freshly written derived paper caches without re-saving a stale papers list. It also keeps deliberate QA follow-up state when the action returns `skipped` to request supervisor route recovery.
- Canonical frontend verification now names the existing `web/` directory instead of `frontend/`.

## Behavioral regression and RED

The new parameterized offline test runs the real QA application/executor, `ResearchQAAgent.run_action`, and the runtime action-node commit path. Only collection answer generation is stubbed; task/report storage and session hydration are real. Cases cover a supported successful answer and an under-supported answer requesting supervisor route recovery.

The test verifies unchanged workflow storage/context before commit; agreement of committed storage with visible task/report/workspace afterward; matching hydration fields; preservation of response warnings and newly refreshed derived paper caches; exactly one task/report commit; and the next workspace-producing action retaining updated TODO/report content.

Command from `/home/myc/Research-Copilot`:

```sh
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest tests/unit/services/test_research_state_commit_boundary.py -q -k qa_action_commits_current_snapshot --tb=short
```

Before implementation: **2 failed, 20 deselected**, both at `assert ctx.task == committed.task`, demonstrating that successful and recovery/skipped actions saved new products but published the old snapshot. Log: `/tmp/final-fix-red.log`.

## GREEN and checks

```sh
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest tests/unit/services/test_research_state_commit_boundary.py tests/unit/services/test_literature_research_service.py tests/unit/services/test_research_supervisor_graph_runtime.py tests/unit/agents/test_research_qa_agent.py tests/unit/services/test_research_architecture_constraints.py -q --tb=short
```

Result: **135 passed**, two baseline third-party deprecation warnings, 9.24 seconds. Log: `/tmp/final-fix-focused.log`.

```sh
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m ruff check .
git diff --check
```

Results: **All checks passed**; diff check clean.

Full final verification command:

```sh
/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider > /tmp/research-copilot-final-fix-pytest.log 2>&1
```

Final full-suite result is recorded below after completion.

## Files and limits

- `agents/research_qa_agent.py`
- `tests/unit/services/test_research_state_commit_boundary.py`
- `services/research/ARCHITECTURE.md`

No live model or external service calls; no user data, environment, dependency, frontend implementation or unrelated architecture changes. The existing distinction between derived tool effects and workflow commits remains intact. The known hyppo/SciPy and numba deprecation warnings are not suppressed.

Final full-suite result: **782 passed, 2 warnings in 86.14 seconds**, exit code 0. The two warnings are the documented pre-existing hyppo/SciPy and numba deprecations. Full log: `/tmp/research-copilot-final-fix-pytest.log`.
