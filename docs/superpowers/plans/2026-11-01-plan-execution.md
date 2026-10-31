# Reliable research plan execution

User-approved scope: repair the existing Plan-and-Execute path, preserve the
Supervisor–Worker architecture, and verify the complete graph offline.

## Constraints

- Continue in the clean development checkout on `refactor/code-structure-20261031`.
- Reuse worker registry, evaluation tools, and runtime state commit boundaries.
- Keep simple single-action routing compatible. No new dependencies or frontend changes.
- Plans execute serially; dependencies control readiness, not parallel scheduling.
- Runtime owns IDs, statuses, and attempts. Model proposals cannot mark work complete.
- Preserve completed results across replanning and bound recovery attempts.
- Do not modify user data, credentials, backups, or resume files.

## Implementation and verification

- [x] Add failing regressions for graph plan retention, first-step consistency,
      task-result correlation, output validation, and bounded recovery.
- [x] Introduce `domain/schemas/research_plan.py` and pure
      `runtime/research/plan_executor.py`: validate proposals, select ready work,
      correlate results, retain completed work, archive superseded steps.
- [x] Integrate explicit plan state updates with supervisor and graph runtime;
      unify first and later dispatch, preserve instruction and input scope.
- [x] Reuse `TaskEvaluation` for acceptance, define skip behavior, recover only
      unfinished work, expose stop reasons and partial completion.
- [x] Project plan state into research context and final metadata; distinguish
      decisions, planning model calls, replans, and worker calls.
- [x] Run focused tests, full offline pytest, Ruff, diff checks, independent
      code review, and resolve findings. Document results and limitations.

## Acceptance scenarios

Three-step execution uses one planning call; unknown/duplicate/cyclic dependencies
fail before dispatch; unrelated compression and stale results cannot complete a
step; first-step params/instruction come from the plan; failed evaluation triggers
replanning; completed work survives replacement; repeated failure stops within
budget; budget exhaustion preserves partial results; single-action tests continue
to pass. Use actual LangGraph compilation with fake models/workers for offline
integration and avoid network access in tests.

## Evidence

Baseline: supervisor and graph runtime tests, 54 passed, two existing dependency
deprecation warnings. Start revision: `3ed7461`.


## Final verification

- Full offline regression: `python -m pytest -q -p no:cacheprovider` —
  **890 passed**, two pre-existing SciPy/Numba deprecation warnings, 92.28 seconds.
- `python -m ruff check .` — passed.
- `git diff --cached --check` — passed.
- `python -m build --wheel --no-isolation` — passed.
- Installed the wheel under `/tmp/research-plan-final-install` without dependencies
  and imported it from `/tmp`; verified plan dispatch, graph schema, and packaged
  supervisor prompt independently of the source checkout.
- Independent final review approved with no remaining Critical or Important
  findings; reviewer independently ran 60 integration/counter cases.

All Python commands used `/home/myc/miniconda3/envs/Research-Copilot/bin/python`.
Test log: `/tmp/research-plan-final-tests.log`.
Build log: `/tmp/research-plan-final-build.log`.

Review corrections include complete Zotero result acceptance, malformed schema
rejection, empty skipped-result rejection for every supported worker action,
and counting only actual replanning model invocations. The production graph
integration test verifies one planning call, four supervisor decisions, and three
worker executions for a three-step plan.

Validation used fake model/worker outputs for deterministic orchestration tests
and the existing offline suite. No live model/provider/Zotero calls were made;
semantic research quality and external side-effect transactions are not implied
by these checks. No frontend changes required a frontend build. Work remains on
`refactor/code-structure-20261031`; no merge or push was requested.
