# Worker context boundary

User-approved scope: execute the proposed multi-agent context improvements to completion.

## Global constraints

- Keep the existing serial Supervisor–Worker graph and `run_action(context, decision)` API.
- Separate JSON model input from live program resources. Do not serialize service clients.
- Each invocation owns its mutable task state, instruction and Skill selection.
- Scope context to the current action and selected papers; bound summaries and history.
- Pass explicit dependency summaries and resolvable output references, not automatic full reports.
- Preserve the full workspace snapshot for persistence. Runtime applies returned state deltas.
- Do not add parallel scheduling, new databases, a second state machine or dependencies.
- Input isolation does not make external side effects transactional or concurrent commits safe.

## Implementation

1. Introduce a typed WorkerTaskInput, action-specific context slicing, bounded dependency summaries,
   and invocation-local artifact references. Test scope, detachment, limits and reference resolution.
2. Fork mutable worker data while retaining service/runtime/registry handles. Clear absent Skills.
   Dispatch with the task slice and retain it in task/result audit messages. Test overlapping calls
   and the runtime's single state-delta application boundary.
3. Propagate task context into final analysis and review LLM inputs, plus existing expert-context
   consumers. Preserve legacy standalone worker entrypoints and full persisted workspace.
4. Review changes, run focused regressions and the complete suite, lint and package smoke tests.
   Document actual behavior and limitations and make a local commit.

## Progress

- Baseline: clean `813fa8d`; prior full suite 890 passed.
- Task-invocation regression tests first failed on missing module (expected RED).
- Typed slices, snapshot isolation, task-local Skills and prompt wiring implemented.
- Added runtime-owned substantive output artifacts after finding that original action metadata
  retained only counts/IDs; dependent prompts receive bounded summaries instead of full output.
- First full suite: 914 passed, 2 existing dependency warnings; Ruff passed.
- Independent review found analysis evidence selection bypassing the slice and collection QA
  forwarding task context only with an active Skill. Fixed both with failing regression tests first.
- Aligned QA selection and metadata-only answer fallback with the explicit task input. Kept
  direct document/image routes and full-workspace persistence intact. Targeted suite: 43 passed.
- Final independent review approved the combined patch with no outstanding actionable findings.
- Full regression caught a compatible explicit document selection being discarded by the new
  QA adapter. Fixed preservation/intersection logic; 31 focused tests passed and review approved.
- Final full suite: **938 passed**, 2 pre-existing SciPy/Numba dependency deprecation warnings
  (90.16 seconds). Includes wheel build/install and import/resource smoke tests outside checkout.
- Repository-wide Ruff and staged/unstaged whitespace checks passed. No live-provider quality
  evaluation was performed; model-boundary tests capture deterministic fake-provider inputs.
