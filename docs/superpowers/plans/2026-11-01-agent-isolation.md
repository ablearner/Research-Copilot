# Agent execution isolation

Approved scope: Supervisor chooses tasks, Runtime dispatches, Workers execute
tools and return results. Workers must not invoke peer Agents or obtain the
dispatch registry through shared runtime/service objects.

## Implementation

1. Split discovery from writing: Scout produces candidates; Runtime dispatches
   Writer for report synthesis and TODO planning. Discovery-only API requests
   retain their report response through search followed by writing.
2. Move paper-figure operations into a reusable tool. QA uses visual/figure
   tools without constructing or invoking ChartAnalysisAgent.
3. Remove Agent construction from research function tools. Analysis uses its
   tool; task decomposition produces a validated proposal without execution.
4. Give Workers explicit allowlisted service/tool ports and no Agent registry.
   Resolve executors from parent runtime state and rebuild KnowledgeAccess
   against the restricted port. Answer tools do not invoke nested RAG Agents.
5. Verify isolation with poisoned peer handles, restricted-port tests, and
   full search/QA/state-commit regressions. Independently review the changes,
   run the full test suite, lint and installed-wheel smoke test.

## Constraints

Preserve API schemas, complete workspace persistence, scoped task inputs and
artifact handoff. No new framework, parallel scheduler or database. This is
an application dependency boundary, not a Python sandbox or process boundary.
Tools may share clients and perform documented external effects; Runtime
remains the sole dispatcher and commits returned workflow deltas.

## Completion

All five implementation steps are complete. Independent review also found and
verified fixes for report-history/workspace preservation, source-warning
handoff, English report headings and saved output-language preference.

Validation: 965 tests passed, including the installed-wheel smoke test; Ruff
and whitespace checks passed. Two existing SciPy/Numba deprecation warnings
remain. Concise fallback reports retain explicit failed-quality status and
their partial report; length/citation requirements were not weakened.
