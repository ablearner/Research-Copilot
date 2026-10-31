# Task 5 — boundary cleanup and final documentation

## Change

- `AgentMessage` and `AgentResultMessage` remain the graph's internal task/result state. `_message_result_update` now constructs the result directly, preserving payload, evaluation, action-output metadata, skill/tool metadata, and agent attribution.
- `serialize_agent_exchange` projects legacy and unified response fields together. The final response and manager decision notice reuse one projection; the redundant core plan serializer is gone. Existing `UnifiedAgentTask`/`UnifiedAgentResult` DTO conversion methods and public unified serializer functions remain for compatibility callers.
- Removed the constructed but unread `unified_runtime_context` field/helper and the unused document-agent DTO result helper, after confirming no production callers. Updated compatibility exports accordingly.
- Updated the blueprint metadata and `services/research/ARCHITECTURE.md` to describe the registry's actual direct specialist dispatch, service state ownership, public projection boundary, and external-effect limits. Updated `evaluation/README.md` with reproducible offline commands and live evaluation limits. Marked the two direct-script bootstrap imports in `evaluation/summarize_benchmark_report.py` with narrow E402 rationales.
- Final repository-wide lint cleanup removed an unused local in `intent_classifier.py` and two unused evaluation-test imports, and annotated E402 only on imports that follow required direct-script path bootstraps in the two migration scripts. Migration execution was not invoked.

## RED / GREEN

- RED: after writing `tests/unit/services/test_research_boundary_projection.py`, `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider tests/unit/services/test_research_boundary_projection.py` stopped at collection: `ImportError: cannot import name 'serialize_agent_exchange'`. This is the missing projection the test required.
- GREEN: `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider tests/unit/services/test_research_boundary_projection.py tests/unit/services/test_unified_runtime.py tests/unit/services/test_research_supervisor_graph_runtime.py tests/unit/agents/test_document_agent.py` → `39 passed, 2 warnings in 9.08s`.
- The existing context isolation test initially failed because it asserted on the removed unread field; it now checks that the field is absent while retaining registry and blueprint isolation checks. The research response integration test now checks that the decision notice and metadata have equal projected values.

## Final verification

- `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider tests/unit/services/test_research_boundary_projection.py tests/unit/services/test_research_supervisor_graph_runtime.py tests/unit/packaging/test_wheel_contents.py` → `33 passed, 2 warnings in 21.58s`. The packaging test builds a wheel, installs it to a temporary target, and imports runtime/resources from a temporary working directory outside this checkout.
- `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider` → `780 passed, 2 warnings in 90.75s`. Full output: `/tmp/research-copilot-task5-full-pytest.log`. This ran before the final unused-name/comment-only lint cleanup.
- `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m ruff check runtime/research/unified_runtime.py runtime/research/context_builder.py runtime/research/agent_protocol/base.py runtime/research/supervisor_graph_runtime_core.py runtime/research/response_formatter.py runtime/research/result_aggregator.py services/research/__init__.py agents/research_document_agent.py domain/schemas/unified_runtime.py tests/unit/services/test_research_boundary_projection.py tests/unit/services/test_research_supervisor_graph_runtime.py evaluation` → `All checks passed!`.
- `git diff --check` → no whitespace errors. The parent had already run `npm run build` including type checking; Task 5 does not change frontend files.
- After the mechanical lint cleanup: `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m ruff check .` → `All checks passed!`; `/home/myc/miniconda3/envs/Research-Copilot/bin/python -m pytest -q -p no:cacheprovider tests/unit/evaluation/test_metrics.py tests/unit/services/test_research_supervisor_graph_runtime.py` → `69 passed, 2 warnings in 8.70s`; `/home/myc/miniconda3/envs/Research-Copilot/bin/python scripts/migrate_memory_json_to_sqlite.py --help` and `/home/myc/miniconda3/envs/Research-Copilot/bin/python scripts/migrate_research_json_to_sqlite.py --help` both exited 0; `git diff --check` clean.

## Limits and self-review

- The two pytest warnings are existing third-party deprecations from hyppo/SciPy and numba. They are not failures and were not suppressed.
- The final repository-wide Ruff check is clean. The final cleanup only removed unused names and added import-order annotations needed to preserve direct script execution.
- Reviewed source diff for response keys, result metadata/evaluation preservation, current callers, removed field usage, and documentation claims. No live model calls or external writes were used for verification. The isolated wheel smoke uses `pip --no-deps`, so it validates package contents/import paths rather than a fresh dependency solve.
