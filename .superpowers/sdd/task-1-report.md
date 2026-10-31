# Task 1 report: reproducible baseline and packaging

## Scope and result

- Added `runtime*` to setuptools discovery. The wheel now contains `runtime`, `runtime.research`, and `runtime.research.agent_protocol`.
- Packaged all tracked prompt templates and built-in `SKILL.md` resources. The default prompt mapping, skill registry, ReAct prompt paths, and graph extraction prompt resolve from an installed wheel when the process starts outside this checkout. Caller-supplied existing relative or absolute prompt paths still take precedence.
- Changed `.gitignore` to allowlist evaluation implementation (`evaluation/*.py`), `evaluation/README.md`, `evaluation/sample_cases.json`, and `evaluation/datasets/core_cases.json`. Downloaded benchmarks, images, generated reports, and miscellaneous local docs remain ignored. Added a note in the evaluation README describing this policy.
- Corrected `services/research/ARCHITECTURE.md` to describe the current action node's direct `specialist.run_action(...)` path through `unified_agent_delegates`; the unified registry remains metadata/inventory rather than the execution path.

## RED/GREEN evidence

1. Wrote `tests/unit/packaging/test_wheel_contents.py` before production changes. `python -m pytest -q tests/unit/packaging/test_wheel_contents.py` failed 2/2: configured package discovery lacked all three `runtime` packages, and the wheel lacked `runtime/research/supervisor_graph_runtime_core.py`.
2. Added discovery and package data. The same command passed 2/2, including a wheel build, install to a temporary target, and resource load from a temporary working directory.
3. Extended the installed-wheel smoke for the ReAct QA and graph prompts. It failed with `prompts/reasoning/react_decide_next_step.txt False` outside the checkout. After adding bundled-path resolution, the packaging test passed 2/2. The final test compares every prompt `.txt`/`.md`, the mapping YAML, and every built-in `SKILL.md` against wheel contents, and asserts the imported runtime module came from the temporary installation.

## Verification

- Full pre-change baseline from parent: `741 passed`, 2 existing third-party deprecation warnings (hyppo/SciPy and numba), in 78.44 seconds. No pre-existing failures. Frontend baseline: `npm run build` passed, with an existing Browserslist data-age notice.
- Focused compatibility command: `python -m pytest -q tests/unit/packaging tests/unit/skills/test_skill_system.py tests/unit/skills/test_research_skill_resolver.py tests/unit/agents/test_research_qa_agent.py tests/unit/services/test_research_supervisor_graph_runtime.py` → `84 passed`, same 2 third-party warnings. The packaging test was rerun after its final assertion changes: `2 passed`.
- `python -m ruff check agents/research_qa_agent.py chains/graph_extraction_chain.py core/prompt_resolver.py core/skill_registry.py core/resource_paths.py tests/unit/packaging/test_wheel_contents.py` → all checks passed.
- `git diff --check` → no whitespace errors.
- `git add -n evaluation` showed only the 16 selected files. `git check-ignore -v` confirmed sample cases and implementation are unignored, while report JSON, benchmark cases, and chart images remain ignored.

## Limits

The installed-wheel smoke uses the existing project Python environment and `pip install --no-deps`; it validates packaging and path resolution, not a clean dependency installation. The parent will run the full suite at the end of the overall refactor.

## Review follow-up: local skill directories

The initial default `SkillRegistry()` chose a working-directory `skills/` tree whenever it existed. An empty or community-only tree therefore hid the wheel's built-in skills. New unit regressions cover an empty local directory, a community directory with a same-name override, and isolation of an explicit `skills_dir`. The installed-wheel smoke now creates both empty and community-only local layouts after installation.

- RED: four focused tests produced `3 failed, 1 passed`; the empty and community-only default scans lacked bundled skills, while the explicit-root isolation test passed. After correcting the smoke's embedded Python string, its installed-wheel assertion also failed specifically because `research-qa` disappeared beside an empty local `skills/` directory.
- GREEN: default scans now load bundled skills first, then the working-directory tree so local skills can override by name. Explicit custom roots scan only that root. Config overrides remain in the working-directory tree for default construction.
- Verification: `python -m pytest -q tests/unit/packaging/test_wheel_contents.py tests/unit/skills/test_skill_system.py` → `54 passed`; `python -m ruff check core/skill_registry.py tests/unit/packaging/test_wheel_contents.py tests/unit/skills/test_skill_system.py` → all checks passed; `git diff --check` → no errors.
- Replaced machine-specific Python paths in `evaluation/README.md` examples with `python`. The two baseline third-party deprecation warnings remain documented above; no dependency changes were made.
