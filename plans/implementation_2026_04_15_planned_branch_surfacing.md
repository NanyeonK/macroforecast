# Implementation Plan — Planned-Branch Surfacing in Staged Selector

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Make planned-but-not-yet-executable branches visible inside `macrocast_single_run()` instead of only discoverable after compile warnings.

Scope
- staged selector option metadata
- staged prompt labeling
- route-preview messaging for planned branch choices
- tests / docs / roadmap sync

Smallest honest implementation
1. Keep YAML inspection and executable staged paths unchanged.
2. Add option-status metadata for staged choices backed by the live registry.
3. Show status labels in staged prompts for finite-option steps.
4. Expand staged options to include the first visible planned branches where appropriate:
   - `feature_builder = factor_pca`
   - `stat_test = mcs`
   - `importance_method = shap`
5. When a planned choice is selected:
   - YAML still rewrites
   - compile/tree preview still refreshes
   - route preview should surface explicit planned-branch messaging rather than a generic blocked message
6. Preserve honest early stop once the selected branch leaves the current executable single-run surface.

Acceptance criteria
- staged current choice exposes option-status metadata
- terminal prompt shows operational/planned labels for finite choices
- selecting `factor_pca`, `mcs`, or `shap` yields explicit planned-branch route preview
- focused and broad tests pass

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
