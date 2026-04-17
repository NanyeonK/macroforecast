# Implementation Plan — Staged Selector Expansion (Preprocessing / Benchmark / Framework)

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Widen `macrocast_single_run()` staged selector beyond route-defining choices into the next executable single-run block: framework, benchmark, and the narrow operational preprocessing choices.

Why this slice
- The first staged selector already handles route-defining choices and immediate route-preview refresh.
- The next highest-value gain is letting users compose most of the current executable single-run subset directly from the staged flow.
- This remains smaller and safer than trying to restore the old archived full wizard.

Smallest honest implementation
1. Keep existing YAML-path inspection mode unchanged.
2. Extend staged selector order after current route-defining choices with:
   - `framework`
   - `benchmark_family`
   - `tcode_policy`
   - `x_missing_policy`
   - `scaling_policy`
   - `preprocess_order`
   - `preprocess_fit_scope`
3. Keep the options grounded in the live executable subset:
   - `framework`: `expanding`, `rolling`
   - `benchmark_family`: `zero_change`, `ar_bic`, `historical_mean`, `custom_benchmark`
   - `tcode_policy`: `raw_only`, `extra_preprocess_without_tcode`
   - `x_missing_policy`: `none`, `em_impute`
   - `scaling_policy`: `none`, `standard`, `robust`
   - `preprocess_order`: `none`, `extra_only`
   - `preprocess_fit_scope`: `not_applicable`, `train_only`
4. Add normalization rules so the wizard keeps preprocessing leaf/fixed values coherent when a user switches between raw-only and extra-preprocess routes.
5. Add benchmark-config prompts only for `custom_benchmark`:
   - `plugin_path`
   - `callable_name`
6. After every completed step:
   - rewrite YAML
   - refresh compile preview / tree context / route preview
7. Preserve honest stop behavior for wrapper routes and non-executable internal sweep routes.

Acceptance criteria
- staged selector supports the new framework / benchmark / preprocessing choices
- switching from `raw_only` to `extra_preprocess_without_tcode` yields a coherent draft path rather than stale incompatible settings
- selecting `custom_benchmark` introduces the required benchmark-config prompts
- compile preview refreshes after every step
- focused and broad tests pass

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
