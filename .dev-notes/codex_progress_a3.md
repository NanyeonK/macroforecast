# A3 tags + axis contribution progress

## Investigation

- Read `.dev-notes/lane_ground_rules.md`, `.dev-notes/workplan_a3_tags_contribution_codex.md`, and `.dev-notes/gcls_axis_audit.md`.
- Confirmed owned code paths:
  - `macroforecast/pipeline/spec.py`: `Arm` currently has `metadata` but no first-class `tags`.
  - `macroforecast/pipeline/run.py`: `_run_one_arm_target` appends `arm` and `contender`; `_spec_echo` is the provenance echo site.
  - `macroforecast/pipeline/result_store.py`: `result_cell_identity` excludes free-form metadata and hashes only compute-relevant arm fields.
- Existing HAC utility to reuse: `macroforecast.data_analysis.newey_west`; it matches statsmodels HAC SEs in `tests/correctness/test_newey_west_hac.py`.

## Decisions

- Tag columns use the flat `tag_<key>` convention for regressability.
- Tag keys must be valid Python identifiers; tag values must be scalar `str`, `int`, `float`, or `bool`.
- Tags are copied and sorted into an immutable mapping at `Arm` construction for stable display and pickling.
- `Arm.tags` is placed after `metadata` to preserve the old positional meaning of `Arm(..., metadata)`.
- Tags remain outside `result_cell_identity`; result-store loads reapply the current arm tags so changing tags reuses cached forecast cells without serving stale tag columns.
- `axis_contribution` lives at `macroforecast.analysis.axis_contribution`, with a top-level lazy export.
- Fixed effects are absorbed as one joint dummy set over the requested columns. The default `("target", "horizon", "date")` therefore implements target-horizon-date cell fixed effects.
- `outcome="r2"` uses row-level pseudo-R2, `1 - e^2 / MSE(reference)`, where the reference MSE is computed within each target-horizon group. The reference can be passed explicitly; when omitted it is inferred deterministically and echoed in result metadata.

## Deviations

- None so far.

## Gates

- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_arm_tags.py tests/pipeline/test_result_store.py tests/analysis/test_contribution.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (18 passed; warnings were multiprocessing fork deprecation and existing undigestible custom-callable result-store warning).
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (253 passed; 64 existing warnings including ragged-coverage, multiprocessing fork deprecation, rescore legacy checkpoint identity, statsmodels frequency, expected failed-cell, and undigestible custom-callable warnings).
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/analysis --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (3 passed).
- PASS: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` (no issues in 113 source files).
- PASS: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` (wrote 37 reference pages).
- PASS: `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` (reference docs up to date).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide` (13 model pages in sync).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` (policy matrix in sync).
- FAIL then fixed: `~/project/macroforecast/.venv/bin/python -m sphinx -W -b html docs docs/_build/html` initially failed because `docs/guide/multi_axis_comparison.md` linked to non-generated `../reference/analysis.md`; changed the guide link to generated `../reference/public_api.md`.
- PASS: `~/project/macroforecast/.venv/bin/python -m sphinx -W -b html docs docs/_build/html` (build succeeded after link fix; pre-existing multiple-toctree consistency notices did not fail).
- PASS rerun after preserving `Arm` positional compatibility: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_arm_tags.py tests/pipeline/test_result_store.py tests/analysis/test_contribution.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (18 passed; same warning classes as first run).
- PASS rerun after preserving `Arm` positional compatibility: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` (no issues in 113 source files).
- PASS rerun after preserving `Arm` positional compatibility: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference && ~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` (wrote 37 pages; reference docs up to date).
- PASS rerun after regenerated reference docs: `~/project/macroforecast/.venv/bin/python -m sphinx -W -b html docs docs/_build/html` (build succeeded).
