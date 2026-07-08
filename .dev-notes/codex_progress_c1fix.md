# C1 Fix Progress

## Scope
- Read `.dev-notes/workplan_c1fix_codex.md`, `.dev-notes/lane_ground_rules.md`, and C1/C3/C4/C6 in `.dev-notes/stage2_review_findings.md`.
- Worktree: `/home/nanyeon99/project/mf-c1fix`.
- Branch: `fix/axis-contribution-clustering`.

## Decisions
- Implement `axis_contribution(vcov=...)` with `driscoll_kraay` as the default and keep `hac_lags` as the Bartlett bandwidth for DK and legacy row-stacked HAC.
- Preserve legacy row-stacked HAC behind `vcov="hac"` for backward compatibility and make `vcov="hc0"` an explicit White covariance route.
- Implement `cluster_by="date"` only, as requested; other cluster keys will raise a focused validation error.
- C3 will rank lasso/elastic-net screens by standardized coefficient magnitude while preserving threshold/nonzero selection semantics.
- S2/S3 are out of scope unless trivially encountered; C1/C3/C4/C6 are the required fixes.

## Running Log
- Initial inspection complete. `axis_contribution` currently sends the stacked panel to `newey_west`; C3/C4/C6 locations match the workplan.
- Implemented `axis_contribution(vcov=..., cluster_by="date")` with DK default, explicit date cluster, explicit HC0, and legacy row-stacked HAC.
- Added a one-date fallback for DK/cluster to HC0 because a literal one-cluster score sandwich is unidentified and collapses to zero; metadata records `single_cluster_fallback=True`.
- Implemented C3 ranking on standardized sparse coefficients, C4 `mad` docstring clarification, and C6 `aicc` validator support.
- Added focused tests for covariance behavior, MC validation, sparse screening ranking, and the public AICc route.

## Gates
- `~/project/macroforecast/.venv/bin/python -m pytest tests/analysis/test_contribution.py tests/feature_engineering/test_features.py tests/model_selection/test_information_criterion.py tests/evaluation/test_metrics.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (`120 passed`).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/analysis --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (`8 passed`).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/model_selection --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (`53 passed`).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/feature_engineering --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (`100 passed`, 2 existing PCA full-sample warnings).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (`65 passed`).
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — PASS.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` — regenerated 37 reference pages.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` — PASS.
