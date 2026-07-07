# Codex progress for issue #442

## S1 investigation decision

- Read `.dev-notes/workplan_442_direct_projection_codex.md` and `.dev-notes/lane_ground_rules.md`; working only in `/home/nanyeon99/project/mf-direct`.
- The supervised direct path (`macroforecast/forecasting/policies/direct.py`) only applies to feature-matrix models through `_fit_one_model_at_origin`. Panel models bypass that path and run through `_run_panel_models` -> `forecast_panel_origin`.
- `var` already owns the `vars::VAR` lag-block construction in `macroforecast/models/timeseries.py` via `_VAR` and `_vars_rhs`, while `forecast_panel_origin` owns the #423 horizon/date emission contract. Therefore Part A will be implemented by adding a `direct` fit mode to `var`/`_VAR`, and by having the panel policy pass `direct=True` only for `model_spec.name == "var"` under `forecast_policy in {"direct", "direct_average"}`. This preserves panel runner labeling and avoids fabricating direct forms for BVAR/DFM/FAVAR.
- Part B reroute cannot mutate `ResolvedTarget.policy` globally because the same target may be shared by a guarded arm and a direct-capable arm. I will store per-cell policy overrides on `PipelineSpec` and have `pipeline/run.py` use the effective arm-target policy when calling `forecasting.run` and when hashing result-store cells.
- Guard reroute uses `forecast_policy="recursive"` for affected panel cells too. For panel-input models this means the native panel forecast path with recursive labels, not construction of a supervised feature-matrix recursion; `forecast_panel_origin` still owns the horizon/date emission contract.

## Gate log

- Deviation: `.dev-notes/policy_matrix_results.json` referenced by the workplan is absent in this worktree. `tools/gen_policy_matrix.py` treats the scan as optional and renders the measured-scan column as `not available`; the generated page states that the scan source was unavailable.
- `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide && ~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` — PASS (`13 model pages in sync`, `1 policy matrix page in sync`).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_var_direct_projection.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (4 passed).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_direct_policy_guard.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (19 passed; statsmodels frequency warnings only).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_panel_horizon_labeling.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (3 passed; statsmodels DFM warnings only).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_runner_golden_snapshot.py::test_panel_routing_intact tests/forecasting/test_panel_input_runner.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (4 passed).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — FAIL initially: golden snapshot's legacy guarded `naive`/`arima` direct cells hit the new default `on_unsupported_direct="error"`. Fixed by making that compatibility snapshot opt into `on_unsupported_direct="warn"`.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_runner_golden_snapshot.py::test_runner_matrix_matches_golden_snapshot --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (1 passed; expected direct-guard and statsmodels warnings).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS on rerun (155 passed; expected statsmodels/default-feature/vintage/direct-guard warnings).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — FAIL initially: `test_provenance_level_basic_matches_pre_change_shape` caught top-level `policy_overrides` in basic provenance. Fixed by keeping overrides only in full `spec_echo`, preserving basic provenance shape.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_default_provenance.py::test_provenance_level_basic_matches_pre_change_shape --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (1 passed).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS on rerun (207 passed; expected warnings only).
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — FAIL initially on missing local ndarray annotations in `_vars_direct_rhs`; PASS after annotation fix (`Success: no issues found in 109 source files`).
- `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide && ~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` — PASS on final run (`13 model pages`, `1 policy matrix page` in sync).
- `git diff -U0 -- macroforecast tools tests docs .github CHANGELOG.md | rg -n "^\\+.*except (Exception|:)|^\\+.*except Exception|^\\+\\s*except:" || true` — PASS (no new blanket `except Exception` or bare `except` lines).
- `git diff --check` — PASS.
- After enabling native panel recursive routing for guarded panel-model reroutes:
  - `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_dfm_unrestricted_midas_prediction_anchor.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (4 passed).
  - `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide && ~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` — PASS (`13 model pages`, `1 policy matrix page` in sync).
  - `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — PASS (`Success: no issues found in 109 source files`).
  - `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (155 passed; expected statsmodels/default-feature/vintage/direct-guard warnings).
  - `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (207 passed; expected runtime/deprecation/statsmodels warnings).
  - `git diff --check` — PASS.
  - `git diff -U0 -- macroforecast tools tests docs .github CHANGELOG.md | rg -n "^\\+.*except (Exception|:)|^\\+.*except Exception|^\\+\\s*except:" || true` — PASS (no output).
