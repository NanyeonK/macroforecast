# A7 standard estimators progress

## S1 inspection

- Branch/worktree: `feat/standard-estimators` in `/home/nanyeon99/project/mf-a7-estimators`.
- Ground rules read: `.dev-notes/lane_ground_rules.md`.
- Workplan read: `.dev-notes/workplan_a7_estimators_codex.md`.
- Reference read: `.dev-notes/phaseB_design_b1_medeiros.md`, especially section 1 "패키지 추가/수정 목록".
- Existing `.dev-notes/{lane_ground_rules,phaseB_design_b1_medeiros,workplan_a7_estimators_codex,codex_run}.md` files are untracked; do not stage them.

## Decisions

- Seed threading: post-#462 pipeline injects a derived per-arm `random_state` only when a registered `ModelSpec.default_params` contains `random_state` and the caller did not supply one. New stochastic model specs must declare `random_state`.
- Model placement: avoid A5-owned `macroforecast/models/linear.py`; put CSR/JMA in a new ensemble-adjacent model module and UCSV in a new target-kind module.
- MAD definition: A7 workplan explicitly requires forecast `mad = median(|e|)`. B1 design table says `median |e - median e|`; implement the A7 package contract and cite the Medeiros inflation-forecasting usage in the docstring as a median absolute forecast-error summary.
- JMA dependency: `scipy>=1.11` is a direct project dependency in `pyproject.toml`; use `scipy.optimize.minimize(method="SLSQP")` for simplex-constrained weights.
- JMA candidate guard: nested candidates stop before saturated OLS designs (`size <= n_obs - 2`) because LOO hat denominators are otherwise undefined; this preserves the general nested method where it is numerically identified.
- UCSV parameterization: package `gamma` is the shared random-walk innovation variance for both log-volatility states. The B1 Medeiros paper-specific `Vtau=Vh=0.12` remains a replication-lane parameter choice, not the package default.
- UCSV direct-policy guard: add `ucsv` to `DIRECT_POLICY_GUARD_MODELS` because it is target-kind and has no explicit direct-projection parameter; recursive/path policies preserve its horizon-invariant final-trend forecast contract.

## Gate log

- `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation/test_metrics.py tests/pipeline/test_evalspec_threading.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`49 passed`, one existing rescore warning).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_standard_estimators.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`4 passed`) after CSR addition.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_standard_estimators.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> failed after JMA addition due to test fixture assuming `x1`/`x2` for one/two-predictor cases; fixed fixture.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_standard_estimators.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`9 passed`) after JMA addition.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_standard_estimators.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`14 passed`, includes UCSV default-draw 500-observation runtime guard in <60s).
- `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide` -> passed (`15 model pages in sync`).
- `~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` -> passed (`1 policy matrix page in sync`).
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` -> passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> failed (`csr` registry `random_state` default did not match callable default); fixed CSR/UCSV callable defaults to `1071` to match stochastic-model registry convention.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_models.py::test_model_spec_defaults_match_callable_signatures --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`194 passed`, existing warnings).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`56 passed`).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> failed (`ucsv` missing from direct-policy guard set); fixed by adding `ucsv` to `DIRECT_POLICY_GUARD_MODELS`.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_direct_policy_guard.py::test_guard_set_matches_model_specs --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`249 passed`, existing warnings).
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` -> failed on missing local ndarray annotations in `models/bayesian.py`; added annotations.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` -> passed.
- Final drift gates: `gen_model_overview --check`, `gen_policy_matrix --check`, and `python -m tools.docgen --check docs/reference` all passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/reference --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`20 passed`).
- `rg -n "except Exception|except:" macroforecast/models/bayesian.py macroforecast/models/model_averaging.py tests/models/test_standard_estimators.py` -> no matches.
