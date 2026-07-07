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

## Gate log

- `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation/test_metrics.py tests/pipeline/test_evalspec_threading.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`49 passed`, one existing rescore warning).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_standard_estimators.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> passed (`4 passed`) after CSR addition.
