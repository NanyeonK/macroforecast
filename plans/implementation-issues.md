
# Macrocast Implementation Issues — Full Option-Universe Build-Out

Status: active implementation tracker
Date: 2026-04-15
Depends on: full-option-universe-registry.md
Purpose: concrete issue-level breakdown for expanding from current ~23% axis coverage to full registry

---

## Notation

- `[BLOCKED]` — cannot start until dependency completes
- `[READY]` — all dependencies met, can start now
- `[IN-PROGRESS]` — actively being worked on
- `[DONE]` — completed and tested
- `pri:critical` / `pri:high` / `pri:medium` / `pri:low`
- `size:S` (<50 LOC) / `size:M` (50-200) / `size:L` (200-500) / `size:XL` (500+)

---

# Epic 0: Registry Architecture Refactor

> Current state: typed registry base classes, per-stage registry packages, auto-discovery loader, and the live 25 axes migrated out of the monolithic dict.
> Next state: expand the new architecture with Stage 0 meta axes like `experiment_unit`, `axis_type`, and `reproducibility_mode`.

## Issue 0-1: Registry entry base classes [DONE]
pri:critical | size:M | deps: none

### What
Create typed base classes that allow per-axis custom fields beyond the current generic `AxisRegistryEntry`.

### Files to create
- `macrocast/registry/base.py`

### Schema
```python
@dataclass(frozen=True)
class BaseRegistryEntry:
    id: str
    description: str
    status: SupportStatus
    priority: Literal["A", "B"]

@dataclass(frozen=True)
class EnumRegistryEntry(BaseRegistryEntry):
    pass

@dataclass(frozen=True)
class AxisDefinition:
    axis_name: str
    layer: str
    axis_type: Literal["enum", "numeric", "callable", "plugin"]
    default_policy: Literal["fixed", "sweep", "conditional"]
    entries: tuple[BaseRegistryEntry, ...]
    compatible_with: dict[str, tuple[str, ...]]
    incompatible_with: dict[str, tuple[str, ...]]
```

### Acceptance criteria
- [x] BaseRegistryEntry, EnumRegistryEntry, AxisDefinition defined
- [x] Existing `AxisRegistryEntry` still works (backward compat during migration)
- [x] Existing tests pass without modification

---

## Issue 0-2: Per-stage registry directory skeleton [DONE]
pri:critical | size:S | deps: 0-1

### What
Create directory structure for per-axis registry files.

### Directories to create
```
macrocast/registry/
  stage0/
    __init__.py
  data/
    __init__.py
  preprocessing/
    __init__.py
  training/
    __init__.py
  evaluation/
    __init__.py
  output/
    __init__.py
  tests/
    __init__.py
  importance/
    __init__.py
```

### Acceptance criteria
- [x] All directories and `__init__.py` files exist
- [x] Package import still works

---

## Issue 0-3: Central registry loader refactor [DONE]
pri:critical | size:L | deps: 0-1, 0-2

### What
Replace the monolithic `_AXIS_REGISTRY` dict in `build.py` with auto-discovery that loads from per-stage directories. Existing `get_axis_registry()`, `get_axis_registry_entry()`, `axis_governance_table()` must keep the same return types.

### Files to modify
- `macrocast/registry/build.py` — rewrite to auto-discover
- `macrocast/registry/__init__.py` — update exports

### Acceptance criteria
- [x] `get_axis_registry()` returns all axes from per-stage files
- [x] `get_axis_registry_entry("model_family")` still works
- [x] `axis_governance_table()` output unchanged
- [x] All 90 existing tests pass
- [x] Compiler pipeline end-to-end still works

---

## Issue 0-4: Migration of existing 25 axes to per-axis files [DONE]
pri:critical | size:L | deps: 0-3

### What
Move every axis currently in the monolithic dict into its own file under the correct stage directory.

### Files to create (one per existing axis)
- `macrocast/registry/stage0/study_mode.py`
- `macrocast/registry/data/dataset.py`
- `macrocast/registry/data/info_set.py`
- `macrocast/registry/data/task.py`
- `macrocast/registry/preprocessing/target_transform_policy.py`
- `macrocast/registry/preprocessing/x_transform_policy.py`
- `macrocast/registry/preprocessing/tcode_policy.py`
- `macrocast/registry/preprocessing/target_missing_policy.py`
- `macrocast/registry/preprocessing/x_missing_policy.py`
- `macrocast/registry/preprocessing/target_outlier_policy.py`
- `macrocast/registry/preprocessing/x_outlier_policy.py`
- `macrocast/registry/preprocessing/scaling_policy.py`
- `macrocast/registry/preprocessing/dimensionality_reduction_policy.py`
- `macrocast/registry/preprocessing/feature_selection_policy.py`
- `macrocast/registry/preprocessing/preprocess_order.py`
- `macrocast/registry/preprocessing/preprocess_fit_scope.py`
- `macrocast/registry/preprocessing/inverse_transform_policy.py`
- `macrocast/registry/preprocessing/evaluation_scale.py`
- `macrocast/registry/training/framework.py`
- `macrocast/registry/training/benchmark_family.py`
- `macrocast/registry/training/model_family.py`
- `macrocast/registry/training/feature_builder.py`
- `macrocast/registry/evaluation/primary_metric.py`
- `macrocast/registry/tests/stat_test.py`
- `macrocast/registry/importance/importance_method.py`

### Acceptance criteria
- [x] Old monolithic dict removed from `build.py`
- [x] All 25 axes loadable from new files
- [ ] All 90 existing tests pass
- [x] Compiler + execution pipeline unchanged

---

# Epic 1: Stage 0 — Meta / Package Grammar

> Current: 1 axis (study_mode), 4 values, 1 operational.
> Target: 6 axes, 45 values, ~15 operational.

## Issue 1-1: Register experiment_unit axis [BLOCKED on 0-4]
pri:critical | size:M | deps: 0-4

### What
Register the `experiment_unit` axis with 12 values. This is the most important Stage 0 addition — it governs route ownership and wrapper behavior.

### File to create
- `macrocast/registry/stage0/experiment_unit.py`

### Values and status
```
single_target_single_model       operational
single_target_model_grid         operational
single_target_full_sweep         operational
multi_target_separate_runs       registry_only
multi_target_shared_design       planned
multi_output_joint_model         registry_only
hierarchical_forecasting_run     registry_only
panel_forecasting_run            registry_only
state_space_run                  registry_only
replication_recipe               registry_only
benchmark_suite                  planned
ablation_study                   planned
```

### Schema (custom fields)
```python
@dataclass(frozen=True)
class ExperimentUnitEntry(EnumRegistryEntry):
    route_owner: Literal["single_run", "wrapper", "orchestrator", "replication"]
    requires_multi_target: bool
    requires_wrapper: bool
```

### Integration points
- `macrocast/compiler/build.py` — `_build_stage0_and_recipe()` must read experiment_unit
- `macrocast/stage0/derive.py` — experiment_unit should inform `execution_posture` derivation
- `macrocast/start.py` — wizard keys must include experiment_unit

### Acceptance criteria
- [ ] Axis registered with all 12 values
- [ ] Compiler reads and validates experiment_unit from recipe YAML
- [ ] Route ownership derivable from experiment_unit
- [ ] Stage 0 frame includes experiment_unit
- [ ] Existing tests pass + new test for experiment_unit validation

---

## Issue 1-2: Register axis_type axis [BLOCKED on 0-4]
pri:high | size:S | deps: 0-4

### What
Register axis_type as a meta-governance axis that describes what role each axis plays in a study.

### File to create
- `macrocast/registry/stage0/axis_type.py`

### Values
```
fixed                operational
sweep                operational
nested_sweep         planned
conditional          operational
derived              operational
eval_only            registry_only
report_only          registry_only
```

### Integration
- Used by `AxisDefinition.default_policy` to validate axis selection mode at compile time
- Compiler should warn if a `fixed`-policy axis is placed in sweep_axes

### Acceptance criteria
- [ ] Axis registered
- [ ] Compiler validates axis_type consistency

---

## Issue 1-3: Register registry_type axis [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### File to create
- `macrocast/registry/stage0/registry_type.py`

### Values
```
enum_registry        operational
numeric_registry     operational
callable_registry    planned
custom_plugin        planned
user_defined_yaml    registry_only
external_adapter     registry_only
```

### Acceptance criteria
- [ ] Axis registered
- [ ] AxisDefinition carries registry_type field

---

## Issue 1-4: Register reproducibility_mode axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File to create
- `macrocast/registry/stage0/reproducibility_mode.py`

### Values
```
strict_reproducible      planned
seeded_reproducible      operational
best_effort              operational
exploratory              registry_only
```

### Integration
- Recipe YAML gets optional `reproducibility_mode` field
- Compiler validates seed presence when `strict_reproducible` or `seeded_reproducible`
- Execution layer respects seed policy

### Acceptance criteria
- [ ] Axis registered
- [ ] Compiler validates seed consistency
- [ ] Manifest records reproducibility_mode

---

## Issue 1-5: Register failure_policy axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File to create
- `macrocast/registry/stage0/failure_policy.py`

### Values
```
fail_fast                operational
skip_failed_cell         planned
skip_failed_model        planned
retry_then_skip          registry_only
fallback_to_default_hp   registry_only
save_partial_results     planned
warn_only                registry_only
hard_error               operational
```

### Integration
- Execution layer (`execution/build.py`) needs try/catch per model per horizon per date
- Currently hard-coded to fail_fast behavior
- `skip_failed_model` requires partial result collection and manifest marking

### Acceptance criteria
- [ ] Axis registered
- [ ] `fail_fast` and `hard_error` work (current behavior preserved)
- [ ] `skip_failed_model` operational with partial manifest
- [ ] `save_partial_results` operational

---

## Issue 1-6: Register compute_mode axis [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### File to create
- `macrocast/registry/stage0/compute_mode.py`

### Values
```
serial                   operational
parallel_by_model        planned
parallel_by_horizon      planned
parallel_by_oos_date     registry_only
parallel_by_trial        registry_only
gpu_single               registry_only
gpu_multi                registry_only
distributed_cluster      registry_only
```

### Integration
- Execution loop in `execution/build.py` currently runs serial only
- `parallel_by_model` needs joblib/concurrent.futures wrapper
- `parallel_by_horizon` needs the same
- GPU/distributed remain registry_only stubs

### Acceptance criteria
- [ ] Axis registered
- [ ] `serial` works (current behavior)
- [ ] `parallel_by_model` operational with joblib backend
- [ ] `parallel_by_horizon` operational with joblib backend

---

# Epic 2: Stage 1 — Data / Task Definition

> Current: 3 axes (dataset, info_set, task), 7 values, all operational.
> Target: 18 axes, ~150 values, ~50 operational.

## Issue 2-1: Register data_domain axis [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### File
`macrocast/registry/data/data_domain.py`

### Values (A-tier)
```
macro                    operational
macro_finance            planned
```
Plus B-tier: housing, energy, labor, regional, panel_macro, text_macro, mixed_domain

### Acceptance criteria
- [ ] Axis registered
- [ ] Compiler accepts data_domain in recipe (optional for v1)

---

## Issue 2-2: Expand dataset_source to full planned set [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/dataset_source.py`

### What
Expand from 3 values to full set. A-tier additions:
```
fred_api_custom          planned
custom_csv               planned
custom_parquet           planned
```
Plus B-tier: bea, bls, census, oecd, imf_ifs, ecb_sdw, bis, world_bank, wrds_macro_finance, survey_spf, etc.

### Acceptance criteria
- [ ] All dataset values registered
- [ ] Existing fred_md/qd/sd behavior preserved
- [ ] custom_csv adapter contract defined (implementation separate issue)

---

## Issue 2-3: Register frequency axis [BLOCKED on 0-4]
pri:high | size:S | deps: 0-4

### File
`macrocast/registry/data/frequency.py`

### Values
```
monthly                  operational
quarterly                operational
daily                    registry_only
weekly                   registry_only
yearly                   registry_only
mixed_frequency          registry_only
```

---

## Issue 2-4: Register information_set_type with vintage integration [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/information_set.py`

### What
Replace current simple `info_set` (revised/real_time) with richer information_set_type axis.

### Values
```
revised                          operational
real_time_vintage                operational
pseudo_oos_revised               planned
pseudo_oos_vintage_aware         registry_only
release_calendar_aware           registry_only
publication_lag_aware            registry_only
```

### Migration
- Current `info_set` axis → rename to `information_set_type`
- Compiler mapping needs update
- Existing recipe YAMLs need migration path or backward-compat alias

---

## Issue 2-5: Register vintage_policy axis [BLOCKED on 0-4]
pri:high | size:S | deps: 0-4

### File
`macrocast/registry/data/vintage_policy.py`

### Values
```
latest_only              operational
single_vintage           operational
rolling_vintage          planned
all_vintages_available   registry_only
vintage_range            registry_only
```

---

## Issue 2-6: Register alignment_rule axis [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### File
`macrocast/registry/data/alignment_rule.py`

---

## Issue 2-7: Register forecast_type axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/forecast_type.py`

### Values (A-tier)
```
direct                   operational
iterated                 planned
```
Plus B-tier: dirrec, mimo, multi_horizon_joint, recursive_state_space, seq2seq

### Integration
- Execution layer needs iterated forecast loop (recursive 1-step)
- Currently only direct is implemented

---

## Issue 2-8: Register forecast_object axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/forecast_object.py`

### Values (A-tier)
```
point_mean               operational
point_median             planned
quantile                 planned
direction                planned
```

### Integration
- Quantile forecasting needs quantile loss + quantile-capable models
- Direction needs threshold + classification metrics

---

## Issue 2-9: Register horizon_target_construction axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/horizon_target.py`

### Values (A-tier)
```
future_level_y_t_plus_h  operational
future_diff              planned
future_logdiff           planned
cumulative_growth_to_h   planned
annualized_growth_to_h   planned
```

### Integration
- Target construction in execution layer currently only supports level
- Each construction type needs a transform function + inverse

---

## Issue 2-10: Register target_family axis [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### File
`macrocast/registry/data/target_family.py`

---

## Issue 2-11: Register predictor_family axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### File
`macrocast/registry/data/predictor_family.py`

### Values (A-tier)
```
target_lags_only         operational
all_macro_vars           operational
all_except_target        planned
category_based           planned
factor_only              planned
```

### Integration
- Feature builder in execution layer needs predictor_family routing
- `category_based` needs FRED-MD category metadata

---

## Issue 2-12: Register overlap_handling axis [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### File
`macrocast/registry/data/overlap_handling.py`

---

## Issue 2-13: Register sample/split axes bundle [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Files (bundle — one issue because tightly coupled)
- `macrocast/registry/data/training_start.py`
- `macrocast/registry/data/oos_period.py`
- `macrocast/registry/data/min_train_size.py`
- `macrocast/registry/data/warmup_rule.py`
- `macrocast/registry/data/structural_break.py`

### Integration
- Execution layer currently uses `_DEFAULT_MINIMUM_TRAIN_SIZE = 5`
- Need to read min_train_size from recipe/registry
- structural_break needs date-based sample segmentation

---

## Issue 2-14: Register target/predictor design axes bundle [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files
- `macrocast/registry/data/contemporaneous_x.py`
- `macrocast/registry/data/own_target_lags.py`
- `macrocast/registry/data/deterministic_components.py`
- `macrocast/registry/data/exogenous_block.py`

---

## Issue 2-15: Register multi-target design axes bundle [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files
- `macrocast/registry/data/x_map_policy.py`
- `macrocast/registry/data/target_to_target.py`
- `macrocast/registry/data/multi_target_architecture.py`

---

## Issue 2-16: Register benchmark_family expansion + regime_task [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Files
- `macrocast/registry/data/benchmark_family.py` (expand from 4 → 10 values)
- `macrocast/registry/data/regime_task.py`
- `macrocast/registry/data/evaluation_scale.py` (move from preprocessing to data/task)

---

# Epic 3: Stage 2 — Preprocessing Governance Completion

> Current: 14 axes, 52 values, 17 operational.
> Target: 22 axes, ~180 values, ~50 operational.
> Critical: 4 mandatory governance fields must be added.

## Issue 3-1: Add representation_policy governance axis [READY — no dep on registry refactor]
pri:critical | size:M | deps: none

### What
This is the first of the 4 mandatory governance fields currently missing.

### File to create
- `macrocast/registry/preprocessing/representation_policy.py`

### Values
```
raw_only                 operational
tcode_only               planned
custom_transform_only    registry_only
```

### Integration
- Add to `PreprocessContract` dataclass
- Add to `build_preprocess_contract()`
- Add to compiler `_build_preprocess_contract()`
- Add to recipe YAML schema
- Validate: if representation_policy=raw_only, tcode_policy must be raw_only or extra_preprocess_without_tcode

### Acceptance criteria
- [ ] Field in PreprocessContract
- [ ] Compiler validates representation_policy
- [ ] Manifest records it
- [ ] Existing tests pass (raw_only is backward-compatible default)

---

## Issue 3-2: Add preprocessing_axis_role governance axis [READY]
pri:critical | size:M | deps: none

### What
Second mandatory governance field. Controls whether preprocessing is fixed, swept, or ablation target.

### File to create
- `macrocast/registry/preprocessing/axis_role.py`

### Values
```
fixed_preprocessing      operational
swept_preprocessing      planned
ablation_preprocessing   planned
```

### Integration
- Add to PreprocessContract
- `check_preprocess_governance()` should enforce: if axis_role=fixed and model_sweep=true, preprocessing must not change across models
- Manifest records preprocessing_axis_role

---

## Issue 3-3: Expand tcode_policy to full governance set [READY]
pri:critical | size:M | deps: none

### What
Current tcode_policy has 6 values but the plan requires explicit target/X separation.
Add the 4 explicit governance values as overlay:

### Add to registry
```
apply_tcode_to_target    planned
apply_tcode_to_X         planned
apply_tcode_to_both      planned
apply_tcode_to_none      operational (alias for raw_only/extra_preprocess_without_tcode)
```

### Decision
Either:
(a) Replace current tcode_policy values with new ones, or
(b) Add a separate `tcode_application_scope` axis alongside existing tcode_policy

Recommended: (b) — add `tcode_application_scope` as separate axis, keep existing tcode_policy for ordering semantics.

---

## Issue 3-4: Expand x_missing values [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### What
Expand from 4 values (none, drop, em_impute, custom) to full A-tier set.

### Values to add
```
mean_impute              planned → operational
median_impute            planned → operational
ffill                    planned → operational
interpolate_linear       planned → operational
drop_rows                planned
drop_columns             planned
drop_if_above_threshold  planned
missing_indicator        planned
```

### Implementation
- Each imputation method needs a function in execution layer
- Must respect train-only fit scope
- EM impute already operational — use as template

---

## Issue 3-5: Expand x_outlier values [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Values to add
```
winsorize                planned → operational
trim                     planned
iqr_clip                 planned → operational
mad_clip                 planned
zscore_clip              planned → operational
outlier_to_missing       planned
```

---

## Issue 3-6: Expand scaling values [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Values to add
```
demean_only              planned
unit_variance_only       planned
minmax                   planned → operational
rank_scale               registry_only
```

---

## Issue 3-7: Implement PCA/static_factor dimensionality reduction [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### What
Move `pca` and `static_factor` from planned to operational.

### Implementation
- sklearn PCA with train-only fit
- Factor extraction using eigendecomposition
- Factor count selection (fixed, variance_explained, BaiNg)
- Integration with feature_builder `factors_plus_AR`

---

## Issue 3-8: Implement correlation_screen / lasso_select feature selection [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

---

## Issue 3-9: Register additional preprocessing axes [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files to create
- `macrocast/registry/preprocessing/target_transform.py` (level, diff, log, logdiff, growth, etc.)
- `macrocast/registry/preprocessing/target_normalization.py`
- `macrocast/registry/preprocessing/target_domain.py`
- `macrocast/registry/preprocessing/scaling_scope.py`
- `macrocast/registry/preprocessing/additional.py` (hp_filter, seasonal_adj, etc.)
- `macrocast/registry/preprocessing/x_lag.py`
- `macrocast/registry/preprocessing/feature_grouping.py`
- `macrocast/registry/preprocessing/recipe_mode.py`

---

# Epic 4: Stage 3 — Training / Forecasting Expansion

> Current: 4 axes, 14 values, 12 operational.
> Target: 22 axes, ~200 values, ~60 operational.

## Issue 4-1: Register outer_window expansion [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### What
Add `anchored_rolling` to operational.

### File
`macrocast/registry/training/outer_window.py`

### Implementation
- Anchored rolling: fixed start, rolling end, expanding train window up to max then rolling
- Modify execution loop in `execution/build.py`

---

## Issue 4-2: Register refit_policy axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Values (A-tier)
```
refit_every_step         operational
refit_every_k_steps      planned → operational
fit_once_predict_many    planned → operational
```

### Implementation
- Currently always refits every step
- `refit_every_k_steps` needs step counter in OOS loop
- `fit_once_predict_many` needs fit-once-then-predict logic

---

## Issue 4-3: Add A-tier model families — OLS, AdaptiveLasso, BayesianRidge [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Models to implement
```python
# OLS — sklearn LinearRegression or statsmodels OLS
# AdaptiveLasso — two-stage: OLS weights → weighted Lasso
# BayesianRidge — sklearn BayesianRidge
# factor_augmented_linear — PCA factors + Ridge/Lasso (depends on 3-7)
```

### Template
Use existing Ridge executor as template. Each model needs:
- sklearn/statsmodels adapter
- HP grid definition
- Importance extraction method
- Compatibility declaration (which feature_builders work)

---

## Issue 4-4: Add A-tier model families — SVR_linear, SVR_rbf [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

---

## Issue 4-5: Add A-tier model families — ExtraTrees, GBM, XGBoost, LightGBM [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Models
```python
# ExtraTrees — sklearn ExtraTreesRegressor
# GradientBoosting — sklearn GradientBoostingRegressor
# XGBoost — xgboost.XGBRegressor
# LightGBM — lightgbm.LGBMRegressor
```

### Each needs
- Import guard (xgboost/lightgbm may not be installed)
- HP grid
- Native importance extraction
- Test with local fixture data

---

## Issue 4-6: Add MLP model [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Implementation
- sklearn MLPRegressor for v1 (no pytorch dep)
- HP grid: hidden_layer_sizes, activation, alpha, learning_rate

---

## Issue 4-7: Add sklearn_adapter / statsmodels_adapter plugin bridge [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### What
Generic adapter that wraps any sklearn estimator or statsmodels model behind the macrocast model executor contract.

### Interface
```python
def sklearn_model_executor(estimator_class, hp_grid, importance_method):
    ...
```

---

## Issue 4-8: Register validation/split axes bundle [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Files
- `macrocast/registry/training/validation_size.py`
- `macrocast/registry/training/validation_location.py`
- `macrocast/registry/training/embargo.py`
- `macrocast/registry/training/split_family.py`
- `macrocast/registry/training/shuffle_rule.py`
- `macrocast/registry/training/alignment_fairness.py`

### Implementation for A-tier
- `blocked_kfold` — sklearn TimeSeriesSplit adapter
- `expanding_cv` — custom expanding window CV
- `rolling_cv` — custom rolling window CV
- `walk_forward_validation` — alias for expanding/rolling OOS

---

## Issue 4-9: Register tuning axes bundle [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Files
- `macrocast/registry/training/search_algorithm.py`
- `macrocast/registry/training/tuning_objective.py`
- `macrocast/registry/training/tuning_budget.py`
- `macrocast/registry/training/hp_space.py`
- `macrocast/registry/training/seed_policy.py`
- `macrocast/registry/training/early_stopping.py`
- `macrocast/registry/training/convergence.py`

### Implementation for A-tier
- `grid_search` — exhaustive sklearn-style
- `random_search` — sklearn RandomizedSearchCV style
- Budget enforcement: max_trials, max_time, early_stop

---

## Issue 4-10: Register feature construction axes [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files
- `macrocast/registry/training/feature_builder.py` (expand)
- `macrocast/registry/training/y_lag_count.py`
- `macrocast/registry/training/factor_count.py`
- `macrocast/registry/training/lookback.py`

---

## Issue 4-11: Register execution runtime axes [BLOCKED on 0-4]
pri:low | size:S | deps: 0-4

### Files
- `macrocast/registry/training/logging_level.py`
- `macrocast/registry/training/checkpointing.py`
- `macrocast/registry/training/cache_policy.py`
- `macrocast/registry/training/execution_backend.py`

---

## Issue 4-12: Register horizon_modelization axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Values (A-tier)
```
separate_model_per_h     operational
recursive_one_step_model planned
```

### Implementation
- `recursive_one_step_model` needs iterated forecast loop (coupled with Issue 2-7 forecast_type=iterated)

---

# Epic 5: Stage 4 — Evaluation Expansion

> Current: 1 axis (primary_metric), 4 values, all operational.
> Target: 17 axes, ~80 values, ~30 operational.

## Issue 5-1: Expand point_forecast_metrics [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Values to add
```
RMSE                     planned → operational
MAE                      planned → operational
MAPE                     planned → operational
```

### Implementation
- Simple metric functions
- Integrate into execution metric computation loop

---

## Issue 5-2: Expand relative_metrics [BLOCKED on 0-4]
pri:high | size:S | deps: 0-4

### Values to add
```
relative_RMSE            planned
relative_MAE             planned
benchmark_win_rate       planned
```

---

## Issue 5-3: Add direction/event metrics [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Values
```
directional_accuracy     planned
sign_accuracy            planned
```

---

## Issue 5-4: Register aggregation axes bundle [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Files
- `macrocast/registry/evaluation/agg_time.py`
- `macrocast/registry/evaluation/agg_horizon.py`
- `macrocast/registry/evaluation/agg_target.py`
- `macrocast/registry/evaluation/ranking.py`
- `macrocast/registry/evaluation/report_style.py`

---

## Issue 5-5: Register benchmark evaluation axes [BLOCKED on 0-4]
pri:medium | size:S | deps: 0-4

### Files
- `macrocast/registry/evaluation/benchmark_window.py`
- `macrocast/registry/evaluation/benchmark_scope.py`

---

## Issue 5-6: Register regime evaluation axes [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files
- `macrocast/registry/evaluation/regime_definition.py`
- `macrocast/registry/evaluation/regime_use.py`
- `macrocast/registry/evaluation/regime_metrics.py`

### Implementation for A-tier
- NBER recession dates lookup table
- Metric computation conditional on regime indicator

---

## Issue 5-7: Register decomposition axes [BLOCKED on 0-4]
pri:low | size:S | deps: 0-4

### Files
- `macrocast/registry/evaluation/decomposition_target.py`
- `macrocast/registry/evaluation/decomposition_order.py`

---

# Epic 6: Stage 5 — Output / Provenance

> Current: 0 registered axes (all hardcoded).
> Target: 4 axes.

## Issue 6-1: Register output axes bundle [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Files
- `macrocast/registry/output/saved_objects.py`
- `macrocast/registry/output/provenance_fields.py`
- `macrocast/registry/output/export_format.py`
- `macrocast/registry/output/artifact_granularity.py`

### Integration
- Current hardcoded manifest fields → driven by provenance_fields registry
- Current JSON-only output → configurable export_format
- Add parquet/csv export alongside JSON

---

## Issue 6-2: Add config_hash + git_commit + package_version to manifest [BLOCKED on 6-1]
pri:high | size:M | deps: 6-1

### What
Compute deterministic hash of recipe config and record git commit + macrocast version.

---

# Epic 7: Stage 6 — Statistical Tests Expansion

> Current: 1 axis, 4 values, 3 operational.
> Target: 9 axes, ~50 values, ~25 operational.

## Issue 7-1: Implement MCS (Model Confidence Set) [BLOCKED on 0-4]
pri:critical | size:XL | deps: 0-4

### What
Most important missing statistical test. Needed for multi-model comparison.

### Implementation
- Hansen, Lunde, Nason (2011) MCS procedure
- Bootstrap-based elimination
- Returns confidence set at given alpha

---

## Issue 7-2: Add DM variants (HLN small-sample, modified DM) [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Implementation
- HLN correction: finite-sample bias adjustment
- Modified DM: long-horizon adjustment

---

## Issue 7-3: Add nested model tests (ENC_NEW, MSE_F, MSE_t) [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

---

## Issue 7-4: Add conditional predictive ability tests [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

### Tests
- Giacomini-White CPA
- Rossi-Sekhposyan forecast stability
- Rolling DM

---

## Issue 7-5: Add White Reality Check / Hansen SPA [BLOCKED on 0-4]
pri:high | size:L | deps: 0-4

---

## Issue 7-6: Register dependence correction axis [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Values to operationalize
- Newey-West HAC
- HAC auto-bandwidth
- Block bootstrap

---

## Issue 7-7: Add residual/calibration diagnostics bundle [BLOCKED on 0-4]
pri:medium | size:L | deps: 0-4

### Tests
- Mincer-Zarnowitz regression
- Ljung-Box on errors
- ARCH-LM on errors
- Bias test

---

## Issue 7-8: Add direction/classification tests [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Tests
- Pesaran-Timmermann
- Binomial hit test

---

# Epic 8: Stage 7 — Importance / Interpretability Expansion

> Current: 1 axis, 3 values, 2 operational.
> Target: 12 axes, ~100 values, ~40 operational.

## Issue 8-1: Implement TreeSHAP [BLOCKED on 0-4, 4-5]
pri:critical | size:L | deps: 0-4, 4-5 (needs tree models)

### Implementation
- `shap.TreeExplainer` for RF/XGBoost/LightGBM
- Per-window SHAP values
- Time-averaged SHAP summary
- Horizon-split SHAP

---

## Issue 8-2: Implement KernelSHAP + LinearSHAP [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Implementation
- KernelSHAP for model-agnostic
- LinearSHAP for Ridge/Lasso/ElasticNet (exact, fast)

---

## Issue 8-3: Implement PFI (permutation feature importance) [BLOCKED on 0-4]
pri:high | size:M | deps: 0-4

### Implementation
- Model-agnostic permutation importance
- Group permutation importance (by FRED category, lag block)
- Conditional permutation importance (registry_only for now)

---

## Issue 8-4: Implement LIME + feature ablation [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

---

## Issue 8-5: Implement PDP / ICE / ALE [BLOCKED on 0-4]
pri:medium | size:L | deps: 0-4

---

## Issue 8-6: Implement grouped importance [BLOCKED on 0-4, 8-1 or 8-3]
pri:high | size:M | deps: 0-4, (8-1 or 8-3)

### What
Aggregate importance by FRED category, economic theme, lag block, factor block.

### Requires
- FRED-MD category metadata table
- Importance values from SHAP or PFI

---

## Issue 8-7: Implement importance stability analysis [BLOCKED on 8-1 or 8-3]
pri:high | size:M | deps: 8-1 or 8-3

### What
- Bootstrap rank stability
- Seed stability
- Window stability
- Model consensus importance
- Rank correlation across runs
- Sign consistency

---

## Issue 8-8: Register remaining importance axes [BLOCKED on 0-4]
pri:medium | size:M | deps: 0-4

### Files
- `macrocast/registry/importance/scope.py`
- `macrocast/registry/importance/model_native.py`
- `macrocast/registry/importance/model_agnostic.py`
- `macrocast/registry/importance/shap.py`
- `macrocast/registry/importance/local_surrogate.py`
- `macrocast/registry/importance/partial_dependence.py`
- `macrocast/registry/importance/grouped.py`
- `macrocast/registry/importance/stability.py`
- `macrocast/registry/importance/aggregation.py`
- `macrocast/registry/importance/output_style.py`
- `macrocast/registry/importance/temporal.py` (B-tier only)
- `macrocast/registry/importance/gradient_path.py` (B-tier only)

---

# Execution Order (Critical Path)

```
Phase 1: Foundation (Epic 0)
  0-1 → 0-2 → 0-3 → 0-4
  Parallel: 3-1, 3-2, 3-3 (preprocessing governance — no dep on registry refactor)

Phase 2: Grammar Lock (Epic 1)
  1-1 (experiment_unit) ← critical
  1-2 (axis_type)
  1-4 (reproducibility_mode)
  1-5 (failure_policy)
  1-6 (compute_mode)
  1-3 (registry_type) ← lowest priority in this phase

Phase 3: Data/Task Semantics (Epic 2)
  2-4 (information_set_type) ← most impactful
  2-7 (forecast_type)
  2-8 (forecast_object)
  2-9 (horizon_target_construction)
  2-13 (sample/split bundle)
  2-2, 2-3, 2-5 (dataset/frequency/vintage expansion)
  2-11 (predictor_family)
  2-16 (benchmark expansion + regime)
  2-1, 2-6, 2-10, 2-12, 2-14, 2-15 (remaining)

Phase 4: Preprocessing Completion (Epic 3)
  3-4 (x_missing expansion) ← operational impact
  3-5 (x_outlier expansion)
  3-7 (PCA/factor)
  3-6, 3-8, 3-9 (remaining)

Phase 5: Training Expansion (Epic 4)
  4-5 (XGBoost/LightGBM/GBM/ExtraTrees) ← highest model impact
  4-3 (OLS/AdaptiveLasso/BayesianRidge)
  4-9 (tuning: grid_search, random_search)
  4-8 (validation/split)
  4-2 (refit_policy)
  4-1 (anchored_rolling)
  4-7 (sklearn/statsmodels adapter)
  4-12 (horizon_modelization)
  4-4, 4-6, 4-10, 4-11 (remaining)

Phase 6: Evaluation Expansion (Epic 5)
  5-1 (RMSE/MAE/MAPE)
  5-4 (aggregation bundle)
  5-6 (regime evaluation)
  5-2, 5-3, 5-5, 5-7 (remaining)

Phase 7: Tests + Output + Importance (Epics 6, 7, 8)
  7-1 (MCS) ← most impactful test
  8-1 (TreeSHAP) ← most impactful importance
  6-1 (output registry)
  7-2, 7-3, 7-4, 7-5 (test expansion)
  8-2, 8-3 (SHAP/PFI)
  8-6, 8-7 (grouped + stability)
  6-2, 7-6, 7-7, 7-8, 8-4, 8-5, 8-8 (remaining)
```

---

# Issue count summary

| Epic | Issues | Size breakdown |
|------|:------:|----------------|
| 0. Registry refactor | 4 | 1S + 1M + 2L |
| 1. Stage 0 grammar | 6 | 2S + 3M + 1M |
| 2. Stage 1 data/task | 16 | 5S + 9M + 2L |
| 3. Stage 2 preprocessing | 9 | 1S + 5M + 2L + 1L |
| 4. Stage 3 training | 12 | 1S + 6M + 4L + 1XL(ish) |
| 5. Stage 4 evaluation | 7 | 2S + 3M + 1L + 1S |
| 6. Stage 5 output | 2 | 2M |
| 7. Stage 6 tests | 8 | 1M + 4L + 1XL + 2M |
| 8. Stage 7 importance | 8 | 1S + 3M + 2L + 1M + 1L |
| **Total** | **72** | |

