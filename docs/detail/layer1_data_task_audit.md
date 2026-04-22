# Layer 1 Data Task Audit

Date: 2026-04-22

Scope: `1_data_task` registry, compiler contract, runtime dispatch, and user docs. Criterion: no Layer 1 value may be merely printable in a recipe while depending on undocumented runtime failure. A value is kept only if it is executable with defaults or has compile-time validation for the extra inputs it requires.

## Result

Layer 1 is closed for the current execution contract, and its semantic boundary
has been narrowed by the layer migration pass.

- Historical audit axis count before semantic migration: 20.
- Historical audit operational value count before semantic migration: 76.
- Compile statuses used: `executable`, `ready_for_sweep_runner`, `ready_for_wrapper_runner`, `ready_for_replication_runner`, `not_supported`, `blocked_by_incompatibility`.
- No printable-only execution class remains.

Migration note: several axes documented in this audit were originally discovered
under Layer 1 but now have a different canonical owner. See
`layer_boundary_contract.md` and `layer_axis_migration_plan.md` before extending
Layer 1.

## Canonical Layer 1 After Migration

Layer 1 now means official data frame and task identity. The canonical Layer 1
registry axes are:

- `dataset`
- `dataset_source`
- `frequency`
- `information_set_type`
- `official_transform_policy`
- `official_transform_scope`
- `task`
- `missing_availability`
- `release_lag_rule`
- `contemporaneous_x_rule`
- `variable_universe`

Target, targets, horizons, sample start/end, data vintage, and official
availability reports remain Layer 1 `leaf_config`/provenance responsibilities.

The following axes were moved out of Layer 1 ownership:

- to Layer 2: `horizon_target_construction`, `deterministic_components`,
  `structural_break_segmentation`
- to Layer 3: `benchmark_family`, `forecast_type`, `forecast_object`,
  `predictor_family`, `min_train_size`, `training_start_rule`
- to Layer 4: `oos_period`
- to Layer 6: `overlap_handling`

## Full Contract Decisions

### 1.1 Source & Frame

- `dataset` keeps `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd`.
- Standalone `fred_sd` requires explicit `frequency`; the source contains monthly and quarterly state series, so an implicit default would hide a research decision.
- `fred_md+fred_sd` is fixed to monthly.
- `fred_qd+fred_sd` is fixed to quarterly.
- Runtime conversion is active and documented: monthly to quarterly uses 3-month average; quarterly to monthly uses linear interpolation. Both paths report provenance warnings.
- `dataset_source=custom_csv/custom_parquet` remains operational but requires `leaf_config.custom_data_path`.
- Official dataset transforms now have canonical Layer 1 axes:
  `official_transform_policy` and `official_transform_scope`.
- Old Layer 2 t-code fields remain accepted as runtime bridge fields. The
  compiler derives the Layer 1 official-transform spec from those fields when
  old recipes omit the new axes, and rejects conflicting new/legacy choices.

### 1.2 Task & Target

- `forecast_type` remains dynamic by `feature_builder`: autoreg paths default to `iterated`, raw/factor panel paths default to `direct`.
- Crossed `forecast_type`/`feature_builder` pairs are `blocked_by_incompatibility`.
- `forecast_object=point_median` and `forecast_object=quantile` now require `model_family=quantile_linear`.
- `model_family=quantile_linear` still requires `forecast_object` in `{point_median, quantile}`.
- This closes the previous silent failure mode where a non-quantile model could compile a `quantile` manifest but emit a point-mean forecast.

### 1.3 Horizon & Evaluation Window

- Existing runtime closure stands: min-train rules, fixed training start, OOS regime filtering, and HAC overlap guards all have concrete execution or compiler blocks.
- No Layer 1 change was needed in this pass.

### 1.4 Benchmark & Predictor Universe

Kept values that require extra user inputs are now compile-time contracts:

- `benchmark_family=multi_benchmark_suite` requires non-empty `leaf_config.benchmark_suite`.
- Multi-benchmark suite members are restricted to `historical_mean`, `zero_change`, `ar_bic`, `rolling_mean`, `ar_fixed_p`, `ardi`.
- `paper_specific_benchmark` requires `leaf_config.paper_forecast_series` entries for the current target(s).
- `survey_forecast` requires `leaf_config.survey_forecast_series` entries for the current target(s).
- `expert_benchmark` requires `leaf_config.benchmark_config.expert_callable`.
- `variable_universe=handpicked_set` requires `leaf_config.variable_universe_columns`.
- `variable_universe=category_subset` requires `leaf_config.variable_universe_category_columns` and `leaf_config.variable_universe_category`.
- `variable_universe=target_specific_subset` requires `leaf_config.target_specific_columns` entries for the current target(s).
- `predictor_family=handpicked_set` requires `leaf_config.handpicked_columns`.
- `predictor_family=category_based` requires `leaf_config.predictor_category_columns` and `leaf_config.predictor_category`.
- `deterministic_components=break_dummies` requires `leaf_config.break_dates`.

### 1.5 Data Handling Policies

- `missing_availability=zero_fill_before_start` is the default policy in the compiler and public experiment defaults.
- `zero_fill_before_start` is sample-period aware: predictor leading missing values are zero-filled, fully missing predictors are zero-filled with warnings, predictor mid-sample missing values are reported, target leading missing values are reported, and target mid-sample missing values block execution.
- `missing_availability=x_impute_only` requires `leaf_config.x_imputation` in `{mean, median, ffill, bfill}`.
- `release_lag_rule=series_specific_lag` requires non-empty `leaf_config.release_lag_per_series`.
- `structural_break_segmentation` remains executable through fixed built-in dates; user-supplied break dates are owned by `deterministic_components=break_dummies`.

## Simple Contract

The simple API remains narrower than the full contract:

- User must choose dataset, target(s), horizons, start, and end.
- MD/QD frequency is inferred.
- Standalone SD frequency is required.
- MD+SD and QD+SD choose monthly/quarterly automatically.
- Default official transform policy is dataset T-code on both target and
  predictors.
- Default missing policy is `zero_fill_before_start`.
- Custom model and custom preprocessor extension points remain the intended first user-facing sweep examples.

Simple docs should stay short; this audit is the detailed contract source for Layer 1.

## Regression Checks Added

- Standalone FRED-SD without explicit frequency fails at compile time.
- FRED-SD composite frequency conflicts fail at compile time.
- Distributional forecast objects with non-quantile models produce `blocked_by_incompatibility`.
- Runtime-only missing leaf_config failures for Layer 1.4 and 1.5 are now compiler validation errors.
- Multi-benchmark suites reject unsupported member families before execution.
- Layer 1 official-transform axes are recorded in `data_task_spec`, and
  conflicting Layer 1 official-transform choices vs legacy Layer 2 t-code bridge
  choices fail at compile time.
