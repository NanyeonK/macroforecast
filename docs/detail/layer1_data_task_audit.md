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

Layer 1 now means official data frame and target identity. The canonical Layer 1
registry axes are:

- `dataset`
- `source_adapter`
- `frequency`
- `information_set_type`
- `official_transform_policy`
- `official_transform_scope`
- `target_structure`
- `missing_availability`
- `raw_missing_policy`
- `raw_outlier_policy`
- `release_lag_rule`
- `variable_universe`

Target, targets, horizons, sample start/end, data vintage, and official
availability reports remain Layer 1 `leaf_config`/provenance responsibilities.

Full mode additionally distinguishes raw-source data defects from
post-transform/model-input defects. A raw-source missing value or raw-source
outlier is already present in the source panel before the official dataset
transform/T-code step. Layer 1 owns the policy and provenance for those raw
defects when the researcher chooses to inspect, fill, drop, winsorize, clip, or
otherwise repair the source panel before official transforms are applied.

Runtime now closes this boundary with `layer1_official_frame_v1`. Every run
writes `layer1_official_frame.json`, records the contract version and compact
summary in `manifest.json`, and registers the sidecar in
`artifact_manifest.json` as a Layer 1 artifact. Layer 2 representation builders
consume this resolved frame contract rather than relying on implicit
`raw_result.data` semantics.

The next Layer 1 runtime frontier should extend this contract rather than
creating parallel provenance fields. Vintage, release-lag, and mixed-source
work all need path-specific evidence in the official-frame sidecar before their
broader cells are marked selectable.

The following axes were moved out of Layer 1 ownership:

- to Layer 2: `horizon_target_construction`, `deterministic_components`,
  `predictor_family`, `structural_break_segmentation`
- to Layer 3: `benchmark_family` as baseline generator role assignment,
  `forecast_type`, `forecast_object`, `min_train_size`, `training_start_rule`
- to Layer 4: `oos_period`
- to Layer 6: `overlap_handling`

## Full Contract Decisions

### 1.1 Source & Frame

- `dataset` keeps `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd`.
- Standalone `fred_sd` requires explicit `frequency`; the source contains monthly and quarterly state series, so an implicit default would hide a research decision.
- `fred_md+fred_sd` is fixed to monthly.
- `fred_qd+fred_sd` is fixed to quarterly.
- Runtime conversion is active and documented: monthly to quarterly uses 3-month average; quarterly to monthly uses linear interpolation. Both paths report provenance warnings.
- FRED-SD's official live/current source remains the St. Louis Fed workbook.
  Local CSV is accepted as a fixture/runtime path for deterministic tests and
  replication bundles that should not require the optional Excel parser. Wide
  CSV columns use the same canonical `VARIABLE_STATE` names produced by the
  workbook adapter.
- `source_adapter=custom_csv/custom_parquet` remains operational but requires `leaf_config.custom_data_path`.
- Official dataset transforms now have canonical Layer 1 axes:
  `official_transform_policy` and `official_transform_scope`.
- Old Layer 2 t-code fields remain accepted as legacy compatibility inputs.
  The compiler derives the Layer 1 official-transform spec from those fields
  when old recipes omit the new axes, derives runtime `PreprocessContract`
  fallback fields from Layer 1 for new recipes, and rejects conflicting
  new/legacy choices.
- Legacy `dataset_source` remains accepted as a compiler alias for canonical
  `source_adapter`. New recipes, manifests, and docs should use
  `source_adapter`.
- Legacy `task` remains accepted as a compiler alias for canonical
  `target_structure`. New recipes, manifests, and docs should use
  `target_structure`.

### 1.2 Target Structure

- Layer 1 `target_structure` only records whether the official frame has one
  target or multiple targets.
- `single_target_point_forecast` requires `leaf_config.target`.
- `multi_target_point_forecast` requires `leaf_config.targets`.
- Layer 0 owns the execution shape derived from target cardinality through
  `experiment_unit`.
- `forecast_type` remains dynamic by the Layer 2 feature runtime: autoregressive paths default to `iterated`, raw/factor panel paths default to `direct`.
- Crossed `forecast_type` / feature-runtime pairs remain `blocked_by_incompatibility`; legacy `feature_builder` recipes are mapped to that runtime for compatibility.
- `forecast_object=point_median` and `forecast_object=quantile` now require `model_family=quantile_linear`.
- `model_family=quantile_linear` still requires `forecast_object` in `{point_median, quantile}`.
- This closes the previous silent failure mode where a non-quantile model could compile a `quantile` manifest but emit a point-mean forecast.
- New compiled specs write `forecast_type` and `forecast_object` under
  `training_spec`, not `data_task_spec`.

### 1.3 Horizon & Evaluation Window

- Existing runtime closure stands: min-train rules, fixed training start, OOS regime filtering, and HAC overlap guards all have concrete execution or compiler blocks.
- New compiled specs write `min_train_size`, `training_start_rule`, and
  `training_start_date` under `training_spec`; execution keeps old
  `data_task_spec` fallback for compatibility.
- New compiled specs write `horizon_target_construction` under
  `layer2_representation_spec.target_representation`, not `data_task_spec`.

### 1.4 Baseline Generator & Predictor Universe

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
- New compiled specs write `predictor_family` and `contemporaneous_x_rule`
  under `layer2_representation_spec.input_panel`.
- New compiled specs write `deterministic_components` and
  `structural_break_segmentation` under
  `layer2_representation_spec.feature_blocks.deterministic_feature_block`.

Layer 1 does not own benchmark selection. The old audit group keeps the
historical contract checks visible, but `benchmark_family` now belongs to Layer
3 as baseline generator role assignment. It remains in this section only to
record the compile-time input contracts discovered during the Layer 1 audit.

### 1.5 Data Handling Policies

- `layer1_official_frame_v1` now includes
  `source_availability_contract_v1`. It records requested dataset, source
  adapter, version mode, requested vintage, actual vintage, data-through,
  observed frame window, local-vs-remote source kind, artifact SHA/size/cache
  status, and component source contracts when a composite dataset is loaded.
- The source-availability contract has deterministic coverage for local-source
  current/vintage runs, cache-hit remote-source simulations that do not touch
  the network, and composite component source contracts.
- `missing_availability=zero_fill_before_start` is the default policy in the compiler and public experiment defaults.
- `zero_fill_before_start` is sample-period aware: predictor leading missing values are zero-filled, fully missing predictors are zero-filled with warnings, predictor mid-sample missing values are reported, target leading missing values are reported, and target mid-sample missing values block execution.
- `missing_availability=x_impute_only` requires `leaf_config.x_imputation` in `{mean, median, ffill, bfill}`.
- `missing_availability=available_case` and `missing_availability=x_impute_only` now write `data_reports.missing_availability` in `layer1_official_frame_v1`, including row-drop counts or predictor-imputation counts.
- `raw_missing_policy=preserve_raw_missing` is the default raw-source missing policy.
- `raw_missing_policy=zero_fill_leading_x_before_tcode` fills predictor leading missing values in the raw source panel before official transforms/T-codes.
- `raw_missing_policy=x_impute_raw` requires `leaf_config.raw_x_imputation` in `{mean, median, ffill, bfill}` and imputes raw predictors before official transforms/T-codes.
- `raw_missing_policy=drop_rows_with_raw_missing` drops rows with any raw-source missing value before official transforms/T-codes.
- `raw_outlier_policy=preserve_raw_outliers` is the default raw-source outlier policy.
- `raw_outlier_policy` values `winsorize_raw`, `iqr_clip_raw`, `mad_clip_raw`, `zscore_clip_raw`, and `raw_outlier_to_missing` operate on raw numeric columns before official transforms/T-codes. `leaf_config.raw_outlier_columns` may restrict the column set.
- `release_lag_rule=series_specific_lag` requires non-empty `leaf_config.release_lag_per_series`.
- `release_lag_rule=fixed_lag_all_series` and `release_lag_rule=series_specific_lag` now write `data_reports.release_lag` in `layer1_official_frame_v1`, including shifted columns, lag map, missing configured columns, maximum lag, and level-source-frame shift status.
- `structural_break_segmentation` remains executable through fixed built-in dates; user-supplied break dates are owned by `deterministic_components=break_dummies`.

Full-mode interpretation:

- `missing_availability` is the current executable Layer 1 availability axis.
  It covers official-frame availability and the compatibility path for simple
  X imputation validation.
- The detailed contract should be read as a two-stage policy. Stage 1 is
  raw-source missing/outlier treatment before `official_transform_policy`
  applies dataset transforms or T-codes.
- Stage 2 is post-transform/model-input missing/outlier treatment in Layer 2.
- Raw-source missing values belong to Layer 1 when the decision is "repair the
  raw source panel, then apply official transforms/T-codes."
- Raw-source outliers follow the same ownership rule. If a researcher clips,
  winsorizes, or converts a raw outlier to missing before T-code construction,
  that is a Layer 1 full-contract decision and must be recorded in provenance.
- Layer 2 may still handle the same observed problem after official transforms
  have been applied. That path is valid but semantically different: it can mix
  raw-source missing/outliers with transform-induced missing values,
  differencing endpoints, and model-input preprocessing artifacts. The numerical
  difference can be small in many empirical settings, but full mode must record
  the phase and order.
- `x_impute_only` remains accepted for migration compatibility. Conceptually,
  imputation applied after the official frame exists belongs to Layer 2.
- Layer 1 now exposes this raw-before-T-code decision through
  `raw_missing_policy` and `raw_outlier_policy`. Layer 2 `x_missing_policy` and
  `x_outlier_policy` remain post-transform/model-input preprocessing.

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
- FRED-SD state and workbook-variable selectors are canonical Layer 1 axes:
  `state_selection=selected_states` reads `leaf_config.sd_states`, and
  `sd_variable_selection=selected_sd_variables` reads
  `leaf_config.sd_variables`. These selectors run at source-load time for the
  FRED-SD component before generic post-load `variable_universe` filtering.
- FRED-SD `support_tier=provisional` now points to remaining mixed-frequency
  and richer grouping controls, not missing live/vintage ingestion or missing
  t-code policy choices.

Simple docs should stay short; this audit is the detailed contract source for Layer 1.

## Regression Checks Added

- Standalone FRED-SD without explicit frequency fails at compile time.
- FRED-SD composite frequency conflicts fail at compile time.
- FRED-SD `state_selection=selected_states` and
  `sd_variable_selection=selected_sd_variables` require explicit
  `leaf_config.sd_states` / `leaf_config.sd_variables`, flow into
  `data_task_spec`, and filter the FRED-SD component at load time.
- Distributional forecast objects with non-quantile models produce `blocked_by_incompatibility`.
- Runtime-only missing leaf_config failures for Layer 1.4 and 1.5 are now compiler validation errors.
- Multi-benchmark suites reject unsupported member families before execution.
- Layer 1 official-transform axes are recorded in `data_task_spec`, and
  conflicting Layer 1 official-transform choices vs legacy Layer 2 t-code bridge
  choices fail at compile time.
- Layer 1 raw missing/outlier axes are recorded in `data_task_spec` and execute
  before official transforms/T-codes.
- Layer 1 official-frame handoff is exported as `layer1_official_frame_v1`,
  written to `layer1_official_frame.json`, summarized in `manifest.json`, and
  listed in `artifact_manifest.json`.
- Local FRED-SD CSV fixtures parse without the optional `openpyxl` dependency,
  preserve state/variable filters, and run in a `fred_md+fred_sd` composite
  execution with component source evidence in `source_availability_contract_v1`.
- Local vintage-path execution records version mode, requested vintage,
  data-through, artifact SHA, release-lag/missing-availability policy, and
  transform-code coverage in `layer1_official_frame_v1`.
