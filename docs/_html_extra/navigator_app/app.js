const layerDefs = [
  { id: "l0", key: "0_meta", name: "Setup", mode: "form", role: "failure policy, reproducibility, compute" },
  { id: "l1", key: "1_data", name: "Data", mode: "form", role: "dataset, targets, horizons, universe" },
  { id: "l1_5", key: "1_5_data_summary", name: "Data Diagnostics", mode: "diagnostic", parent: "l1", role: "raw data summary" },
  { id: "l2", key: "2_preprocessing", name: "Preprocessing", mode: "form", role: "clean panel construction" },
  { id: "l2_5", key: "2_5_pre_post_preprocessing", name: "Pre/Post Diagnostics", mode: "diagnostic", parent: "l2", role: "pre/post comparison" },
  { id: "l3", key: "3_feature_engineering", name: "Feature Builder", mode: "dag", role: "ordered feature blocks and target construction" },
  { id: "l3_5", key: "3_5_feature_diagnostics", name: "Feature Diagnostics", mode: "diagnostic", parent: "l3", role: "feature checks" },
  { id: "l4", key: "4_forecasting_model", name: "Forecast Builder", mode: "dag", role: "model, benchmark, prediction, combination selections" },
  { id: "l4_5", key: "4_5_generator_diagnostics", name: "Generator Diagnostics", mode: "diagnostic", parent: "l4", role: "model-fit diagnostics" },
  { id: "l5", key: "5_evaluation", name: "Evaluation", mode: "form", role: "metrics, aggregation, ranking" },
  { id: "l6", key: "6_statistical_tests", name: "Statistical Tests", mode: "form-toggle", role: "inferential tests" },
  { id: "l7", key: "7_interpretation", name: "Interpretation Builder", mode: "dag-toggle", role: "importance and attribution selections" },
  { id: "l8", key: "8_output", name: "Output", mode: "form", role: "saved objects and provenance" }
];

const AXIS_OPTIONS = {
  l1: {
    custom_source_policy: ["official_only", "custom_panel_only", "official_plus_custom"],
    dataset: ["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"],
    frequency: ["monthly", "quarterly"],
    information_set_type: ["final_revised_data", "pseudo_oos_on_revised_data"],
    release_lag_rule: ["ignore_release_lag", "fixed_lag_all_series", "series_specific_lag"],
    contemporaneous_x_rule: ["allow_same_period_predictors", "forbid_same_period_predictors"],
    target_structure: ["single_target", "multi_target"],
    variable_universe: ["all_variables", "core_variables", "category_variables", "target_specific_variables", "explicit_variable_list"],
    fred_sd_frequency_policy: ["report_only", "allow_mixed_frequency", "reject_mixed_known_frequency", "require_single_known_frequency"],
    fred_sd_state_group: [
      "all_states",
      "census_region_northeast",
      "census_region_midwest",
      "census_region_south",
      "census_region_west",
      "census_division_new_england",
      "census_division_middle_atlantic",
      "census_division_east_north_central",
      "census_division_west_north_central",
      "census_division_south_atlantic",
      "census_division_east_south_central",
      "census_division_west_south_central",
      "census_division_mountain",
      "census_division_pacific",
      "contiguous_48_plus_dc",
      "custom_state_group"
    ],
    state_selection: ["all_states", "selected_states"],
    fred_sd_variable_group: [
      "all_sd_variables",
      "labor_market_core",
      "employment_sector",
      "gsp_output",
      "housing",
      "trade",
      "income",
      "direct_analog_high_confidence",
      "provisional_analog_medium",
      "semantic_review_outputs",
      "no_reliable_analog",
      "custom_sd_variable_group"
    ],
    sd_variable_selection: ["all_sd_variables", "selected_sd_variables"],
    raw_missing_policy: ["preserve_raw_missing", "zero_fill_leading_predictor_missing_before_tcode", "impute_raw_predictors", "drop_raw_missing_rows"],
    raw_outlier_policy: ["preserve_raw_outliers", "winsorize_raw", "iqr_clip_raw", "mad_clip_raw", "zscore_clip_raw", "set_raw_outliers_to_missing"],
    official_transform_policy: ["apply_official_tcode", "keep_official_raw_scale"],
    official_transform_scope: ["target_only", "predictors_only", "target_and_predictors", "none"],
    missing_availability: ["require_complete_rows", "keep_available_rows", "impute_predictors_only", "zero_fill_leading_predictor_gaps"]
  },
  l2: {
    fred_sd_mixed_frequency_representation: ["calendar_aligned_frame", "drop_unknown_native_frequency", "drop_non_target_native_frequency", "native_frequency_block_payload", "mixed_frequency_model_adapter"],
    horizon_target_construction: ["future_target_level_t_plus_h", "future_diff", "future_logdiff", "average_growth_1_to_h", "path_average_growth_1_to_h", "average_difference_1_to_h", "path_average_difference_1_to_h", "average_log_growth_1_to_h", "path_average_log_growth_1_to_h"],
    target_transform: ["level", "difference", "log", "log_difference", "growth_rate"],
    target_normalization: ["none", "zscore_train_only", "robust_zscore", "minmax", "unit_variance"],
    tcode_policy: ["raw_only", "official_tcode_only", "official_tcode_then_extra_preprocess", "extra_preprocess_only", "extra_preprocess_then_official_tcode", "custom_transform_sequence"],
    x_missing_policy: ["none", "drop", "em_impute", "mean_impute", "median_impute", "ffill", "interpolate_linear", "drop_rows", "drop_columns", "drop_if_above_threshold", "missing_indicator", "custom"],
    x_outlier_policy: ["none", "clip", "outlier_to_nan", "winsorize", "trim", "iqr_clip", "mad_clip", "zscore_clip", "outlier_to_missing", "custom"],
    scaling_policy: ["none", "standard", "robust", "minmax", "demean_only", "unit_variance_only", "rank_scale", "custom"],
    target_lag_block: ["none", "fixed_target_lags", "ic_selected_target_lags", "horizon_specific_target_lags", "custom_target_lags"],
    x_lag_feature_block: ["none", "fixed_predictor_lags", "variable_specific_predictor_lags", "category_specific_predictor_lags", "cv_selected_predictor_lags", "custom_predictor_lags"],
    factor_feature_block: ["none", "pca_static_factors", "pca_factor_lags", "supervised_factors", "custom_factors"],
    level_feature_block: ["none", "target_level_addback", "x_level_addback", "selected_level_addbacks", "level_growth_pairs"],
    temporal_feature_block: ["none", "moving_average_features", "rolling_moments", "local_temporal_factors", "volatility_features", "custom_temporal_features"],
    rotation_feature_block: ["none", "marx_rotation", "maf_rotation", "moving_average_rotation", "custom_rotation"],
    feature_block_combination: ["replace_with_selected_blocks", "append_to_base_predictors", "append_to_target_lags", "concatenate_named_blocks", "custom_feature_combiner"],
    feature_selection_policy: ["none", "correlation_filter", "lasso_selection", "mutual_information_screen", "custom"],
    feature_selection_semantics: ["select_before_factor", "select_after_factor", "select_after_custom_feature_blocks"],
    evaluation_scale: ["original_scale", "raw_level", "transformed_scale", "both"],
    feature_builder: ["target_lag_features", "factors_plus_target_lags", "raw_feature_panel", "raw_predictors_only", "pca_factor_features", "sequence_tensor"]
  },
  l5: {
    primary_metric: ["mse", "rmse", "mae", "mape", "medae", "theil_u1", "theil_u2", "relative_mse", "r2_oos", "relative_mae", "mse_reduction", "log_score", "crps"],
    point_metrics: ["mse", "rmse", "mae", "mape", "medae", "theil_u1", "theil_u2"],
    density_metrics: ["log_score", "crps", "interval_score", "coverage_rate"],
    direction_metrics: ["success_ratio", "pesaran_timmermann_metric"],
    relative_metrics: ["relative_mse", "r2_oos", "relative_mae", "mse_reduction"],
    benchmark_window: ["full_oos", "rolling", "expanding"],
    benchmark_scope: ["all_targets_horizons", "per_target_horizon"],
    agg_time: ["mean", "median", "per_subperiod"],
    agg_horizon: ["pool_horizons", "per_horizon_separate"],
    agg_target: ["pool_targets", "per_target_separate"],
    agg_state: ["pool_states", "per_state_separate"],
    oos_period: ["full_oos", "fixed_dates", "multiple_subperiods"],
    regime_use: ["pooled", "per_regime", "both"],
    decomposition_target: ["none", "by_horizon", "by_target", "by_state", "by_regime"],
    decomposition_order: ["marginal", "sequential"],
    ranking: ["by_primary_metric", "by_relative_metric", "by_average_rank", "mcs_inclusion"],
    report_style: ["single_table", "per_target_horizon_panel", "latex_table"]
  },
  l6: {
    test_scope: ["per_target_horizon", "per_target", "per_horizon", "pooled"],
    dependence_correction: ["newey_west", "andrews", "parzen_kernel", "none"],
    overlap_handling: ["nw_with_h_minus_1_lag", "west_1996_adjustment", "none"],
    equal_predictive_test: ["dm_diebold_mariano", "gw_giacomini_white", "multi", "none"],
    model_pair_strategy: ["vs_benchmark_only", "all_pairs", "user_list"]
  },
  l8: {
    export_format: ["json", "csv", "parquet", "json_csv", "json_parquet", "latex_tables", "markdown_report", "html_report", "all"],
    compression: ["none", "gzip", "zip"],
    saved_objects: [
      "forecasts",
      "forecast_intervals",
      "metrics",
      "ranking",
      "decomposition",
      "regime_metrics",
      "state_metrics",
      "model_artifacts",
      "combination_weights",
      "feature_metadata",
      "clean_panel",
      "raw_panel",
      "diagnostics_l1_5",
      "diagnostics_l2_5",
      "diagnostics_l3_5",
      "diagnostics_l4_5",
      "diagnostics_all",
      "tests",
      "importance",
      "transformation_attribution"
    ],
    model_artifacts_format: ["pickle", "joblib", "onnx", "pmml"],
    provenance_fields: [
      "recipe_yaml_full",
      "recipe_hash",
      "package_version",
      "python_version",
      "r_version",
      "julia_version",
      "dependency_lockfile",
      "git_commit_sha",
      "git_branch_name",
      "data_revision_tag",
      "random_seed_used",
      "runtime_environment",
      "runtime_duration",
      "cell_resolved_axes"
    ],
    manifest_format: ["json", "yaml", "json_lines"],
    artifact_granularity: ["per_cell", "per_target", "per_horizon", "per_target_horizon", "flat"],
    naming_convention: ["cell_id", "descriptive", "recipe_hash", "custom"]
  }
};

const AXIS_DESCRIPTIONS = {
  custom_source_policy: "Choose FRED-only, custom-only, or FRED plus custom source data.",
  dataset: "FRED source panel. Hidden when the study is custom-only.",
  frequency: "Final analysis frequency. Required for standalone FRED-SD and custom-only routes.",
  information_set_type: "Data revision or pseudo-real-time vintage regime.",
  release_lag_rule: "Publication lag rule for predictor availability.",
  contemporaneous_x_rule: "Controls whether same-period predictors are allowed.",
  target_structure: "Single target uses target; multi target uses targets.",
  variable_universe: "FRED-MD/QD predictor universe before Layer 2 representation choices.",
  fred_sd_frequency_policy: "Native-frequency evidence policy for FRED-SD source columns.",
  fred_sd_state_group: "State group selector for FRED-SD.",
  state_selection: "Explicit FRED-SD state-list switch.",
  fred_sd_variable_group: "Workbook-series group selector for FRED-SD.",
  sd_variable_selection: "Explicit FRED-SD series-list switch.",
  raw_missing_policy: "Raw-source missing-value handling before official transforms.",
  raw_outlier_policy: "Raw-source outlier handling before official transforms.",
  official_transform_policy: "Whether to apply official FRED-MD/QD transform codes.",
  official_transform_scope: "Target/predictor scope for official transforms.",
  missing_availability: "Source-frame availability-gap policy after loading and transforms.",
  fred_sd_mixed_frequency_representation: "Layer 2 representation rule for FRED-SD mixed native frequencies.",
  horizon_target_construction: "How future y is constructed for each horizon.",
  target_transform: "Target-side transformation before modeling.",
  target_normalization: "Target-side normalization policy.",
  tcode_policy: "Research preprocessing transform policy.",
  x_missing_policy: "Predictor missing-data policy after Layer 1.",
  x_outlier_policy: "Predictor outlier policy after Layer 1.",
  scaling_policy: "Predictor scaling policy.",
  primary_metric: "Main metric used for ranking and summaries.",
  ranking: "How forecast generators are ordered in the evaluation report.",
  export_format: "External artifact format emitted by Layer 8.",
  saved_objects: "Artifacts included in the output bundle.",
  provenance_fields: "Manifest fields used for reproducibility."
};

const AXIS_TITLES = {
  custom_source_policy: "Data source",
  dataset: "FRED dataset",
  frequency: "Analysis frequency",
  information_set_type: "Vintage setting",
  release_lag_rule: "Release lag",
  contemporaneous_x_rule: "Same-period predictors",
  target_structure: "Target mode",
  variable_universe: "Predictor universe",
  fred_sd_frequency_policy: "FRED-SD frequency handling",
  fred_sd_state_group: "FRED-SD state group",
  state_selection: "State list",
  fred_sd_variable_group: "FRED-SD variable group",
  sd_variable_selection: "FRED-SD variable list",
  raw_missing_policy: "Raw missing values",
  raw_outlier_policy: "Raw outliers",
  official_transform_policy: "Official transforms",
  official_transform_scope: "Transform scope",
  missing_availability: "Availability gaps",
  fred_sd_mixed_frequency_representation: "Mixed-frequency representation",
  horizon_target_construction: "Forecast target construction",
  target_transform: "Target transform",
  target_normalization: "Target normalization",
  tcode_policy: "Transform-code policy",
  x_missing_policy: "Predictor missing values",
  x_outlier_policy: "Predictor outliers",
  scaling_policy: "Predictor scaling",
  target_lag_block: "Target lag features",
  x_lag_feature_block: "Predictor lag features",
  factor_feature_block: "Factor features",
  level_feature_block: "Level features",
  temporal_feature_block: "Temporal features",
  rotation_feature_block: "Rotations",
  feature_block_combination: "Feature combination",
  feature_selection_policy: "Feature selection",
  feature_selection_semantics: "Selection timing",
  evaluation_scale: "Evaluation scale",
  feature_builder: "Feature recipe",
  primary_metric: "Main score",
  point_metrics: "Point forecast scores",
  density_metrics: "Distribution scores",
  direction_metrics: "Direction scores",
  relative_metrics: "Benchmark-relative scores",
  benchmark_window: "Benchmark window",
  benchmark_scope: "Benchmark scope",
  agg_time: "Time aggregation",
  agg_horizon: "Horizon aggregation",
  agg_target: "Target aggregation",
  agg_state: "State aggregation",
  oos_period: "OOS period",
  regime_use: "Regime split",
  decomposition_target: "Decomposition dimension",
  decomposition_order: "Decomposition order",
  ranking: "Model ranking",
  report_style: "Report layout",
  test_scope: "Test scope",
  dependence_correction: "Serial-correlation correction",
  overlap_handling: "Overlapping horizons",
  equal_predictive_test: "Equal-predictive-ability test",
  model_pair_strategy: "Model pairs",
  export_format: "Export format",
  compression: "Compression",
  saved_objects: "Saved artifacts",
  model_artifacts_format: "Model artifact format",
  provenance_fields: "Provenance fields",
  manifest_format: "Manifest format",
  artifact_granularity: "Artifact split",
  naming_convention: "File naming"
};

const OPTION_LABELS = {
  official_only: "Use built-in FRED data",
  custom_panel_only: "Use my own panel only",
  official_plus_custom: "Combine FRED with my panel",
  fred_md: "FRED-MD monthly macro panel",
  fred_qd: "FRED-QD quarterly macro panel",
  fred_sd: "FRED-SD state-level panel",
  "fred_md+fred_sd": "FRED-MD plus state data",
  "fred_qd+fred_sd": "FRED-QD plus state data",
  monthly: "Monthly",
  quarterly: "Quarterly",
  final_revised_data: "Use final revised data",
  pseudo_oos_on_revised_data: "Pseudo OOS on revised data",
  ignore_release_lag: "Ignore release lag",
  fixed_lag_all_series: "Apply one fixed lag",
  series_specific_lag: "Use series-specific lags",
  allow_same_period_predictors: "Allow same-period predictors",
  forbid_same_period_predictors: "Forbid same-period predictors",
  single_target: "Single target",
  multi_target: "Multiple targets",
  all_variables: "All variables",
  core_variables: "Core variables",
  category_variables: "One category",
  target_specific_variables: "Target-specific universe",
  explicit_variable_list: "Manual variable list",
  report_only: "Report mixed frequencies only",
  allow_mixed_frequency: "Allow mixed frequencies",
  reject_mixed_known_frequency: "Reject mixed known frequencies",
  require_single_known_frequency: "Require one known frequency",
  all_states: "All states",
  selected_states: "Selected states only",
  all_sd_variables: "All FRED-SD variables",
  labor_market_core: "Core labor market",
  employment_sector: "Employment by sector",
  gsp_output: "State output / GSP",
  housing: "Housing",
  trade: "Trade",
  income: "Income",
  direct_analog_high_confidence: "High-confidence FRED-MD analogs",
  provisional_analog_medium: "Medium-confidence analogs",
  semantic_review_outputs: "Needs semantic review",
  no_reliable_analog: "No reliable analog",
  custom_sd_variable_group: "Custom variable group",
  selected_sd_variables: "Selected FRED-SD variables",
  preserve_raw_missing: "Keep raw missing values",
  zero_fill_leading_predictor_missing_before_tcode: "Zero-fill leading predictor gaps",
  impute_raw_predictors: "Impute raw predictors",
  drop_raw_missing_rows: "Drop rows with raw missing values",
  preserve_raw_outliers: "Keep raw outliers",
  winsorize_raw: "Winsorize raw values",
  iqr_clip_raw: "IQR-clip raw values",
  mad_clip_raw: "MAD-clip raw values",
  zscore_clip_raw: "Z-score clip raw values",
  set_raw_outliers_to_missing: "Set raw outliers to missing",
  apply_official_tcode: "Apply official transform codes",
  keep_official_raw_scale: "Keep raw scale",
  target_only: "Target only",
  predictors_only: "Predictors only",
  target_and_predictors: "Target and predictors",
  require_complete_rows: "Require complete rows",
  keep_available_rows: "Keep available rows",
  impute_predictors_only: "Impute predictors only",
  zero_fill_leading_predictor_gaps: "Zero-fill leading predictor gaps",
  calendar_aligned_frame: "Calendar-aligned table",
  drop_unknown_native_frequency: "Drop unknown-frequency series",
  drop_non_target_native_frequency: "Keep target frequency only",
  native_frequency_block_payload: "Keep native-frequency blocks",
  mixed_frequency_model_adapter: "Use mixed-frequency model adapter",
  future_target_level_t_plus_h: "Future level at t+h",
  future_diff: "Future difference",
  future_logdiff: "Future log difference",
  average_growth_1_to_h: "Average growth to h",
  path_average_growth_1_to_h: "Path average growth to h",
  average_difference_1_to_h: "Average difference to h",
  path_average_difference_1_to_h: "Path average difference to h",
  average_log_growth_1_to_h: "Average log growth to h",
  path_average_log_growth_1_to_h: "Path average log growth to h",
  level: "Level",
  difference: "Difference",
  log: "Log level",
  log_difference: "Log difference",
  growth_rate: "Growth rate",
  none: "None",
  zscore_train_only: "Train-only z-score",
  robust_zscore: "Robust z-score",
  minmax: "Min-max scale",
  unit_variance: "Unit variance",
  raw_only: "Raw only",
  official_tcode_only: "Official t-code only",
  official_tcode_then_extra_preprocess: "Official t-code, then extra preprocessing",
  extra_preprocess_only: "Extra preprocessing only",
  extra_preprocess_then_official_tcode: "Extra preprocessing, then official t-code",
  custom_transform_sequence: "Custom transform sequence",
  drop: "Drop missing values",
  em_impute: "EM imputation",
  mean_impute: "Mean imputation",
  median_impute: "Median imputation",
  ffill: "Forward fill",
  interpolate_linear: "Linear interpolation",
  drop_rows: "Drop rows",
  drop_columns: "Drop columns",
  drop_if_above_threshold: "Drop above threshold",
  missing_indicator: "Add missing indicators",
  custom: "Custom",
  clip: "Clip",
  outlier_to_nan: "Outlier to NaN",
  winsorize: "Winsorize",
  trim: "Trim tails",
  iqr_clip: "IQR clip",
  mad_clip: "MAD clip",
  zscore_clip: "Z-score clip",
  outlier_to_missing: "Outlier to missing",
  standard: "Standard scale",
  robust: "Robust scale",
  demean_only: "Demean only",
  unit_variance_only: "Unit variance only",
  rank_scale: "Rank scale",
  fixed_target_lags: "Fixed target lags",
  ic_selected_target_lags: "IC-selected target lags",
  horizon_specific_target_lags: "Horizon-specific target lags",
  custom_target_lags: "Custom target lags",
  fixed_predictor_lags: "Fixed predictor lags",
  variable_specific_predictor_lags: "Variable-specific predictor lags",
  category_specific_predictor_lags: "Category-specific predictor lags",
  cv_selected_predictor_lags: "CV-selected predictor lags",
  custom_predictor_lags: "Custom predictor lags",
  pca_static_factors: "PCA static factors",
  pca_factor_lags: "PCA factor lags",
  supervised_factors: "Supervised factors",
  custom_factors: "Custom factors",
  target_level_addback: "Add target level",
  x_level_addback: "Add predictor levels",
  selected_level_addbacks: "Selected level addbacks",
  level_growth_pairs: "Level-growth pairs",
  moving_average_features: "Moving-average features",
  rolling_moments: "Rolling moments",
  local_temporal_factors: "Local temporal factors",
  volatility_features: "Volatility features",
  custom_temporal_features: "Custom temporal features",
  marx_rotation: "MARX rotation",
  maf_rotation: "MAF rotation",
  moving_average_rotation: "Moving-average rotation",
  custom_rotation: "Custom rotation",
  replace_with_selected_blocks: "Use selected blocks only",
  append_to_base_predictors: "Append to base predictors",
  append_to_target_lags: "Append to target lags",
  concatenate_named_blocks: "Concatenate named blocks",
  custom_feature_combiner: "Custom feature combiner",
  correlation_filter: "Correlation filter",
  lasso_selection: "Lasso selection",
  mutual_information_screen: "Mutual information screen",
  select_before_factor: "Select before factor extraction",
  select_after_factor: "Select after factor extraction",
  select_after_custom_feature_blocks: "Select after custom blocks",
  original_scale: "Original scale",
  raw_level: "Raw level",
  transformed_scale: "Transformed scale",
  both: "Both",
  target_lag_features: "Target lags only",
  factors_plus_target_lags: "Factors plus target lags",
  raw_feature_panel: "Raw feature panel",
  raw_predictors_only: "Raw predictors only",
  pca_factor_features: "PCA factor features",
  sequence_tensor: "Sequence tensor",
  mse: "Mean squared error",
  rmse: "Root mean squared error",
  mae: "Mean absolute error",
  mape: "Mean absolute percentage error",
  medae: "Median absolute error",
  theil_u1: "Theil U1",
  theil_u2: "Theil U2",
  relative_mse: "Relative MSE vs benchmark",
  r2_oos: "Out-of-sample R2",
  relative_mae: "Relative MAE vs benchmark",
  mse_reduction: "MSE reduction",
  log_score: "Log score",
  crps: "CRPS",
  interval_score: "Interval score",
  coverage_rate: "Coverage rate",
  success_ratio: "Direction success ratio",
  pesaran_timmermann_metric: "Pesaran-Timmermann direction test",
  full_oos: "Full OOS period",
  rolling: "Rolling",
  expanding: "Expanding",
  all_targets_horizons: "All target-horizon cells",
  per_target_horizon: "Each target-horizon cell",
  mean: "Mean",
  median: "Median",
  per_subperiod: "Separate by subperiod",
  pool_horizons: "Pool horizons",
  per_horizon_separate: "Separate horizons",
  pool_targets: "Pool targets",
  per_target_separate: "Separate targets",
  pool_states: "Pool states",
  per_state_separate: "Separate states",
  fixed_dates: "Fixed dates",
  multiple_subperiods: "Multiple subperiods",
  pooled: "Pooled",
  per_regime: "Separate by regime",
  by_horizon: "By horizon",
  by_target: "By target",
  by_state: "By state",
  by_regime: "By regime",
  marginal: "Marginal contribution",
  sequential: "Sequential contribution",
  by_primary_metric: "Rank by main score",
  by_relative_metric: "Rank by relative score",
  by_average_rank: "Rank by average rank",
  mcs_inclusion: "Model confidence set inclusion",
  single_table: "Single table",
  per_target_horizon_panel: "Target-horizon panel",
  latex_table: "LaTeX table",
  per_target: "Each target",
  per_horizon: "Each horizon",
  newey_west: "Newey-West HAC",
  andrews: "Andrews automatic bandwidth",
  parzen_kernel: "Parzen kernel HAC",
  nw_with_h_minus_1_lag: "NW with h-1 lags",
  west_1996_adjustment: "West 1996 adjustment",
  dm_diebold_mariano: "Diebold-Mariano",
  gw_giacomini_white: "Giacomini-White",
  multi: "Multiple tests",
  vs_benchmark_only: "Compare models to benchmark only",
  all_pairs: "Compare all model pairs",
  user_list: "Use manual pair list",
  json: "JSON",
  csv: "CSV",
  parquet: "Parquet",
  json_csv: "JSON plus CSV",
  json_parquet: "JSON plus Parquet",
  latex_tables: "LaTeX tables",
  markdown_report: "Markdown report",
  html_report: "HTML report",
  all: "All supported outputs",
  gzip: "Gzip",
  zip: "Zip",
  forecasts: "Forecasts",
  forecast_intervals: "Forecast intervals",
  metrics: "Metrics",
  ranking: "Ranking",
  decomposition: "Decomposition",
  regime_metrics: "Regime metrics",
  state_metrics: "State metrics",
  model_artifacts: "Model artifacts",
  combination_weights: "Combination weights",
  feature_metadata: "Feature metadata",
  clean_panel: "Clean panel",
  raw_panel: "Raw panel",
  diagnostics_all: "All diagnostics",
  tests: "Statistical test results",
  importance: "Importance scores",
  transformation_attribution: "Transformation attribution",
  pickle: "Pickle",
  joblib: "Joblib",
  onnx: "ONNX",
  pmml: "PMML",
  recipe_yaml_full: "Full recipe YAML",
  recipe_hash: "Recipe hash",
  package_version: "Package version",
  python_version: "Python version",
  r_version: "R version",
  julia_version: "Julia version",
  dependency_lockfile: "Dependency lockfile",
  git_commit_sha: "Git commit SHA",
  git_branch_name: "Git branch name",
  data_revision_tag: "Data revision tag",
  random_seed_used: "Random seed used",
  runtime_environment: "Runtime environment",
  runtime_duration: "Runtime duration",
  cell_resolved_axes: "Resolved axes per cell",
  yaml: "YAML",
  json_lines: "JSON lines",
  per_cell: "One artifact per forecast cell",
  flat: "Single flat bundle",
  cell_id: "Cell ID",
  descriptive: "Readable descriptive names",
  recipe_hash: "Recipe hash",
};

const OPTION_DESCRIPTIONS = {
  multi_target: "Default. Use when the study should evaluate several macro targets together.",
  single_target: "Use when the recipe should forecast one target series only.",
  fred_md: "Default broad monthly macro panel. Good starting point for non-technical users.",
  fred_sd: "Use when state-level predictors are central to the question.",
  all_variables: "Default broad universe. Keeps the first setup wide instead of hand-picking predictors.",
  factors_plus_target_lags: "Default broad feature recipe: summarize predictors with factors and keep target history.",
  per_target_horizon: "Default. Tests each target and forecast horizon separately.",
  newey_west: "Default HAC correction for serial correlation in forecast errors.",
  nw_with_h_minus_1_lag: "Default for overlapping h-step forecast errors.",
  mse: "Default main score. Penalizes large misses more strongly.",
  relative_mse: "Shows performance relative to a benchmark; below 1 is better than benchmark.",
  r2_oos: "Out-of-sample R2; positive values mean improvement over benchmark.",
  by_primary_metric: "Default. Uses the selected main score for ranking.",
  json_csv: "Default. JSON preserves structure; CSV is easy to inspect.",
  per_cell: "Default. Keeps artifacts separated by target, horizon, and model cell."
};

const DAG_PRESETS = {
  l3: [
    { key: "target_horizons", label: "Target Horizons", type: "step", op: "target_construction", params: { horizons: [1, 3, 6, 12] }, help: "Build y(t+h) for each forecast horizon." },
    { key: "predictor_lags", label: "Predictor Lag Block", type: "step", op: "lag", params: { n_lag: 4 }, help: "Create broad lagged predictors." },
    { key: "pca_factors", label: "PCA Factors", type: "step", op: "pca_factors", params: { n_factors: 8 }, help: "Compress many predictors into a smaller factor block." },
    { key: "feature_sink", label: "Feature Sink", type: "sink", op: "sink", params: {}, help: "Final L3 output passed into forecasting models." }
  ],
  l4: [
    { key: "fit_model", label: "Fit Model", type: "step", op: "fit_model", params: { family: "ridge" }, help: "Train one forecast model." },
    { key: "fit_benchmark", label: "Fit Benchmark", type: "step", op: "fit_model", params: { family: "autoregressive_bic", is_benchmark: true }, help: "Train benchmark used for relative metrics." },
    { key: "predict", label: "Predict", type: "step", op: "predict", params: {}, help: "Generate forecasts from a fitted model." },
    { key: "combine", label: "Combine Forecasts", type: "combine", op: "forecast_combination", params: { method: "equal_weight" }, help: "Combine multiple forecast streams." },
    { key: "forecast_sink", label: "Forecast Sink", type: "sink", op: "sink", params: {}, help: "Final L4 forecast bundle." }
  ],
  l7: [
    { key: "importance", label: "Importance", type: "step", op: "permutation_importance", params: { scope: "global" }, help: "Estimate global predictor importance." },
    { key: "attribution_sink", label: "Attribution Sink", type: "sink", op: "sink", params: {}, help: "Final L7 interpretation artifact." }
  ]
};

const DAG_GUIDES = {
  l3: {
    purpose: "L3 turns the cleaned panel from L2 into modeling inputs. It should end with a feature sink that L4 can consume.",
    defaultTemplate: "broad_features",
    steps: [
      "Start from L2 predictors and L2 target.",
      "Build target horizons from the target input.",
      "Build predictor features from the predictor input: lags, factors, or both.",
      "Send both feature and target streams into the L3 feature sink.",
      "Do not add forecast models here. Forecasting belongs in L4."
    ],
    required: ["At least one source node", "At least one feature-building step", "A sink named l3_features_v1"],
    avoid: ["Forecast combination", "Model fitting", "Evaluation metrics"]
  },
  l4: {
    purpose: "L4 trains forecast generators and writes forecast outputs. It consumes L3 features and produces forecast sinks for evaluation.",
    defaultTemplate: "ridge_vs_benchmark",
    steps: [
      "Start from X_final and y_final from L3.",
      "Fit at least one candidate model.",
      "Fit or select a benchmark model if relative metrics are used.",
      "Connect fitted models to prediction nodes.",
      "Send forecast nodes into l4_forecasts_v1."
    ],
    required: ["Feature source from L3", "Target source from L3", "At least one fit_model step", "At least one predict step", "A sink named l4_forecasts_v1"],
    avoid: ["Raw data cleaning", "Target construction", "Metric selection"]
  },
  l7: {
    purpose: "L7 is optional. It reads model or forecast artifacts and emits interpretation artifacts such as importance or attribution.",
    defaultTemplate: "importance",
    steps: [
      "Enable L7 only if interpretation artifacts are needed.",
      "Start from L4 model artifacts or forecast outputs.",
      "Add an importance or attribution method.",
      "Send outputs into l7_importance_v1."
    ],
    required: ["L4 artifact source", "One interpretation step", "A sink named l7_importance_v1"],
    avoid: ["Changing forecasts", "Changing evaluation scores"]
  }
};

const DAG_TEMPLATES = {
  l3: {
    broad_features: {
      label: "Apply Broad Feature Template",
      description: "L2 predictors -> lags and PCA factors; L2 target -> horizons; both feed l3_features_v1.",
      nodes: [
        { id: "src_x", type: "source", op: "source", label: "L2 predictors", x: 90, y: 180, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "predictors" } },
        { id: "src_y", type: "source", op: "source", label: "L2 target", x: 90, y: 360, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "target" } },
        { id: "x_lag", type: "step", op: "lag", label: "Predictor lags", x: 350, y: 150, params: { n_lag: 4 } },
        { id: "x_factor", type: "step", op: "pca_factors", label: "PCA factor block", x: 350, y: 270, params: { n_factors: 8 } },
        { id: "y_h", type: "step", op: "target_construction", label: "Target horizons", x: 350, y: 410, params: { horizons: [1, 3, 6, 12] } },
        { id: "features", type: "sink", op: "sink", label: "l3_features_v1", x: 660, y: 280, params: { X_final: ["x_lag", "x_factor"], y_final: "y_h" } }
      ],
      edges: [
        { from: "src_x", to: "x_lag" },
        { from: "src_x", to: "x_factor" },
        { from: "src_y", to: "y_h" },
        { from: "x_lag", to: "features" },
        { from: "x_factor", to: "features" },
        { from: "y_h", to: "features" }
      ]
    },
    simple_lags: {
      label: "Apply Simple Lag Template",
      description: "Minimal L3: predictor lags plus target horizons feeding one feature sink.",
      nodes: [
        { id: "src_x", type: "source", op: "source", label: "L2 predictors", x: 90, y: 210, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "predictors" } },
        { id: "src_y", type: "source", op: "source", label: "L2 target", x: 90, y: 350, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "target" } },
        { id: "x_lag", type: "step", op: "lag", label: "Predictor lags", x: 360, y: 210, params: { n_lag: 4 } },
        { id: "y_h", type: "step", op: "target_construction", label: "Target horizons", x: 360, y: 350, params: { horizons: [1, 3, 6, 12] } },
        { id: "features", type: "sink", op: "sink", label: "l3_features_v1", x: 650, y: 280, params: { X_final: "x_lag", y_final: "y_h" } }
      ],
      edges: [
        { from: "src_x", to: "x_lag" },
        { from: "src_y", to: "y_h" },
        { from: "x_lag", to: "features" },
        { from: "y_h", to: "features" }
      ]
    }
  },
  l4: {
    ridge_vs_benchmark: {
      label: "Apply Ridge vs Benchmark Template",
      description: "L3 features feed Ridge and AR-BIC benchmark; both predict into l4_forecasts_v1.",
      nodes: [
        { id: "src_X", type: "source", op: "source", label: "X_final", x: 80, y: 170, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "X_final" } },
        { id: "src_y", type: "source", op: "source", label: "y_final", x: 80, y: 350, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "y_final" } },
        { id: "fit_ridge", type: "step", op: "fit_model", label: "Fit ridge", x: 330, y: 150, params: { family: "ridge" } },
        { id: "fit_benchmark", type: "step", op: "fit_model", label: "Fit AR-BIC benchmark", x: 330, y: 330, params: { family: "autoregressive_bic", is_benchmark: true } },
        { id: "predict_ridge", type: "step", op: "predict", label: "Predict ridge", x: 590, y: 150, params: {} },
        { id: "predict_benchmark", type: "step", op: "predict", label: "Predict benchmark", x: 590, y: 330, params: {} },
        { id: "forecasts", type: "sink", op: "sink", label: "l4_forecasts_v1", x: 850, y: 240, params: { forecasts: ["predict_ridge", "predict_benchmark"] } }
      ],
      edges: [
        { from: "src_X", to: "fit_ridge" },
        { from: "src_y", to: "fit_ridge" },
        { from: "src_y", to: "fit_benchmark" },
        { from: "fit_ridge", to: "predict_ridge" },
        { from: "src_X", to: "predict_ridge" },
        { from: "fit_benchmark", to: "predict_benchmark" },
        { from: "predict_ridge", to: "forecasts" },
        { from: "predict_benchmark", to: "forecasts" }
      ]
    },
    model_ensemble: {
      label: "Apply Ensemble Template",
      description: "Two candidate models plus benchmark, with candidate forecasts combined before the forecast sink.",
      nodes: [
        { id: "src_X", type: "source", op: "source", label: "X_final", x: 70, y: 190, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "X_final" } },
        { id: "src_y", type: "source", op: "source", label: "y_final", x: 70, y: 390, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "y_final" } },
        { id: "fit_ridge", type: "step", op: "fit_model", label: "Fit ridge", x: 310, y: 130, params: { family: "ridge" } },
        { id: "fit_lasso", type: "step", op: "fit_model", label: "Fit lasso", x: 310, y: 260, params: { family: "lasso" } },
        { id: "fit_benchmark", type: "step", op: "fit_model", label: "Fit benchmark", x: 310, y: 410, params: { family: "autoregressive_bic", is_benchmark: true } },
        { id: "predict_ridge", type: "step", op: "predict", label: "Predict ridge", x: 560, y: 130, params: {} },
        { id: "predict_lasso", type: "step", op: "predict", label: "Predict lasso", x: 560, y: 260, params: {} },
        { id: "predict_benchmark", type: "step", op: "predict", label: "Predict benchmark", x: 560, y: 410, params: {} },
        { id: "combine_candidates", type: "combine", op: "forecast_combination", label: "Combine candidates", x: 800, y: 200, params: { method: "equal_weight" } },
        { id: "forecasts", type: "sink", op: "sink", label: "l4_forecasts_v1", x: 1040, y: 300, params: { forecasts: ["combine_candidates", "predict_benchmark"] } }
      ],
      edges: [
        { from: "src_X", to: "fit_ridge" },
        { from: "src_y", to: "fit_ridge" },
        { from: "src_X", to: "fit_lasso" },
        { from: "src_y", to: "fit_lasso" },
        { from: "src_y", to: "fit_benchmark" },
        { from: "fit_ridge", to: "predict_ridge" },
        { from: "fit_lasso", to: "predict_lasso" },
        { from: "fit_benchmark", to: "predict_benchmark" },
        { from: "predict_ridge", to: "combine_candidates" },
        { from: "predict_lasso", to: "combine_candidates" },
        { from: "combine_candidates", to: "forecasts" },
        { from: "predict_benchmark", to: "forecasts" }
      ]
    }
  },
  l7: {
    importance: {
      label: "Apply Importance Template",
      description: "L4 model artifacts feed permutation importance and produce l7_importance_v1.",
      nodes: [
        { id: "src_model", type: "source", op: "source", label: "L4 model artifacts", x: 120, y: 230, params: { layer_ref: "l4", sink_name: "l4_model_artifacts_v1" } },
        { id: "importance", type: "step", op: "permutation_importance", label: "Permutation importance", x: 390, y: 230, params: { scope: "global" } },
        { id: "importance_sink", type: "sink", op: "sink", label: "l7_importance_v1", x: 680, y: 230, params: {} }
      ],
      edges: [
        { from: "src_model", to: "importance" },
        { from: "importance", to: "importance_sink" }
      ]
    }
  }
};

const state = {
  selectedLayer: "map",
  mapFocusLayer: "l1",
  selectedNode: null,
  connectingFrom: null,
  bottomTab: "yaml",
  recipeName: "broad-multitarget-fred-md",
  layers: {
    l0: {
      fixed_axes: { failure_policy: "fail_fast", reproducibility_mode: "seeded_reproducible", compute_mode: "serial" },
      leaf_config: { random_seed: 42 }
    },
    l1: {
      fixed_axes: {
        custom_source_policy: "official_only",
        dataset: "fred_md",
        frequency: "monthly",
        information_set_type: "final_revised_data",
        release_lag_rule: "ignore_release_lag",
        contemporaneous_x_rule: "forbid_same_period_predictors",
        target_structure: "multi_target",
        variable_universe: "all_variables",
        fred_sd_frequency_policy: "report_only",
        fred_sd_state_group: "all_states",
        state_selection: "all_states",
        fred_sd_variable_group: "all_sd_variables",
        sd_variable_selection: "all_sd_variables",
        raw_missing_policy: "preserve_raw_missing",
        raw_outlier_policy: "preserve_raw_outliers",
        official_transform_policy: "apply_official_tcode",
        official_transform_scope: "target_and_predictors",
        missing_availability: "zero_fill_leading_predictor_gaps"
      },
      leaf_config: {
        target: "INDPRO",
        targets: ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "RPI"],
        horizons: [1, 3, 6, 12],
        sample_start_date: "1960-01",
        sample_end_date: "",
        custom_source_path: "",
        sd_states: [],
        sd_variables: []
      }
    },
    l2: {
      fixed_axes: {
        fred_sd_mixed_frequency_representation: "calendar_aligned_frame",
        horizon_target_construction: "future_target_level_t_plus_h",
        target_transform: "level",
        target_normalization: "none",
        tcode_policy: "official_tcode_only",
        x_missing_policy: "em_impute",
        x_outlier_policy: "iqr_clip",
        scaling_policy: "standard",
        target_lag_block: "fixed_target_lags",
        x_lag_feature_block: "fixed_predictor_lags",
        factor_feature_block: "pca_static_factors",
        level_feature_block: "none",
        temporal_feature_block: "none",
        rotation_feature_block: "none",
        feature_block_combination: "append_to_base_predictors",
        feature_selection_policy: "none",
        feature_selection_semantics: "select_after_factor",
        evaluation_scale: "original_scale",
        feature_builder: "factors_plus_target_lags"
      },
      leaf_config: {}
    },
    l5: {
      fixed_axes: {
        primary_metric: "mse",
        point_metrics: ["mse", "mae"],
        density_metrics: ["log_score", "crps"],
        direction_metrics: [],
        relative_metrics: ["relative_mse", "r2_oos"],
        benchmark_window: "full_oos",
        benchmark_scope: "all_targets_horizons",
        agg_time: "mean",
        agg_horizon: "per_horizon_separate",
        agg_target: "per_target_separate",
        agg_state: "pool_states",
        oos_period: "full_oos",
        regime_use: "pooled",
        regime_metrics: [],
        decomposition_target: "none",
        decomposition_order: "marginal",
        ranking: "by_primary_metric",
        report_style: "single_table"
      },
      leaf_config: {}
    },
    l6: {
      enabled: false,
      test_scope: "per_target_horizon",
      dependence_correction: "newey_west",
      overlap_handling: "nw_with_h_minus_1_lag",
      sub_layers: {
        L6_A_equal_predictive: {
          enabled: false,
          fixed_axes: {
            equal_predictive_test: "dm_diebold_mariano",
            model_pair_strategy: "vs_benchmark_only"
          }
        }
      },
      leaf_config: {}
    },
    l8: {
      fixed_axes: {
        export_format: "json_csv",
        compression: "none",
        saved_objects: ["forecasts", "metrics", "ranking"],
        model_artifacts_format: "pickle",
        provenance_fields: ["recipe_yaml_full", "recipe_hash", "package_version", "python_version", "git_commit_sha", "random_seed_used", "runtime_environment", "runtime_duration", "cell_resolved_axes"],
        manifest_format: "json",
        artifact_granularity: "per_cell",
        naming_convention: "descriptive"
      },
      leaf_config: {
        output_directory: "./macrocast_output/default_recipe/timestamp/",
        descriptive_naming_template: "{model_family}_{forecast_strategy}_h{horizon}"
      }
    }
  },
  diagnostics: {
    l1_5: { enabled: false, preset: "minimal", fixed_axes: {} },
    l2_5: { enabled: false, preset: "minimal", fixed_axes: {} },
    l3_5: { enabled: false, preset: "minimal", fixed_axes: {} },
    l4_5: { enabled: false, preset: "minimal", fixed_axes: {} }
  },
  dags: {
    l3: {
      enabled: true,
      nodes: [
        { id: "src_x", type: "source", op: "source", label: "L2 predictors", x: 90, y: 110, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "predictors" } },
        { id: "src_y", type: "source", op: "source", label: "L2 target", x: 90, y: 270, params: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", role: "target" } },
        { id: "x_lag", type: "step", op: "lag", label: "Broad lag block", x: 360, y: 120, params: { n_lag: 4 } },
        { id: "y_h", type: "step", op: "target_construction", label: "Target horizons", x: 360, y: 280, params: { horizons: [1, 3, 6, 12] } },
        { id: "features", type: "sink", op: "sink", label: "l3_features_v1", x: 650, y: 190, params: { X_final: "x_lag", y_final: "y_h" } }
      ],
      edges: [
        { from: "src_x", to: "x_lag" },
        { from: "src_y", to: "y_h" },
        { from: "x_lag", to: "features" },
        { from: "y_h", to: "features" }
      ]
    },
    l4: {
      enabled: true,
      nodes: [
        { id: "src_X", type: "source", op: "source", label: "X_final", x: 80, y: 120, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "X_final" } },
        { id: "src_y", type: "source", op: "source", label: "y_final", x: 80, y: 280, params: { layer_ref: "l3", sink_name: "l3_features_v1", component: "y_final" } },
        { id: "fit_ridge", type: "step", op: "fit_model", label: "Ridge model", x: 340, y: 120, params: { family: "ridge" } },
        { id: "fit_benchmark", type: "step", op: "fit_model", label: "AR-BIC benchmark", x: 340, y: 280, params: { family: "autoregressive_bic", is_benchmark: true } },
        { id: "predict_ridge", type: "step", op: "predict", label: "Predict ridge", x: 600, y: 120, params: {} },
        { id: "predict_benchmark", type: "step", op: "predict", label: "Predict benchmark", x: 600, y: 280, params: {} },
        { id: "forecasts", type: "sink", op: "sink", label: "l4_forecasts_v1", x: 860, y: 190, params: { forecasts: ["predict_ridge", "predict_benchmark"] } }
      ],
      edges: [
        { from: "src_X", to: "fit_ridge" },
        { from: "src_y", to: "fit_ridge" },
        { from: "src_y", to: "fit_benchmark" },
        { from: "fit_ridge", to: "predict_ridge" },
        { from: "src_X", to: "predict_ridge" },
        { from: "fit_benchmark", to: "predict_benchmark" },
        { from: "predict_ridge", to: "forecasts" },
        { from: "predict_benchmark", to: "forecasts" }
      ]
    },
    l7: {
      enabled: false,
      nodes: [
        { id: "src_model", type: "source", op: "source", label: "L4 artifacts", x: 120, y: 180, params: { layer_ref: "l4", sink_name: "l4_model_artifacts_v1" } },
        { id: "importance", type: "step", op: "permutation_importance", label: "Importance", x: 390, y: 180, params: { scope: "global" } },
        { id: "importance_sink", type: "sink", op: "sink", label: "l7_importance_v1", x: 660, y: 180, params: {} }
      ],
      edges: [
        { from: "src_model", to: "importance" },
        { from: "importance", to: "importance_sink" }
      ]
    }
  }
};

const $ = (selector) => document.querySelector(selector);
const layerById = (id) => layerDefs.find((layer) => layer.id === id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const DAG_CANVAS_TOP = 292;

function resetState() {
  window.location.reload();
}

function render() {
  $("#recipeName").value = state.recipeName;
  renderLayerRail();
  renderWorkspace();
  renderInspector();
  renderBottomPanel();
  renderValidationBadge();
}

function renderLayerRail() {
  const rail = $("#layerRail");
  rail.innerHTML = "";
  const mapButton = document.createElement("button");
  mapButton.className = `layer-row ${state.selectedLayer === "map" ? "active" : ""}`;
  mapButton.innerHTML = `<span class="layer-id">MAP</span><span class="layer-name">Contract Map</span><span class="layer-status on"></span>`;
  mapButton.addEventListener("click", () => selectLayer("map"));
  rail.appendChild(mapButton);

  for (const layer of layerDefs) {
    const row = document.createElement("button");
    const status = layerStatus(layer.id);
    row.className = `layer-row ${layer.mode === "diagnostic" ? "diagnostic" : ""} ${state.selectedLayer === layer.id ? "active" : ""}`;
    row.innerHTML = `<span class="layer-id">${layer.id.toUpperCase()}</span><span class="layer-name">${layer.name}</span><span class="layer-status ${status}"></span>`;
    row.addEventListener("click", () => selectLayer(layer.id));
    rail.appendChild(row);
  }
}

function layerStatus(id) {
  if (state.diagnostics[id]) return state.diagnostics[id].enabled ? "on" : "off";
  if (state.dags[id]) return state.dags[id].enabled ? "on" : "off";
  if (id === "l6") return state.layers.l6.enabled ? "on" : "off";
  return "on";
}

function displayMode(layer) {
  if (layer.mode === "dag") return "builder";
  if (layer.mode === "dag-toggle") return "builder-toggle";
  return layer.mode;
}

function selectLayer(id) {
  state.selectedLayer = id;
  state.selectedNode = null;
  state.connectingFrom = null;
  render();
}

function renderWorkspace() {
  const layer = layerById(state.selectedLayer);
  $("#workspaceEyebrow").textContent = layer ? `${layer.id.toUpperCase()} ${displayMode(layer)}` : "Contract";
  $("#workspaceTitle").textContent = layer ? layer.name : "Layer Map";
  $("#workspaceActions").innerHTML = "";
  const body = $("#workspaceBody");
  body.innerHTML = "";

  if (!layer) {
    renderMap(body);
    return;
  }

  if (layer.mode === "dag" || layer.mode === "dag-toggle") {
    renderDagWorkspace(layer, body);
  } else if (layer.mode === "diagnostic") {
    renderDiagnosticForm(layer, body);
  } else {
    renderFormLayer(layer, body);
  }
}

function renderMap(body) {
  const shell = document.createElement("div");
  shell.className = "contract-flow";

  const lanes = document.createElement("div");
  lanes.className = "flow-lanes";
  const mainLayers = layerDefs.filter((layer) => !layer.id.includes("_5"));
  for (const layer of mainLayers) {
    lanes.appendChild(flowLayerCard(layer));
    const diagnostics = layerDefs.filter((candidate) => candidate.parent === layer.id);
    for (const diagnostic of diagnostics) lanes.appendChild(flowDiagnosticChip(diagnostic));
  }

  const legend = document.createElement("div");
  legend.className = "flow-legend";
  legend.innerHTML = `
    <span><i class="legend-dot configured"></i>configured</span>
    <span><i class="legend-dot optional"></i>optional/off</span>
    <span><i class="legend-dot dag"></i>Builder layer</span>
    <span><i class="legend-dot diagnostic"></i>.5 diagnostic</span>
  `;

  const focus = layerById(state.mapFocusLayer) || layerById("l1");
  const strip = document.createElement("div");
  strip.className = "focus-strip";
  strip.innerHTML = `
    <div>
      <div class="eyebrow">Selected Contract Layer</div>
      <h2>${focus.id.toUpperCase()} ${focus.name}</h2>
      <p>${focus.role}</p>
    </div>
    <button class="primary-button" data-action="edit-focus">Edit ${focus.id.toUpperCase()}</button>
  `;
  strip.querySelector("[data-action='edit-focus']").addEventListener("click", () => selectLayer(focus.id));

  shell.appendChild(legend);
  shell.appendChild(lanes);
  shell.appendChild(strip);
  body.appendChild(shell);
}

function flowLayerCard(layer) {
  const card = document.createElement("button");
  const isFocus = state.mapFocusLayer === layer.id;
  const isDag = layer.mode.includes("dag");
  card.className = `flow-card ${isFocus ? "focused" : ""} ${isDag ? "dag" : ""} ${layerStatus(layer.id)}`;
  card.innerHTML = `
    <div class="flow-card-head">
      <span class="flow-id">${layer.id.toUpperCase()}</span>
      <span class="pill">${displayMode(layer)}</span>
    </div>
    <div class="flow-title">${layer.name}</div>
    <div class="flow-role">${layer.role}</div>
    ${flowLayerFacts(layer.id)}
  `;
  card.addEventListener("click", () => {
    state.mapFocusLayer = layer.id;
    render();
  });
  return card;
}

function flowDiagnosticChip(layer) {
  const chip = document.createElement("button");
  const status = layerStatus(layer.id);
  chip.className = `flow-diagnostic ${state.mapFocusLayer === layer.id ? "focused" : ""} ${status}`;
  chip.innerHTML = `<span>${layer.id.toUpperCase()}</span><span>${layer.name}</span>`;
  chip.addEventListener("click", () => {
    state.mapFocusLayer = layer.id;
    render();
  });
  return chip;
}

function flowLayerFacts(layerId) {
  if (layerId === "l1") {
    const fixed = state.layers.l1.fixed_axes;
    const leaf = state.layers.l1.leaf_config;
    const targets = fixed.target_structure === "single_target" ? leaf.target : `${leaf.targets.length} targets`;
    return `<div class="flow-facts"><span>${fixed.dataset || "custom"}</span><span>${targets}</span><span>h=${leaf.horizons.join(",")}</span></div>`;
  }
  if (layerId === "l3" || layerId === "l4" || layerId === "l7") {
    const dag = state.dags[layerId];
    return `<div class="flow-facts"><span>${dag.nodes.length} nodes</span><span>${dag.edges.length} edges</span><span>${dag.enabled ? "on" : "off"}</span></div>`;
  }
  if (layerId === "l5") {
    const fixed = state.layers.l5.fixed_axes;
    return `<div class="flow-facts"><span>${fixed.primary_metric}</span><span>${fixed.ranking}</span></div>`;
  }
  if (layerId === "l6") {
    return `<div class="flow-facts"><span>${state.layers.l6.enabled ? "enabled" : "off"}</span><span>${state.layers.l6.test_scope}</span></div>`;
  }
  if (layerId === "l8") {
    return `<div class="flow-facts"><span>${state.layers.l8.fixed_axes.export_format}</span><span>${state.layers.l8.fixed_axes.artifact_granularity}</span></div>`;
  }
  return `<div class="flow-facts"><span>${layerStatus(layerId)}</span></div>`;
}

function renderFormLayer(layer, body) {
  const grid = document.createElement("div");
  grid.className = "form-grid";
  if (layer.id === "l0") grid.appendChild(sectionFromFields("Runtime policy", [
    selectField("failure_policy", state.layers.l0.fixed_axes.failure_policy, ["fail_fast", "continue_on_failure"], (v) => state.layers.l0.fixed_axes.failure_policy = v),
    selectField("reproducibility_mode", state.layers.l0.fixed_axes.reproducibility_mode, ["seeded_reproducible", "exploratory"], (v) => state.layers.l0.fixed_axes.reproducibility_mode = v),
    selectField("compute_mode", state.layers.l0.fixed_axes.compute_mode, ["serial", "parallel"], (v) => state.layers.l0.fixed_axes.compute_mode = v),
    textField("random_seed", state.layers.l0.leaf_config.random_seed, (v) => state.layers.l0.leaf_config.random_seed = Number(v) || 42)
  ]));

  if (layer.id === "l1") {
    const sourceFields = [selectAxis("l1", "custom_source_policy")];
    if (usesFredSource()) sourceFields.push(selectAxis("l1", "dataset"));
    if (needsFrequencyChoice()) sourceFields.push(selectAxis("l1", "frequency"));
    if (usesCustomSource()) {
      sourceFields.push(textField("custom_source_path", state.layers.l1.leaf_config.custom_source_path, (v) => state.layers.l1.leaf_config.custom_source_path = v));
    }
    grid.appendChild(sectionFromFields("Data source mode / frequency", sourceFields));

    const timingFields = [];
    if (!isCustomOnly()) {
      timingFields.push(selectAxis("l1", "information_set_type"));
      timingFields.push(selectAxis("l1", "release_lag_rule"));
    }
    timingFields.push(selectAxis("l1", "contemporaneous_x_rule"));
    grid.appendChild(sectionFromFields("Forecast-time information", timingFields));

    const targetFields = [
      selectAxis("l1", "target_structure"),
    ];
    if (state.layers.l1.fixed_axes.target_structure === "single_target") {
      targetFields.push(textField("target", state.layers.l1.leaf_config.target, (v) => state.layers.l1.leaf_config.target = v));
    } else {
      targetFields.push(textAreaField("targets", state.layers.l1.leaf_config.targets.join(", "), (v) => state.layers.l1.leaf_config.targets = splitCsv(v)));
    }
    targetFields.push(textAreaField("horizons", state.layers.l1.leaf_config.horizons.join(", "), (v) => state.layers.l1.leaf_config.horizons = splitCsv(v).map(Number).filter(Boolean)));
    targetFields.push(textField("sample_start_date", state.layers.l1.leaf_config.sample_start_date, (v) => state.layers.l1.leaf_config.sample_start_date = v));
    targetFields.push(textField("sample_end_date", state.layers.l1.leaf_config.sample_end_date, (v) => state.layers.l1.leaf_config.sample_end_date = v));
    if (usesFredMdQdMetadata()) targetFields.push(selectAxis("l1", "variable_universe"));
    grid.appendChild(sectionFromFields("Target and predictor definitions", targetFields));

    if (hasFredSdSource()) {
      const fredSdFields = [
        selectAxis("l1", "fred_sd_frequency_policy"),
        selectAxis("l1", "fred_sd_state_group"),
        selectAxis("l1", "state_selection")
      ];
      if (state.layers.l1.fixed_axes.state_selection === "selected_states") {
        fredSdFields.push(textAreaField("sd_states", state.layers.l1.leaf_config.sd_states.join(", "), (v) => state.layers.l1.leaf_config.sd_states = splitCsv(v)));
      }
      fredSdFields.push(selectAxis("l1", "fred_sd_variable_group"));
      fredSdFields.push(selectAxis("l1", "sd_variable_selection"));
      if (state.layers.l1.fixed_axes.sd_variable_selection === "selected_sd_variables") {
        fredSdFields.push(textAreaField("sd_variables", state.layers.l1.leaf_config.sd_variables.join(", "), (v) => state.layers.l1.leaf_config.sd_variables = splitCsv(v)));
      }
      grid.appendChild(sectionFromFields("FRED-SD predictor scope", fredSdFields));
    }

    const sourceQualityFields = [
      selectAxis("l1", "raw_missing_policy"),
      selectAxis("l1", "raw_outlier_policy")
    ];
    if (usesFredMdQdMetadata()) {
      sourceQualityFields.push(selectAxis("l1", "official_transform_policy"));
      sourceQualityFields.push(selectAxis("l1", "official_transform_scope"));
    }
    sourceQualityFields.push(selectAxis("l1", "missing_availability"));
    grid.appendChild(sectionFromFields("Raw source / official transform / availability", sourceQualityFields));
  }

  if (layer.id === "l2") {
    grid.appendChild(sectionFromFields("FRED-SD mixed frequency", [
      selectAxis("l2", "fred_sd_mixed_frequency_representation")
    ]));
    grid.appendChild(sectionFromFields("Target construction", [
      selectAxis("l2", "horizon_target_construction"),
      selectAxis("l2", "target_transform"),
      selectAxis("l2", "target_normalization")
    ]));
    grid.appendChild(sectionFromFields("Transform and cleaning", [
      selectAxis("l2", "tcode_policy"),
      selectAxis("l2", "x_missing_policy"),
      selectAxis("l2", "x_outlier_policy"),
      selectAxis("l2", "scaling_policy")
    ]));
    grid.appendChild(sectionFromFields("Feature blocks", [
      selectAxis("l2", "target_lag_block"),
      selectAxis("l2", "x_lag_feature_block"),
      selectAxis("l2", "factor_feature_block"),
      selectAxis("l2", "level_feature_block"),
      selectAxis("l2", "temporal_feature_block"),
      selectAxis("l2", "rotation_feature_block")
    ]));
    grid.appendChild(sectionFromFields("Composition, selection, handoff", [
      selectAxis("l2", "feature_block_combination"),
      selectAxis("l2", "feature_selection_policy"),
      selectAxis("l2", "feature_selection_semantics"),
      selectAxis("l2", "evaluation_scale"),
      selectAxis("l2", "feature_builder")
    ]));
  }

  if (layer.id === "l5") {
    grid.appendChild(sectionFromFields("Metric specification", [
      selectAxis("l5", "primary_metric"),
      textAreaField("point_metrics", state.layers.l5.fixed_axes.point_metrics.join(", "), (v) => state.layers.l5.fixed_axes.point_metrics = splitCsv(v)),
      textAreaField("density_metrics", state.layers.l5.fixed_axes.density_metrics.join(", "), (v) => state.layers.l5.fixed_axes.density_metrics = splitCsv(v)),
      textAreaField("direction_metrics", state.layers.l5.fixed_axes.direction_metrics.join(", "), (v) => state.layers.l5.fixed_axes.direction_metrics = splitCsv(v)),
      textAreaField("relative_metrics", state.layers.l5.fixed_axes.relative_metrics.join(", "), (v) => state.layers.l5.fixed_axes.relative_metrics = splitCsv(v))
    ]));
    grid.appendChild(sectionFromFields("Benchmark and aggregation", [
      selectAxis("l5", "benchmark_window"),
      selectAxis("l5", "benchmark_scope"),
      selectAxis("l5", "agg_time"),
      selectAxis("l5", "agg_horizon"),
      selectAxis("l5", "agg_target"),
      selectAxis("l5", "agg_state")
    ]));
    grid.appendChild(sectionFromFields("Slicing, decomposition, ranking", [
      selectAxis("l5", "oos_period"),
      selectAxis("l5", "regime_use"),
      textAreaField("regime_metrics", state.layers.l5.fixed_axes.regime_metrics.join(", "), (v) => state.layers.l5.fixed_axes.regime_metrics = splitCsv(v)),
      selectAxis("l5", "decomposition_target"),
      selectAxis("l5", "decomposition_order"),
      selectAxis("l5", "ranking"),
      selectAxis("l5", "report_style")
    ]));
  }

  if (layer.id === "l6") {
    grid.appendChild(toggleSection("Statistical tests", state.layers.l6.enabled, (checked) => state.layers.l6.enabled = checked, [
      selectTopLevelAxis("l6", "test_scope"),
      selectTopLevelAxis("l6", "dependence_correction"),
      selectTopLevelAxis("l6", "overlap_handling"),
      toggleField("L6_A_equal_predictive.enabled", state.layers.l6.sub_layers.L6_A_equal_predictive.enabled, (checked) => state.layers.l6.sub_layers.L6_A_equal_predictive.enabled = checked),
      selectSubLayerAxis("L6_A_equal_predictive", "equal_predictive_test"),
      selectSubLayerAxis("L6_A_equal_predictive", "model_pair_strategy")
    ]));
  }

  if (layer.id === "l8") grid.appendChild(sectionFromFields("Output", [
    selectAxis("l8", "export_format"),
    selectAxis("l8", "compression"),
    textAreaField("saved_objects", state.layers.l8.fixed_axes.saved_objects.join(", "), (v) => state.layers.l8.fixed_axes.saved_objects = splitCsv(v)),
    selectAxis("l8", "model_artifacts_format"),
    textAreaField("provenance_fields", state.layers.l8.fixed_axes.provenance_fields.join(", "), (v) => state.layers.l8.fixed_axes.provenance_fields = splitCsv(v)),
    selectAxis("l8", "manifest_format"),
    selectAxis("l8", "artifact_granularity"),
    selectAxis("l8", "naming_convention"),
    textField("output_directory", state.layers.l8.leaf_config.output_directory, (v) => state.layers.l8.leaf_config.output_directory = v),
    textField("descriptive_naming_template", state.layers.l8.leaf_config.descriptive_naming_template, (v) => state.layers.l8.leaf_config.descriptive_naming_template = v)
  ]));

  body.appendChild(grid);
}

function renderDiagnosticForm(layer, body) {
  const diagnostic = state.diagnostics[layer.id];
  const grid = document.createElement("div");
  grid.className = "form-grid";
  grid.appendChild(toggleSection(`${layer.name}`, diagnostic.enabled, (checked) => diagnostic.enabled = checked, [
    selectField("preset", diagnostic.preset, ["minimal", "full", "custom"], (v) => diagnostic.preset = v)
  ], "When off: no nodes, no sink. When on: L8 can save this diagnostic artifact."));
  body.appendChild(grid);
}

function renderDagWorkspace(layer, body) {
  const dag = state.dags[layer.id];
  const actions = $("#workspaceActions");
  actions.innerHTML = `
    <button class="icon-button" data-action="add-step">+ Blank Block</button>
    <button class="icon-button" data-action="layout">Reset visual order</button>
  `;
  actions.querySelector('[data-action="add-step"]').addEventListener("click", () => addNode(layer.id, "step"));
  actions.querySelector('[data-action="layout"]').addEventListener("click", () => autoLayout(layer.id));

  if (layer.mode === "dag-toggle" && !dag.enabled) {
    const off = document.createElement("div");
    off.className = "empty-state";
    off.innerHTML = `<p>${layer.name} is off. Enable it in the inspector to create nodes and sinks.</p>`;
    body.appendChild(off);
    return;
  }

  const shell = document.createElement("div");
  shell.className = "builder-shell";
  shell.innerHTML = `
    <section class="builder-guide">
      ${dagGuideHtml(layer.id)}
    </section>
    <section class="builder-section">
      <div class="builder-section-head">
        <div>
          <div class="eyebrow">${layer.id.toUpperCase()} Builder</div>
          <h2>${layer.id === "l3" ? "Ordered Feature Blocks" : "Selected Components"}</h2>
          <p>${layer.id === "l3" ? "Order matters here: earlier feature blocks feed later composition choices." : "Order is shown for readability; dependencies are inferred from the selected component type."}</p>
        </div>
        <div class="builder-template-bar">${dagTemplateButtons(layer.id)}</div>
      </div>
      <div class="builder-columns">
        <div>
          <h3>Available blocks</h3>
          <div class="builder-palette">${dagPresetButtons(layer.id)}</div>
        </div>
        <div>
          <h3>Current selection</h3>
          <div class="ordered-block-list">${orderedBlockList(layer.id)}</div>
        </div>
      </div>
    </section>
  `;
  body.appendChild(shell);
  shell.querySelectorAll("[data-template]").forEach((button) => {
    button.addEventListener("click", () => applyDagTemplate(layer.id, button.dataset.template));
  });
  shell.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => addPresetNode(layer.id, button.dataset.preset));
  });
  shell.querySelectorAll("[data-move]").forEach((button) => {
    button.addEventListener("click", () => moveDagNode(layer.id, button.dataset.node, Number(button.dataset.move)));
  });
  shell.querySelectorAll("[data-remove-node]").forEach((button) => {
    button.addEventListener("click", () => removeDagNode(layer.id, button.dataset.removeNode));
  });
  shell.querySelectorAll("[data-select-node]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedNode = button.dataset.selectNode;
      render();
    });
  });
}

function dagPresetButtons(layerId) {
  return (DAG_PRESETS[layerId] || []).map((preset) => (
    `<button class="builder-palette-button" data-preset="${preset.key}" title="${preset.help}"><strong>${preset.label}</strong><span>${preset.help}</span></button>`
  )).join("");
}

function dagTemplateButtons(layerId) {
  const templates = DAG_TEMPLATES[layerId] || {};
  return Object.entries(templates).map(([key, template]) => (
    `<button class="template-button" data-template="${key}" title="${template.description}">${template.label}</button>`
  )).join("");
}

function dagGuideHtml(layerId) {
  const guide = DAG_GUIDES[layerId];
  if (!guide) return "";
  const template = DAG_TEMPLATES[layerId]?.[guide.defaultTemplate];
  return `
    <div class="dag-guide-main">
      <div class="eyebrow">How to build this layer</div>
      <h2>${layerId.toUpperCase()} recommended structure</h2>
      <p>${guide.purpose}</p>
      ${template ? `<p class="dag-guide-default">Recommended start: ${template.label}. ${template.description}</p>` : ""}
    </div>
    <div class="dag-guide-columns">
      ${dagGuideList("Build order", guide.steps)}
      ${dagGuideList("Must include", guide.required)}
      ${dagGuideList("Do not put here", guide.avoid)}
    </div>
  `;
}

function dagGuideList(title, items) {
  return `
    <div class="dag-guide-list">
      <h3>${title}</h3>
      <ol>${items.map((item) => `<li>${item}</li>`).join("")}</ol>
    </div>
  `;
}

function orderedBlockList(layerId) {
  const blocks = orderedDagBlocks(layerId);
  if (!blocks.length) return `<div class="empty-state">No blocks selected.</div>`;
  return blocks.map((node, index) => blockRowHtml(layerId, node, index, blocks.length)).join("");
}

function orderedDagBlocks(layerId) {
  return state.dags[layerId].nodes
    .filter((node) => !["source", "sink"].includes(node.type))
    .sort((a, b) => a.x - b.x || a.y - b.y);
}

function blockRowHtml(layerId, node, index, count) {
  const inputs = incomingLabels(layerId, node.id);
  const outputs = outgoingLabels(layerId, node.id);
  const orderControls = layerId === "l3"
    ? `<button class="mini-button" data-node="${node.id}" data-move="-1" ${index === 0 ? "disabled" : ""}>Up</button><button class="mini-button" data-node="${node.id}" data-move="1" ${index === count - 1 ? "disabled" : ""}>Down</button>`
    : "";
  return `
    <article class="ordered-block ${state.selectedNode === node.id ? "selected" : ""}">
      <div class="ordered-block-index">${index + 1}</div>
      <div class="ordered-block-main">
        <div class="ordered-block-title">${node.label}</div>
        <div class="ordered-block-meta">${nodeTypeLabel(node.type)} · ${optionLabel(node.op)} · YAML key: ${node.id}</div>
        <p>${nodeHelp(layerId, node)}</p>
        <div class="io-line"><span>Inputs</span>${inputs || "auto / none"}</div>
        <div class="io-line"><span>Outputs</span>${outputs || "auto / final sink"}</div>
      </div>
      <div class="ordered-block-actions">
        ${orderControls}
        <button class="mini-button" data-select-node="${node.id}">Edit</button>
        <button class="mini-button danger" data-remove-node="${node.id}">Remove</button>
      </div>
    </article>
  `;
}

function incomingLabels(layerId, nodeId) {
  const dag = state.dags[layerId];
  return dag.edges
    .filter((edge) => edge.to === nodeId)
    .map((edge) => dag.nodes.find((node) => node.id === edge.from)?.label || edge.from)
    .join(", ");
}

function outgoingLabels(layerId, nodeId) {
  const dag = state.dags[layerId];
  return dag.edges
    .filter((edge) => edge.from === nodeId)
    .map((edge) => dag.nodes.find((node) => node.id === edge.to)?.label || edge.to)
    .join(", ");
}

function drawEdges(svg, dag) {
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.innerHTML = "";
  for (const edge of dag.edges) {
    const from = dag.nodes.find((node) => node.id === edge.from);
    const to = dag.nodes.find((node) => node.id === edge.to);
    if (!from || !to) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const x1 = from.x + 190;
    const y1 = from.y + DAG_CANVAS_TOP + 37;
    const x2 = to.x;
    const y2 = to.y + DAG_CANVAS_TOP + 37;
    const mid = Math.max(40, (x2 - x1) / 2);
    line.setAttribute("d", `M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`);
    line.setAttribute("stroke", "#4a5260");
    line.setAttribute("fill", "none");
    line.setAttribute("stroke-width", "1.5");
    svg.appendChild(line);
  }
}

function drawNodes(canvas, layerId, dag) {
  canvas.innerHTML = "";
  for (const node of dag.nodes) {
    const el = document.createElement("div");
    el.className = `dag-node ${node.type} ${state.selectedNode === node.id ? "selected" : ""}`;
    el.style.left = `${node.x}px`;
    el.style.top = `${node.y + DAG_CANVAS_TOP}px`;
    el.innerHTML = `
      <div class="node-strip"></div>
      <div class="node-content">
        <div class="node-title">${node.label}</div>
        <div><span class="node-kind">${nodeTypeLabel(node.type)}</span></div>
        <div class="node-meta">${node.id} · ${optionLabel(node.op)}</div>
        <div class="node-help">${nodeHelp(layerId, node)}</div>
        <div class="node-ports">
          <button class="port-button ${state.connectingFrom === node.id ? "connecting" : ""}" data-port="out">Start link</button>
          <button class="port-button" data-port="in">Link here</button>
        </div>
      </div>
    `;
    el.addEventListener("pointerdown", (event) => startDrag(event, layerId, node.id));
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selectedNode = node.id;
      render();
    });
    el.querySelector('[data-port="out"]').addEventListener("click", (event) => {
      event.stopPropagation();
      state.connectingFrom = node.id;
      render();
    });
    el.querySelector('[data-port="in"]').addEventListener("click", (event) => {
      event.stopPropagation();
      if (state.connectingFrom && state.connectingFrom !== node.id) {
        dag.edges.push({ from: state.connectingFrom, to: node.id });
        state.connectingFrom = null;
      }
      render();
    });
    canvas.appendChild(el);
  }
}

function nodeTypeLabel(type) {
  return {
    source: "Input",
    step: "Transform / model step",
    combine: "Combine",
    sink: "Output"
  }[type] || type;
}

function nodeHelp(layerId, node) {
  const preset = (DAG_PRESETS[layerId] || []).find((item) => item.op === node.op || item.label === node.label);
  if (preset) return preset.help;
  if (node.type === "source") return "Input from an earlier layer.";
  if (node.type === "sink") return "Named output that later layers can consume.";
  return "Custom step. Edit op and params in the inspector.";
}

function startDrag(event, layerId, nodeId) {
  if (event.target.tagName === "BUTTON") return;
  const dag = state.dags[layerId];
  const node = dag.nodes.find((item) => item.id === nodeId);
  const startX = event.clientX;
  const startY = event.clientY;
  const originalX = node.x;
  const originalY = node.y;
  event.currentTarget.setPointerCapture(event.pointerId);
  const move = (moveEvent) => {
    node.x = Math.max(8, originalX + moveEvent.clientX - startX);
    node.y = Math.max(8, originalY + moveEvent.clientY - startY);
    renderWorkspace();
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

function addNode(layerId, type) {
  const dag = state.dags[layerId];
  if (layerId === "l3" && type === "combine") {
    // Combine is allowed in L3 only for feature combination, not forecast combination.
  }
  const count = dag.nodes.length + 1;
  const id = `${type}_${count}`;
  dag.nodes.push({ id, type, op: type === "sink" ? "sink" : type, label: `${type} ${count}`, x: 180 + count * 24, y: 120 + count * 18, params: {} });
  state.selectedNode = id;
  render();
}

function addPresetNode(layerId, presetKey) {
  const preset = (DAG_PRESETS[layerId] || []).find((item) => item.key === presetKey);
  if (!preset) return;
  const dag = state.dags[layerId];
  const count = dag.nodes.filter((node) => node.op === preset.op).length + 1;
  const id = `${preset.key}_${count}`;
  const order = orderedDagBlocks(layerId).length;
  dag.nodes.push({
    id,
    type: preset.type,
    op: preset.op,
    label: preset.label,
    x: 300 + order * 40,
    y: 200 + order * 20,
    params: clone(preset.params)
  });
  state.selectedNode = id;
  render();
}

function moveDagNode(layerId, nodeId, delta) {
  const blocks = orderedDagBlocks(layerId);
  const index = blocks.findIndex((node) => node.id === nodeId);
  const target = index + delta;
  if (index < 0 || target < 0 || target >= blocks.length) return;
  const current = blocks[index];
  const other = blocks[target];
  [current.x, other.x] = [other.x, current.x];
  [current.y, other.y] = [other.y, current.y];
  render();
}

function removeDagNode(layerId, nodeId) {
  const dag = state.dags[layerId];
  const node = dag.nodes.find((item) => item.id === nodeId);
  if (!node || ["source", "sink"].includes(node.type)) return;
  dag.nodes = dag.nodes.filter((item) => item.id !== nodeId);
  dag.edges = dag.edges.filter((edge) => edge.from !== nodeId && edge.to !== nodeId);
  if (state.selectedNode === nodeId) state.selectedNode = null;
  render();
}

function applyDagTemplate(layerId, templateKey) {
  const template = DAG_TEMPLATES[layerId]?.[templateKey];
  if (!template) return;
  const dag = state.dags[layerId];
  dag.nodes = clone(template.nodes);
  dag.edges = clone(template.edges);
  dag.enabled = true;
  state.selectedNode = null;
  state.connectingFrom = null;
  render();
}

function autoLayout(layerId) {
  const dag = state.dags[layerId];
  dag.nodes.forEach((node, index) => {
    const col = index % 4;
    const row = Math.floor(index / 4);
    node.x = 80 + col * 250;
    node.y = 100 + row * 150;
  });
  render();
}

function renderInspector() {
  const body = $("#inspectorBody");
  body.innerHTML = "";
  const layer = layerById(state.selectedLayer);
  if (!layer) {
    body.appendChild(mapFocusInspector());
    return;
  }

  if (state.dags[layer.id] && state.selectedNode) {
    const dag = state.dags[layer.id];
    const node = dag.nodes.find((item) => item.id === state.selectedNode);
    if (node) {
      body.appendChild(nodeInspector(layer.id, node));
      return;
    }
  }

  if (state.dags[layer.id]) {
    body.appendChild(layerDagInspector(layer));
    return;
  }

  if (state.diagnostics[layer.id]) {
    body.appendChild(diagnosticInspector(layer));
    return;
  }

  body.appendChild(contractCard(`${layer.id.toUpperCase()} ${layer.name}`, layer.role));
}

function nodeInspector(layerId, node) {
  return sectionFromFields(`Node ${node.id}`, [
    readonlyField("What this node does", nodeHelp(layerId, node)),
    textField("label", node.label, (v) => node.label = v),
    textField("op", node.op, (v) => node.op = v),
    selectField("type", node.type, ["source", "step", "combine", "sink"], (v) => node.type = v),
    textAreaField("params JSON", JSON.stringify(node.params, null, 2), (v) => {
      try {
        node.params = JSON.parse(v || "{}");
      } catch {
        node.params_error = true;
      }
    }),
    buttonField("Delete node", () => {
      const dag = state.dags[layerId];
      dag.nodes = dag.nodes.filter((item) => item.id !== node.id);
      dag.edges = dag.edges.filter((edge) => edge.from !== node.id && edge.to !== node.id);
      state.selectedNode = null;
    })
  ]);
}

function layerDagInspector(layer) {
  const dag = state.dags[layer.id];
  const fields = [];
  if (layer.mode === "dag-toggle") {
    fields.push(toggleField("enabled", dag.enabled, (checked) => dag.enabled = checked));
  }
  const guide = DAG_GUIDES[layer.id];
  if (guide) {
    fields.push(readonlyField("recommended structure", guide.steps.join(" -> ")));
    fields.push(readonlyField("must include", guide.required.join(", ")));
  }
  fields.push(readonlyField("required sinks", requiredSinks(layer.id).join(", ")));
  fields.push(readonlyField("nodes", String(dag.nodes.length)));
  fields.push(readonlyField("edges", String(dag.edges.length)));
  fields.push(readonlyField("how to connect", "Start link on source node, then Link here on target node."));
  fields.push(edgeListField(layer.id));
  return sectionFromFields(`${layer.id.toUpperCase()} Builder`, fields);
}

function edgeListField(layerId) {
  const field = document.createElement("div");
  field.className = "field edge-list-field";
  const label = document.createElement("label");
  label.textContent = "Current links";
  field.appendChild(label);
  const list = document.createElement("div");
  list.className = "edge-list";
  const dag = state.dags[layerId];
  if (!dag.edges.length) {
    list.textContent = "No links yet.";
  } else {
    for (const edge of dag.edges) {
      const from = dag.nodes.find((node) => node.id === edge.from);
      const to = dag.nodes.find((node) => node.id === edge.to);
      const row = document.createElement("div");
      row.className = "edge-row";
      row.innerHTML = `<span>${from?.label || edge.from} -> ${to?.label || edge.to}</span>`;
      const remove = document.createElement("button");
      remove.className = "edge-remove";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        dag.edges = dag.edges.filter((item) => item !== edge);
        render();
      });
      row.appendChild(remove);
      list.appendChild(row);
    }
  }
  field.appendChild(list);
  return field;
}

function diagnosticInspector(layer) {
  const diagnostic = state.diagnostics[layer.id];
  return sectionFromFields(`${layer.name}`, [
    toggleField("enabled", diagnostic.enabled, (checked) => diagnostic.enabled = checked),
    readonlyField("off contract", "no nodes, no sink"),
    readonlyField("L8 saved object", diagnostic.enabled ? `diagnostics_${layer.id}` : "inactive")
  ]);
}

function contractSummary() {
  const issues = validateState();
  return sectionFromFields("Workspace", [
    readonlyField("preset", "Broad Multi-Target FRED-MD"),
    readonlyField("validation", issues.some((issue) => issue.level === "error") ? "has errors" : "valid"),
    readonlyField("targets", state.layers.l1.leaf_config.targets.join(", ")),
    readonlyField("horizons", state.layers.l1.leaf_config.horizons.join(", "))
  ]);
}

function mapFocusInspector() {
  const layer = layerById(state.mapFocusLayer) || layerById("l1");
  const fields = [
    readonlyField("layer", `${layer.id.toUpperCase()} ${layer.name}`),
    readonlyField("mode", displayMode(layer)),
    readonlyField("status", layerStatus(layer.id)),
    readonlyField("role", layer.role),
    buttonField(`Edit ${layer.id.toUpperCase()}`, () => selectLayer(layer.id))
  ];
  if (state.dags[layer.id]) {
    const dag = state.dags[layer.id];
    fields.splice(3, 0, readonlyField("nodes", String(dag.nodes.length)));
    fields.splice(4, 0, readonlyField("edges", String(dag.edges.length)));
    fields.splice(5, 0, readonlyField("required sinks", requiredSinks(layer.id).join(", ")));
  }
  if (layer.id === "l1") {
    const fixed = state.layers.l1.fixed_axes;
    const leaf = state.layers.l1.leaf_config;
    fields.splice(3, 0, readonlyField("source", `${fixed.custom_source_policy} / ${fixed.dataset || "custom"}`));
    fields.splice(4, 0, readonlyField("targets", fixed.target_structure === "single_target" ? leaf.target : leaf.targets.join(", ")));
  }
  return sectionFromFields("Map Selection", fields);
}

function renderBottomPanel() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === state.bottomTab);
    tab.onclick = () => {
      state.bottomTab = tab.dataset.tab;
      renderBottomPanel();
    };
  });
  const body = $("#bottomBody");
  if (state.bottomTab === "yaml") body.innerHTML = `<pre>${escapeHtml(generateYaml())}</pre>`;
  if (state.bottomTab === "validation") body.innerHTML = validationHtml();
  if (state.bottomTab === "contract") body.innerHTML = contractHtml();
  if (state.bottomTab === "run") body.innerHTML = `<pre>macrocast-navigate resolve recipe.yaml\nmacrocast-navigate run recipe.yaml --output-root ${state.layers.l8.leaf_config.output_directory}</pre>`;
}

function renderValidationBadge() {
  const badge = $("#validationBadge");
  const invalid = validateState().some((issue) => issue.level === "error");
  badge.textContent = invalid ? "invalid" : "valid";
  badge.className = `status-badge ${invalid ? "invalid" : "valid"}`;
}

function validateState() {
  const issues = [];
  if (state.layers.l1.fixed_axes.target_structure === "single_target" && !state.layers.l1.leaf_config.target) {
    issues.push({ level: "error", where: "L1", message: "target is required for single_target." });
  }
  if (state.layers.l1.fixed_axes.target_structure === "multi_target" && state.layers.l1.leaf_config.targets.length < 2) {
    issues.push({ level: "error", where: "L1", message: "At least two targets are required for multi_target." });
  }
  if (!state.layers.l1.leaf_config.horizons.length) issues.push({ level: "error", where: "L1", message: "At least one horizon is required." });
  if (state.layers.l1.fixed_axes.custom_source_policy !== "official_only" && !state.layers.l1.leaf_config.custom_source_path) {
    issues.push({ level: "error", where: "L1", message: "custom_source_path is required when custom_source_policy is not official_only." });
  }
  if (state.layers.l1.fixed_axes.state_selection === "selected_states" && !state.layers.l1.leaf_config.sd_states.length) {
    issues.push({ level: "error", where: "L1", message: "sd_states is required when state_selection=selected_states." });
  }
  if (state.layers.l1.fixed_axes.sd_variable_selection === "selected_sd_variables" && !state.layers.l1.leaf_config.sd_variables.length) {
    issues.push({ level: "error", where: "L1", message: "sd_variables is required when sd_variable_selection=selected_sd_variables." });
  }
  for (const layerId of ["l3", "l4"]) {
    const dag = state.dags[layerId];
    const sinks = requiredSinks(layerId);
    for (const sink of sinks) {
      if (!dag.nodes.some((node) => node.type === "sink" && node.label === sink)) {
        issues.push({ level: "warning", where: layerId.toUpperCase(), message: `Expected sink ${sink} is not explicitly represented.` });
      }
    }
  }
  const l3Bad = state.dags.l3.nodes.find((node) => /forecast/i.test(node.op) && node.type === "combine");
  if (l3Bad) issues.push({ level: "error", where: "L3", message: "Forecast combination belongs in L4, not L3." });
  for (const [id, diagnostic] of Object.entries(state.diagnostics)) {
    if (!diagnostic.enabled) issues.push({ level: "info", where: id.toUpperCase(), message: "Diagnostic off: no nodes, no sink." });
  }
  return issues;
}

function validationHtml() {
  const issues = validateState();
  if (!issues.length) return `<div class="empty-state">No validation messages.</div>`;
  return issues.map((issue) => `<div class="issue"><div class="issue-level ${issue.level}">${issue.level}</div><div><strong>${issue.where}</strong><br>${issue.message}</div></div>`).join("");
}

function contractHtml() {
  return `
    <div class="form-grid">
      ${layerDefs.map((layer) => `<div class="contract-card form-section"><h2>${layer.id.toUpperCase()} ${layer.name}</h2><p>${layer.role}</p><p class="field-hint">Mode: ${displayMode(layer)} · YAML: ${layer.key}</p></div>`).join("")}
    </div>
  `;
}

function generateYaml() {
  const out = {};
  out.recipe_id = state.recipeName;
  out[state.layers ? "0_meta" : "0_meta"] = state.layers.l0;
  out["1_data"] = l1Yaml();
  out["2_preprocessing"] = state.layers.l2;
  out["3_feature_engineering"] = dagYaml("l3");
  out["4_forecasting_model"] = dagYaml("l4");
  out["5_evaluation"] = state.layers.l5;
  if (state.layers.l6.enabled) out["6_statistical_tests"] = state.layers.l6;
  if (state.dags.l7.enabled) out["7_interpretation"] = dagYaml("l7");
  out["8_output"] = state.layers.l8;
  for (const [id, diagnostic] of Object.entries(state.diagnostics)) {
    if (diagnostic.enabled) out[layerById(id).key] = diagnostic;
  }
  return toYaml(out);
}

function l1Yaml() {
  const layer = clone(state.layers.l1);
  const fixed = layer.fixed_axes;
  const leaf = layer.leaf_config;

  if (!usesFredSource()) delete fixed.dataset;
  if (!needsFrequencyChoice()) delete fixed.frequency;
  if (!usesCustomSource()) delete leaf.custom_source_path;
  if (isCustomOnly()) {
    delete fixed.information_set_type;
    delete fixed.release_lag_rule;
  }
  if (!usesFredMdQdMetadata()) {
    delete fixed.variable_universe;
    delete fixed.official_transform_policy;
    delete fixed.official_transform_scope;
  }
  if (!hasFredSdSource()) {
    delete fixed.fred_sd_frequency_policy;
    delete fixed.fred_sd_state_group;
    delete fixed.state_selection;
    delete fixed.fred_sd_variable_group;
    delete fixed.sd_variable_selection;
    delete leaf.sd_states;
    delete leaf.sd_variables;
  } else {
    if (fixed.state_selection !== "selected_states") delete leaf.sd_states;
    if (fixed.sd_variable_selection !== "selected_sd_variables") delete leaf.sd_variables;
  }
  if (fixed.target_structure === "single_target") {
    delete leaf.targets;
  } else {
    delete leaf.target;
  }
  for (const [key, value] of Object.entries(leaf)) {
    if (value === "" || (Array.isArray(value) && value.length === 0)) delete leaf[key];
  }
  return layer;
}

function dagYaml(layerId) {
  const dag = state.dags[layerId];
  const nodesById = Object.fromEntries(dag.nodes.map((node) => [node.id, {
    enabled: true,
    name: node.label,
    type: node.type,
    operation: node.op,
    config: node.params,
    inputs: dag.edges.filter((edge) => edge.to === node.id).map((edge) => edge.from),
    outputs: dag.edges.filter((edge) => edge.from === node.id).map((edge) => edge.to)
  }]));
  const blocks = orderedDagBlocks(layerId).map((node) => ({
    id: node.id,
    block: node.op,
    enabled: true,
    config: node.params
  }));
  const contract = {
    builder_view: layerId === "l3" ? "ordered_feature_blocks" : "selected_components",
    nodes: nodesById,
    edges: dag.edges,
    sinks: Object.fromEntries(dag.nodes.filter((node) => node.type === "sink").map((node) => [node.label, node.id]))
  };
  if (layerId === "l3") {
    contract.sequence = blocks;
  } else {
    contract.selected = blocks;
  }
  return contract;
}

function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    if (value.every(isScalarValue)) return `[${value.map(scalarToYaml).join(", ")}]`;
    return value.map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return `${pad}- ${toYaml(item, indent + 2).trim()}`;
      }
      const entries = Object.entries(item);
      if (!entries.length) return `${pad}- {}`;
      return entries.map(([key, val], index) => {
        const prefix = index === 0 ? `${pad}- ` : `${pad}  `;
        if (isInlineYamlValue(val)) return `${prefix}${key}: ${toYaml(val, indent + 2).trim()}`;
        return `${prefix}${key}:\n${toYaml(val, indent + 4)}`;
      }).join("\n");
    }).join("\n");
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return "{}";
    return Object.entries(value).map(([key, val]) => {
      if (isInlineYamlValue(val)) return `${pad}${key}: ${toYaml(val, indent + 2).trim()}`;
      if (val && typeof val === "object") return `${pad}${key}:\n${toYaml(val, indent + 2)}`;
      return `${pad}${key}: ${toYaml(val, indent + 2).trim()}`;
    }).join("\n");
  }
  if (isScalarValue(value)) return scalarToYaml(value);
  return String(value);
}

function isScalarValue(value) {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function scalarToYaml(value) {
  if (value === null) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === "") return '""';
  return /^[A-Za-z0-9_+.-]+$/.test(value) ? value : JSON.stringify(value);
}

function isInlineYamlValue(value) {
  return isScalarValue(value)
    || (Array.isArray(value) && value.every(isScalarValue))
    || (value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0);
}

function requiredSinks(layerId) {
  if (layerId === "l3") return ["l3_features_v1", "l3_metadata_v1"];
  if (layerId === "l4") return ["l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"];
  if (layerId === "l7") return ["l7_importance_v1"];
  return [];
}

function isCustomOnly() {
  return state.layers.l1.fixed_axes.custom_source_policy === "custom_panel_only";
}

function usesFredSource() {
  return !isCustomOnly();
}

function usesCustomSource() {
  return state.layers.l1.fixed_axes.custom_source_policy !== "official_only";
}

function hasFredSdSource() {
  return usesFredSource() && state.layers.l1.fixed_axes.dataset.includes("fred_sd");
}

function usesFredMdQdMetadata() {
  if (!usesFredSource()) return false;
  return state.layers.l1.fixed_axes.dataset.includes("fred_md") || state.layers.l1.fixed_axes.dataset.includes("fred_qd");
}

function needsFrequencyChoice() {
  return isCustomOnly() || state.layers.l1.fixed_axes.dataset === "fred_sd";
}

function sectionFromFields(title, fields) {
  const section = document.createElement("section");
  section.className = "form-section";
  section.innerHTML = `<h2>${title}</h2>`;
  for (const field of fields) section.appendChild(field);
  return section;
}

function toggleSection(title, checked, onChange, fields, hint = "") {
  const section = sectionFromFields(title, [toggleField("enabled", checked, onChange), ...fields]);
  if (hint) {
    const note = document.createElement("div");
    note.className = "field-hint";
    note.textContent = hint;
    section.appendChild(note);
  }
  return section;
}

function textField(label, value, onChange) {
  const field = baseField(label);
  const input = document.createElement("input");
  input.value = value ?? "";
  input.addEventListener("input", () => {
    onChange(input.value);
    renderAfterEdit();
  });
  field.appendChild(input);
  return field;
}

function textAreaField(label, value, onChange) {
  const field = baseField(label);
  const input = document.createElement("textarea");
  input.value = value ?? "";
  input.addEventListener("input", () => {
    onChange(input.value);
    renderAfterEdit();
  });
  field.appendChild(input);
  const options = axisOptionsByName(label);
  if (options.length) {
    const choices = document.createElement("div");
    choices.className = "choice-help";
    choices.innerHTML = options.map((option) => `<span title="${option}">${escapeHtml(optionLabel(option))}</span>`).join("");
    field.appendChild(choices);
  }
  return field;
}

function selectAxis(layerId, axisName) {
  const layer = state.layers[layerId];
  return selectField(axisName, layer.fixed_axes[axisName], AXIS_OPTIONS[layerId][axisName], (v) => layer.fixed_axes[axisName] = v, optionDescription(axisName));
}

function selectTopLevelAxis(layerId, axisName) {
  const layer = state.layers[layerId];
  return selectField(axisName, layer[axisName], AXIS_OPTIONS[layerId][axisName], (v) => layer[axisName] = v, optionDescription(axisName));
}

function selectSubLayerAxis(subLayerName, axisName) {
  const subLayer = state.layers.l6.sub_layers[subLayerName];
  return selectField(`${subLayerName}.${axisName}`, subLayer.fixed_axes[axisName], AXIS_OPTIONS.l6[axisName], (v) => subLayer.fixed_axes[axisName] = v, optionDescription(axisName));
}

function selectField(label, value, options, onChange, describe = null) {
  const field = baseField(label);
  const select = document.createElement("select");
  for (const option of options) {
    const el = document.createElement("option");
    el.value = option;
    el.textContent = optionLabel(option);
    if (option === value) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => {
    onChange(select.value);
    render();
  });
  field.appendChild(select);
  const description = document.createElement("div");
  description.className = "field-description";
  description.textContent = describe ? describe(value) : defaultOptionDescription(label, value);
  field.appendChild(description);
  const code = document.createElement("div");
  code.className = "field-code";
  code.textContent = `YAML value: ${value}`;
  field.appendChild(code);
  return field;
}

function optionDescription(axisName) {
  return (value) => {
    const axis = AXIS_DESCRIPTIONS[axisName] || axisTitle(axisName);
    const option = OPTION_DESCRIPTIONS[value] || `Selected option: ${optionLabel(value)}.`;
    return `${axis} ${option}`;
  };
}

function defaultOptionDescription(label, value) {
  if (value === undefined || value === null) return "";
  return `Selected: ${optionLabel(String(value))}.`;
}

function optionLabel(value) {
  return OPTION_LABELS[value] || String(value).replaceAll("_", " ");
}

function axisTitle(value) {
  const clean = String(value).split(".").pop();
  return AXIS_TITLES[clean] || clean.replaceAll("_", " ");
}

function toggleField(label, checked, onChange) {
  const field = document.createElement("div");
  field.className = "field toggle-line";
  field.innerHTML = `<label>${label}</label><label class="switch"><input type="checkbox" ${checked ? "checked" : ""}><span></span></label>`;
  field.querySelector("input").addEventListener("change", (event) => {
    onChange(event.target.checked);
    render();
  });
  return field;
}

function readonlyField(label, value) {
  const field = baseField(label);
  const input = document.createElement("input");
  input.value = value;
  input.readOnly = true;
  field.appendChild(input);
  return field;
}

function buttonField(label, onClick) {
  const field = document.createElement("div");
  field.className = "field";
  const button = document.createElement("button");
  button.className = "icon-button";
  button.textContent = label;
  button.addEventListener("click", () => {
    onClick();
    render();
  });
  field.appendChild(button);
  return field;
}

function baseField(label) {
  const field = document.createElement("div");
  field.className = "field";
  const labelEl = document.createElement("label");
  const title = axisTitle(label);
  labelEl.innerHTML = `${escapeHtml(title)}${title !== label ? ` <span>${escapeHtml(label)}</span>` : ""}`;
  field.appendChild(labelEl);
  return field;
}

function renderAfterEdit() {
  renderLayerRail();
  renderBottomPanel();
  renderValidationBadge();
}

function splitCsv(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function axisOptionsByName(axisName) {
  for (const layerOptions of Object.values(AXIS_OPTIONS)) {
    if (layerOptions[axisName]) return layerOptions[axisName];
  }
  return [];
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

function exportYaml() {
  const blob = new Blob([generateYaml()], { type: "text/yaml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.recipeName || "macroforecast-recipe"}.yaml`;
  a.click();
  URL.revokeObjectURL(url);
}

$("#recipeName").addEventListener("input", (event) => {
  state.recipeName = event.target.value;
  renderAfterEdit();
});
$("#resetBtn").addEventListener("click", resetState);
$("#exportBtn").addEventListener("click", exportYaml);

render();
