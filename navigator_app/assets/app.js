const state = {
  data: null,
  sampleIndex: 0,
  axisFilter: "",
  layerFilter: "0_meta",
  activeAxis: null,
  engineState: null,
  loadedSource: null,
  activeTopologyLayer: "l0",
  activeSubLayer: null,
  activeCanonicalAxis: null,
  canonicalSelections: {},
  dagSelections: {},
};

const els = {
  sampleSelect: document.getElementById("sample-select"),
  axisSearch: document.getElementById("axis-search"),
  axisList: document.getElementById("axis-list"),
  optionList: document.getElementById("option-list"),
  axisExplainer: document.getElementById("axis-explainer"),
  axisTitle: document.getElementById("axis-title"),
  axisLayer: document.getElementById("axis-layer"),
  axisSelected: document.getElementById("axis-selected"),
  activeLayerTitle: document.getElementById("active-layer-title"),
  activeLayerDescription: document.getElementById("active-layer-description"),
  executionStatus: document.getElementById("execution-status"),
  blockedCount: document.getElementById("blocked-count"),
  disabledCount: document.getElementById("disabled-count"),
  yamlPreview: document.getElementById("yaml-preview"),
  copyYaml: document.getElementById("copy-yaml"),
  downloadYaml: document.getElementById("download-yaml"),
  importYaml: document.getElementById("import-yaml"),
  importYamlInput: document.getElementById("import-yaml-input"),
  resetPath: document.getElementById("reset-path"),
  pathSource: document.getElementById("path-source"),
  treePath: document.getElementById("tree-path"),
  runtimeSupport: document.getElementById("runtime-support"),
  layerTopology: document.getElementById("layer-topology"),
  layerDetail: document.getElementById("layer-detail"),
};

const CANONICAL_SUB_LAYERS = {
  l0: ["L0.A Execution policy"],
  l1: ["L1.A Source selection", "L1.B Target definition", "L1.C Predictor universe", "L1.D Geography scope", "L1.E Sample window", "L1.F Horizon set", "L1.G Regime definition"],
  l1_5: ["L1.5.A sample coverage", "L1.5.B univariate summary", "L1.5.C stationarity", "L1.5.D missing and outlier", "L1.5.E correlation", "L1.5.Z export"],
  l2: ["L2.A FRED-SD frequency alignment", "L2.B Transform", "L2.C Outlier handling", "L2.D Imputation", "L2.E Frame edge"],
  l2_5: ["L2.5.A comparison", "L2.5.B distribution shift", "L2.5.C correlation shift", "L2.5.D cleaning summary", "L2.5.Z export"],
  l3: ["L3.A Target construction", "L3.B Feature pipelines", "L3.C Pipeline combine", "L3.D Feature selection"],
  l3_5: ["L3.5.A comparison", "L3.5.B factor inspection", "L3.5.C feature correlation", "L3.5.D lag inspection", "L3.5.E selection", "L3.5.Z export"],
  l4: ["L4.A Model selection", "L4.B Forecast strategy", "L4.C Training window", "L4.D Tuning"],
  l4_5: ["L4.5.A fit", "L4.5.B scale", "L4.5.C window stability", "L4.5.D tuning", "L4.5.E ensemble", "L4.5.Z export"],
  l5: ["L5.A metrics", "L5.B benchmark", "L5.C aggregation", "L5.D slicing and decomposition", "L5.E ranking"],
  l6: ["L6 globals", "L6_A_equal_predictive", "L6_B_nested", "L6_C_cpa", "L6_D_multiple_model", "L6_E_density_interval", "L6_F_direction", "L6_G_residual"],
  l7: ["L7.A Importance DAG", "L7.B Output shape"],
  l8: ["L8_A_export_format", "L8_B_saved_objects", "L8_C_provenance", "L8_D_artifact_granularity"],
};

const CANONICAL_AXIS_GROUPS = {
  l0: {
    "L0.A Execution policy": ["failure_policy", "reproducibility_mode", "compute_mode"],
  },
  l1: {
    "L1.A Source selection": ["custom_source_policy", "dataset", "frequency", "vintage_policy"],
    "L1.B Target definition": ["target_structure"],
    "L1.C Predictor universe": ["variable_universe"],
    "L1.D Geography scope": ["target_geography_scope", "predictor_geography_scope"],
    "L1.E Sample window": ["sample_start_rule", "sample_end_rule"],
    "L1.F Horizon set": ["horizon_set"],
    "L1.G Regime definition": ["regime_definition", "regime_estimation_temporal_rule"],
  },
  l2: {
    "L2.A FRED-SD frequency alignment": ["sd_series_frequency_filter", "quarterly_to_monthly_rule", "monthly_to_quarterly_rule"],
    "L2.B Transform": ["transform_policy", "transform_scope"],
    "L2.C Outlier handling": ["outlier_policy", "outlier_action", "outlier_scope"],
    "L2.D Imputation": ["imputation_policy", "imputation_temporal_rule", "imputation_scope"],
    "L2.E Frame edge": ["frame_edge_policy", "frame_edge_scope"],
  },
  l5: {
    "L5.A metrics": ["primary_metric", "point_metrics", "density_metrics", "direction_metrics", "relative_metrics"],
    "L5.B benchmark": ["benchmark_window", "benchmark_scope"],
    "L5.C aggregation": ["agg_time", "agg_horizon", "agg_target", "agg_state"],
    "L5.D slicing and decomposition": ["oos_period", "regime_use", "regime_metrics", "decomposition_target", "decomposition_order"],
    "L5.E ranking": ["ranking", "report_style"],
  },
  l6: {
    "L6 globals": ["enabled", "test_scope", "dependence_correction", "overlap_handling"],
    "L6_A_equal_predictive": ["equal_predictive_test", "loss_function", "model_pair_strategy", "hln_correction"],
    "L6_B_nested": ["nested_test", "nested_pair_strategy", "cw_adjustment", "enc_test_one_sided"],
    "L6_C_cpa": ["cpa_test", "cpa_window_type", "cpa_conditioning_info", "cpa_critical_value_method"],
    "L6_D_multiple_model": ["multiple_model_test", "mcs_alpha", "mmt_loss_function", "bootstrap_method", "bootstrap_n_replications", "bootstrap_block_length", "mcs_t_statistic", "spa_studentization", "stepm_alpha"],
    "L6_E_density_interval": ["density_test", "interval_test", "coverage_levels", "pit_n_bins", "pit_test_horizon_dependence"],
    "L6_F_direction": ["direction_test", "direction_threshold", "direction_alpha"],
    "L6_G_residual": ["residual_test", "residual_lag_count", "residual_test_scope", "residual_alpha"],
  },
  l7: {
    "L7.A Importance DAG": ["enabled"],
    "L7.B Output shape": ["output_table_format", "figure_type", "top_k_features_to_show", "precision_digits", "figure_dpi", "figure_format", "latex_table_export", "markdown_table_export"],
  },
  l8: {
    "L8_A_export_format": ["export_format", "compression"],
    "L8_B_saved_objects": ["saved_objects", "model_artifacts_format"],
    "L8_C_provenance": ["provenance_fields", "manifest_format"],
    "L8_D_artifact_granularity": ["artifact_granularity", "naming_convention"],
  },
};

const TREE_LAYER_ALIASES = {
  l0: "0_meta",
  l1: "1_data_task",
  l2: "2_preprocessing",
  l5: "4_evaluation",
  l6: "6_stat_tests",
  l7: "7_importance",
  l8: "5_output_provenance",
};

const GRAPH_LAYER_SECTIONS = {
  l3: {
    "L3.A Target construction": {
      summary: "Build y at the requested forecast horizon before feature pipelines feed models.",
      columns: [
        { label: "Sources", items: [
          { name: "L2 cleaned target", kind: "source", summary: "Standard y input after L2 preprocessing." },
          { name: "L1 raw target", kind: "source", summary: "Optional raw-level target access for level pipelines." },
        ] },
        { label: "Target step", items: [
          { name: "target_construction", kind: "step", summary: "Creates direct, cumulative, or horizon-specific target series." },
        ] },
        { label: "Rules", items: [
          { name: "L3.A only", kind: "validation", summary: "target_construction belongs in the target-construction section." },
        ] },
      ],
    },
    "L3.B Feature pipelines": {
      summary: "Compose parallel feature branches from cleaned, raw, regime, or prior pipeline outputs.",
      columns: [
        { label: "Sources", items: [
          { name: "l2_clean_panel_v1", kind: "source", summary: "Default predictor panel." },
          { name: "l1_data_definition_v1 raw", kind: "source", summary: "Raw feature access for level or transformation attribution pipelines." },
          { name: "l1_regime_metadata_v1", kind: "gated source", summary: "Available when regime support is active." },
        ] },
        { label: "Transform nodes", items: [
          { name: "lag / seasonal_lag", kind: "step", summary: "Build target or predictor lag blocks." },
          { name: "pca / scaled_pca / dfm", kind: "step", summary: "Factor and dimensionality-reduction blocks." },
          { name: "ma_increasing_order", kind: "step", summary: "MARX increasing-order moving-average block." },
        ] },
        { label: "Cascade", items: [
          { name: "parallel branches", kind: "DAG pattern", summary: "Multiple feature pipelines can feed one final feature sink." },
          { name: "pipeline output source", kind: "beta", summary: "One pipeline can feed another within maximum cascade depth." },
        ] },
      ],
    },
    "L3.C Pipeline combine": {
      summary: "Merge parallel feature blocks into the final X matrix while preserving lineage.",
      columns: [
        { label: "Combine nodes", items: [
          { name: "concat", kind: "combine", summary: "Append feature blocks column-wise." },
          { name: "weighted_concat / simple_average", kind: "combine", summary: "Structured feature block combination." },
          { name: "hierarchical_pca", kind: "combine", summary: "Reduce grouped blocks after combining." },
        ] },
        { label: "Metadata", items: [
          { name: "column lineage", kind: "metadata", summary: "Tracks source node, pipeline, step, and transform history." },
        ] },
      ],
    },
    "L3.D Feature selection": {
      summary: "Optional post-combine feature subset selection before emitting L3 sinks.",
      columns: [
        { label: "Selection", items: [
          { name: "feature_selection", kind: "step", summary: "Variance, correlation, lasso, or user-list selection policies." },
          { name: "future selectors", kind: "future", summary: "Boruta, RFE, lasso path, stability, and genetic search are future/schema-only." },
        ] },
        { label: "Sinks", items: [
          { name: "l3_features_v1", kind: "sink", summary: "Final X/y feature matrices." },
          { name: "l3_metadata_v1", kind: "sink", summary: "Lineage and pipeline metadata for L5/L7." },
        ] },
      ],
    },
  },
  l4: {
    "L4.A Model selection": {
      summary: "Choose model family nodes, benchmark flags, and optional forecast-combination nodes.",
      columns: [
        { label: "Sources", items: [
          { name: "l3_features_v1", kind: "source", summary: "Training and forecast design matrices." },
          { name: "l3_metadata_v1", kind: "source", summary: "Feature lineage for model artifacts." },
        ] },
        { label: "Model nodes", items: [
          { name: "fit_model", kind: "step", summary: "Family-specific model fitting." },
          { name: "is_benchmark", kind: "flag", summary: "Benchmark detection uses the L4 artifact flag, not an L5 axis." },
          { name: "weighted_average_forecast / median_forecast", kind: "combine", summary: "Optional ensemble combination inside L4.A." },
        ] },
        { label: "Rules", items: [
          { name: "model_id lineage", kind: "metadata", summary: "Every forecast and artifact keeps model_id for L5/L6/L7." },
        ] },
      ],
    },
    "L4.B Forecast strategy": {
      summary: "Define direct, iterated, path-average, or forecast-object behavior.",
      columns: [
        { label: "Forecast nodes", items: [
          { name: "predict_direct", kind: "step", summary: "Direct h-step forecasts." },
          { name: "predict_iterated", kind: "step", summary: "Recursive or iterated forecast path." },
          { name: "predict_path_average", kind: "step", summary: "Path-average forecast strategy." },
        ] },
        { label: "Objects", items: [
          { name: "point", kind: "forecast_object", summary: "Point forecast table." },
          { name: "quantile / density", kind: "forecast_object", summary: "Enables L5 density metrics and L6 density tests." },
        ] },
      ],
    },
    "L4.C Training window": {
      summary: "Set expanding, rolling, fixed, or OOS-origin training window behavior.",
      columns: [
        { label: "Window policies", items: [
          { name: "expanding_window", kind: "strategy", summary: "Train on all data available at each origin." },
          { name: "rolling_window", kind: "strategy", summary: "Train on a fixed-size recent history." },
          { name: "fixed_window", kind: "strategy", summary: "Use one fixed training sample." },
        ] },
        { label: "Metadata", items: [
          { name: "l4_training_metadata_v1", kind: "sink", summary: "Records window plans, durations, warnings, and seeds." },
        ] },
      ],
    },
    "L4.D Tuning": {
      summary: "Configure optional hyperparameter search for model-family nodes.",
      columns: [
        { label: "Search", items: [
          { name: "search_algorithm", kind: "axis", summary: "None, grid, random, Bayesian, or family-specific search." },
          { name: "tuning_history", kind: "metadata", summary: "Search traces when tuning is active." },
        ] },
        { label: "Sinks", items: [
          { name: "l4_forecasts_v1", kind: "sink", summary: "Forecast outputs." },
          { name: "l4_model_artifacts_v1", kind: "sink", summary: "Fitted artifacts and benchmark flags." },
          { name: "l4_training_metadata_v1", kind: "sink", summary: "Training and tuning metadata." },
        ] },
      ],
    },
  },
  l7: {
    "L7.A Importance DAG": {
      summary: "Interpretation is a source -> importance step -> aggregation -> sink DAG.",
      columns: [
        { label: "Sources", items: [
          { name: "L4 model artifacts", kind: "source", summary: "Fitted models selected by id, ranking, or MCS inclusion." },
          { name: "L3 features", kind: "source", summary: "X/y matrices and feature metadata." },
        ] },
        { label: "Steps", items: [
          { name: "shap_tree / permutation", kind: "step", summary: "Feature importance methods." },
          { name: "group_aggregate", kind: "step", summary: "FRED-MD/QD/state/user block aggregation." },
          { name: "transformation_attribution", kind: "step", summary: "Cross-cell transformation decomposition." },
        ] },
        { label: "Sinks", items: [
          { name: "l7_importance_v1", kind: "sink", summary: "Main importance results." },
          { name: "l7_transformation_attribution_v1", kind: "sink", summary: "Coulombe-style transformation attribution." },
        ] },
      ],
    },
  },
};

const MULTI_SELECT_AXES = new Set([
  "point_metrics",
  "density_metrics",
  "direction_metrics",
  "relative_metrics",
  "summary_metrics",
  "distribution_metric",
  "coverage_levels",
  "residual_test",
  "saved_objects",
  "provenance_fields",
]);

const DEFAULT_MULTI_SELECTIONS = {
  point_metrics: ["mse", "mae"],
  density_metrics: ["log_score", "crps"],
  direction_metrics: [],
  relative_metrics: ["relative_mse", "r2_oos"],
  summary_metrics: ["mean", "sd", "min", "max", "n_missing"],
  distribution_metric: ["mean_change", "sd_change", "ks_statistic"],
  coverage_levels: ["0.5", "0.9", "0.95"],
  residual_test: ["ljung_box_q", "arch_lm", "jarque_bera_normality"],
  saved_objects: ["forecasts", "metrics", "ranking"],
  provenance_fields: [
    "recipe_yaml_full",
    "recipe_hash",
    "package_version",
    "python_version",
    "dependency_lockfile",
    "data_revision_tag",
    "random_seed_used",
    "runtime_environment",
    "runtime_duration",
    "cell_resolved_axes",
  ],
};

const DEFAULT_SINGLE_SELECTIONS = {
  enabled: "false",
  primary_metric: "mse",
  benchmark_window: "full_oos",
  benchmark_scope: "all_targets_horizons",
  agg_time: "mean",
  agg_horizon: "per_horizon_separate",
  agg_target: "per_target_separate",
  agg_state: "pool_states",
  oos_period: "full_oos",
  regime_use: "pooled",
  regime_metrics: "same_as_primary",
  decomposition_target: "none",
  decomposition_order: "marginal",
  ranking: "by_primary_metric",
  report_style: "single_table",
  test_scope: "per_target_horizon",
  dependence_correction: "newey_west",
  overlap_handling: "nw_with_h_minus_1_lag",
  equal_predictive_test: "dm_diebold_mariano",
  loss_function: "squared",
  model_pair_strategy: "vs_benchmark_only",
  hln_correction: "true",
  nested_test: "clark_west",
  nested_pair_strategy: "vs_benchmark_auto",
  cw_adjustment: "true",
  enc_test_one_sided: "one_sided",
  cpa_test: "giacomini_rossi_2010",
  cpa_window_type: "rolling_window",
  cpa_conditioning_info: "none",
  cpa_critical_value_method: "simulated",
  multiple_model_test: "mcs_hansen",
  mcs_alpha: "0.10",
  mmt_loss_function: "squared",
  bootstrap_method: "stationary_bootstrap",
  bootstrap_n_replications: "1000",
  bootstrap_block_length: "auto",
  mcs_t_statistic: "t_max",
  spa_studentization: "consistent",
  stepm_alpha: "0.10",
  density_test: "pit_berkowitz",
  interval_test: "christoffersen_conditional_coverage",
  pit_n_bins: "10",
  pit_test_horizon_dependence: "nw_correction",
  direction_test: "pesaran_timmermann_1992",
  direction_threshold: "zero",
  direction_alpha: "0.05",
  residual_lag_count: "derived",
  residual_test_scope: "per_model_target_horizon",
  residual_alpha: "0.05",
  output_table_format: "long",
  figure_type: "auto",
  top_k_features_to_show: "20",
  precision_digits: "4",
  figure_dpi: "300",
  figure_format: "pdf",
  latex_table_export: "true",
  markdown_table_export: "false",
  export_format: "json_csv",
  compression: "none",
  model_artifacts_format: "pickle",
  manifest_format: "json",
  artifact_granularity: "per_cell",
  naming_convention: "descriptive",
  coverage_view: "multi",
  summary_split: "full_sample",
  stationarity_test: "none",
  stationarity_test_scope: "target_and_predictors",
  missing_view: "multi",
  outlier_view: "iqr_flag",
  correlation_method: "pearson",
  correlation_view: "none",
  comparison_pair: "raw_vs_final_clean",
  comparison_output_form: "multi",
  distribution_view: "multi",
  correlation_shift: "none",
  cleaning_summary_view: "multi",
  t_code_application_log: "summary",
  diagnostic_format: "pdf",
  attach_to_manifest: "true",
  latex_export: "true",
  comparison_stages: "cleaned_vs_features",
  factor_view: "multi",
  dfm_diagnostics: "multi",
  feature_correlation: "cross_block",
  lag_view: "multi",
  marx_view: "weight_decay_visualization",
  selection_view: "multi",
  stability_metric: "jaccard",
  fit_view: "multi",
  fit_per_origin: "last_origin_only",
  forecast_scale_view: "both_overlay",
  back_transform_method: "auto",
  window_view: "multi",
  coef_view_models: "all_linear_models",
  tuning_view: "multi",
  ensemble_view: "multi",
  weights_over_time_method: "stacked_area",
};

const CANONICAL_AXIS_OPTIONS = {
  primary_metric: ["mse", "rmse", "mae", "relative_mse", "r2_oos", "log_score", "crps"],
  point_metrics: ["mse", "rmse", "mae", "mape", "medae", "theil_u1", "theil_u2"],
  density_metrics: ["log_score", "crps", "interval_score", "coverage_rate"],
  direction_metrics: ["success_ratio", "pesaran_timmermann_metric"],
  relative_metrics: ["relative_mse", "r2_oos", "relative_mae", "mse_reduction"],
  benchmark_window: ["full_oos", "rolling", "expanding"],
  benchmark_scope: ["all_targets_horizons", "per_target", "per_horizon", "per_target_horizon"],
  agg_time: ["mean", "median", "weighted_recent", "per_subperiod"],
  agg_horizon: ["per_horizon_separate", "mean", "per_horizon_then_mean"],
  agg_target: ["per_target_separate", "mean", "weighted"],
  agg_state: ["pool_states", "per_state_separate", "weighted_average", "top_k_worst"],
  oos_period: ["full_oos", "fixed_dates", "rolling_window", "multiple_subperiods"],
  regime_use: ["pooled", "per_regime", "both"],
  regime_metrics: ["same_as_primary", "all_metrics", "custom_list"],
  decomposition_target: ["none", "by_target", "by_horizon", "by_predictor_block", "by_oos_period", "by_state", "by_regime"],
  decomposition_order: ["marginal", "sequential", "shapley", "interaction_first_order"],
  ranking: ["by_primary_metric", "by_relative_metric", "by_average_rank", "borda_count", "mcs_inclusion"],
  report_style: ["single_table", "per_target_horizon_panel", "heatmap", "forest_plot", "latex_table", "markdown_table"],
  enabled: ["true", "false"],
  test_scope: ["per_target_horizon", "per_target", "per_horizon", "pooled"],
  dependence_correction: ["newey_west", "andrews", "parzen_kernel", "none"],
  overlap_handling: ["nw_with_h_minus_1_lag", "west_1996_adjustment", "none"],
  equal_predictive_test: ["dm_diebold_mariano", "gw_giacomini_white", "multi"],
  loss_function: ["squared", "absolute", "lin_lin_asymmetric", "custom"],
  model_pair_strategy: ["vs_benchmark_only", "all_pairs", "user_list"],
  hln_correction: ["true", "false"],
  nested_test: ["clark_west", "enc_new", "enc_t", "multi"],
  nested_pair_strategy: ["vs_benchmark_auto", "auto_detect", "user_list"],
  cw_adjustment: ["true", "false"],
  enc_test_one_sided: ["one_sided", "two_sided"],
  cpa_test: ["giacomini_rossi_2010", "rossi_sekhposyan", "multi"],
  cpa_window_type: ["rolling_window", "recursive"],
  cpa_conditioning_info: ["none", "lagged_loss_difference", "regime", "external_indicator"],
  cpa_critical_value_method: ["simulated", "bootstrap", "asymptotic"],
  multiple_model_test: ["mcs_hansen", "spa_hansen", "reality_check_white", "step_m_romano_wolf", "multi"],
  mcs_alpha: ["0.10", "0.05", "0.25"],
  mmt_loss_function: ["squared", "absolute"],
  bootstrap_method: ["stationary_bootstrap", "block_bootstrap", "circular_bootstrap"],
  bootstrap_n_replications: ["1000", "5000", "custom"],
  bootstrap_block_length: ["auto", "custom_int"],
  mcs_t_statistic: ["t_max", "t_range"],
  spa_studentization: ["consistent", "lower", "upper"],
  stepm_alpha: ["0.10", "0.05", "0.25"],
  density_test: ["pit_kolmogorov_smirnov", "pit_berkowitz", "pit_anderson_darling", "pit_ljung_box", "multi"],
  interval_test: ["kupiec_unconditional_coverage", "christoffersen_independence", "christoffersen_conditional_coverage", "dynamic_quantile_test", "multi"],
  coverage_levels: ["0.5", "0.9", "0.95", "custom_list"],
  pit_n_bins: ["10", "20", "custom_int"],
  pit_test_horizon_dependence: ["nw_correction", "none"],
  direction_test: ["pesaran_timmermann_1992", "henriksson_merton", "multi"],
  direction_threshold: ["zero", "median", "user_defined"],
  direction_alpha: ["0.05", "0.10", "0.01"],
  residual_test: ["ljung_box_q", "arch_lm", "jarque_bera_normality", "breusch_godfrey_serial_correlation", "durbin_watson", "multi"],
  residual_lag_count: ["derived", "10", "4", "custom_int"],
  residual_test_scope: ["per_model", "per_model_target_horizon"],
  residual_alpha: ["0.05", "0.10", "0.01"],
  output_table_format: ["wide", "long", "multi"],
  figure_type: ["auto", "bar_global", "beeswarm", "heatmap", "multi"],
  top_k_features_to_show: ["20", "10", "50", "custom_int"],
  precision_digits: ["4", "3", "6"],
  figure_dpi: ["300", "150", "600"],
  figure_format: ["png", "pdf", "svg", "multi"],
  latex_table_export: ["true", "false"],
  markdown_table_export: ["true", "false"],
  export_format: ["json", "csv", "parquet", "json_csv", "json_parquet", "latex_tables", "markdown_report", "html_report", "all"],
  compression: ["none", "gzip", "zip"],
  saved_objects: ["forecasts", "forecast_intervals", "metrics", "ranking", "decomposition", "regime_metrics", "state_metrics", "model_artifacts", "combination_weights", "feature_metadata", "clean_panel", "raw_panel", "diagnostics_l1_5", "diagnostics_l2_5", "diagnostics_l3_5", "diagnostics_l4_5", "diagnostics_all", "tests", "importance", "transformation_attribution"],
  model_artifacts_format: ["pickle", "joblib", "onnx", "pmml"],
  provenance_fields: ["recipe_yaml_full", "recipe_hash", "package_version", "python_version", "r_version", "julia_version", "dependency_lockfile", "git_commit_sha", "git_branch_name", "data_revision_tag", "random_seed_used", "runtime_environment", "runtime_duration", "cell_resolved_axes"],
  manifest_format: ["json", "yaml", "json_lines"],
  artifact_granularity: ["per_cell", "per_target", "per_horizon", "per_target_horizon", "flat"],
  naming_convention: ["cell_id", "descriptive", "recipe_hash", "custom"],
  coverage_view: ["per_series_start_end", "panel_balance_matrix", "observation_count", "multi"],
  summary_metrics: ["mean", "sd", "min", "max", "skew", "kurtosis", "n_obs", "n_missing"],
  summary_split: ["full_sample", "pre_oos_only", "per_decade", "per_regime"],
  stationarity_test: ["none", "adf", "pp", "kpss", "multi"],
  stationarity_test_scope: ["target_only", "predictors_only", "target_and_predictors"],
  missing_view: ["heatmap", "per_series_count", "longest_gap", "multi"],
  outlier_view: ["none", "zscore_flag", "iqr_flag", "multi"],
  correlation_method: ["pearson", "spearman", "kendall"],
  correlation_view: ["none", "full_matrix", "clustered_heatmap", "top_k_per_target", "top_k"],
  comparison_pair: ["raw_vs_final_clean", "raw_vs_tcoded", "raw_vs_outlier_handled", "raw_vs_imputed", "multi_stage"],
  comparison_output_form: ["side_by_side_summary", "overlay_timeseries", "difference_table", "distribution_shift", "multi"],
  distribution_metric: ["mean_change", "sd_change", "skew_change", "kurtosis_change", "ks_statistic"],
  distribution_view: ["summary_table", "qq_plot", "histogram_overlay", "multi"],
  correlation_shift: ["none", "delta_matrix", "pre_post_overlay"],
  cleaning_summary_view: ["n_imputed_per_series", "n_outliers_flagged", "n_truncated_obs", "multi"],
  t_code_application_log: ["none", "summary", "per_series_detail"],
  diagnostic_format: ["png", "pdf", "html", "json", "latex_table", "csv", "multi"],
  attach_to_manifest: ["true", "false"],
  latex_export: ["true", "false"],
  comparison_stages: ["cleaned_vs_features", "raw_vs_cleaned_vs_features", "features_only"],
  factor_view: ["scree_plot", "cumulative_variance", "loadings_heatmap", "factor_timeseries", "multi"],
  dfm_diagnostics: ["none", "idiosyncratic_acf", "factor_var_stability", "multi"],
  feature_correlation: ["none", "within_block", "cross_block", "with_target", "multi"],
  lag_view: ["autocorrelation_per_lag", "partial_autocorrelation", "lag_correlation_decay", "multi"],
  marx_view: ["none", "weight_decay_visualization"],
  selection_view: ["selected_list", "selection_count_per_origin", "selection_stability", "multi"],
  stability_metric: ["jaccard", "kuncheva"],
  fit_view: ["fitted_vs_actual", "residual_time", "residual_acf", "residual_qq", "multi"],
  fit_per_origin: ["last_origin_only", "every_n_origins", "all_origins"],
  forecast_scale_view: ["transformed_only", "back_transformed_only", "both_overlay"],
  back_transform_method: ["auto", "manual_function"],
  window_view: ["rolling_train_loss", "rolling_coef", "first_vs_last_window_forecast", "parameter_stability", "multi"],
  coef_view_models: ["all_linear_models", "user_list"],
  tuning_view: ["objective_trace", "hyperparameter_path", "cv_score_distribution", "multi"],
  ensemble_view: ["weights_over_time", "weight_concentration", "member_contribution", "multi"],
  weights_over_time_method: ["line_plot", "stacked_area", "heatmap"],
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function currentSample() {
  return state.data.samples[state.sampleIndex];
}

function currentSource() {
  return state.loadedSource || currentSample();
}

function currentView() {
  return currentSource().view || currentSample().view;
}

function resetEngineState() {
  state.engineState = NavigatorStateEngine.createStateFromRecipe(state.data, currentSource().recipe || {});
}

function loadSource(source) {
  state.loadedSource = source;
  state.activeAxis = null;
  resetEngineState();
}

function allAxes() {
  return NavigatorStateEngine.visibleTree(state.data, state.engineState);
}

function axisMatchesLayer(axis) {
  return axis.layer === state.layerFilter;
}

function layerLabel(layer) {
  const labels = {
    "0_meta": "L0 Study Scope",
    "1_data_task": "L1 Data + y/x",
    "2_preprocessing": "L2 Representation",
    "3_training": "L3 Generator",
    "4_evaluation": "L4 Evaluation",
    "5_output_provenance": "L5 Outputs",
    "6_stat_tests": "L6 Tests",
    "7_importance": "L7 Interpretation",
  };
  return labels[layer] || (state.data && state.data.layer_labels && state.data.layer_labels[layer]) || layer || "Layer";
}

function humanizeToken(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function axisPresentation(axisName) {
  return ((state.data && state.data.axis_presentation) || {})[axisName] || {};
}

function valuePresentation(axisName, value) {
  const presentation = axisPresentation(axisName);
  return ((presentation.values || {})[value]) || {};
}

function axisDisplayName(axis) {
  const presentation = axisPresentation(axis.axis);
  return presentation.label || humanizeToken(axis.axis);
}

function axisQuestion(axis) {
  const presentation = axisPresentation(axis.axis);
  return presentation.question || `Choose ${humanizeToken(axis.axis).toLowerCase()}.`;
}

function axisSummary(axis) {
  const presentation = axisPresentation(axis.axis);
  return presentation.summary || axis.layer_label || layerLabel(axis.layer);
}

function valueDisplayName(axisName, value) {
  const presentation = valuePresentation(axisName, value);
  return presentation.label || humanizeToken(value);
}

function valueSummary(axisName, value) {
  return valuePresentation(axisName, value).summary || "";
}

function defaultValue(axisName) {
  return axisPresentation(axisName).default_value || null;
}

function isDefaultSelection(axisName, value) {
  const fallback = defaultValue(axisName);
  return fallback !== null && String(fallback) === String(value);
}

function runtimeSupportSpec() {
  return (state.data && state.data.runtime_support) || {};
}

function runtimeStatusKey(option) {
  const status = option && option.status ? option.status : "registry_only";
  return (runtimeSupportSpec().status_map || {})[status] || "schema_only";
}

function runtimeStatusMeta(key) {
  const legend = runtimeSupportSpec().legend || {};
  return legend[key] || { label: humanizeToken(key), summary: "" };
}

function runtimeBadge(option) {
  const key = runtimeStatusKey(option);
  const meta = runtimeStatusMeta(key);
  return `<span class="runtime-badge runtime-${escapeHtml(key)}" title="${escapeHtml(meta.summary || meta.label)}">${escapeHtml(meta.label)}</span>`;
}

function renderRuntimeSupport() {
  if (!els.runtimeSupport) return;
  const spec = runtimeSupportSpec();
  const layerNote = ((spec.layer_notes || {})[state.layerFilter]) || {};
  const legend = spec.legend || {};
  const layerAxes = allAxes().filter((axis) => axis.layer === state.layerFilter);
  const selectedCounts = layerAxes.reduce((acc, axis) => {
    const selected = axisSelectedOption(axis);
    const key = runtimeStatusKey(selected);
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const chips = Object.entries(legend).map(([key, meta]) => `
    <span class="support-chip runtime-${escapeHtml(key)}">
      <strong>${escapeHtml(String(selectedCounts[key] || 0))}</strong>
      ${escapeHtml(meta.label)}
    </span>
  `).join("");
  els.runtimeSupport.innerHTML = `
    <div>
      <p class="eyebrow">Runtime support</p>
      <h2>${escapeHtml(layerNote.label || "Runtime status")}</h2>
      <p class="source-note">${escapeHtml(layerNote.summary || "Selected options are classified by current runtime support, separate from schema validity.")}</p>
    </div>
    <div class="support-grid">${chips}</div>
  `;
}

function axisSelectedOption(axis) {
  return axis.options.find((option) => option.value === axis.selected);
}

function axisRequiresSelection(axis) {
  const option = axisSelectedOption(axis);
  return Boolean(option && option.disabled_reason && isDefaultSelection(axis.axis, axis.selected));
}

function selectedDisplayLabel(axisName, value, axis) {
  if (axis && axisRequiresSelection(axis)) return "Selection required";
  const label = valueDisplayName(axisName, value);
  return isDefaultSelection(axisName, value) ? `${label} [default]` : label;
}

function docsLink(axisName) {
  return axisPresentation(axisName).docs_url || "";
}

function layerDescription(layer) {
  const descriptions = {
    "0_meta": "Study runtime policy: failure handling, reproducibility, and compute layout. study_scope is derived from recipe shape.",
    "1_data_task": "Data contract: source, target, predictor universe, geography, sample window, horizons, and regimes.",
    "2_preprocessing": "Cleaned panel construction: frequency alignment, transforms, outliers, imputation, and frame edge handling. Scaling belongs in L3.",
    "3_training": "Feature engineering DAG: target construction, feature pipelines, pipeline combine, and optional feature selection.",
    "4_evaluation": "Forecast DAG: model selection, forecast strategy, training window, and tuning.",
    "5_output_provenance": "Evaluation choices: metrics, benchmark comparison, aggregation, slicing, decomposition, ranking, and reports.",
    "6_stat_tests": "Statistical tests over forecasts, losses, density or interval outputs, direction, and residual diagnostics.",
    "7_importance": "Interpretation and importance outputs: method family, scope, aggregation, temporal shape, and detailed reports.",
  };
  return descriptions[layer] || "";
}

function layerNumber(layer) {
  const match = String(layer || "").match(/^(\d+)/);
  return match ? match[1] : "";
}

function layerTopologySpec() {
  return (state.data && state.data.layer_topology) || { nodes: [], edges: [], main_flow: [] };
}

function topologyNodeById() {
  return new Map((layerTopologySpec().nodes || []).map((node) => [node.id, node]));
}

function displayLayerId(id) {
  return String(id || "").toUpperCase().replace("_", ".");
}

function topologyNodeCard(node) {
  const modeLabel = node.ui_mode === "graph" ? "DAG" : "List";
  const active = node.id === state.activeTopologyLayer ? " active" : "";
  return `
    <article class="topology-node${active} topology-${escapeHtml(node.group || node.category)} topology-mode-${escapeHtml(node.ui_mode)}" data-topology-layer="${escapeHtml(node.id)}" role="button" tabindex="0">
      <span class="topology-layer-id">${escapeHtml(displayLayerId(node.id))}</span>
      <span class="topology-node-main">
        <strong>${escapeHtml(node.label || node.name)}</strong>
        <em>${escapeHtml(humanizeToken(node.category))} · ${escapeHtml(modeLabel)}</em>
      </span>
      <span class="topology-count">${escapeHtml(String(node.axis_count || 0))}</span>
    </article>
  `;
}

function formatList(items, emptyText) {
  const values = items || [];
  if (!values.length) return `<p class="empty-note">${escapeHtml(emptyText)}</p>`;
  return `<div class="detail-chip-grid">${values.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function subLayersForNode(node) {
  const explicit = node.sub_layers || [];
  if (explicit.length) return explicit;
  return CANONICAL_SUB_LAYERS[node.id] || [];
}

function treeAxesForNode(node) {
  const treeLayer = TREE_LAYER_ALIASES[node.id];
  if (!treeLayer) return [];
  return allAxes()
    .filter((axis) => axis.layer === treeLayer)
    .map((axis) => axis.axis);
}

function effectiveAxesForNode(node) {
  const declared = node.axes || [];
  if (declared.length) return declared;
  return treeAxesForNode(node);
}

function normalizeSubLayerName(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function axesForSubLayer(node, subLayer) {
  if ((GRAPH_LAYER_SECTIONS[node.id] || {})[subLayer]) return [];
  if (node.sub_layer_axes && node.sub_layer_axes[subLayer]) return node.sub_layer_axes[subLayer];
  const grouped = CANONICAL_AXIS_GROUPS[node.id] || {};
  const axes = effectiveAxesForNode(node);
  if (grouped[subLayer]) return grouped[subLayer].filter((axis) => axes.includes(axis));
  const normalized = normalizeSubLayerName(subLayer);
  const letterMatch = normalized.match(/\bl\d+(?:\s+5)?\s+([a-z])\b/);
  const letter = letterMatch ? letterMatch[1] : "";
  const keywordMatches = axes.filter((axis) => {
    const a = normalizeSubLayerName(axis);
    return normalized.split(" ").some((token) => token.length > 3 && a.includes(token));
  });
  if (keywordMatches.length) return keywordMatches;
  if (!letter) return axes;
  const subLayers = subLayersForNode(node);
  const index = Math.max(0, subLayers.indexOf(subLayer));
  const chunkSize = Math.max(1, Math.ceil(axes.length / Math.max(1, subLayers.length)));
  return axes.slice(index * chunkSize, (index + 1) * chunkSize);
}

function graphSectionForSubLayer(node, subLayer) {
  return (GRAPH_LAYER_SECTIONS[node.id] || {})[subLayer] || null;
}

function graphItemCount(node, subLayer) {
  const section = graphSectionForSubLayer(node, subLayer);
  if (!section) return 0;
  return (section.columns || []).reduce((total, column) => total + ((column.items || []).length), 0);
}

function activeNode() {
  const byId = topologyNodeById();
  return byId.get(state.activeTopologyLayer) || byId.get("l0") || (layerTopologySpec().nodes || [])[0];
}

function ensureActiveSubLayer(node) {
  const subLayers = subLayersForNode(node);
  if (!subLayers.length) {
    state.activeSubLayer = null;
    return null;
  }
  if (!state.activeSubLayer || !subLayers.includes(state.activeSubLayer)) {
    state.activeSubLayer = subLayers[0];
  }
  return state.activeSubLayer;
}

function renderSubLayerButton(node, subLayer, idx) {
  const axes = axesForSubLayer(node, subLayer);
  const graphCount = graphItemCount(node, subLayer);
  const active = subLayer === state.activeSubLayer ? " active" : "";
  const countLabel = graphCount ? `${graphCount} DAG items` : `${axes.length} axes`;
  return `
    <button type="button" class="sublayer-card${active}" data-sub-layer="${escapeHtml(subLayer)}">
      <span>${escapeHtml(String(idx + 1).padStart(2, "0"))}</span>
      <strong>${escapeHtml(subLayer)}</strong>
      <em>${escapeHtml(countLabel)}</em>
    </button>
  `;
}

function renderAxisButton(axisName) {
  const active = axisName === state.activeCanonicalAxis ? " active" : "";
  const options = optionRecordsForAxis(axisName);
  return `
    <button type="button" class="canonical-axis${active}" data-canonical-axis="${escapeHtml(axisName)}">
      <span>${escapeHtml(axisName)}</span>
      <em>${escapeHtml(String(options.length))} options</em>
    </button>
  `;
}

function optionRecordsForAxis(axisName) {
  const topologyOptions = (layerTopologySpec().nodes || [])
    .map((node) => (node.axis_options || {})[axisName])
    .find((records) => Array.isArray(records) && records.length);
  if (topologyOptions) return topologyOptions;
  const hardcoded = CANONICAL_AXIS_OPTIONS[axisName];
  if (hardcoded) {
    return hardcoded.map((value) => ({
      value,
      status: value === "onnx" || value === "pmml" ? "future" : "operational",
      enabled: value !== "onnx" && value !== "pmml",
      disabled_reason: value === "onnx" || value === "pmml" ? "future option" : null,
    }));
  }
  const treeAxis = allAxes().find((axis) => axis.axis === axisName);
  if (treeAxis && treeAxis.options) return treeAxis.options;
  const catalog = state.data && state.data.axis_catalog && state.data.axis_catalog[axisName];
  if (catalog && catalog.allowed_values) {
    return catalog.allowed_values.map((value) => ({
      value,
      status: (catalog.current_status || {})[value] || "operational",
      enabled: ((catalog.current_status || {})[value] || "operational") !== "future",
      disabled_reason: ((catalog.current_status || {})[value] || "operational") === "future" ? "future option" : null,
    }));
  }
  return [];
}

function isMultiSelectAxis(axisName) {
  return MULTI_SELECT_AXES.has(axisName);
}

function selectedCanonicalValues(axisName, records) {
  const stored = state.canonicalSelections[axisName];
  if (Array.isArray(stored)) return stored;
  if (stored) return [stored];
  if (!records.length) return "";
  if (isMultiSelectAxis(axisName)) {
    const defaults = DEFAULT_MULTI_SELECTIONS[axisName];
    if (defaults) return defaults.filter((value) => records.some((record) => String(record.value) === String(value)));
    return records.filter((record) => record.enabled !== false).slice(0, 1).map((record) => record.value);
  }
  const explicitDefault = DEFAULT_SINGLE_SELECTIONS[axisName];
  if (explicitDefault && records.some((record) => String(record.value) === String(explicitDefault) && record.enabled !== false)) {
    return [explicitDefault];
  }
  const recordDefault = records.find((record) => record.default && record.enabled !== false);
  if (recordDefault) return [recordDefault.value];
  const presentedDefault = defaultValue(axisName);
  if (presentedDefault && records.some((record) => String(record.value) === String(presentedDefault) && record.enabled !== false)) {
    return [presentedDefault];
  }
  const enabled = records.find((record) => record.enabled !== false);
  return [(enabled || records[0]).value];
}

function selectionSummary(axisName, records) {
  const values = selectedCanonicalValues(axisName, records);
  if (!Array.isArray(values)) return "";
  if (!values.length) return "none";
  return values.join(", ");
}

function renderCanonicalOption(record, axisName, selectedValues) {
  const selected = selectedValues.some((value) => String(record.value) === String(value)) ? " selected" : "";
  const disabled = record.enabled === false ? " disabled" : "";
  const disabledAttr = record.enabled === false ? " disabled" : "";
  return `
    <button type="button" class="axis-choice${selected}${disabled}" data-canonical-option="${escapeHtml(record.value)}"${disabledAttr}>
      <strong>${escapeHtml(record.value)}</strong>
      <span>${escapeHtml(selected ? (isMultiSelectAxis(axisName) ? "selected - click to toggle" : "selected") : (record.status || "operational"))}</span>
      ${record.disabled_reason ? `<em>${escapeHtml(record.disabled_reason)}</em>` : ""}
    </button>
  `;
}

function renderGraphSubLayerPanel(node, subLayer) {
  const section = graphSectionForSubLayer(node, subLayer);
  if (!section) return "";
  const selected = new Set(selectedDagItems(node.id));
  return `
    <div class="dag-panel">
      <div class="axis-option-head">
        <span>DAG section</span>
        <strong>${escapeHtml(subLayer)}</strong>
      </div>
      <p class="source-note">${escapeHtml(section.summary || "Graph layer section.")}</p>
      <p class="source-note">Click DAG items to include them in the generated YAML template.</p>
      <div class="dag-flow">
        ${(section.columns || []).map((column) => `
          <section class="dag-column">
            <h4>${escapeHtml(column.label)}</h4>
            ${(column.items || []).map((item) => `
              <button type="button" class="dag-node${selected.has(dagItemKey(subLayer, item)) ? " selected" : ""}" data-dag-layer="${escapeHtml(node.id)}" data-dag-sub-layer="${escapeHtml(subLayer)}" data-dag-item="${escapeHtml(item.name)}">
                <span>${escapeHtml(item.kind || "node")}</span>
                <strong>${escapeHtml(item.name)}</strong>
                <p>${escapeHtml(item.summary || "")}</p>
              </button>
            `).join("")}
          </section>
        `).join("")}
      </div>
    </div>
  `;
}

function renderCanonicalOptionsPanel(axisName) {
  if (!axisName) return `<p class="empty-note">Select an axis to see available options.</p>`;
  const records = optionRecordsForAxis(axisName);
  if (!records.length) {
    return `
      <div class="axis-option-panel">
        <div class="axis-option-head">
          <span>Selected axis</span>
          <strong>${escapeHtml(axisName)}</strong>
        </div>
        <p class="empty-note">No fixed option list is registered for this axis. Configure it with leaf_config or DAG node params.</p>
      </div>
    `;
  }
  const selectedValues = selectedCanonicalValues(axisName, records);
  const multi = isMultiSelectAxis(axisName);
  return `
    <div class="axis-option-panel">
      <div class="axis-option-head">
        <span>${multi ? "Multi-select axis" : "Selected axis"}</span>
        <strong>${escapeHtml(axisName)}</strong>
        <em>${multi ? "selected values" : "selected"}: ${escapeHtml(selectionSummary(axisName, records))}</em>
      </div>
      <div class="axis-choice-grid">
        ${records.map((record) => renderCanonicalOption(record, axisName, selectedValues)).join("")}
      </div>
    </div>
  `;
}

function applyCanonicalSelection(axisName, value) {
  if (isMultiSelectAxis(axisName)) {
    const current = selectedCanonicalValues(axisName, optionRecordsForAxis(axisName));
    const next = current.some((item) => String(item) === String(value))
      ? current.filter((item) => String(item) !== String(value))
      : [...current, value];
    state.canonicalSelections[axisName] = next;
    return;
  }
  state.canonicalSelections[axisName] = value;
  const axis = allAxes().find((item) => item.axis === axisName);
  if (axis) {
    state.engineState = NavigatorStateEngine.selectOption(state.data, state.engineState, axisName, value);
  }
}

function selectedDagItems(layerId) {
  return state.dagSelections[layerId] || [];
}

function dagItemKey(subLayer, item) {
  return `${subLayer}::${item.name}`;
}

function toggleDagItem(layerId, subLayer, itemName) {
  const current = selectedDagItems(layerId);
  const key = `${subLayer}::${itemName}`;
  state.dagSelections[layerId] = current.includes(key)
    ? current.filter((item) => item !== key)
    : [...current, key];
}

function renderLayerDetail() {
  if (!els.layerDetail) return;
  const node = activeNode();
  if (!node) {
    els.layerDetail.innerHTML = `<p class="muted">No registered layer metadata available.</p>`;
    return;
  }
  const subLayers = subLayersForNode(node);
  const activeSubLayer = ensureActiveSubLayer(node);
  const subLayerAxes = activeSubLayer ? axesForSubLayer(node, activeSubLayer) : (node.axes || []);
  const graphSection = activeSubLayer ? graphSectionForSubLayer(node, activeSubLayer) : null;
  const selectableAxes = [...(node.layer_globals || []), ...subLayerAxes];
  if (!state.activeCanonicalAxis || !selectableAxes.includes(state.activeCanonicalAxis)) {
    state.activeCanonicalAxis = subLayerAxes[0] || null;
  }
  const modeText = node.ui_mode === "graph"
    ? "Graph/DAG layer: users compose source, step, and sink nodes."
    : "List layer: users resolve ordered axes and sub-layer sections.";
  const graphSections = GRAPH_LAYER_SECTIONS[node.id] || {};
  const graphItemTotal = Object.keys(graphSections).reduce((total, subLayer) => total + graphItemCount(node, subLayer), 0);
  const layerAxes = effectiveAxesForNode(node);
  const layerControlCount = layerAxes.length || graphItemTotal;
  const inputCount = (node.expected_inputs || []).length;
  const sinkCount = (node.produces || []).length;
  els.layerDetail.innerHTML = `
    <div class="layer-hero">
      <div class="layer-hero-id">${escapeHtml(displayLayerId(node.id))}</div>
      <div class="layer-hero-copy">
        <p class="eyebrow">Canonical layer workbench</p>
        <h2>${escapeHtml(node.label || node.name)}</h2>
        <p>${escapeHtml(modeText)}</p>
      </div>
      <div class="layer-metrics">
        <span><strong>${escapeHtml(String(inputCount))}</strong> inputs</span>
        <span><strong>${escapeHtml(String(sinkCount))}</strong> sinks</span>
        <span><strong>${escapeHtml(String(node.sub_layer_count || 0))}</strong> sub-layers</span>
        <span><strong>${escapeHtml(String(layerControlCount))}</strong> ${node.ui_mode === "graph" ? "DAG items" : "axes"}</span>
      </div>
    </div>

    <div class="definition-grid">
      <section class="definition-wide">
        <h3>Sub-layers</h3>
        ${subLayers.length ? `<div class="sublayer-grid">${subLayers.map((subLayer, idx) => renderSubLayerButton(node, subLayer, idx)).join("")}</div>` : `<p class="empty-note">No explicit sub-layer sections.</p>`}
      </section>
      <section>
        <h3>Layer globals</h3>
        ${(node.layer_globals || []).length ? `<div class="canonical-axis-grid">${(node.layer_globals || []).map(renderAxisButton).join("")}</div>` : `<p class="empty-note">No layer-global axes.</p>`}
      </section>
      <section>
        <h3>Selected sub-layer</h3>
        <p class="selected-sublayer">${escapeHtml(activeSubLayer || "Layer-level controls")}</p>
        <p class="source-note">${escapeHtml(graphSection ? "This sub-layer is configured as a DAG section." : (subLayerAxes.length ? `${subLayerAxes.length} axis/control entries available.` : "This sub-layer is configured by DAG nodes or runtime metadata."))}</p>
      </section>
      <section class="definition-wide">
        <h3>${graphSection ? "DAG nodes / flow" : "Axes / output controls"}</h3>
        ${graphSection ? renderGraphSubLayerPanel(node, activeSubLayer) : (subLayerAxes.length ? `<div class="canonical-axis-grid">${subLayerAxes.map(renderAxisButton).join("")}</div>` : `<p class="empty-note">No fixed axes for this sub-layer.</p>`)}
        ${graphSection ? "" : renderCanonicalOptionsPanel(state.activeCanonicalAxis)}
      </section>
    </div>

    <div class="handoff-grid">
      <section class="handoff-card">
        <h3>Inputs</h3>
        ${formatList(node.expected_inputs, "No upstream sink inputs.")}
      </section>
      <section class="handoff-card">
        <h3>Produces</h3>
        ${formatList(node.produces, "No produced sinks registered.")}
      </section>
    </div>
  `;
}

function renderLayerTopology() {
  if (!els.layerTopology) return;
  const topology = layerTopologySpec();
  const byId = topologyNodeById();
  const mainNodes = (topology.main_flow || []).map((id) => byId.get(id)).filter(Boolean);
  const diagnostics = (topology.nodes || []).filter((node) => node.category === "diagnostic");
  const graphLayers = (topology.nodes || []).filter((node) => node.ui_mode === "graph");
  els.layerTopology.innerHTML = `
    <div class="topology-stats">
      <span><strong>${escapeHtml(String(mainNodes.length))}</strong> main</span>
      <span><strong>${escapeHtml(String(diagnostics.length))}</strong> diagnostics</span>
      <span><strong>${escapeHtml(String(graphLayers.length))}</strong> DAG</span>
    </div>
    <div class="topology-main-flow">
      ${mainNodes.map(topologyNodeCard).join("")}
    </div>
    <div class="diagnostic-heading">
      <p class="eyebrow">Default-off hooks</p>
    </div>
    <div class="topology-diagnostics">
      ${diagnostics.map(topologyNodeCard).join("")}
    </div>
  `;
}

function orderedLayers() {
  return Object.keys((state.data && state.data.tree_axes) || {});
}

function axesByLayer(axes) {
  return axes.reduce((acc, axis) => {
    if (!acc[axis.layer]) acc[axis.layer] = [];
    acc[axis.layer].push(axis);
    return acc;
  }, {});
}

function hierarchyLevelLabel(level) {
  const labels = {
    primary_decision: "Primary",
    primary_policy: "Primary policy",
    derived_or_required: "Derived/required",
    conditional_subgroup: "Conditional",
    conditional_subdecision: "Conditional",
    contract_derived: "Contract-derived",
    secondary_policy: "Policy",
    timing_policy: "Timing policy",
  };
  return labels[level] || humanizeToken(level || "decision");
}

function layerAxisGroups(layer, axes) {
  const configured = (((state.data || {}).layer_axis_groups || {})[layer]) || [];
  if (!configured.length) {
    return [{
      id: `${layer}-steps`,
      label: layerLabel(layer),
      level: "primary_decision",
      summary: "",
      condition: "",
      axisItems: axes,
    }];
  }
  const byAxis = new Map(axes.map((axis) => [axis.axis, axis]));
  const seen = new Set();
  const groups = configured
    .map((group) => {
      const axisItems = (group.axes || [])
        .map((axisName) => byAxis.get(axisName))
        .filter(Boolean);
      axisItems.forEach((axis) => seen.add(axis.axis));
      return { ...group, axisItems };
    })
    .filter((group) => group.axisItems.length);
  const ungrouped = axes.filter((axis) => !seen.has(axis.axis));
  if (ungrouped.length) {
    groups.push({
      id: `${layer}-other`,
      label: "Other Layer Steps",
      level: "secondary_policy",
      summary: "",
      condition: "",
      axisItems: ungrouped,
    });
  }
  return groups;
}

function setLayerFilter(layer) {
  state.layerFilter = layer;
  document.querySelectorAll("[data-layer]").forEach((item) => {
    item.classList.toggle("active", item.dataset.layer === layer);
  });
  state.activeAxis = null;
}

function filteredAxes() {
  const q = state.axisFilter.trim().toLowerCase();
  return allAxes().filter((axis) => {
    const selectedLabel = valueDisplayName(axis.axis, axis.selected);
    const text = [
      axis.axis,
      axis.layer,
      axis.selected ?? "",
      axisDisplayName(axis),
      axisQuestion(axis),
      axisSummary(axis),
      selectedLabel,
    ].join(" ").toLowerCase();
    return axisMatchesLayer(axis) && (!q || text.includes(q));
  });
}

function setDefaultAxis() {
  const axes = filteredAxes();
  if (!axes.length) {
    state.activeAxis = null;
    return;
  }
  if (!state.activeAxis || !axes.some((axis) => axis.axis === state.activeAxis)) {
    state.activeAxis = axes[0].axis;
  }
}

function renderSampleSelect() {
  els.sampleSelect.innerHTML = state.data.samples
    .map((sample, idx) => `<option value="${idx}">${escapeHtml(sample.label)}</option>`)
    .join("");
  els.sampleSelect.value = String(state.sampleIndex);
}

function renderSummary() {
  const view = currentView();
  const preview = view.compile_preview || {};
  const selectedDisabled = NavigatorStateEngine.selectedDisabledReasons(state.data, state.engineState);
  const disabled = view.tree.reduce((count, axis) => count + axis.options.filter((option) => !option.enabled).length, 0);
  const liveDisabled = allAxes().reduce((count, axis) => count + axis.options.filter((option) => !option.enabled).length, 0);
  els.executionStatus.textContent = selectedDisabled.length ? "browser_blocked" : (Object.keys(state.engineState.edits).length ? "browser_preview" : (preview.execution_status || "-"));
  els.blockedCount.textContent = String(selectedDisabled.length);
  els.disabledCount.textContent = String(liveDisabled || disabled);
  els.pathSource.textContent = state.loadedSource
    ? `loaded: ${state.loadedSource.label || state.loadedSource.id}`
    : `sample: ${currentSample().label}`;
}

function renderPathHeader() {
  const fullLabel = (state.data.layer_labels && state.data.layer_labels[state.layerFilter]) || layerLabel(state.layerFilter);
  els.activeLayerTitle.textContent = fullLabel;
  els.activeLayerDescription.textContent = layerDescription(state.layerFilter);
}

function renderAxisList() {
  const axes = filteredAxes();
  const groups = layerAxisGroups(state.layerFilter, axes);
  let step = 0;
  els.axisList.innerHTML = groups
    .map((group) => `
      <section class="axis-group" data-axis-group="${escapeHtml(group.id)}">
        <div class="axis-group-head">
          <span class="axis-group-level">${escapeHtml(hierarchyLevelLabel(group.level))}</span>
          <strong>${escapeHtml(group.label)}</strong>
          ${group.condition ? `<span class="axis-group-condition">${escapeHtml(group.condition)}</span>` : ""}
          ${group.summary ? `<p>${escapeHtml(group.summary)}</p>` : ""}
        </div>
        ${group.axisItems.map((axis) => {
          step += 1;
      const disabled = axis.options.filter((option) => !option.enabled).length;
      const active = axis.axis === state.activeAxis ? " active" : "";
      const edited = axis.edited ? `<span class="axis-edited">edited</span>` : "";
      const selectedLabel = selectedDisplayLabel(axis.axis, axis.selected, axis);
      const summary = axisSummary(axis);
      return `
        <button type="button" class="axis-button${active}" data-axis="${escapeHtml(axis.axis)}">
          <span class="axis-step">${step}</span>
          <span class="axis-button-body">
            <span class="axis-name">${escapeHtml(axisDisplayName(axis))}</span>
            <span class="axis-question">${escapeHtml(axisQuestion(axis))}</span>
            <span class="axis-meta">level: ${escapeHtml(hierarchyLevelLabel(axis.axis_level || axis.group_level))} | selected: ${escapeHtml(selectedLabel || "-")} | disabled options: ${disabled}</span>
            <span class="axis-summary">${escapeHtml(summary)}</span>
            ${edited}
          </span>
        </button>
      `;
        }).join("")}
      </section>
    `)
    .join("");
}

function renderOptions() {
  const axis = allAxes().find((item) => item.axis === state.activeAxis);
  if (!axis) {
    els.axisTitle.textContent = "No axis";
    els.axisLayer.textContent = "No matching axes";
    els.axisSelected.textContent = "selected: -";
    els.axisExplainer.innerHTML = `<p class="muted">Select a layer step to see its decision rule and choices.</p>`;
    els.optionList.innerHTML = "";
    return;
  }
  const layerAxes = allAxes().filter((item) => item.layer === axis.layer);
  const stepIndex = Math.max(0, layerAxes.findIndex((item) => item.axis === axis.axis));
  const groupAxes = layerAxes.filter((item) => item.group_id === axis.group_id);
  const groupStepIndex = Math.max(0, groupAxes.findIndex((item) => item.axis === axis.axis));
  const presentation = axisPresentation(axis.axis);
  const selectedLabel = selectedDisplayLabel(axis.axis, axis.selected, axis) || "-";
  const selectedSummary = valueSummary(axis.axis, axis.selected);
  const docs = docsLink(axis.axis);
  const defaultLabel = defaultValue(axis.axis) ? valueDisplayName(axis.axis, defaultValue(axis.axis)) : "";
  const showDefaultLabel = defaultLabel && !axisRequiresSelection(axis);
  els.axisTitle.textContent = axisDisplayName(axis);
  els.axisLayer.textContent = `${layerLabel(axis.layer)} | ${axis.group_label || "Steps"} | group step ${groupStepIndex + 1} of ${groupAxes.length} | layer step ${stepIndex + 1} of ${layerAxes.length}`;
  els.axisSelected.textContent = `selected: ${selectedLabel}`;
  els.axisExplainer.innerHTML = `
    <p class="decision-question">${escapeHtml(axisQuestion(axis))}</p>
    <p class="decision-summary">${escapeHtml(axisSummary(axis))}</p>
    <div class="decision-meta">
      <span><strong>Current selection:</strong> ${escapeHtml(selectedLabel)}</span>
      ${showDefaultLabel ? `<span><strong>Default:</strong> ${escapeHtml(defaultLabel)}</span>` : ""}
      <span><strong>YAML key:</strong> ${escapeHtml(axis.axis)}</span>
      ${axis.group_label ? `<span><strong>Group:</strong> ${escapeHtml(axis.group_label)}</span>` : ""}
      ${(axis.axis_level || axis.group_level) ? `<span><strong>Level:</strong> ${escapeHtml(hierarchyLevelLabel(axis.axis_level || axis.group_level))}</span>` : ""}
      ${presentation.selection_kind ? `<span><strong>Selection type:</strong> ${escapeHtml(humanizeToken(presentation.selection_kind))}</span>` : ""}
      <span><strong>Runtime:</strong> ${runtimeBadge(axisSelectedOption(axis))}</span>
    </div>
    ${selectedSummary ? `<p class="decision-selected">${escapeHtml(selectedSummary)}</p>` : ""}
    ${presentation.warning ? `<p class="decision-warning"><strong>Check docs:</strong> ${escapeHtml(presentation.warning)}</p>` : ""}
    ${presentation.contract ? `<p class="decision-contract"><strong>Contract:</strong> ${escapeHtml(presentation.contract)}</p>` : ""}
    ${docs ? `<a class="decision-docs" href="${escapeHtml(docs)}">Open detailed docs</a>` : ""}
  `;
  els.optionList.innerHTML = axis.options
    .map((option) => {
      const stateClass = option.enabled ? "enabled" : "disabled";
      const selected = option.value === axis.selected ? " selected" : "";
      const reason = option.disabled_reason ? `<p class="reason">${escapeHtml(option.disabled_reason)}</p>` : "";
      const disabledAttr = option.enabled ? "" : " disabled";
      const summary = valueSummary(axis.axis, option.value);
      const optionLabel = valueDisplayName(axis.axis, option.value);
      const needsSelection = axisRequiresSelection(axis) && option.value === axis.selected;
      const statusLabel = needsSelection ? "required" : (isDefaultSelection(axis.axis, option.value) ? `${option.status} · default` : option.status);
      return `
        <button type="button" class="option-card ${stateClass}${selected}" data-option="${escapeHtml(option.value)}"${disabledAttr}>
          <div class="option-value">
            <span>${escapeHtml(optionLabel)}</span>
            <span class="status-stack"><span class="status">${escapeHtml(statusLabel)}</span>${runtimeBadge(option)}</span>
          </div>
          <div class="option-code">YAML value: ${escapeHtml(option.value)}</div>
          ${summary ? `<p class="option-summary">${escapeHtml(summary)}</p>` : ""}
          ${reason}
          <div class="effect">Path records: ${escapeHtml(axisDisplayName(axis))} -> ${escapeHtml(optionLabel)}</div>
          <div class="effect-code">${escapeHtml(option.canonical_path_effect)}</div>
        </button>
      `;
    })
    .join("");
}

function renderTreePath() {
  const axes = allAxes();
  if (!axes.length) {
    els.treePath.innerHTML = `<p class="muted">No path axes available.</p>`;
    return;
  }
  const groupedAxes = axesByLayer(axes);
  const layers = orderedLayers();
  const currentLayerIndex = Math.max(0, layers.indexOf(state.layerFilter));
  els.treePath.innerHTML = `
    <div class="path-flow" aria-label="Selected path so far">
      ${layers.slice(0, currentLayerIndex + 1).map((layer) => {
        const layerAxes = groupedAxes[layer] || [];
        const activeAxisIndex = Math.max(0, layerAxes.findIndex((axis) => axis.axis === state.activeAxis));
        const visibleAxes = layer === state.layerFilter ? layerAxes.slice(0, activeAxisIndex + 1) : layerAxes;
        const activeLayer = layer === state.layerFilter ? " active" : "";
        const blockedCount = layerAxes.reduce((count, axis) => {
          const selectedOption = axis.options.find((option) => option.value === axis.selected);
          return count + (selectedOption && selectedOption.disabled_reason ? 1 : 0);
        }, 0);
        const editedCount = layerAxes.filter((axis) => axis.edited).length;
        return `
          <section class="path-layer${activeLayer}" data-tree-layer="${escapeHtml(layer)}" tabindex="0" role="button">
            <div class="path-layer-marker">${escapeHtml(layerNumber(layer))}</div>
            <div class="path-layer-card">
              <div class="path-layer-head">
                <div>
                  <p class="eyebrow">${escapeHtml(layerLabel(layer))}</p>
                  <h3>${escapeHtml((state.data.layer_labels && state.data.layer_labels[layer]) || layerLabel(layer))}</h3>
                </div>
                <div class="path-layer-stats">
                  <span>${visibleAxes.length}/${layerAxes.length} steps shown</span>
                  ${blockedCount ? `<span class="danger-chip">${blockedCount} blocked</span>` : ""}
                  ${editedCount ? `<span class="path-edited">${editedCount} edited</span>` : ""}
                </div>
              </div>
              <div class="path-axis-grid">
                ${renderPathAxisGroups(layer, visibleAxes)}
              </div>
            </div>
          </section>
        `;
      }).join("")}
    </div>
  `;
}

function renderPathAxisGroups(layer, visibleAxes) {
  const groups = layerAxisGroups(layer, visibleAxes);
  let step = 0;
  return groups
    .map((group) => `
      <section class="path-axis-group">
        <div class="path-axis-group-head">
          <span>${escapeHtml(hierarchyLevelLabel(group.level))}</span>
          <strong>${escapeHtml(group.label)}</strong>
          ${group.condition ? `<em>${escapeHtml(group.condition)}</em>` : ""}
        </div>
        <div class="path-axis-group-grid">
          ${group.axisItems.map((axis) => {
            step += 1;
            return renderPathAxis(axis, step, axis.axis === state.activeAxis);
          }).join("")}
        </div>
      </section>
    `)
    .join("");
}

function renderPathAxis(axis, axisIdx, current) {
  const selectedOption = axisSelectedOption(axis);
  const disabledReason = selectedOption && selectedOption.disabled_reason;
  const blocked = disabledReason ? " blocked" : "";
  const active = axis.axis === state.activeAxis ? " active" : "";
  const edited = axis.edited ? `<span class="path-edited">edited</span>` : "";
  const docs = docsLink(axis.axis);
  const valueLabel = valueDisplayName(axis.axis, axis.selected);
  const selectedValueLabel = selectedDisplayLabel(axis.axis, axis.selected, axis);
  const selectedSummary = valueSummary(axis.axis, axis.selected);
  const presentation = axisPresentation(axis.axis);
  const contract = presentation.contract ? `<span class="path-contract"><strong>Contract:</strong> ${escapeHtml(presentation.contract)}</span>` : "";
  const status = selectedOption && selectedOption.status ? `<span class="status">${escapeHtml(selectedOption.status)}</span>` : "";
  const detail = current
    ? `
      <span class="path-question">${escapeHtml(axisQuestion(axis))}</span>
      ${status}
      <span class="path-summary">${escapeHtml(selectedSummary || axisSummary(axis))}</span>
      ${contract}
      <span class="path-effect">Current decision step</span>
      ${docs ? `<a class="path-docs" href="${escapeHtml(docs)}">Detailed docs</a>` : ""}
      ${disabledReason ? `<span class="path-reason">${escapeHtml(disabledReason)}</span>` : ""}
    `
    : "";
  return `
    <div class="tree-path-item${blocked}${active}" data-tree-axis="${escapeHtml(axis.axis)}" data-tree-layer="${escapeHtml(axis.layer)}" tabindex="0" role="button">
      <span class="path-step">${axisIdx}</span>
      <span class="path-body">
        <span class="path-axis">${escapeHtml(axisDisplayName(axis))}</span>
        <span class="path-value">${escapeHtml(selectedValueLabel || valueLabel || "-")}</span>
        <span class="path-raw">YAML: ${escapeHtml(axis.axis)} = ${escapeHtml(axis.selected ?? "-")}</span>
        ${edited}
        ${detail}
      </span>
    </div>
  `;
}

const LAYER_YAML_KEYS = {
  l0: "0_meta",
  l1: "1_data",
  l2: "2_preprocessing",
  l1_5: "1_5_data_summary",
  l2_5: "2_5_pre_post_preprocessing",
  l3_5: "3_5_feature_diagnostics",
  l4_5: "4_5_generator_diagnostics",
  l5: "5_evaluation",
  l6: "6_statistical_tests",
  l7: "7_interpretation",
  l8: "8_output",
};

function coerceYamlScalar(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  if (/^-?\d+$/.test(String(value))) return Number(value);
  if (/^-?\d+\.\d+$/.test(String(value))) return Number(value);
  return value;
}

function selectedAxisYamlValue(axisName) {
  const records = optionRecordsForAxis(axisName);
  const selected = selectedCanonicalValues(axisName, records);
  if (Array.isArray(selected)) {
    if (isMultiSelectAxis(axisName)) return selected.map(coerceYamlScalar);
    return coerceYamlScalar(selected[0]);
  }
  return coerceYamlScalar(selected);
}

function fixedAxesForNode(node, axes) {
  return (axes || []).reduce((acc, axisName) => {
    const records = optionRecordsForAxis(axisName);
    if (!records.length) return acc;
    acc[axisName] = selectedAxisYamlValue(axisName);
    return acc;
  }, {});
}

function selectedDatasetIncludes(token) {
  const dataset = String(selectedAxisYamlValue("dataset") || "");
  return dataset.split("+").includes(token);
}

function l5FixedAxes(node) {
  const axes = fixedAxesForNode(node, node.axes || []);
  const regime = selectedAxisYamlValue("regime_definition");
  const targetStructure = selectedAxisYamlValue("target_structure");

  // The template L4 DAG emits point forecasts. Drop density-only controls
  // unless a density/quantile forecast DAG is added later.
  delete axes.density_metrics;
  // Benchmark-relative controls require execution context from L4 artifacts.
  // The navigator emits a portable starter recipe, so leave these to L5
  // defaults or explicit user edits once benchmark artifacts exist.
  delete axes.relative_metrics;
  delete axes.benchmark_window;
  delete axes.benchmark_scope;

  if (targetStructure !== "multi_series_target") {
    delete axes.agg_target;
  }
  if (!selectedDatasetIncludes("fred_sd")) {
    delete axes.agg_state;
  }
  if (regime === "none") {
    delete axes.regime_use;
    delete axes.regime_metrics;
  } else if (!["per_regime", "both"].includes(axes.regime_use)) {
    delete axes.regime_metrics;
  }
  if (axes.decomposition_target === "none") {
    delete axes.decomposition_order;
  }
  return axes;
}

function diagnosticExportAxes() {
  return {
    diagnostic_format: selectedAxisYamlValue("diagnostic_format"),
    attach_to_manifest: selectedAxisYamlValue("attach_to_manifest"),
    figure_dpi: selectedAxisYamlValue("figure_dpi"),
    latex_export: selectedAxisYamlValue("latex_export"),
  };
}

function diagnosticFixedAxes(layerId, node) {
  if (layerId === "l3_5") {
    return {
      comparison_stages: selectedAxisYamlValue("comparison_stages"),
      comparison_output_form: selectedAxisYamlValue("comparison_output_form"),
      feature_correlation: selectedAxisYamlValue("feature_correlation"),
      correlation_method: selectedAxisYamlValue("correlation_method"),
      correlation_view: "clustered_heatmap",
      ...diagnosticExportAxes(),
    };
  }
  if (layerId === "l4_5") {
    return {
      fit_view: selectedAxisYamlValue("fit_view"),
      fit_per_origin: selectedAxisYamlValue("fit_per_origin"),
      forecast_scale_view: selectedAxisYamlValue("forecast_scale_view"),
      back_transform_method: selectedAxisYamlValue("back_transform_method"),
      window_view: selectedAxisYamlValue("window_view"),
      ...diagnosticExportAxes(),
    };
  }
  return fixedAxesForNode(node, node.axes || []);
}

function yamlQuote(value) {
  if (typeof value !== "string") return String(value);
  if (!value || /[:#\[\]{},&*?|\-<>=!%@`]/.test(value) || /^\d/.test(value) || ["true", "false", "null"].includes(value)) {
    return JSON.stringify(value);
  }
  return value;
}

function yamlDump(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return value.map((item) => {
      if (item && typeof item === "object") {
        const dumped = yamlDump(item, indent + 2);
        return `${pad}- ${dumped.trimStart()}`;
      }
      return `${pad}- ${yamlQuote(item)}`;
    }).join("\n");
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value).filter(([, item]) => item !== undefined);
    if (!entries.length) return "{}";
    return entries.map(([key, item]) => {
      if (Array.isArray(item) && !item.length) return `${pad}${key}: []`;
      if (item && typeof item === "object" && !Array.isArray(item) && !Object.keys(item).length) return `${pad}${key}: {}`;
      if (item && typeof item === "object") {
        const dumped = yamlDump(item, indent + 2);
        return `${pad}${key}:\n${dumped}`;
      }
      return `${pad}${key}: ${yamlQuote(item)}`;
    }).join("\n");
  }
  return `${pad}${yamlQuote(value)}`;
}

function selectedDagSummary(layerId) {
  return selectedDagItems(layerId).map((item) => item.replace("::", " / "));
}

function l3DagYaml() {
  return {
    nodes: [
      { id: "src_X", type: "source", selector: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", subset: { role: "predictors" } } },
      { id: "src_y", type: "source", selector: { layer_ref: "l2", sink_name: "l2_clean_panel_v1", subset: { role: "target" } } },
      { id: "y_h", type: "step", op: "target_construction", params: { horizon: 1, mode: "point_forecast", method: "direct" }, inputs: ["src_y"] },
      { id: "X_lag", type: "step", op: "lag", params: { n_lag: 1 }, inputs: ["src_X"], pipeline_id: "baseline_lag" },
    ],
    sinks: {
      l3_features_v1: { X_final: "X_lag", y_final: "y_h" },
      l3_metadata_v1: "auto",
    },
    leaf_config: { navigator_selected_dag_items: selectedDagSummary("l3") },
  };
}

function l4DagYaml() {
  return {
    nodes: [
      { id: "src_features", type: "source", selector: { layer_ref: "l3", sink_name: "l3_features_v1" } },
      { id: "fit_ar_benchmark", type: "step", op: "fit_model", params: { family: "ar_p", forecast_strategy: "direct", training_start_rule: "expanding", search_algorithm: "none" }, inputs: ["src_features"], is_benchmark: true },
      { id: "fit_ridge", type: "step", op: "fit_model", params: { family: "ridge", forecast_strategy: "direct", training_start_rule: "expanding", search_algorithm: "none" }, inputs: ["src_features"], is_benchmark: false },
      { id: "predict_ridge", type: "step", op: "predict", params: { forecast_object: "point" }, inputs: ["fit_ridge"] },
    ],
    sinks: {
      l4_forecasts_v1: "predict_ridge",
      l4_model_artifacts_v1: ["fit_ar_benchmark", "fit_ridge"],
      l4_training_metadata_v1: "auto",
    },
    leaf_config: { navigator_selected_dag_items: selectedDagSummary("l4") },
  };
}

function l7DagYaml(node) {
  const outputAxes = fixedAxesForNode(node, (node.sub_layer_axes || {})["L7.B Output shape"] || CANONICAL_AXIS_GROUPS.l7["L7.B Output shape"]);
  return {
    enabled: selectedAxisYamlValue("enabled"),
    nodes: [
      { id: "src_model", type: "source", selector: { layer_ref: "l4", sink_name: "l4_model_artifacts_v1", subset: { model_id: "fit_ridge" } } },
      { id: "src_X", type: "source", selector: { layer_ref: "l3", sink_name: "l3_features_v1", subset: { component: "X_final" } } },
      { id: "importance", type: "step", op: "shap_kernel", params: { n_samples_background: 100, link_function: "identity" }, inputs: ["src_model", "src_X"] },
    ],
    sinks: { l7_importance_v1: { global: "importance" } },
    fixed_axes: outputAxes,
    leaf_config: { navigator_selected_dag_items: selectedDagSummary("l7") },
  };
}

function l6Yaml(node) {
  const subLayerAxes = node.sub_layer_axes || CANONICAL_AXIS_GROUPS.l6 || {};
  const out = {
    enabled: selectedAxisYamlValue("enabled"),
    test_scope: selectedAxisYamlValue("test_scope"),
    dependence_correction: selectedAxisYamlValue("dependence_correction"),
    overlap_handling: selectedAxisYamlValue("overlap_handling"),
    sub_layers: {},
  };
  Object.entries(subLayerAxes).forEach(([subLayer, axes]) => {
    if (subLayer === "L6 globals") return;
    out.sub_layers[subLayer] = {
      enabled: false,
      fixed_axes: fixedAxesForNode(node, axes),
    };
  });
  return out;
}

function canonicalRecipeYaml() {
  const byId = topologyNodeById();
  const recipe = { recipe_id: "macrocast-navigator-designed" };
  ["l0", "l1", "l2"].forEach((layerId) => {
    const node = byId.get(layerId);
    recipe[LAYER_YAML_KEYS[layerId]] = { fixed_axes: fixedAxesForNode(node, node.axes || []) };
    if (layerId === "l1") {
      recipe[LAYER_YAML_KEYS[layerId]].leaf_config = { target: "INDPRO" };
    }
  });
  recipe["3_feature_engineering"] = l3DagYaml();
  recipe["4_forecasting_model"] = l4DagYaml();
  const l5 = byId.get("l5");
  recipe[LAYER_YAML_KEYS.l5] = { fixed_axes: l5FixedAxes(l5) };
  const l6 = byId.get("l6");
  recipe[LAYER_YAML_KEYS.l6] = l6Yaml(l6);
  const l7 = byId.get("l7");
  recipe[LAYER_YAML_KEYS.l7] = l7DagYaml(l7);
  const l8 = byId.get("l8");
  recipe[LAYER_YAML_KEYS.l8] = { fixed_axes: fixedAxesForNode(l8, l8.axes || []) };
  ["l1_5", "l2_5", "l3_5", "l4_5"].forEach((layerId) => {
    const node = byId.get(layerId);
    recipe[LAYER_YAML_KEYS[layerId]] = {
      enabled: selectedAxisYamlValue("enabled"),
      fixed_axes: diagnosticFixedAxes(layerId, node),
    };
  });
  return `${yamlDump(recipe)}\n`;
}

function renderYaml() {
  els.yamlPreview.textContent = canonicalRecipeYaml();
}

function render() {
  renderSampleSelect();
  renderLayerTopology();
  renderLayerDetail();
  if (els.pathSource) {
    els.pathSource.textContent = state.loadedSource
      ? `loaded: ${state.loadedSource.label || state.loadedSource.id}`
      : `sample: ${currentSample().label}`;
  }
  renderYaml();
}

function sanitizeFilename(value) {
  return String(value || "navigator-recipe")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "navigator-recipe";
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/yaml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function importYamlFile(file) {
  const text = await file.text();
  const recipe = NavigatorStateEngine.recipeFromYaml(text);
  loadSource({
    id: `imported:${file.name}`,
    label: (recipe && recipe.recipe_id) || file.name,
    path: file.name,
    recipe,
    recipe_yaml: text.endsWith("\n") ? text : `${text}\n`,
    source_type: "imported_yaml",
  });
  render();
}

function bindEvents() {
  els.sampleSelect.addEventListener("change", (event) => {
    state.sampleIndex = Number(event.target.value);
    state.loadedSource = null;
    state.activeAxis = null;
    state.activeCanonicalAxis = null;
    resetEngineState();
    render();
  });

  if (els.axisSearch) els.axisSearch.addEventListener("input", (event) => {
    state.axisFilter = event.target.value;
    render();
  });

  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      setLayerFilter(button.dataset.layer);
      render();
    });
  });

  if (els.layerTopology) {
    els.layerTopology.addEventListener("click", (event) => {
      const card = event.target.closest("[data-topology-layer]");
      if (!card) return;
      state.activeTopologyLayer = card.dataset.topologyLayer;
      state.activeSubLayer = null;
      state.activeCanonicalAxis = null;
      render();
    });
    els.layerTopology.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const card = event.target.closest("[data-topology-layer]");
      if (!card) return;
      event.preventDefault();
      state.activeTopologyLayer = card.dataset.topologyLayer;
      state.activeSubLayer = null;
      state.activeCanonicalAxis = null;
      render();
    });
  }

  if (els.layerDetail) {
    els.layerDetail.addEventListener("click", (event) => {
      const subLayerButton = event.target.closest("[data-sub-layer]");
      if (subLayerButton) {
        state.activeSubLayer = subLayerButton.dataset.subLayer;
        state.activeCanonicalAxis = null;
        render();
        return;
      }
      const axisButton = event.target.closest("[data-canonical-axis]");
      if (axisButton) {
        state.activeCanonicalAxis = axisButton.dataset.canonicalAxis;
        render();
        return;
      }
      const optionButton = event.target.closest("[data-canonical-option]");
      if (optionButton && state.activeCanonicalAxis) {
        applyCanonicalSelection(state.activeCanonicalAxis, optionButton.dataset.canonicalOption);
        render();
        return;
      }
      const dagButton = event.target.closest("[data-dag-item]");
      if (dagButton) {
        toggleDagItem(dagButton.dataset.dagLayer, dagButton.dataset.dagSubLayer, dagButton.dataset.dagItem);
        render();
      }
    });
    els.layerDetail.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const subLayerButton = event.target.closest("[data-sub-layer]");
      const axisButton = event.target.closest("[data-canonical-axis]");
      const optionButton = event.target.closest("[data-canonical-option]");
      const dagButton = event.target.closest("[data-dag-item]");
      if (!subLayerButton && !axisButton && !optionButton && !dagButton) return;
      event.preventDefault();
      if (subLayerButton) {
        state.activeSubLayer = subLayerButton.dataset.subLayer;
        state.activeCanonicalAxis = null;
      } else if (axisButton) {
        state.activeCanonicalAxis = axisButton.dataset.canonicalAxis;
      } else if (optionButton && state.activeCanonicalAxis) {
        applyCanonicalSelection(state.activeCanonicalAxis, optionButton.dataset.canonicalOption);
      } else if (dagButton) {
        toggleDagItem(dagButton.dataset.dagLayer, dagButton.dataset.dagSubLayer, dagButton.dataset.dagItem);
      }
      render();
    });
  }

  if (els.axisList) els.axisList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-axis]");
    if (!button) return;
    state.activeAxis = button.dataset.axis;
    render();
  });

  if (els.optionList) els.optionList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-option]");
    if (!button || button.disabled) return;
    const axis = allAxes().find((item) => item.axis === state.activeAxis);
    if (!axis) return;
    state.engineState = NavigatorStateEngine.selectOption(state.data, state.engineState, axis.axis, button.dataset.option);
    render();
  });

  if (els.treePath) els.treePath.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    const axisItem = event.target.closest("[data-tree-axis]");
    const layerItem = event.target.closest("[data-tree-layer]");
    if (layerItem && layerItem.dataset.treeLayer !== state.layerFilter) {
      setLayerFilter(layerItem.dataset.treeLayer);
    }
    if (axisItem) state.activeAxis = axisItem.dataset.treeAxis;
    render();
  });

  if (els.treePath) els.treePath.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    if (event.target.closest("a")) return;
    const axisItem = event.target.closest("[data-tree-axis]");
    const layerItem = event.target.closest("[data-tree-layer]");
    if (!axisItem && !layerItem) return;
    event.preventDefault();
    if (layerItem && layerItem.dataset.treeLayer !== state.layerFilter) {
      setLayerFilter(layerItem.dataset.treeLayer);
    }
    if (axisItem) state.activeAxis = axisItem.dataset.treeAxis;
    render();
  });

  els.resetPath.addEventListener("click", () => {
    resetEngineState();
    state.canonicalSelections = {};
    state.dagSelections = {};
    render();
  });

  els.downloadYaml.addEventListener("click", () => {
    const source = currentSource();
    const filename = `${sanitizeFilename((source.recipe || {}).recipe_id || source.label || "navigator-recipe")}.yaml`;
    downloadText(filename, els.yamlPreview.textContent);
  });

  els.importYaml.addEventListener("click", () => {
    els.importYamlInput.click();
  });

  els.importYamlInput.addEventListener("change", async () => {
    const file = els.importYamlInput.files && els.importYamlInput.files[0];
    if (!file) return;
    try {
      await importYamlFile(file);
    } catch (error) {
      els.pathSource.textContent = `import failed: ${error.message}`;
    } finally {
      els.importYamlInput.value = "";
    }
  });

  els.copyYaml.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(els.yamlPreview.textContent);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = els.yamlPreview.textContent;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    els.copyYaml.textContent = "Copied";
    setTimeout(() => {
      els.copyYaml.textContent = "Copy";
    }, 900);
  });
}

async function boot() {
  bindEvents();
  const response = await fetch("assets/navigator_ui_data.json?v=20260501-dag-yaml-diagnostics");
  if (!response.ok) throw new Error(`Failed to load navigator data: ${response.status}`);
  state.data = await response.json();
  resetEngineState();
  render();
}

boot().catch((error) => {
  document.body.innerHTML = `<main class="panel" style="margin: 24px; padding: 18px;"><h1>Navigator failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
