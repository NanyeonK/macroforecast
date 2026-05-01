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
  l0: ["L0.A study scope", "L0.B execution policy", "L0.C reproducibility", "L0.D compute mode"],
  l1: ["L1.A data source", "L1.B target and horizons", "L1.C predictors", "L1.D geography", "L1.G regimes"],
  l1_5: ["L1.5.A sample coverage", "L1.5.B univariate summary", "L1.5.C stationarity", "L1.5.D missing and outlier", "L1.5.E correlation", "L1.5.Z export"],
  l2: ["L2.A target construction", "L2.B transforms", "L2.C missing and outliers", "L2.D scaling", "L2.E features"],
  l2_5: ["L2.5.A comparison", "L2.5.B distribution shift", "L2.5.C correlation shift", "L2.5.D cleaning summary", "L2.5.Z export"],
  l3: ["L3.A source nodes", "L3.B feature DAG", "L3.C sinks"],
  l3_5: ["L3.5.A comparison", "L3.5.B factor inspection", "L3.5.C feature correlation", "L3.5.D lag inspection", "L3.5.E selection", "L3.5.Z export"],
  l4: ["L4.A model DAG", "L4.B forecasts", "L4.C model artifacts", "L4.D training metadata"],
  l4_5: ["L4.5.A fit", "L4.5.B scale", "L4.5.C window stability", "L4.5.D tuning", "L4.5.E ensemble", "L4.5.Z export"],
  l5: ["L5.A metrics", "L5.B benchmark", "L5.C aggregation", "L5.D slicing and decomposition", "L5.E ranking"],
  l6: ["L6 globals", "L6_A_equal_predictive", "L6_B_nested", "L6_C_cpa", "L6_D_multiple_model", "L6_E_density_interval", "L6_F_direction", "L6_G_residual"],
  l7: ["L7.A importance DAG", "L7.B output shape"],
  l8: ["L8_A_export_format", "L8_B_saved_objects", "L8_C_provenance", "L8_D_artifact_granularity"],
};

const CANONICAL_AXIS_GROUPS = {
  l0: {
    "L0.A study scope": ["study_scope"],
    "L0.B execution policy": ["failure_policy"],
    "L0.C reproducibility": ["reproducibility_mode"],
    "L0.D compute mode": ["compute_mode"],
  },
  l1: {
    "L1.A data source": ["custom_source_policy", "dataset", "frequency", "information_set_type", "release_lag_rule", "contemporaneous_x_rule"],
    "L1.B target and horizons": ["target_structure"],
    "L1.C predictors": ["variable_universe", "fred_sd_variable_group", "sd_variable_selection"],
    "L1.D geography": ["fred_sd_state_group", "state_selection"],
    "L1.G regimes": ["regime_definition"],
  },
  l2: {
    "L2.A target construction": ["horizon_target_construction", "target_transform", "target_normalization"],
    "L2.B transforms": ["tcode_policy", "transform_policy", "transform_scope", "fred_sd_mixed_frequency_representation"],
    "L2.C missing and outliers": ["x_missing_policy", "x_outlier_policy", "outlier_policy", "outlier_action", "imputation_policy"],
    "L2.D scaling": ["scaling_policy"],
    "L2.E features": ["target_lag_block", "x_lag_feature_block", "factor_feature_block", "level_feature_block"],
  },
  l3: {
    "L3.A source nodes": ["model_family", "benchmark_family"],
    "L3.B feature DAG": ["framework", "outer_window", "refit_policy", "min_train_size", "training_start_rule"],
    "L3.C sinks": ["forecast_type", "forecast_object", "exogenous_x_path_policy", "recursive_x_model_family", "midasr_weight_family"],
  },
  l4: {
    "L4.A model DAG": ["model_family", "benchmark_family", "midasr_weight_family"],
    "L4.B forecasts": ["forecast_type", "forecast_object", "exogenous_x_path_policy", "recursive_x_model_family"],
    "L4.C model artifacts": ["framework", "outer_window", "refit_policy"],
    "L4.D training metadata": ["min_train_size", "training_start_rule"],
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
    "L7.A importance DAG": ["enabled"],
    "L7.B output shape": ["output_table_format", "figure_type", "top_k_features_to_show", "precision_digits", "figure_dpi", "figure_format", "latex_table_export", "markdown_table_export"],
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
  l3: "3_training",
  l4: "3_training",
  l5: "4_evaluation",
  l6: "6_stat_tests",
  l7: "7_importance",
  l8: "5_output_provenance",
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
  agg_horizon: ["pooled", "per_horizon", "short_medium_long"],
  agg_target: ["pooled", "per_target", "target_group"],
  agg_state: ["pooled", "per_state", "per_region"],
  oos_period: ["full", "user_defined", "rolling_origin"],
  regime_use: ["pooled", "per_regime", "regime_interaction"],
  regime_metrics: ["same_metrics", "separate_metrics"],
  decomposition_target: ["none", "by_predictor_block", "by_period", "by_regime", "by_state"],
  decomposition_order: ["sequential", "shapley", "leave_one_out"],
  ranking: ["primary_metric", "all_metrics", "pareto"],
  report_style: ["compact", "full", "paper_table"],
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
  saved_objects: ["forecasts", "forecast_intervals", "metrics", "ranking", "decomposition", "regime_metrics", "state_metrics", "model_artifacts", "combination_weights", "feature_metadata", "clean_panel", "raw_panel", "diagnostics_all", "tests", "importance", "transformation_attribution"],
  model_artifacts_format: ["pickle", "joblib", "onnx", "pmml"],
  provenance_fields: ["recipe_yaml_full", "recipe_hash", "package_version", "python_version", "r_version", "julia_version", "dependency_lockfile", "git_commit_sha", "git_branch_name", "data_revision_tag", "random_seed_used", "runtime_environment", "runtime_duration", "cell_resolved_axes"],
  manifest_format: ["json", "yaml", "json_lines"],
  artifact_granularity: ["per_cell", "per_target", "per_horizon", "per_target_horizon", "flat"],
  naming_convention: ["cell_id", "descriptive", "recipe_hash", "custom"],
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
    "0_meta": "4 user-facing decisions in order: study scope, failure handling, reproducibility, and compute layout. axis_type is internal YAML grammar.",
    "1_data_task": "Data source, target y, and predictor x source frame: source mode, frequency, forecast-time information, target y, predictor x, raw source quality, transforms, and availability.",
    "2_preprocessing": "Representation construction after the Layer 1 source frame: t-codes, target construction, feature blocks, scaling, selection, and custom preprocessing.",
    "3_training": "Forecast generation: model, benchmark, forecast object, future-X path, windows, and tuning.",
    "4_evaluation": "Evaluation choices: metrics, benchmark comparison, aggregation, ranking, regimes, decomposition, and OOS period.",
    "5_output_provenance": "Output and provenance: export format, saved objects, provenance fields, and artifact granularity.",
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
  const active = subLayer === state.activeSubLayer ? " active" : "";
  return `
    <button type="button" class="sublayer-card${active}" data-sub-layer="${escapeHtml(subLayer)}">
      <span>${escapeHtml(String(idx + 1).padStart(2, "0"))}</span>
      <strong>${escapeHtml(subLayer)}</strong>
      <em>${escapeHtml(String(axes.length))} axes</em>
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

function selectedCanonicalOption(axisName, records) {
  if (state.canonicalSelections[axisName]) return state.canonicalSelections[axisName];
  if (!records.length) return "";
  const enabled = records.find((record) => record.enabled !== false);
  return (enabled || records[0]).value;
}

function renderCanonicalOption(record, axisName, selectedValue) {
  const selected = String(record.value) === String(selectedValue) ? " selected" : "";
  const disabled = record.enabled === false ? " disabled" : "";
  const disabledAttr = record.enabled === false ? " disabled" : "";
  return `
    <button type="button" class="axis-choice${selected}${disabled}" data-canonical-option="${escapeHtml(record.value)}"${disabledAttr}>
      <strong>${escapeHtml(record.value)}</strong>
      <span>${escapeHtml(record.status || "operational")}</span>
      ${record.disabled_reason ? `<em>${escapeHtml(record.disabled_reason)}</em>` : ""}
    </button>
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
  const selected = selectedCanonicalOption(axisName, records);
  return `
    <div class="axis-option-panel">
      <div class="axis-option-head">
        <span>Selected axis</span>
        <strong>${escapeHtml(axisName)}</strong>
        <em>selected: ${escapeHtml(selected)}</em>
      </div>
      <div class="axis-choice-grid">
        ${records.map((record) => renderCanonicalOption(record, axisName, selected)).join("")}
      </div>
    </div>
  `;
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
  if (!state.activeCanonicalAxis || !subLayerAxes.includes(state.activeCanonicalAxis)) {
    state.activeCanonicalAxis = subLayerAxes[0] || null;
  }
  const modeText = node.ui_mode === "graph"
    ? "Graph/DAG layer: users compose source, step, and sink nodes."
    : "List layer: users resolve ordered axes and sub-layer sections.";
  const layerAxes = effectiveAxesForNode(node);
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
        <span><strong>${escapeHtml(String(layerAxes.length))}</strong> axes</span>
      </div>
    </div>

    <div class="definition-grid">
      <section class="definition-wide">
        <h3>Sub-layers</h3>
        ${subLayers.length ? `<div class="sublayer-grid">${subLayers.map((subLayer, idx) => renderSubLayerButton(node, subLayer, idx)).join("")}</div>` : `<p class="empty-note">No explicit sub-layer sections.</p>`}
      </section>
      <section>
        <h3>Layer globals</h3>
        ${formatList(node.layer_globals, "No layer-global axes.")}
      </section>
      <section>
        <h3>Selected sub-layer</h3>
        <p class="selected-sublayer">${escapeHtml(activeSubLayer || "Layer-level controls")}</p>
        <p class="source-note">${escapeHtml(subLayerAxes.length ? `${subLayerAxes.length} axis/control entries available.` : "This sub-layer is configured by DAG nodes or runtime metadata.")}</p>
      </section>
      <section class="definition-wide">
        <h3>Axes / output controls</h3>
        ${subLayerAxes.length ? `<div class="canonical-axis-grid">${subLayerAxes.map(renderAxisButton).join("")}</div>` : `<p class="empty-note">No fixed axes for this sub-layer.</p>`}
        ${renderCanonicalOptionsPanel(state.activeCanonicalAxis)}
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

function renderYaml() {
  els.yamlPreview.textContent = NavigatorStateEngine.recipeYaml(state.data, currentSource(), state.engineState);
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
        state.canonicalSelections[state.activeCanonicalAxis] = optionButton.dataset.canonicalOption;
        render();
      }
    });
    els.layerDetail.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const subLayerButton = event.target.closest("[data-sub-layer]");
      const axisButton = event.target.closest("[data-canonical-axis]");
      const optionButton = event.target.closest("[data-canonical-option]");
      if (!subLayerButton && !axisButton && !optionButton) return;
      event.preventDefault();
      if (subLayerButton) {
        state.activeSubLayer = subLayerButton.dataset.subLayer;
        state.activeCanonicalAxis = null;
      } else if (axisButton) {
        state.activeCanonicalAxis = axisButton.dataset.canonicalAxis;
      } else if (state.activeCanonicalAxis) {
        state.canonicalSelections[state.activeCanonicalAxis] = optionButton.dataset.canonicalOption;
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
  const response = await fetch("assets/navigator_ui_data.json?v=20260501-canonical-layer-redesign");
  if (!response.ok) throw new Error(`Failed to load navigator data: ${response.status}`);
  state.data = await response.json();
  resetEngineState();
  render();
}

boot().catch((error) => {
  document.body.innerHTML = `<main class="panel" style="margin: 24px; padding: 18px;"><h1>Navigator failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
