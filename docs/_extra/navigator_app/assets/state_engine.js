(function attachNavigatorStateEngine(root) {
  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function asSet(values) {
    return new Set(values || []);
  }

  function statusDisabledReason(data, status) {
    if (status === "operational" || status === "operational_narrow") return null;
    const reasons = data.state_engine.status_disabled_reasons || {};
    return reasons[status] || null;
  }

  function axisLayer(data, axisName) {
    for (const [layer, axes] of Object.entries(data.tree_axes || {})) {
      if ((axes || []).includes(axisName)) return layer;
    }
    const catalog = data.axis_catalog || {};
    return catalog[axisName] ? catalog[axisName].layer : null;
  }

  function canonicalPathEffect(data, axisName, value) {
    const layer = axisLayer(data, axisName) || "";
    if (axisName === "exogenous_x_path_policy" || axisName === "recursive_x_model_family") {
      return `path.${layer}.leaf_config.${axisName} = '${value}'`;
    }
    return `path.${layer}.fixed_axes.${axisName} = '${value}'`;
  }

  function createState(data, sample) {
    const defaults = data.state_engine.default_selections || {};
    const selected = (((sample || {}).view || {}).compatibility || {}).selected || {};
    return {
      selections: { ...defaults, ...selected },
      edits: {},
    };
  }

  function selectedStatTests(data, selections, overrideAxis, overrideValue) {
    const spec = data.state_engine.stat_tests || {};
    const splitAxes = spec.split_axes || [];
    const legacyMap = spec.legacy_to_split || {};
    const values = {};
    splitAxes.forEach((axis) => {
      values[axis] = String(selections[axis] || "none");
    });
    const legacy = String(selections.stat_test || "none");
    if (legacy !== "none" && legacyMap[legacy]) {
      const route = legacyMap[legacy];
      if ((values[route.axis] || "none") === "none") values[route.axis] = route.value;
    }
    if (overrideAxis === "stat_test") {
      Object.keys(values).forEach((axis) => {
        values[axis] = "none";
      });
      if (overrideValue && overrideValue !== "none" && legacyMap[overrideValue]) {
        const route = legacyMap[overrideValue];
        values[route.axis] = route.value;
      }
    } else if (Object.prototype.hasOwnProperty.call(values, overrideAxis)) {
      values[overrideAxis] = overrideValue;
    }
    return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== "none"));
  }

  function selectedImportanceMethods(data, engineState, overrideAxis, overrideValue) {
    const spec = data.state_engine.importance || {};
    const defaults = spec.default_spec || {};
    const splitAxes = spec.split_axes || [];
    const legacyMap = spec.legacy_to_axis || {};
    const raw = {};
    Object.keys(defaults).forEach((axis) => {
      raw[axis] = engineState.selections[axis] ?? defaults[axis];
    });
    if (overrideAxis === "importance_method") {
      splitAxes.forEach((axis) => {
        raw[axis] = "none";
      });
    }
    if (overrideAxis && Object.prototype.hasOwnProperty.call(raw, overrideAxis)) {
      raw[overrideAxis] = overrideValue;
    }
    const legacy = String(raw.importance_method || "none");
    if (legacy !== "none" && legacyMap[legacy]) {
      const route = legacyMap[legacy];
      if ((raw[route.axis] || "none") === "none") raw[route.axis] = route.value;
    }
    const methods = [];
    splitAxes.forEach((axis) => {
      const method = raw[axis];
      if (method && method !== "none" && !methods.includes(method)) methods.push(method);
    });
    return methods;
  }

  function compatibilityReason(data, engineState, axisName, value) {
    const selected = engineState.selections;
    const groups = data.state_engine.model_groups || {};
    const forecastRules = data.state_engine.forecast_object_rules || {};
    const treeModels = asSet(groups.tree_models);
    const linearModels = asSet(groups.linear_models);
    const deepModels = asSet(groups.deep_sequence_models);
    const rawPanelBuilders = asSet(groups.raw_panel_builders);
    const hacCompatible = asSet(((data.state_engine.stat_tests || {}).hac_compatible) || []);
    const importanceSplitAxes = asSet(((data.state_engine.importance || {}).split_axes) || []);
    const importanceMetaAxes = asSet(((data.state_engine.importance || {}).meta_axes) || []);
    const statSplitAxes = asSet(((data.state_engine.stat_tests || {}).split_axes) || []);
    const localImportance = asSet(((data.state_engine.importance || {}).local_methods) || []);
    const defaultImportance = ((data.state_engine.importance || {}).default_spec) || {};

    const model = String(selected.model_family || "");
    const featureBuilder = String(selected.feature_builder || "");
    const forecastType = String(selected.forecast_type || "direct");
    const forecastObject = String(selected.forecast_object || "point_mean");
    const xPath = String(selected.exogenous_x_path_policy || "unavailable");
    const importance = String(selected.importance_method || "none");
    const importanceMethods = selectedImportanceMethods(data, engineState);

    if (axisName === "feature_builder") {
      if (deepModels.has(model)) {
        if (value === "autoreg_lagged_target") return null;
        if (value === "sequence_tensor") return "full multivariate sequence_tensor remains gated; current deep slice is univariate target-history sequence";
        return `model_family=${model} consumes the current univariate sequence/autoreg path, not ${value}`;
      }
      if (model === "ar" && rawPanelBuilders.has(value)) {
        return "model_family=ar is target-lag/autoreg only; raw-panel Z is incompatible";
      }
    }

    if (axisName === "model_family") {
      if (rawPanelBuilders.has(featureBuilder) && value === "ar") return "raw-panel feature builders cannot feed the AR-BIC target-lag generator";
      if (featureBuilder === "sequence_tensor" && !deepModels.has(value)) return "sequence_tensor is reserved for sequence/tensor generators";
      if ((importance === "tree_shap" || importanceMethods.includes("tree_shap")) && !treeModels.has(value)) return "importance_method=tree_shap requires a tree model";
      if ((importance === "linear_shap" || importanceMethods.includes("linear_shap")) && !linearModels.has(value)) return "importance_method=linear_shap requires a linear estimator";
      if (forecastObject === "quantile" && value !== forecastRules.quantile_model) return "forecast_object=quantile currently requires model_family=quantile_linear";
    }

    if (axisName === "forecast_object") {
      if (value === "quantile" && model && model !== forecastRules.quantile_model) return "quantile forecasts currently require model_family=quantile_linear";
      if ((value === "interval" || value === "density") && String(selected.target_normalization || "none") !== "none") {
        return "interval/density payload wrappers currently require target_normalization=none";
      }
    }

    if (axisName === "importance_method" || importanceSplitAxes.has(axisName)) {
      const methods = selectedImportanceMethods(data, engineState, axisName, value);
      if (methods.includes("tree_shap") && model && !treeModels.has(model)) return "tree_shap requires a tree model";
      if (methods.includes("linear_shap") && model && !linearModels.has(model)) return "linear_shap requires a linear estimator";
      if (methods.includes("minimal_importance") && !rawPanelBuilders.has(featureBuilder)) return "minimal_importance currently requires a raw-panel feature builder";
    }

    if (importanceMetaAxes.has(axisName)) {
      const methods = selectedImportanceMethods(data, engineState);
      const defaultValue = defaultImportance[axisName];
      if (!methods.length && value !== defaultValue) return "Layer 7 detail axes are active only when an importance family is selected";
      if (axisName === "importance_scope") {
        const allLocal = methods.length > 0 && methods.every((method) => localImportance.has(method));
        const hasLocal = methods.some((method) => localImportance.has(method));
        if (value === "global" && allLocal) return "local-only importance methods require importance_scope=local";
        if (value === "local" && methods.length > 0 && !hasLocal) return "global-only importance methods require importance_scope=global";
      }
    }

    if (axisName === "exogenous_x_path_policy") {
      if (forecastType !== "iterated" && value !== "unavailable") return "future-X path policies apply only when forecast_type=iterated";
      if (value !== "unavailable" && !rawPanelBuilders.has(featureBuilder)) return "raw-panel iterated future-X paths require a raw-panel feature builder";
    }

    if (axisName === "recursive_x_model_family") {
      if (xPath !== "recursive_x_model" && value !== "none") return "recursive_x_model_family is active only for exogenous_x_path_policy=recursive_x_model";
      if (xPath === "recursive_x_model" && value !== "ar1") return "only recursive_x_model_family=ar1 is currently operational";
    }

    if (axisName === "stat_test") {
      if (forecastObject === "direction" && !(forecastRules.direction_stats || []).includes(value)) return "direction forecast objects should use direction-family tests";
      if ((forecastObject === "interval" || forecastObject === "density") && value !== "none") return "interval/density calibration tests live on the density_interval axis, not legacy stat_test";
      if (forecastObject === "quantile" && !(forecastRules.quantile_stats || []).includes(value)) return "quantile tasks should avoid legacy point-forecast-only tests";
      if (String(selected.overlap_handling || "allow_overlap") === "evaluate_with_hac") {
        const activeTests = selectedStatTests(data, selected, axisName, value);
        if (Object.values(activeTests).some((test) => !hacCompatible.has(test))) return "evaluate_with_hac requires HAC-capable Layer 6 tests";
      }
    }

    if (axisName === "density_interval" && !["interval", "density", "quantile"].includes(forecastObject) && value !== "none") {
      return "density/interval tests require interval, density, or quantile forecast objects";
    }
    if (axisName === "direction" && forecastObject !== "direction" && value !== "none") {
      return "direction tests require forecast_object=direction";
    }
    if (statSplitAxes.has(axisName) && String(selected.overlap_handling || "allow_overlap") === "evaluate_with_hac") {
      const activeTests = selectedStatTests(data, selected, axisName, value);
      if (Object.values(activeTests).some((test) => !hacCompatible.has(test))) return "evaluate_with_hac requires HAC-capable Layer 6 tests";
    }
    if (axisName === "dependence_correction") {
      const activeTests = selectedStatTests(data, selected);
      const correctionActive = ["nw_hac", "nw_hac_auto", "block_bootstrap"].includes(value);
      if (correctionActive && !Object.keys(activeTests).length) return "dependence corrections are active only when a Layer 6 test is selected";
      if (correctionActive && Object.values(activeTests).some((test) => !hacCompatible.has(test))) {
        return "dependence corrections require HAC/bootstrap-compatible Layer 6 tests";
      }
    }
    if (axisName === "overlap_handling" && value === "evaluate_with_hac") {
      const activeTests = selectedStatTests(data, selected);
      if (Object.values(activeTests).some((test) => !hacCompatible.has(test))) return "evaluate_with_hac requires HAC-capable Layer 6 tests";
    }

    return null;
  }

  function axisView(data, engineState, layer, axisName) {
    const catalog = data.axis_catalog[axisName];
    const values = (catalog && catalog.allowed_values) || [];
    const currentStatus = (catalog && catalog.current_status) || {};
    const options = values.map((value) => {
      const status = currentStatus[value] || "unknown";
      const statusReason = statusDisabledReason(data, status);
      const compatReason = compatibilityReason(data, engineState, axisName, value);
      return {
        value,
        status,
        enabled: !statusReason && !compatReason,
        disabled_reason: compatReason || statusReason,
        canonical_path_effect: canonicalPathEffect(data, axisName, value),
      };
    });
    return {
      layer,
      layer_label: data.layer_labels[layer] || layer,
      axis: axisName,
      selected: engineState.selections[axisName],
      edited: Object.prototype.hasOwnProperty.call(engineState.edits, axisName),
      options,
    };
  }

  function buildTree(data, engineState) {
    const out = [];
    Object.entries(data.tree_axes || {}).forEach(([layer, axes]) => {
      (axes || []).forEach((axisName) => {
        if (data.axis_catalog[axisName]) out.push(axisView(data, engineState, layer, axisName));
      });
    });
    return out;
  }

  function selectOption(data, engineState, axisName, value) {
    const next = clone(engineState);
    const importanceSplitAxes = new Set(((data.state_engine.importance || {}).split_axes) || []);
    const statSplitAxes = new Set(((data.state_engine.stat_tests || {}).split_axes) || []);
    if (axisName === "importance_method") {
      importanceSplitAxes.forEach((axis) => {
        next.selections[axis] = "none";
        next.edits[axis] = "none";
      });
      const methods = selectedImportanceMethods(data, { selections: { ...next.selections, importance_method: value }, edits: next.edits }, axisName, value);
      const localMethods = new Set(((data.state_engine.importance || {}).local_methods) || []);
      const allLocal = methods.length > 0 && methods.every((method) => localMethods.has(method));
      if (allLocal && !Object.prototype.hasOwnProperty.call(next.edits, "importance_scope")) {
        next.selections.importance_scope = "local";
      }
    } else if (importanceSplitAxes.has(axisName) && value !== "none") {
      next.selections.importance_method = "none";
      next.edits.importance_method = "none";
    } else if (axisName === "stat_test") {
      statSplitAxes.forEach((axis) => {
        next.selections[axis] = "none";
        next.edits[axis] = "none";
      });
    } else if (statSplitAxes.has(axisName) && value !== "none") {
      next.selections.stat_test = "none";
      next.edits.stat_test = "none";
    }
    next.selections[axisName] = value;
    next.edits[axisName] = value;
    return next;
  }

  function selectedDisabledReasons(data, engineState) {
    return buildTree(data, engineState)
      .map((axis) => {
        const option = axis.options.find((item) => item.value === axis.selected);
        if (option && !option.enabled) return { axis: axis.axis, value: axis.selected, reason: option.disabled_reason };
        return null;
      })
      .filter(Boolean);
  }

  function compatibility(data, engineState) {
    const selected = engineState.selections;
    const groups = data.state_engine.model_groups || {};
    const rawPanelBuilders = asSet(groups.raw_panel_builders);
    const deepModels = asSet(groups.deep_sequence_models);
    const importanceMethods = selectedImportanceMethods(data, engineState);
    const rules = [];
    const recommendations = [];
    if (deepModels.has(String(selected.model_family || ""))) {
      rules.push({ rule: "deep_model_current_sequence_slice", effect: "keep current univariate target-history sequence/autoreg path; full sequence_tensor remains gated" });
    }
    if (importanceMethods.length) rules.push({ rule: "layer7_importance_split_contract", effect: "split importance-family axes materialize Layer 7 artifacts" });
    if (importanceMethods.includes("tree_shap")) rules.push({ rule: "tree_shap_requires_tree_model", effect: "model_family restricted to tree generators" });
    if (importanceMethods.includes("linear_shap")) rules.push({ rule: "linear_shap_requires_linear_model", effect: "model_family restricted to linear estimators" });
    if (selected.forecast_object === "quantile") {
      rules.push({ rule: "quantile_requires_quantile_generator", effect: "model_family=quantile_linear" });
      recommendations.push("Use quantile-oriented metrics/tests such as pinball/coverage families when those downstream axes are active.");
    }
    if (selected.forecast_object === "direction") recommendations.push("Use direction-family tests such as pesaran_timmermann or binomial_hit.");
    if (selected.forecast_object === "interval" || selected.forecast_object === "density") recommendations.push("Use density_interval tests; interval/density payloads are baseline wrappers over scalar generators.");
    if (["parquet", "all"].includes(String(selected.export_format || "json"))) recommendations.push("Parquet output writes sidecar artifact files in addition to the always-written CSV prediction table.");
    if (String(selected.regime_definition || "none") !== "none") recommendations.push("Regime evaluation is post-forecast evaluation filtering; regime_use beyond eval_only remains a separate runtime gate.");
    if (selected.forecast_type === "iterated" && rawPanelBuilders.has(String(selected.feature_builder || ""))) {
      rules.push({ rule: "raw_panel_iterated_future_x_path", effect: "leaf_config.exogenous_x_path_policy selects hold, observed, scheduled-known, or recursive-X/ar1 path" });
    }
    return {
      selected: Object.fromEntries(Object.entries(selected).sort()),
      active_rules: rules,
      recommendations,
    };
  }

  function applyEditToRecipe(data, recipe, axisName, value) {
    const layer = axisLayer(data, axisName);
    if (!layer) return;
    recipe.path = recipe.path || {};
    recipe.path[layer] = recipe.path[layer] || {};
    if (axisName === "exogenous_x_path_policy" || axisName === "recursive_x_model_family") {
      recipe.path[layer].leaf_config = recipe.path[layer].leaf_config || {};
      recipe.path[layer].leaf_config[axisName] = value;
    } else {
      recipe.path[layer].fixed_axes = recipe.path[layer].fixed_axes || {};
      recipe.path[layer].fixed_axes[axisName] = value;
    }
  }

  function recipeWithEdits(data, sample, engineState) {
    const recipe = clone(sample.recipe || {});
    Object.entries(engineState.edits).forEach(([axisName, value]) => applyEditToRecipe(data, recipe, axisName, value));
    return recipe;
  }

  function yamlScalar(value) {
    if (value === null) return "null";
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    const text = String(value);
    if (!text || /[:#\n{}\[\],&*?|\-<>=!%@`]/.test(text) || /^\s|\s$/.test(text)) return JSON.stringify(text);
    return text;
  }

  function toYaml(value, indent = 0) {
    const pad = " ".repeat(indent);
    if (Array.isArray(value)) {
      if (!value.length) return "[]";
      return value.map((item) => {
        if (item && typeof item === "object") return `${pad}-\n${toYaml(item, indent + 2)}`;
        return `${pad}- ${yamlScalar(item)}`;
      }).join("\n");
    }
    if (value && typeof value === "object") {
      return Object.entries(value).map(([key, item]) => {
        if (item && typeof item === "object") {
          const rendered = toYaml(item, indent + 2);
          return `${pad}${key}:${rendered === "[]" ? " []" : `\n${rendered}`}`;
        }
        return `${pad}${key}: ${yamlScalar(item)}`;
      }).join("\n");
    }
    return `${pad}${yamlScalar(value)}`;
  }

  function recipeYaml(data, sample, engineState) {
    if (!Object.keys(engineState.edits).length) return sample.recipe_yaml || "";
    return `${toYaml(recipeWithEdits(data, sample, engineState))}\n`;
  }

  const api = {
    createState,
    buildTree,
    selectOption,
    selectedDisabledReasons,
    compatibility,
    recipeWithEdits,
    recipeYaml,
  };

  root.NavigatorStateEngine = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
