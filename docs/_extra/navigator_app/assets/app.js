const state = {
  data: null,
  sampleIndex: 0,
  axisFilter: "",
  layerFilter: "0_meta",
  activeAxis: null,
  engineState: null,
  loadedSource: null,
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
  return NavigatorStateEngine.buildTree(state.data, state.engineState);
}

function axisMatchesLayer(axis) {
  return axis.layer === state.layerFilter;
}

function layerLabel(layer) {
  const labels = {
    "0_meta": "L0 Study Scope",
    "1_data_task": "L1 Data task",
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

function selectedDisplayLabel(axisName, value) {
  const label = valueDisplayName(axisName, value);
  return isDefaultSelection(axisName, value) ? `${label} [default]` : label;
}

function docsLink(axisName) {
  return axisPresentation(axisName).docs_url || "";
}

function layerDescription(layer) {
  const descriptions = {
    "0_meta": "4 user-facing decisions in order: study scope, failure handling, reproducibility, and compute layout. axis_type is internal YAML grammar.",
    "1_data_task": "Official data task and source frame: source, target structure, availability, raw source policy, and official transforms.",
    "2_preprocessing": "Representation construction after the official frame: t-codes, target construction, feature blocks, scaling, selection, and custom preprocessing.",
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
      const selectedLabel = selectedDisplayLabel(axis.axis, axis.selected);
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
  const selectedLabel = selectedDisplayLabel(axis.axis, axis.selected) || "-";
  const selectedSummary = valueSummary(axis.axis, axis.selected);
  const docs = docsLink(axis.axis);
  const defaultLabel = defaultValue(axis.axis) ? valueDisplayName(axis.axis, defaultValue(axis.axis)) : "";
  els.axisTitle.textContent = axisDisplayName(axis);
  els.axisLayer.textContent = `${layerLabel(axis.layer)} | ${axis.group_label || "Steps"} | group step ${groupStepIndex + 1} of ${groupAxes.length} | layer step ${stepIndex + 1} of ${layerAxes.length}`;
  els.axisSelected.textContent = `selected: ${selectedLabel}`;
  els.axisExplainer.innerHTML = `
    <p class="decision-question">${escapeHtml(axisQuestion(axis))}</p>
    <p class="decision-summary">${escapeHtml(axisSummary(axis))}</p>
    <div class="decision-meta">
      <span><strong>Current selection:</strong> ${escapeHtml(selectedLabel)}</span>
      ${defaultLabel ? `<span><strong>Default:</strong> ${escapeHtml(defaultLabel)}</span>` : ""}
      <span><strong>YAML key:</strong> ${escapeHtml(axis.axis)}</span>
      ${axis.group_label ? `<span><strong>Group:</strong> ${escapeHtml(axis.group_label)}</span>` : ""}
      ${(axis.axis_level || axis.group_level) ? `<span><strong>Level:</strong> ${escapeHtml(hierarchyLevelLabel(axis.axis_level || axis.group_level))}</span>` : ""}
      ${presentation.selection_kind ? `<span><strong>Selection type:</strong> ${escapeHtml(humanizeToken(presentation.selection_kind))}</span>` : ""}
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
      const statusLabel = isDefaultSelection(axis.axis, option.value) ? `${option.status} · default` : option.status;
      return `
        <button type="button" class="option-card ${stateClass}${selected}" data-option="${escapeHtml(option.value)}"${disabledAttr}>
          <div class="option-value">
            <span>${escapeHtml(optionLabel)}</span>
            <span class="status">${escapeHtml(statusLabel)}</span>
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
  const selectedOption = axis.options.find((option) => option.value === axis.selected);
  const disabledReason = selectedOption && selectedOption.disabled_reason;
  const blocked = disabledReason ? " blocked" : "";
  const active = axis.axis === state.activeAxis ? " active" : "";
  const edited = axis.edited ? `<span class="path-edited">edited</span>` : "";
  const docs = docsLink(axis.axis);
  const valueLabel = valueDisplayName(axis.axis, axis.selected);
  const selectedValueLabel = selectedDisplayLabel(axis.axis, axis.selected);
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
  setDefaultAxis();
  renderSampleSelect();
  renderSummary();
  renderPathHeader();
  renderAxisList();
  renderOptions();
  renderTreePath();
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
    resetEngineState();
    render();
  });

  els.axisSearch.addEventListener("input", (event) => {
    state.axisFilter = event.target.value;
    render();
  });

  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      setLayerFilter(button.dataset.layer);
      render();
    });
  });

  els.axisList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-axis]");
    if (!button) return;
    state.activeAxis = button.dataset.axis;
    render();
  });

  els.optionList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-option]");
    if (!button || button.disabled) return;
    const axis = allAxes().find((item) => item.axis === state.activeAxis);
    if (!axis) return;
    state.engineState = NavigatorStateEngine.selectOption(state.data, state.engineState, axis.axis, button.dataset.option);
    render();
  });

  els.treePath.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    const axisItem = event.target.closest("[data-tree-axis]");
    const layerItem = event.target.closest("[data-tree-layer]");
    if (layerItem && layerItem.dataset.treeLayer !== state.layerFilter) {
      setLayerFilter(layerItem.dataset.treeLayer);
    }
    if (axisItem) state.activeAxis = axisItem.dataset.treeAxis;
    render();
  });

  els.treePath.addEventListener("keydown", (event) => {
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
  const response = await fetch("assets/navigator_ui_data.json");
  if (!response.ok) throw new Error(`Failed to load navigator data: ${response.status}`);
  state.data = await response.json();
  resetEngineState();
  render();
}

boot().catch((error) => {
  document.body.innerHTML = `<main class="panel" style="margin: 24px; padding: 18px;"><h1>Navigator failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
