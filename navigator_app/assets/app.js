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
  axisTitle: document.getElementById("axis-title"),
  axisLayer: document.getElementById("axis-layer"),
  axisSelected: document.getElementById("axis-selected"),
  executionStatus: document.getElementById("execution-status"),
  blockedCount: document.getElementById("blocked-count"),
  disabledCount: document.getElementById("disabled-count"),
  compatibilityView: document.getElementById("compatibility-view"),
  replicationList: document.getElementById("replication-list"),
  yamlPreview: document.getElementById("yaml-preview"),
  copyYaml: document.getElementById("copy-yaml"),
  downloadYaml: document.getElementById("download-yaml"),
  importYaml: document.getElementById("import-yaml"),
  importYamlInput: document.getElementById("import-yaml-input"),
  resetPath: document.getElementById("reset-path"),
  pathSource: document.getElementById("path-source"),
  resolverPreview: document.getElementById("resolver-preview"),
  treePath: document.getElementById("tree-path"),
  treePathLayer: document.getElementById("tree-path-layer"),
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
  if (state.layerFilter === "downstream") return !["0_meta", "1_data_task", "2_preprocessing", "3_training"].includes(axis.layer);
  return axis.layer === state.layerFilter;
}

function layerLabel(layer) {
  const labels = {
    "0_meta": "L0 Study setup",
    "1_data_task": "L1 Data task",
    "2_preprocessing": "L2 Representation",
    "3_training": "L3 Generator",
    downstream: "L4-7 Evaluate",
  };
  return labels[layer] || layer || "Layer";
}

function filteredAxes() {
  const q = state.axisFilter.trim().toLowerCase();
  return allAxes().filter((axis) => {
    const text = `${axis.axis} ${axis.layer} ${axis.selected ?? ""}`.toLowerCase();
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

function renderAxisList() {
  const axes = filteredAxes();
  els.axisList.innerHTML = axes
    .map((axis) => {
      const disabled = axis.options.filter((option) => !option.enabled).length;
      const active = axis.axis === state.activeAxis ? " active" : "";
      const edited = axis.edited ? " | edited" : "";
      return `
        <button type="button" class="axis-button${active}" data-axis="${escapeHtml(axis.axis)}">
          <span class="axis-name">${escapeHtml(axis.axis)}</span>
          <span class="axis-meta">${escapeHtml(axis.layer)} | selected: ${escapeHtml(axis.selected ?? "-")} | disabled: ${disabled}${edited}</span>
        </button>
      `;
    })
    .join("");
}

function renderOptions() {
  const axis = allAxes().find((item) => item.axis === state.activeAxis);
  if (!axis) {
    els.axisTitle.textContent = "No axis";
    els.axisLayer.textContent = "No matching axes";
    els.axisSelected.textContent = "selected: -";
    els.optionList.innerHTML = "";
    return;
  }
  els.axisTitle.textContent = axis.axis;
  els.axisLayer.textContent = axis.layer_label || axis.layer;
  els.axisSelected.textContent = `selected: ${axis.selected ?? "-"}`;
  els.optionList.innerHTML = axis.options
    .map((option) => {
      const stateClass = option.enabled ? "enabled" : "disabled";
      const selected = option.value === axis.selected ? " selected" : "";
      const reason = option.disabled_reason ? `<p class="reason">${escapeHtml(option.disabled_reason)}</p>` : "";
      const disabledAttr = option.enabled ? "" : " disabled";
      return `
        <button type="button" class="option-card ${stateClass}${selected}" data-option="${escapeHtml(option.value)}"${disabledAttr}>
          <div class="option-value">
            <span>${escapeHtml(option.value)}</span>
            <span class="status">${escapeHtml(option.status)}</span>
          </div>
          ${reason}
          <div class="effect">${escapeHtml(option.canonical_path_effect)}</div>
        </button>
      `;
    })
    .join("");
}

function renderTreePath() {
  const axes = allAxes().filter(axisMatchesLayer);
  els.treePathLayer.textContent = layerLabel(state.layerFilter);
  if (!axes.length) {
    els.treePath.innerHTML = `<p class="muted">No axes in this layer.</p>`;
    return;
  }
  els.treePath.innerHTML = `
    <ol class="tree-path-list">
      ${axes.map((axis, idx) => {
        const selectedOption = axis.options.find((option) => option.value === axis.selected);
        const disabledReason = selectedOption && selectedOption.disabled_reason;
        const pathEffect = selectedOption && selectedOption.canonical_path_effect
          ? selectedOption.canonical_path_effect
          : `selected: ${axis.selected ?? "-"}`;
        const blocked = disabledReason ? " blocked" : "";
        const edited = axis.edited ? `<span class="path-edited">edited</span>` : "";
        return `
          <li class="tree-path-item${blocked}" data-tree-axis="${escapeHtml(axis.axis)}">
            <button type="button">
              <span class="path-step">${idx + 1}</span>
              <span class="path-body">
                <span class="path-axis">${escapeHtml(axis.axis)}</span>
                <span class="path-value">${escapeHtml(axis.selected ?? "-")}</span>
                ${edited}
                <span class="path-effect">${escapeHtml(pathEffect)}</span>
                ${disabledReason ? `<span class="path-reason">${escapeHtml(disabledReason)}</span>` : ""}
              </span>
            </button>
          </li>
        `;
      }).join("")}
    </ol>
  `;
}

function renderCompatibility() {
  const compatibility = NavigatorStateEngine.compatibility(state.data, state.engineState);
  const selectedDisabled = NavigatorStateEngine.selectedDisabledReasons(state.data, state.engineState);
  const rules = compatibility.active_rules || [];
  const recommendations = compatibility.recommendations || [];
  const selectedDisabledHtml = selectedDisabled.length
    ? selectedDisabled.map((item) => `<div class="rule blocked"><strong>${escapeHtml(item.axis)}=${escapeHtml(item.value)}</strong><p>${escapeHtml(item.reason)}</p></div>`).join("")
    : "";
  const ruleHtml = rules.length
    ? rules.map((rule) => `<div class="rule"><strong>${escapeHtml(rule.rule)}</strong><p>${escapeHtml(rule.effect)}</p></div>`).join("")
    : `<p class="muted">No active compatibility rules for this sample.</p>`;
  const recHtml = recommendations.length
    ? recommendations.map((item) => `<div class="rule"><strong>Recommendation</strong><p>${escapeHtml(item)}</p></div>`).join("")
    : "";
  els.compatibilityView.innerHTML = selectedDisabledHtml + ruleHtml + recHtml;
}

function renderReplications() {
  els.replicationList.innerHTML = state.data.replications
    .map((entry) => {
      const outputs = (entry.expected_outputs || []).join(", ");
      const path = (entry.exact_tree_path || []).join(" | ");
      const deviations = (entry.deviations_from_original_paper || []).join(" ");
      return `
        <article class="replication-card" data-replication-card="${escapeHtml(entry.id)}">
          <strong>${escapeHtml(entry.id)}</strong>
          <p>${escapeHtml(entry.paper_name)}</p>
          <p>${escapeHtml(entry.short_description)}</p>
          <p><b>Path:</b> ${escapeHtml(path)}</p>
          <p><b>Outputs:</b> ${escapeHtml(outputs)}</p>
          <p><b>Deviations:</b> ${escapeHtml(deviations)}</p>
          <button type="button" class="secondary-action" data-replication-id="${escapeHtml(entry.id)}">Load path</button>
        </article>
      `;
    })
    .join("");
}

function renderYaml() {
  els.yamlPreview.textContent = NavigatorStateEngine.recipeYaml(state.data, currentSource(), state.engineState);
}

function renderResolverPreview() {
  const selectedDisabled = NavigatorStateEngine.selectedDisabledReasons(state.data, state.engineState);
  const source = currentSource();
  const filename = `${sanitizeFilename((source.recipe || {}).recipe_id || source.label || "navigator-recipe")}.yaml`;
  const status = selectedDisabled.length ? "browser_blocked" : (Object.keys(state.engineState.edits).length || state.loadedSource ? "browser_preview" : (currentView().compile_preview || {}).execution_status || "unknown");
  const blockedHtml = selectedDisabled.length
    ? selectedDisabled.map((item) => `<div class="rule blocked"><strong>${escapeHtml(item.axis)}=${escapeHtml(item.value)}</strong><p>${escapeHtml(item.reason)}</p></div>`).join("")
    : `<p class="muted">No selected branch is blocked in the browser state engine.</p>`;
  const snapshot = !state.loadedSource && !Object.keys(state.engineState.edits).length && currentView().compile_preview
    ? `<p class="muted">Compiler snapshot: ${escapeHtml(currentView().compile_preview.status || currentView().compile_preview.execution_status || "available")}</p>`
    : `<p class="muted">Browser preview only. Run the resolver command against the downloaded YAML before execution.</p>`;
  els.resolverPreview.innerHTML = `
    <div class="rule">
      <strong>${escapeHtml(status)}</strong>
      ${snapshot}
      <div class="effect">macrocast-navigate resolve ${escapeHtml(filename)}</div>
      <div class="effect">macrocast-navigate run ${escapeHtml(filename)} --output-root results/${escapeHtml(filename.replace(/\.ya?ml$/, ""))}</div>
    </div>
    ${blockedHtml}
  `;
}

function render() {
  setDefaultAxis();
  renderSampleSelect();
  renderSummary();
  renderAxisList();
  renderOptions();
  renderTreePath();
  renderCompatibility();
  renderReplications();
  renderResolverPreview();
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
      state.layerFilter = button.dataset.layer;
      document.querySelectorAll("[data-layer]").forEach((item) => item.classList.toggle("active", item === button));
      state.activeAxis = null;
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
    const item = event.target.closest("[data-tree-axis]");
    if (!item) return;
    state.activeAxis = item.dataset.treeAxis;
    render();
  });

  els.resetPath.addEventListener("click", () => {
    resetEngineState();
    render();
  });

  els.replicationList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-replication-id]");
    if (!button) return;
    const entry = state.data.replications.find((item) => item.id === button.dataset.replicationId);
    if (!entry || !entry.recipe) return;
    loadSource({
      id: entry.id,
      label: entry.id,
      path: entry.id,
      recipe: entry.recipe,
      recipe_yaml: entry.recipe_yaml,
      source_type: "replication",
      replication: entry,
    });
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
      els.resolverPreview.innerHTML = `<div class="rule blocked"><strong>import_failed</strong><p>${escapeHtml(error.message)}</p></div>`;
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
