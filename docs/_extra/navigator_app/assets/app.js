const state = {
  data: null,
  sampleIndex: 0,
  axisFilter: "",
  layerFilter: "all",
  activeAxis: null,
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

function currentView() {
  return currentSample().view;
}

function allAxes() {
  return currentView().tree;
}

function axisMatchesLayer(axis) {
  if (state.layerFilter === "all") return true;
  if (state.layerFilter === "downstream") return !["0_meta", "1_data_task", "2_preprocessing", "3_training"].includes(axis.layer);
  return axis.layer === state.layerFilter;
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
  const blocked = preview.blocked_reasons || [];
  const disabled = view.tree.reduce((count, axis) => count + axis.options.filter((option) => !option.enabled).length, 0);
  els.executionStatus.textContent = preview.execution_status || "-";
  els.blockedCount.textContent = String(blocked.length);
  els.disabledCount.textContent = String(disabled);
}

function renderAxisList() {
  const axes = filteredAxes();
  els.axisList.innerHTML = axes
    .map((axis) => {
      const disabled = axis.options.filter((option) => !option.enabled).length;
      const active = axis.axis === state.activeAxis ? " active" : "";
      return `
        <button type="button" class="axis-button${active}" data-axis="${escapeHtml(axis.axis)}">
          <span class="axis-name">${escapeHtml(axis.axis)}</span>
          <span class="axis-meta">${escapeHtml(axis.layer)} | selected: ${escapeHtml(axis.selected ?? "-")} | disabled: ${disabled}</span>
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
      return `
        <article class="option-card ${stateClass}${selected}">
          <div class="option-value">
            <span>${escapeHtml(option.value)}</span>
            <span class="status">${escapeHtml(option.status)}</span>
          </div>
          ${reason}
          <div class="effect">${escapeHtml(option.canonical_path_effect)}</div>
        </article>
      `;
    })
    .join("");
}

function renderCompatibility() {
  const compatibility = currentView().compatibility || {};
  const rules = compatibility.active_rules || [];
  const recommendations = compatibility.recommendations || [];
  const ruleHtml = rules.length
    ? rules.map((rule) => `<div class="rule"><strong>${escapeHtml(rule.rule)}</strong><p>${escapeHtml(rule.effect)}</p></div>`).join("")
    : `<p class="muted">No active compatibility rules for this sample.</p>`;
  const recHtml = recommendations.length
    ? recommendations.map((item) => `<div class="rule"><strong>Recommendation</strong><p>${escapeHtml(item)}</p></div>`).join("")
    : "";
  els.compatibilityView.innerHTML = ruleHtml + recHtml;
}

function renderReplications() {
  els.replicationList.innerHTML = state.data.replications
    .map((entry) => {
      const outputs = (entry.expected_outputs || []).join(", ");
      return `
        <article class="replication-card">
          <strong>${escapeHtml(entry.id)}</strong>
          <p>${escapeHtml(entry.short_description)}</p>
          <p><b>Outputs:</b> ${escapeHtml(outputs)}</p>
        </article>
      `;
    })
    .join("");
}

function renderYaml() {
  els.yamlPreview.textContent = currentSample().recipe_yaml || "";
}

function render() {
  setDefaultAxis();
  renderSampleSelect();
  renderSummary();
  renderAxisList();
  renderOptions();
  renderCompatibility();
  renderReplications();
  renderYaml();
}

function bindEvents() {
  els.sampleSelect.addEventListener("change", (event) => {
    state.sampleIndex = Number(event.target.value);
    state.activeAxis = null;
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
      render();
    });
  });

  els.axisList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-axis]");
    if (!button) return;
    state.activeAxis = button.dataset.axis;
    render();
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
  render();
}

boot().catch((error) => {
  document.body.innerHTML = `<main class="panel" style="margin: 24px; padding: 18px;"><h1>Navigator failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
});
