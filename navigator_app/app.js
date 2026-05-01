const layerDefs = [
  { id: "l0", key: "0_meta", name: "Setup", mode: "form", role: "failure policy, reproducibility, compute" },
  { id: "l1", key: "1_data", name: "Data", mode: "form", role: "dataset, targets, horizons, universe" },
  { id: "l1_5", key: "1_5_data_summary", name: "Data Diagnostics", mode: "diagnostic", parent: "l1", role: "raw data summary" },
  { id: "l2", key: "2_preprocessing", name: "Preprocessing", mode: "form", role: "clean panel construction" },
  { id: "l2_5", key: "2_5_pre_post_preprocessing", name: "Pre/Post Diagnostics", mode: "diagnostic", parent: "l2", role: "pre/post comparison" },
  { id: "l3", key: "3_feature_engineering", name: "Feature DAG", mode: "dag", role: "features and target construction" },
  { id: "l3_5", key: "3_5_feature_diagnostics", name: "Feature Diagnostics", mode: "diagnostic", parent: "l3", role: "feature checks" },
  { id: "l4", key: "4_forecasting_model", name: "Forecast DAG", mode: "dag", role: "fit, predict, benchmark, combine" },
  { id: "l4_5", key: "4_5_generator_diagnostics", name: "Generator Diagnostics", mode: "diagnostic", parent: "l4", role: "model-fit diagnostics" },
  { id: "l5", key: "5_evaluation", name: "Evaluation", mode: "form", role: "metrics, aggregation, ranking" },
  { id: "l6", key: "6_statistical_tests", name: "Statistical Tests", mode: "form-toggle", role: "inferential tests" },
  { id: "l7", key: "7_interpretation", name: "Interpretation DAG", mode: "dag-toggle", role: "importance and attribution" },
  { id: "l8", key: "8_output", name: "Output", mode: "form", role: "saved objects and provenance" }
];

const state = {
  selectedLayer: "map",
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
      fixed_axes: { dataset: "fred_md", target_structure: "multi_target", variable_universe: "all_variables" },
      leaf_config: { targets: ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "RPI"], horizons: [1, 3, 6, 12] }
    },
    l2: {
      fixed_axes: { preprocessing_profile: "minimal_official_when_available", scaling_policy: "standard", missing_policy: "broad_safe_default" },
      leaf_config: {}
    },
    l5: {
      fixed_axes: { metrics: ["msfe", "rmse", "mae"], ranking: "by_primary_metric", primary_metric: "msfe" },
      leaf_config: {}
    },
    l6: { enabled: false, fixed_axes: { equal_predictive: "none" }, leaf_config: {} },
    l8: {
      fixed_axes: { saved_objects: ["forecasts", "metrics", "ranking", "manifest", "recipe_yaml"], provenance: "full" },
      leaf_config: { output_root: "macrocast_output" }
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

function selectLayer(id) {
  state.selectedLayer = id;
  state.selectedNode = null;
  state.connectingFrom = null;
  render();
}

function renderWorkspace() {
  const layer = layerById(state.selectedLayer);
  $("#workspaceEyebrow").textContent = layer ? `${layer.id.toUpperCase()} ${layer.mode}` : "Contract";
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
  const grid = document.createElement("div");
  grid.className = "map-grid";
  for (const layer of layerDefs) {
    const card = document.createElement("button");
    card.className = `map-card ${layer.mode === "diagnostic" ? "diagnostic" : ""}`;
    card.innerHTML = `
      <div class="map-card-title">
        <span>${layer.id.toUpperCase()} ${layer.name}</span>
        <span class="pill">${layer.mode}</span>
      </div>
      <div class="map-card-body">${layer.role}<br>Status: ${layerStatus(layer.id)}</div>
    `;
    card.addEventListener("click", () => selectLayer(layer.id));
    grid.appendChild(card);
  }
  body.appendChild(grid);
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

  if (layer.id === "l1") grid.appendChild(sectionFromFields("Broad Multi-Target FRED-MD", [
    selectField("dataset", state.layers.l1.fixed_axes.dataset, ["fred_md", "fred_qd", "fred_sd"], (v) => state.layers.l1.fixed_axes.dataset = v),
    selectField("target_structure", state.layers.l1.fixed_axes.target_structure, ["multi_target", "single_target"], (v) => state.layers.l1.fixed_axes.target_structure = v),
    textAreaField("targets", state.layers.l1.leaf_config.targets.join(", "), (v) => state.layers.l1.leaf_config.targets = splitCsv(v)),
    textAreaField("horizons", state.layers.l1.leaf_config.horizons.join(", "), (v) => state.layers.l1.leaf_config.horizons = splitCsv(v).map(Number).filter(Boolean))
  ]));

  if (layer.id === "l2") grid.appendChild(sectionFromFields("Preprocessing defaults", [
    selectField("profile", state.layers.l2.fixed_axes.preprocessing_profile, ["minimal_official_when_available", "raw_passthrough", "train_only_scaled"], (v) => state.layers.l2.fixed_axes.preprocessing_profile = v),
    selectField("scaling_policy", state.layers.l2.fixed_axes.scaling_policy, ["standard", "none", "robust"], (v) => state.layers.l2.fixed_axes.scaling_policy = v),
    selectField("missing_policy", state.layers.l2.fixed_axes.missing_policy, ["broad_safe_default", "keep_available_rows", "require_complete_rows"], (v) => state.layers.l2.fixed_axes.missing_policy = v)
  ]));

  if (layer.id === "l5") grid.appendChild(sectionFromFields("Evaluation", [
    textAreaField("metrics", state.layers.l5.fixed_axes.metrics.join(", "), (v) => state.layers.l5.fixed_axes.metrics = splitCsv(v)),
    selectField("primary_metric", state.layers.l5.fixed_axes.primary_metric, ["msfe", "rmse", "mae"], (v) => state.layers.l5.fixed_axes.primary_metric = v),
    selectField("ranking", state.layers.l5.fixed_axes.ranking, ["by_primary_metric", "by_average_rank"], (v) => state.layers.l5.fixed_axes.ranking = v)
  ]));

  if (layer.id === "l6") {
    grid.appendChild(toggleSection("Statistical tests", state.layers.l6.enabled, (checked) => state.layers.l6.enabled = checked, [
      selectField("equal_predictive", state.layers.l6.fixed_axes.equal_predictive, ["none", "dm_hln", "cw"], (v) => state.layers.l6.fixed_axes.equal_predictive = v)
    ]));
  }

  if (layer.id === "l8") grid.appendChild(sectionFromFields("Output", [
    textAreaField("saved_objects", state.layers.l8.fixed_axes.saved_objects.join(", "), (v) => state.layers.l8.fixed_axes.saved_objects = splitCsv(v)),
    selectField("provenance", state.layers.l8.fixed_axes.provenance, ["full", "minimal"], (v) => state.layers.l8.fixed_axes.provenance = v),
    textField("output_root", state.layers.l8.leaf_config.output_root, (v) => state.layers.l8.leaf_config.output_root = v)
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
    <button class="icon-button" data-action="add-source">Source</button>
    <button class="icon-button" data-action="add-step">Step</button>
    <button class="icon-button" data-action="add-combine">Combine</button>
    <button class="icon-button" data-action="add-sink">Sink</button>
    <button class="icon-button" data-action="layout">Auto layout</button>
  `;
  actions.querySelector('[data-action="add-source"]').addEventListener("click", () => addNode(layer.id, "source"));
  actions.querySelector('[data-action="add-step"]').addEventListener("click", () => addNode(layer.id, "step"));
  actions.querySelector('[data-action="add-combine"]').addEventListener("click", () => addNode(layer.id, "combine"));
  actions.querySelector('[data-action="add-sink"]').addEventListener("click", () => addNode(layer.id, "sink"));
  actions.querySelector('[data-action="layout"]').addEventListener("click", () => autoLayout(layer.id));

  if (layer.mode === "dag-toggle" && !dag.enabled) {
    const off = document.createElement("div");
    off.className = "empty-state";
    off.innerHTML = `<p>${layer.name} is off. Enable it in the inspector to create nodes and sinks.</p>`;
    body.appendChild(off);
    return;
  }

  const shell = document.createElement("div");
  shell.className = "dag-shell";
  shell.innerHTML = `<svg class="edge-layer"></svg><div class="dag-canvas"></div>`;
  body.appendChild(shell);
  drawEdges(shell.querySelector("svg"), dag);
  drawNodes(shell.querySelector(".dag-canvas"), layer.id, dag);
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
    const y1 = from.y + 37;
    const x2 = to.x;
    const y2 = to.y + 37;
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
    el.style.top = `${node.y}px`;
    el.innerHTML = `
      <div class="node-strip"></div>
      <div class="node-content">
        <div class="node-title">${node.label}</div>
        <div><span class="node-kind">${node.type}</span></div>
        <div class="node-meta">${node.id} · ${node.op}</div>
        <div class="node-ports">
          <button class="port-button" data-port="out">connect</button>
          <button class="port-button ${state.connectingFrom === node.id ? "connecting" : ""}" data-port="in">target</button>
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
    body.appendChild(contractSummary());
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
  fields.push(readonlyField("required sinks", requiredSinks(layer.id).join(", ")));
  fields.push(readonlyField("nodes", String(dag.nodes.length)));
  fields.push(readonlyField("edges", String(dag.edges.length)));
  return sectionFromFields(`${layer.id.toUpperCase()} DAG`, fields);
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
  if (state.bottomTab === "run") body.innerHTML = `<pre>macrocast-navigate resolve recipe.yaml\nmacrocast-navigate run recipe.yaml --output-root ${state.layers.l8.leaf_config.output_root}</pre>`;
}

function renderValidationBadge() {
  const badge = $("#validationBadge");
  const invalid = validateState().some((issue) => issue.level === "error");
  badge.textContent = invalid ? "invalid" : "valid";
  badge.className = `status-badge ${invalid ? "invalid" : "valid"}`;
}

function validateState() {
  const issues = [];
  if (!state.layers.l1.leaf_config.targets.length) issues.push({ level: "error", where: "L1", message: "At least one target is required." });
  if (!state.layers.l1.leaf_config.horizons.length) issues.push({ level: "error", where: "L1", message: "At least one horizon is required." });
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
      ${layerDefs.map((layer) => `<div class="contract-card form-section"><h2>${layer.id.toUpperCase()} ${layer.name}</h2><p>${layer.role}</p><p class="field-hint">Mode: ${layer.mode} · YAML: ${layer.key}</p></div>`).join("")}
    </div>
  `;
}

function generateYaml() {
  const out = {};
  out.recipe_id = state.recipeName;
  out[state.layers ? "0_meta" : "0_meta"] = state.layers.l0;
  out["1_data"] = state.layers.l1;
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

function dagYaml(layerId) {
  const dag = state.dags[layerId];
  return {
    nodes: dag.nodes.map((node) => ({ id: node.id, type: node.type, op: node.op, params: node.params })),
    edges: dag.edges,
    sinks: Object.fromEntries(dag.nodes.filter((node) => node.type === "sink").map((node) => [node.label, node.id]))
  };
}

function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    if (value.every((item) => typeof item !== "object")) return `[${value.join(", ")}]`;
    return value.map((item) => `${pad}- ${toYaml(item, indent + 2).trimStart()}`).join("\n");
  }
  if (value && typeof value === "object") {
    return Object.entries(value).map(([key, val]) => {
      if (val && typeof val === "object" && !Array.isArray(val)) return `${pad}${key}:\n${toYaml(val, indent + 2)}`;
      return `${pad}${key}: ${toYaml(val, indent + 2).trim()}`;
    }).join("\n");
  }
  if (typeof value === "string") return value;
  return String(value);
}

function requiredSinks(layerId) {
  if (layerId === "l3") return ["l3_features_v1", "l3_metadata_v1"];
  if (layerId === "l4") return ["l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"];
  if (layerId === "l7") return ["l7_importance_v1"];
  return [];
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
  return field;
}

function selectField(label, value, options, onChange) {
  const field = baseField(label);
  const select = document.createElement("select");
  for (const option of options) {
    const el = document.createElement("option");
    el.value = option;
    el.textContent = option;
    if (option === value) el.selected = true;
    select.appendChild(el);
  }
  select.addEventListener("change", () => {
    onChange(select.value);
    renderAfterEdit();
  });
  field.appendChild(select);
  return field;
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
  labelEl.textContent = label;
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
