# Post-PR70 Runtime Roadmap

Date: 2026-04-26

PR #70 closed the implicit Layer 1 to Layer 2 data-frame handoff by writing
`layer1_official_frame_v1` for every run. The next work should preserve that
discipline: a branch becomes selectable only when the registry status, compiler
validation, runtime behavior, manifest/artifact contract, Navigator data, docs,
and tests agree.

## Current Clean State

- Server1 `main` is clean at `108a838`.
- `feature_block_set` is a public Layer 2 representation-family axis.
- `Layer2Representation` is explicit as `layer2_representation_v1`.
- Remaining operational-narrow surfaces are centralized in
  `OPERATIONAL_NARROW_CONTRACTS` and exported through
  `navigator_ui_data_v1`.
- `predictions.csv` has `prediction_row_schema_v1`.
- Layer 1 official-frame handoff is explicit as
  `layer1_official_frame_v1`.

## Priority Order

### 1. Layer 1 Runtime Widening

Use `layer1_official_frame_v1` as the extension point before opening broader
data-task cells.

Candidate slices:

1. Vintage-path contract hardening for FRED-MD/FRED-QD.
2. Release-lag evidence in the official-frame contract.
3. Mixed-source provenance for MD/SD and QD/SD composite panels.

Acceptance gate:

- compiler validation for required user inputs;
- runtime sidecar fields in `layer1_official_frame.json`;
- manifest summary fields only for compact routing metadata;
- artifact-manifest entry unchanged and still Layer 1-owned;
- no-lookahead or availability evidence when timing is involved;
- targeted regression test plus full suite;
- Navigator UI data regenerated/checked if registry status changes.

### 2. Layer 3 Gated Cells

Layer 3 is mostly operational for scalar tabular generators, path-average
execution, and raw-panel iterated narrow paths. Remaining work should stay
behind explicit contracts:

- multivariate `sequence_tensor` Layer 2 handoff;
- `sequence_forecast_payload_v1`;
- richer `recursive_x_model_family` values beyond `ar1`;
- non-point raw-panel iterated payloads;
- transformed or normalized target-scale composition;
- direct tabular generator protocol if `forecast_payload_v1` becomes too small.

### 3. Layer 2 Remaining Contracts

Layer 2 method-research extension points are broad enough for current custom
feature blocks, custom combiners, final-`Z` selection, MARX rotations, and
factor-score rotations. Remaining items are feature work:

- custom inverse transform contract;
- broader custom rotation families;
- broader factor-block fit-state/leakage evidence;
- selected full-grammar preprocessing values that remain registry-only.

### 4. Layer 5 And Wrapper Work

Output/provenance is closed for aggregated run directories. Still gated:

- per-target, per-horizon, and hierarchical artifact layouts;
- wrapper/orchestrator handoff for `multi_target_separate_runs`,
  `benchmark_suite`, and `ablation_study`;
- branch deltas for `single_target_full_sweep` and
  `single_target_generator_grid`.

### 5. Docs And Navigator UI

Do not prioritize deeper UI affordances over runtime truth. When runtime work
changes registry status or compatibility logic, update:

- generated `navigator_ui_data_v1`;
- compatibility disabled-branch reasons;
- replication route metadata if a recipe path changes;
- docs pages that own the affected layer.

Deferred UI affordances:

- path diff/history;
- shareable URL state;
- richer route comparison;
- browser-only convenience beyond authoritative CLI resolver/run commands.

## First Runtime Widening Slice

Start with Layer 1 vintage/release-lag/mixed-source audit and implement the
smallest executable widening that can be tested without external network
fragility. The safest first candidate is a contract-only hardening pass for
vintage/local-source paths: assert the sidecar records version mode, vintage,
data-through, artifact SHA, transform-code coverage, and availability policy
for local vintage fixtures before live-source widening.

Implemented first slice:

- `layer1_official_frame_v1` records `information_set_contract`;
- `layer1_official_frame_v1` records `transform_code_coverage`;
- local vintage FRED-MD execution is covered by a regression test without live
  network dependency.
- `layer1_official_frame_v1` records `source_availability_contract_v1`;
- current and vintage local-source FRED-MD execution now expose source kind,
  artifact SHA/size/cache evidence, requested vintage, actual vintage, and the
  observed data window without live network dependency.
- local-source FRED-SD CSV fixtures now provide an openpyxl-free deterministic
  path for `fred_md+fred_sd` composite execution while the official live source
  remains the St. Louis Fed workbook.
- cache-hit current/vintage paths now simulate remote-source contract evidence
  deterministically without touching the network;
- composite source contracts now have deterministic component-level coverage.
