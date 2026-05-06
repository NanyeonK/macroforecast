# macroforecast

> Fair, reproducible macro forecasting benchmarking. One YAML recipe → end-to-end study with bit-exact replication.

[![ci-core](https://github.com/NanyeonK/macroforecast/actions/workflows/ci-core.yml/badge.svg)](https://github.com/NanyeonK/macroforecast/actions/workflows/ci-core.yml)
[![ci-docs](https://github.com/NanyeonK/macroforecast/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/NanyeonK/macroforecast/actions/workflows/ci-docs.yml)
[![python](https://img.shields.io/badge/python-3.10+-blue)](#)

> **v0.6.0** — 953 tests passing locally as of 2026-05-05. The CI badges above
> reflect live build status and replace the previously static
> "tests-N passing / version-X" badges to keep the README from going stale.
>
> **Renamed from `macrocast` -> `macroforecast`** in v0.6.0 (PyPI
> namespace ownership). See ``CHANGELOG.md`` for the migration diff.

## Install

```bash
pip install macroforecast                    # core
pip install 'macroforecast[deep]'            # + torch / captum (LSTM / GRU / Transformer)
pip install 'macroforecast[xgboost,lightgbm]'  # + optional gradient-boosting backends
pip install 'macroforecast[tuning]'          # + optuna for bayesian_optimization
pip install 'macroforecast[shap]'            # + shap package for richer L7 figures
```

Or pin to a tagged release directly from GitHub:

```bash
pip install "git+https://github.com/NanyeonK/macroforecast.git@v0.6.3"
```

For development:

```bash
git clone https://github.com/NanyeonK/macroforecast.git
cd macroforecast
pip install -e ".[dev]"
```

## 5-line quickstart

```python
import macroforecast

result = macroforecast.run("recipe.yaml", output_directory="out/")
print(result.cells[0].sink_hashes)            # per-cell sink hashes
replication = macroforecast.replicate("out/manifest.json")
assert replication.sink_hashes_match           # bit-exact replication
```

A minimal recipe:

```yaml
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
1_data:
  fixed_axes: {custom_source_policy: custom_panel_only, frequency: monthly, horizon_set: custom_list}
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks: {l3_features_v1: {X_final: lag_x, y_final: y_h}, l3_metadata_v1: auto}
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit
      type: step
      op: fit_model
      params: {family: ridge, alpha: 0.1, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit, src_X]}
  sinks: {l4_forecasts_v1: predict, l4_model_artifacts_v1: fit, l4_training_metadata_v1: auto}
5_evaluation:
  fixed_axes: {primary_metric: mse}
```

## Architecture (12 layers)

```
L0 study setup → L1 data → L2 preprocess → L3 features (DAG, 37 ops)
                ↓                                ↓
                L1.5 / L2.5 / L3.5 (diagnostics, default-off)
                                                 ↓
              L4 model (35+ families) → L4.5 → L5 evaluation → L6 tests
                                                                    ↓
                                                  L7 interpretation → L8 output
```

See `plans/design/part1-4` for the canonical design tables.

## Operational coverage

> Before relying on advanced families/tests in a paper workflow, check
> [`docs/getting_started/runtime_support.md`](docs/getting_started/runtime_support.md)
> for the exact current path coverage. Some listed families are wired
> through legacy/specialized paths or optional extras, not necessarily
> through the minimal core runtime end-to-end.

* **35+ L4 families** — linear (8), tree / boosting (8), SVM (3), kNN, MLP, deep
  NN (3, opt-in via `[deep]`), AR_p, factor_augmented_ar, BVAR Minnesota / NIW,
  FAVAR, MRF GTVP (Coulombe 2024), DFM (Mariano-Murasawa MQ Kalman),
  quantile_regression_forest, bagging.
* **18 L7 figure types** — bar / heatmap / pdp / ALE / SHAP family /
  attribution / IRF with CI / decomp stacked / state choropleth.
* **L6 tests** — Diebold-Mariano (with HLN + HAC kernels), Clark-West,
  Giacomini-Rossi (simulated CVs), MCS / SPA / RC / StepM via stationary
  bootstrap, Pesaran-Timmermann, residual battery, density tests
  (PIT-Berkowitz / KS / Kupiec / Christoffersen / Engle-Manganelli DQ),
  Diebold-Mariano-Pesaran joint multi-horizon.
* **L1.G regimes** — none / NBER / user-provided / Hamilton MS / Tong SETAR /
  Bai-Perron breaks.
* **3 sweep kinds** — param-level (`{sweep: [...]}`), recipe-level (external
  axis), node-level (`sweep_groups`). Combine via grid (default) or zip.
* **Sub-cell parallelism** — `parallel_unit ∈ {cells, models, oos_dates,
  horizons, targets}`.
* **Bit-exact replication** — `replicate()` re-executes and verifies
  per-cell sink hashes match.

## Recipe gallery

`examples/recipes/` ships ~50 reference recipes; new in v0.3:

* `l4_minimal_ridge.yaml` — minimal ridge on a custom panel.
* `l4_random_forest.yaml`, `l4_xgboost.yaml`, `l4_lightgbm.yaml` (when extras installed).
* `l4_quantile_regression_forest.yaml` — Meinshausen QRF with quantile bands.
* `l4_bagging.yaml` — bootstrap-aggregated ridge.
* `l4_dfm_mariano_murasawa.yaml` — mixed-frequency DFM.
* `l4_macroeconomic_random_forest.yaml` — Coulombe MRF GTVP.
* `l4_ensemble_ridge_xgb_vs_ar1.yaml` — horse race with benchmark.

A replication script for Coulombe (2024) MRF on FRED-MD lives at
`examples/replication/coulombe_2024_mrf_fred_md.py`.

## Status levels

Two-value vocabulary (defined in `macroforecast.core.status`):
* **`operational`** — runtime executes the full design-spec procedure.
* **`future`** — schema-only; validator hard-rejects, runtime raises
  `NotImplementedError`.

The package shipped 19 honesty-pass demotions in v0.1.1; **all of them have
real implementations in v0.2 / v0.25 / v0.3** (every `future` flag in the
v0.1.1 audit table is now `operational`).

## Citing

If you use macroforecast in published work, please cite:

> macroforecast: Fair, reproducible macro forecasting benchmarking. v0.6.0, 2026.

## License

MIT
