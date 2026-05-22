# Example Walkthrough — minimal ridge

This walkthrough opens the smallest bundled recipe
[`examples/recipes/l4_minimal_ridge.yaml`](https://github.com/NanyeonK/macroforecast/blob/main/examples/recipes/l4_minimal_ridge.yaml)
and explains every layer choice end-to-end. Use it as a **template** when
writing your own replication page (see [study_1.md](study_1.md) etc.).

## Reproduce in two commands

```bash
macroforecast run examples/recipes/l4_minimal_ridge.yaml -o out/walkthrough
macroforecast replicate out/walkthrough/manifest.json
```

The first command writes per-cell artifacts and `manifest.json` to
`out/walkthrough/`. The second re-runs from the manifest and verifies every
sink hash matches bit-for-bit.

## Layer 0 — study setup

```yaml
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
  leaf_config: {random_seed: 42}
```

- `failure_policy: fail_fast` — abort the sweep on first cell failure
  (default while developing a recipe; switch to `continue_on_failure`
  for large unsupervised sweeps).
- `reproducibility_mode: seeded_reproducible` + `random_seed: 42` — every
  stochastic step receives a deterministic seed derived from the L0 seed
  plus the cell index. This is what makes
  `macroforecast.replicate(manifest_path)` bit-exact.

## Layer 1 — data

```yaml
1_data:
  fixed_axes: {custom_source_policy: custom_panel_only, frequency: monthly, horizon_set: custom_list}
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, ..., 2018-12-01]
      y:   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      x1:  [0.5, 1.0, 1.5, ..., 6.0]
```

- `custom_source_policy: custom_panel_only` — bypass the FRED loaders;
  use the inline panel. Real recipes use `dataset: fred_md` (or
  `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd`) plus
  `start` / `end`.
- `frequency: monthly` — the panel calendar; required when not derivable
  from the dataset.
- `horizon_set: custom_list` + `target_horizons: [1]` — predict h=1 only.
  Multi-horizon recipes use `[1, 3, 6, 12]` (monthly) or `[1, 2, 4, 8]`
  (quarterly).

## Layer 2 — preprocessing

```yaml
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
```

The minimal recipe disables every L2 stage (panel is already clean). A
realistic FRED-MD recipe would use:

- `transform_policy: apply_official_tcode` — apply McCracken-Ng codes.
- `outlier_policy: mccracken_ng_iqr` + `outlier_action: flag_as_nan`.
- `imputation_policy: em_factor` — McCracken-Ng PCA-EM imputation.
- `frame_edge_policy: truncate_to_balanced`.

For FRED-SD / mixed-frequency studies see
[`mixed_frequency_representation`](../encyclopedia/l2/axes/mixed_frequency_representation.md).

## Layer 3 — feature engineering DAG

```yaml
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
```

- L3 is a DAG. `src_X` / `src_y` pull predictors / target from the L2
  clean panel.
- `lag_x` step: a single 1-period lag of every predictor column.
- `y_h` step: build the L3 target as a direct h=1 forecast (no lead /
  cumulative).
- Sinks: the DAG terminates at `l3_features_v1` (an `(X_final,
  y_final)` pair) plus `l3_metadata_v1` (lineage for L7 attribution).

Real recipes compose richer DAGs: `pca` reduction,
`ma_increasing_order` (MARX), `scaled_pca`, `feature_selection`, etc.
See the [encyclopedia L3 page](../encyclopedia/l3/index.md) for the 37
operational ops.

## Layer 4 — forecasting model

```yaml
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params:
        family: ridge
        alpha: 1.0
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      is_benchmark: true
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
```

- `family: ridge` with `alpha: 1.0` — standard L2-regularised OLS.
  Replace with `lasso`, `elastic_net`, `ar_p`, `random_forest`,
  `xgboost`, `bayesian_ridge`, `bvar_minnesota`,
  `macroeconomic_random_forest`, `dfm_mixed_mariano_murasawa`, ... see
  the [encyclopedia L4 page](../encyclopedia/l4/index.md) for all 35+.
- `forecast_strategy: direct` — train one model per horizon (vs.
  `iterated` which recursively rolls h=1 forecasts).
- `training_start_rule: expanding` + `refit_policy: every_origin` —
  expanding-window walk-forward, refit at every OOS origin.
- `search_algorithm: none` — no hyperparameter tuning. Set to
  `bayesian_optimization` or `cv_path` for tuning.
- `is_benchmark: true` — flags this model as the L5 / L6 reference.
  Required when comparing models via `compare_models([...])`.

## Layer 5 — evaluation

The minimal recipe defaults to `primary_metric: mse`. Realistic
recipes add:

```yaml
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
    relative_metrics: [relative_mse, r2_oos]
    benchmark_scope: per_target_horizon
    ranking: by_relative_metric
```

For statistical inference (DM / MCS / SPA / Reality Check), enable
L6 — see the [encyclopedia L6 page](../encyclopedia/l6/index.md).

## Output (Layer 8)

`manifest.json` records the canonical-key-ordered recipe, per-cell
sink hashes, and provenance (Python / package versions, lockfile, git
SHA, OS, CPU). `out/walkthrough/cell_001/` carries:

- `forecasts.csv` — y_true / y_pred per origin
- `metrics.json` — point / relative metrics
- `cell_manifest.json`
- `figures/` — when L7 importance is enabled

## Replication contract

```bash
macroforecast replicate out/walkthrough/manifest.json
```

Returns a `ReplicationResult` with `recipe_match=True` and
`sink_hashes_match=True` when every artifact reproduces bit-for-bit.
This is the package's core promise; if it fails, file an issue.

## Programmatic equivalent (`mf.Experiment`)

```python
import macroforecast as mf

# One-shot
result = mf.forecast(
    dataset="fred_md",
    target="INDPRO",
    horizons=[1, 3, 6],
    model_family="ridge",
    output_directory="out/quickstart",
)
print(result.metrics)

# Builder (multi-cell horse race)
exp = (
    mf.Experiment(dataset="fred_md", target="INDPRO", horizons=[1, 3, 6])
      .compare_models(["ridge", "lasso", "ar_p"])
)
horse_race = exp.run(output_directory="out/horse_race")
print(horse_race.ranking)
print(horse_race.replicate().sink_hashes_match)  # True
```

## How to write your own replication page

1. Copy this file's structure into `study_<n>.md`.
2. Fill **Paper / Year / Source** in the page header.
3. Drop the recipe YAML into
   `examples/recipes/replications/study_<n>.yaml` and commit it.
4. Run the recipe + replicate; capture the per-cell metrics in the
   "Expected artifacts" section.
5. Note which axes are paper-faithful vs simplified (e.g. an
   approximation of the published method).

See [study_1.md](study_1.md), [study_2.md](study_2.md),
[study_3.md](study_3.md), [study_4.md](study_4.md) for the four
maintainer replications (filled in as each study runs).
