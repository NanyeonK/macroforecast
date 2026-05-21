# Partial Layer Execution

Most recipe-author work is iterative: tweak one knob, rerun, inspect.
``macroforecast.run(...)`` executes the entire L1 → L8 cell loop, which is
overkill when you only care about whether the L2 outlier policy actually
flagged what you expected, or whether your new L3 op produces the right
``X_final``.

The ``macroforecast.core`` runtime exposes per-layer materialization helpers
that do exactly that. Each helper accepts the parsed recipe dict and the
upstream artifacts, and returns the same artifact dataclasses that the full
pipeline would have produced -- so you can inspect intermediate sinks
without invoking L4 / L5 / L6 / L7 / L8.

> See also: [Custom hooks](custom_hooks.md) -- developing a custom hook
> almost always involves L1+L2 once and then iterating on the layer the
> hook is registered against.

## Why this exists

| Use case | Helper(s) |
|---|---|
| "Did L2 actually flag my outliers?" | ``materialize_l1`` + ``materialize_l2``, then read ``L2CleanPanelArtifact.cleaning_log['steps']``. |
| "Does my new L3 op produce the X_final I expect?" | ``materialize_l1`` + ``materialize_l2`` + ``materialize_l3_minimal``; iterate on L3 only. |
| "Walk forward through L1 → L5 once, no L6/L7/L8" | ``execute_minimal_forecast`` -- the same helper that the integration tests use. |
| "Bridge from a custom-panel YAML straight to the L2 sink" | ``execute_l1_l2`` -- L1 + L2 only, no L3+ overhead. |
| "Replay one DAG node from cache" | ``execute_node`` -- foundation primitive used by ``execute_recipe``. |

## Public API surface

All six helpers live on ``macroforecast.core``:

```python
from macroforecast.core import (
    materialize_l1,
    materialize_l2,
    materialize_l3_minimal,
    materialize_l4_minimal,
    materialize_l5_minimal,
    execute_l1_l2,
    execute_minimal_forecast,
    execute_node,
)
```

| Function | Input | Returns |
|---|---|---|
| ``materialize_l1(recipe_root)`` | ``dict`` (parsed recipe) | ``(L1DataDefinitionArtifact, L1RegimeMetadataArtifact, dict[str, Any] resolved_axes)`` |
| ``materialize_l2(recipe_root, l1_artifact)`` | ``dict``, L1 artifact | ``(L2CleanPanelArtifact, L2ResolvedAxes)`` |
| ``materialize_l3_minimal(recipe_root, l1_artifact, l2_artifact)`` | ``dict``, L1, L2 | ``(L3FeaturesArtifact, L3MetadataArtifact)`` |
| ``materialize_l4_minimal(recipe_root, l3_features)`` | ``dict``, L3 features | ``(L4ForecastsArtifact, L4ModelArtifactsArtifact, L4TrainingMetadataArtifact)`` |
| ``materialize_l5_minimal(recipe_root, l1_artifact, l3_features, l4_forecasts, l4_models)`` | as listed | ``L5EvaluationArtifact`` |
| ``execute_l1_l2(recipe)`` | ``dict`` or YAML ``str`` | ``RuntimeResult`` with ``l1_data_definition_v1`` + ``l1_regime_metadata_v1`` + ``l2_clean_panel_v1`` (plus L1.5 / L2.5 diagnostics if enabled). |
| ``execute_minimal_forecast(recipe)`` | ``dict`` or YAML ``str`` | ``RuntimeResult`` with L1 → L5 sinks + any enabled L1.5 / L2.5 / L3.5 / L4.5 / L6 / L7 / L8 sinks. |
| ``execute_node(node, dag, runtime_context, cache_dir)`` | one DAG ``Node`` | the materialized node value (cached on disk). |

``RuntimeResult`` (from ``macroforecast.core``) is a frozen dataclass with
``artifacts: dict[str, Any]`` (sink_name → artifact),
``resolved_axes: dict[str, dict]`` (per-layer resolved axis values), and
``runtime_durations: dict[str, float]`` (L1 / L2 / L3 / ... wall-clock
seconds). Access a single sink with ``rt.sink("l2_clean_panel_v1")``.

## Worked sequence

The example below uses the same 10-row inline custom panel as
``examples/recipes/l4_minimal_ridge.yaml`` and walks through L1 → L3 by
hand.

```python
import macroforecast as mf
from macroforecast.core import (
    materialize_l1, materialize_l2, materialize_l3_minimal,
    materialize_l4_minimal, materialize_l5_minimal,
)

recipe = mf.core.parse_recipe_yaml(open("examples/recipes/l4_minimal_ridge.yaml").read())

# --- L1 ---------------------------------------------------------------
l1_artifact, regime_artifact, l1_axes = materialize_l1(recipe)
print("L1 frequency :", l1_artifact.frequency)
print("L1 target    :", l1_artifact.target)
print("L1 raw_panel :", l1_artifact.raw_panel.data.shape, "rows x cols")
print("L1 axes keys :", sorted(l1_axes)[:6])

# --- L2 ---------------------------------------------------------------
l2_artifact, l2_axes = materialize_l2(recipe, l1_artifact)
print("L2 panel     :", l2_artifact.panel.data.shape)
print("L2 cleaning_log steps:", [step for step in l2_artifact.cleaning_log["steps"]])
print("L2 n_outliers:", l2_artifact.n_outliers_flagged)
print("L2 n_imputed :", l2_artifact.n_imputed_cells)

# --- L3 ---------------------------------------------------------------
l3_features, l3_metadata = materialize_l3_minimal(recipe, l1_artifact, l2_artifact)
print("L3 X_final   :", l3_features.X_final.data.shape)
print("L3 y_final   :", l3_features.y_final.shape, l3_features.y_final.name)
print("L3 horizons  :", l3_features.horizon_set)
print("L3 sample_ix :", l3_features.sample_index[:3].tolist())
```

Expected output (the inline panel is deterministic):

```text
L1 frequency : monthly
L1 target    : y
L1 raw_panel : (12, 2) rows x cols
L1 axes keys : ['custom_source_policy', 'dataset', 'frequency', ...]
L2 panel     : (12, 2)
L2 cleaning_log steps: [{'transform': 'no_transform'}, {'outlier': 'none'}, ...]
L2 n_outliers: 0
L2 n_imputed : 0
L3 X_final   : (10, 1)
L3 y_final   : (10,) y
L3 horizons  : (1,)
L3 sample_ix : [Timestamp('2018-02-01 00:00:00'), Timestamp('2018-03-01 00:00:00'), ...]
```

The L3 step drops the first two rows (lag 1 + h=1 target shift), giving 10
rows of ``X_final`` / ``y_final``. From here you could continue:

```python
l4_forecasts, l4_models, l4_training = materialize_l4_minimal(recipe, l3_features)
print("L4 model_ids :", l4_forecasts.model_ids)
print("L4 forecasts :", list(l4_forecasts.forecasts.values())[:3])

l5_eval = materialize_l5_minimal(recipe, l1_artifact, l3_features, l4_forecasts, l4_models)
print("L5 metrics   :", l5_eval.metrics_table.head())
```

## Convenience helpers

When you do not need the artifact dataclasses directly, two helpers wrap the
materialize calls and return a ``RuntimeResult``:

```python
from macroforecast.core import execute_l1_l2, execute_minimal_forecast

# L1 + L2 only -- no L3+ overhead. Good for "did the cleaner do its job?"
rt = execute_l1_l2(open("examples/recipes/l2_minimal.yaml").read())
print("sinks       :", sorted(rt.artifacts))
panel = rt.sink("l2_clean_panel_v1").panel.data
print("panel shape :", panel.shape)
print("L2 axes     :", sorted(rt.resolved_axes["l2"])[:6])

# L1 → L5 (plus any enabled L1.5 / L2.5 / L3.5 / L4.5 / L6 / L7 / L8 sinks).
rt5 = execute_minimal_forecast(open("examples/recipes/l4_minimal_ridge.yaml").read())
print("durations   :", rt5.runtime_durations)
print("forecasts   :", rt5.sink("l4_forecasts_v1").model_ids)
```

Use ``execute_l1_l2`` while debugging L2 settings; use
``execute_minimal_forecast`` when you want a full minimal end-to-end pass
without going through ``execute_recipe`` (which writes a manifest and
manages the cell loop).

For the full multi-cell ``run(...)`` API see ``macroforecast.core.execute_recipe``.

## Schemas of the intermediate sinks

The artifacts are frozen dataclasses defined in ``macroforecast/core/types.py``.

### ``L1DataDefinitionArtifact``

| Field | Type | Notes |
|---|---|---|
| ``custom_source_policy`` | ``Literal["official_only", "custom_panel_only", "official_plus_custom"]`` | Resolved from L1 fixed_axes. |
| ``dataset`` | ``Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"] \| None`` | None for ``custom_panel_only``. |
| ``frequency`` | ``Literal["monthly", "quarterly"]`` | Resolved frequency. |
| ``vintage_policy`` | ``Literal["current_vintage", "real_time_alfred"] \| None`` | None for custom-panel runs. |
| ``target_structure`` | ``Literal["single_target", "multi_series_target"]`` | -- |
| ``target`` | ``str \| None`` | The single-target name (or first of ``targets``). |
| ``targets`` | ``tuple[str, ...]`` | The full list when ``target_structure='multi_series_target'``. |
| ``variable_universe`` | enum or ``None`` | -- |
| ``target_geography_scope`` / ``predictor_geography_scope`` | enums or ``None`` | FRED-SD only. |
| ``sample_start_rule`` / ``sample_end_rule`` | enums | -- |
| ``horizon_set`` / ``target_horizons`` | str / ``tuple[int, ...]`` | -- |
| ``regime_definition`` | ``str`` | ``"none"`` unless a regime axis is set. |
| ``raw_panel`` | ``Panel`` | The materialized predictor + target frame. ``raw_panel.data`` is a ``pd.DataFrame`` indexed by ``DatetimeIndex``; ``raw_panel.metadata.values`` carries the ``transform_codes`` dict when official t-codes are loaded. |
| ``leaf_config`` | ``dict[str, Any]`` | Echo of L1.leaf_config; useful for reading ``custom_panel_inline``, ``target_transformer``, etc. |

There is no separate ``target_series`` field; the target column lives inside
``raw_panel.data[target]`` until the L3 stage splits it out.

### ``L1RegimeMetadataArtifact``

| Field | Type | When ``None`` |
|---|---|---|
| ``definition`` | ``Literal["none", "external_nber", "external_user_provided", "estimated_markov_switching", "estimated_threshold", "estimated_structural_break"]`` | Always set. |
| ``n_regimes`` | ``int`` | -- |
| ``regime_label_series`` | ``Series \| None`` | ``None`` when ``definition='none'``. |
| ``regime_probabilities`` | ``Series \| None`` | ``None`` for non-MS regimes. |
| ``transition_matrix`` | ``Any \| None`` | ``None`` outside Markov-switching. |
| ``estimation_temporal_rule`` | ``str \| None`` | ``None`` for external regimes. |
| ``estimation_metadata`` | ``dict`` | Empty for external regimes. |

### ``L2CleanPanelArtifact``

Inherits from ``Panel``; therefore exposes ``data``, ``shape``, ``column_names``, ``index``, ``metadata`` directly **and** repeats them through the ``panel`` field.

| Field | Type | Notes |
|---|---|---|
| ``panel`` | ``Panel`` | The cleaned panel. ``panel.data`` is the post-pipeline DataFrame (``DatetimeIndex``, ``float64`` + ``pd.NA``). |
| ``column_metadata`` | ``dict[str, Any]`` | Per-column dtype string and other column-level audit info. |
| ``cleaning_log`` | ``dict[str, Any]`` | ``{"runtime": "core_l1_l2_materialization", "steps": [...]}``. Each step entry is a dict produced by the relevant stage (``transform``, ``outlier``, ``imputation``, ``frame_edge``, plus any ``custom_preprocessor`` / ``custom_postprocessor`` entries). |
| ``n_imputed_cells`` | ``int`` | Total cells the imputer filled. |
| ``n_outliers_flagged`` | ``int`` | Total cells the outlier policy touched. |
| ``n_truncated_obs`` | ``int`` | Rows the frame-edge policy dropped. |
| ``transform_map_applied`` | ``dict[str, int]`` | ``column -> applied tcode``. |
| ``cleaning_temporal_rules`` | ``dict[str, str]`` | Records the per-stage temporal rule (``imputation``, ``outlier``, ``frame_edge``). |
| ``upstream_hashes`` | ``dict[str, str]`` | Populated by the cell loop only -- empty in raw materialize calls. |

### ``L3FeaturesArtifact``

| Field | Type | Notes |
|---|---|---|
| ``X_final`` | ``Panel \| LaggedPanel \| Factor`` | The final predictor matrix. ``X_final.data`` is a ``pd.DataFrame`` with the post-DAG features. |
| ``y_final`` | ``Series`` | The final target series; ``y_final.name`` is the target column, ``y_final.metadata.values["data"]`` carries the raw ``pd.Series`` (and ``["raw_data"]`` when a target transformer is active). |
| ``sample_index`` | ``pd.DatetimeIndex \| None`` | The aligned index of ``X_final`` ∩ ``y_final`` after dropna. |
| ``horizon_set`` | ``tuple[int, ...]`` | Per-recipe target horizons. |
| ``upstream_hashes`` | ``dict[str, str]`` | Populated by the cell loop only. |

### ``L3MetadataArtifact``

| Field | Type | Notes |
|---|---|---|
| ``column_lineage`` | ``dict[str, ColumnLineage]`` | column → ``(source_variable_ids, step_chain, pipeline_id, cascade_depth, output_type)``. |
| ``pipeline_definitions`` | ``dict[str, PipelineDefinition]`` | One entry per L3 pipeline. |
| ``cascade_graph`` | ``dict[str, tuple[str, ...]]`` | Cascade-DAG adjacency. |
| ``transform_chain`` | ``dict[str, tuple[StepRef, ...]]`` | Per-column step chain. |
| ``source_variables`` | ``dict[str, tuple[str, ...]]`` | Per-column source variable ids. |

### ``L4ForecastsArtifact``

| Field | Type | Notes |
|---|---|---|
| ``forecasts`` | ``dict[tuple[str, str, int, Any], float]`` | ``(model_id, target, horizon, origin) -> point forecast``. |
| ``forecast_intervals`` | ``dict[tuple[str, str, int, Any, float], float]`` | ``(model_id, target, horizon, origin, alpha) -> quantile``. Empty for point recipes. |
| ``forecast_object`` | ``Literal["point", "quantile", "density"]`` | -- |
| ``sample_index`` | ``pd.DatetimeIndex \| None`` | Sorted unique forecast origins. |
| ``targets`` / ``horizons`` / ``model_ids`` | ``tuple[str, ...]`` / ``tuple[int, ...]`` / ``tuple[str, ...]`` | -- |
| ``upstream_hashes`` | ``dict[str, str]`` | Populated by the cell loop only. |

### ``L4ModelArtifactsArtifact``

| Field | Type | Notes |
|---|---|---|
| ``artifacts`` | ``dict[str, ModelArtifact]`` | model_id → fitted ``ModelArtifact`` (``family``, ``framework``, ``fitted_object``, ``fit_metadata``, ``feature_names``). |
| ``is_benchmark`` | ``dict[str, bool]`` | model_id → ``is_benchmark`` flag. |
| ``upstream_hashes`` | ``dict[str, str]`` | -- |

### ``L4TrainingMetadataArtifact``

Records ``forecast_origins``, ``refit_origins``, ``training_window_per_origin``,
``runtime_per_origin``, ``cache_hits_per_origin``, ``tuning_log``,
``upstream_hashes`` -- one row per ``(model_id, origin)`` walk-forward step.

### ``L5EvaluationArtifact``

| Field | Type | When empty |
|---|---|---|
| ``metrics_table`` | ``pd.DataFrame`` | Per-(model, target, horizon) metric rows. |
| ``ranking_table`` | ``pd.DataFrame`` | Sorted by primary metric. |
| ``benchmark_relative_metrics`` | ``dict`` | -- |
| ``per_regime_metrics`` | ``dict \| None`` | ``None`` when ``regime_definition='none'``. |
| ``decomposition_results`` | ``dict \| None`` | ``None`` when no ``decomposition`` axis is set. |
| ``per_state_metrics`` | ``dict \| None`` | FRED-SD only. |
| ``report_artifacts`` | ``dict[str, Any]`` | -- |
| ``per_origin_loss_panel`` | ``pd.DataFrame`` | Empty when L5 took the summary-only fallback path. |
| ``l5_axis_resolved`` | ``dict`` | Resolved L5 axes. |

## Use case 1: Did my outlier policy actually flag values?

```python
import macroforecast as mf
from macroforecast.core import materialize_l1, materialize_l2

recipe_str = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
1_data:
  fixed_axes: {custom_source_policy: custom_panel_only, frequency: monthly, horizon_set: custom_list}
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
             2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 99.0]
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: zscore_threshold
    outlier_action: flag_as_nan
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
"""
recipe = mf.core.parse_recipe_yaml(recipe_str)
l1_artifact, _, _ = materialize_l1(recipe)
l2_artifact, _ = materialize_l2(recipe, l1_artifact)

print("flagged cells :", l2_artifact.n_outliers_flagged)
for step in l2_artifact.cleaning_log["steps"]:
    print(" -", step)
```

The ``cleaning_log['steps']`` entry for the outlier stage tells you exactly
which policy ran, what action it took, and how many cells it flagged.

## Use case 2: Iterating on L3 only

```python
import macroforecast as mf
from macroforecast.core import materialize_l1, materialize_l2, materialize_l3_minimal

recipe = mf.core.parse_recipe_yaml(open("examples/recipes/l3_minimal_lag_only.yaml").read())

# Run L1 + L2 once; cache the artifacts.
l1_artifact, _, _ = materialize_l1(recipe)
l2_artifact, _ = materialize_l2(recipe, l1_artifact)

# Iterate on L3 -- swap ops, change params, re-run only this step.
recipe["3_feature_engineering"]["nodes"][2]["params"]["n_lag"] = 3
l3_features, l3_metadata = materialize_l3_minimal(recipe, l1_artifact, l2_artifact)
print("X_final shape:", l3_features.X_final.data.shape)

recipe["3_feature_engineering"]["nodes"][2]["params"]["n_lag"] = 6
l3_features, l3_metadata = materialize_l3_minimal(recipe, l1_artifact, l2_artifact)
print("X_final shape:", l3_features.X_final.data.shape)
```

Each L3 iteration reuses the same ``l1_artifact`` and ``l2_artifact``, so
the experiment is bounded by L3 cost rather than full L1 → L8 cost.

When developing a custom L3 ``feature_block`` or ``feature_combiner``
([Custom hooks](custom_hooks.md)), this loop is the canonical inner cycle:
register the callable once, then call ``materialize_l3_minimal`` repeatedly
with different parameter values.

## ``execute_node`` -- the cache-aware primitive

``execute_node(node, dag, runtime_context, cache_dir)`` is the foundation
primitive that ``execute_recipe`` calls per DAG node. It hashes the node +
its inputs, checks the on-disk cache at
``cache_dir/nodes/<node_hash>/result.pickle``, returns the cached value if
present, and otherwise computes and caches the result. Most recipe authors
do not need ``execute_node`` directly -- the materialize helpers above cover
inspection use cases. Reach for it only when you are writing a custom
runtime layer (rare).
