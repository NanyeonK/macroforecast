# macroforecast.pipeline

[Back to reference](index.md)

Comprehensive pseudo-out-of-sample pipeline specs, execution, evaluation, interpretation, and result stores.

Guide context: [../guide/index.md](../guide/index.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `TCODE_TARGET_MAP` | data | dict() -> new empty dictionary |
| `auto_parallelism` | function | Return ``(cell_workers, model_threads)`` saturating ``cores``. |
| `rescore` | function | Re-score a saved pipeline run from its checkpoint directory alone. |
| `result_store_summary` | function | Summarise result-store manifests, one row per readable cell manifest. |
| `purge_result_store` | function | Delete result-store cells matching the supplied filters and return a count. |
| `run_arms` | function | Execute every cell and concatenate into one master forecast frame. |
| `run_pipeline` | function | Execute the full pipeline: run arms, evaluate, and assemble a PipelineReport. |
| `interpret_pipeline` | function | Interpret the interpretable arms of a completed pipeline run. |
| `PipelineReport` | class | Standard pipeline output (mutable: interpretation is filled in later). |
| `evaluate` | function | Run the full evaluation: combinations -> accuracy + significance + MCS |
| `evaluate_cross_policy` | function | Score every ``(arm, forecast_policy)`` contender against ONE benchmark fixed |
| `apply_combinations` | function | Append cross-arm combination contenders to the master forecast frame. |
| `Arm` | class | A target-agnostic configuration: preprocessing + features + a single model. |
| `CombinationContender` | class | A forecast combination that becomes an additional contender. |
| `DIRECT_POLICY_GUARD_MODELS` | data | frozenset() -> empty frozenset object |
| `EvalSpec` | class | Automatic evaluation and significance-testing configuration. |
| `InterpretSpec` | class | ML interpretation request for an arm (deferred, multi-method). |
| `PipelineSpec` | class | Validated, frozen configuration produced by :func:`pipeline_spec`. |
| `ResolvedTarget` | class | A target with its forecast policy and transform resolved. |
| `SubsampleWindow` | class | Evaluation-window filter applied to forecast target dates. |
| `TargetSpec` | class | A forecast target and how its forecast object is defined. |
| `contender_names` | function | Display contender labels for an arm. A contender IS exactly an arm. |
| `is_vintage_aware` | function | Return whether a pipeline spec runs against per-origin vintage data. |
| `model_arms` | function | Build one :class:`Arm` per model for a pure model comparison. |
| `pipeline_spec` | function | Validate and build a :class:`PipelineSpec`. |
| `resolve_target` | function | Resolve a target to its (forecast_policy, target_transform). |

## Data And Module Values

### `TCODE_TARGET_MAP`

Kind: `data`

```python
TCODE_TARGET_MAP = dict(7 entries: 1, 2, 3, 4, 5, 6, 7)
```
### `DIRECT_POLICY_GUARD_MODELS`

Kind: `data`

```python
DIRECT_POLICY_GUARD_MODELS = frozenset({'arima', 'auto_arima', 'bvar_minnesota', 'bvar_normal_inverse_wishart', 'dfm_mixed_mariano_murasawa', 'dfm_unrestricted_midas', 'ets', 'favar', 'holt_winters', 'naive', 'random_walk_drift', 'seasonal_naive', 'stlf', 'theta_met...
```

## Callable And Class Reference

### auto_parallelism

Qualified name: `macroforecast.pipeline.parallelism.auto_parallelism`

#### Signature

```python
macroforecast.pipeline.auto_parallelism(n_cells: int, *, cores: int | None = None, reserve: int = 0) -> tuple[int, int]
```

#### Description

Return ``(cell_workers, model_threads)`` saturating ``cores``.

Cell-level parallelism comes first: one worker per cell up to the core budget.
Whatever cores remain become per-cell model-internal threads for the
parallelizable models (random_forest, gradient_boosting, xgboost, lightgbm).
The product ``cell_workers * model_threads`` is always ``<= cores``, so the CPU
is never oversubscribed.

Parameters
n_cells:
    Number of independent ``(arm x target x horizon)`` cells to schedule.
cores:
    Core budget. Defaults to the affinity count
    (``len(os.sched_getaffinity(0))``) -- the cores this process may actually
    run on, which respects cgroup/taskset pinning.
reserve:
    Cores to hold back (e.g. for the parent process / other work) before the
    split. ``cores`` is reduced by ``reserve`` (floored at 1).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_cells` | positional or keyword | `int` | `required` |
| `cores` | keyword only | `int \| None` | `None` |
| `reserve` | keyword only | `int` | `0` |

#### Returns

`tuple[int, int]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.auto_parallelism(...)
```
### rescore

Qualified name: `macroforecast.pipeline.rescore.rescore`

#### Signature

```python
macroforecast.pipeline.rescore(checkpoint_dir: str | Path, spec: "'Any'") -> "'Any'"
```

#### Description

Re-score a saved pipeline run from its checkpoint directory alone.

Walks every (target, arm, horizon) cell ``spec`` describes, loads that cell's
persisted lean forecast records from
``<checkpoint_dir>/<target>__<arm>/h<h>/origin_*.parquet`` (the exact layout
``run_pipeline(spec)`` writes when ``spec.checkpoint_dir`` is set -- see
``pipeline/run.py::_cell_checkpoint_path`` and ``forecasting/runner.py``'s
per-horizon ``h<h>`` subdirectory), reassembles the master forecast frame
(attaching ``arm``/``contender`` from ``spec`` itself -- the lean checkpoint
schema does not carry them, only ``target``/``horizon``/``model``/etc.), and
runs the standard ``evaluate()`` used by ``run_pipeline``.

Parameters
checkpoint_dir:
    The directory ``spec.checkpoint_dir`` pointed at during the original run
    (or any directory with that same layout).
spec:
    The ``PipelineSpec`` the checkpointed run used -- NOT necessarily the same
    object with ``checkpoint_dir`` set to this path; ``rescore`` reads records
    from ``checkpoint_dir`` regardless of what ``spec.checkpoint_dir`` says.
    Every field that determines a cell's identity (targets, arms, horizons)
    must match the original run, or cells will not be found.

Returns
PipelineReport
    The same report type ``run_pipeline`` returns, with the evaluation fields
    (``forecasts``, ``accuracy``, ``significance``, ``mcs``, ``density``,
    ``calibration``) populated exactly as a live run would produce from the
    same forecasts -- ``density``/``calibration`` are only non-empty when
    ``spec.evaluation.metrics``/``tests`` requests a density metric or
    calibration test AND the checkpointed forecasts actually carry the
    needed ``variance_prediction``/``quantile_predictions`` columns (see
    ``forecasting/checkpoint.py``'s lean schema). Fields that require
    having actually EXECUTED the run are explicitly absent/best-effort:

    - ``interpretation`` is always ``None`` (interpretation needs the fitted
      model; re-fit via ``interpret_pipeline`` on a live run instead).
    - ``failed_cells`` is always empty -- a cell that failed during the
      original run wrote no checkpoint files and is indistinguishable here
      from a cell that never ran.
    - ``empty_cells`` is best-effort: a (target, horizon) is reported empty
      only when NONE of its arms produced any checkpoint rows; an arm that
      failed outright (vs. produced zero rows) cannot be distinguished from
      one that was simply never run with this checkpoint_dir.
    - ``provenance``/``leakage_audit`` carry a ``rescored_from`` marker and a
      note that they were not recomputed from a live run.

Raises
ValueError
    If no cell under ``checkpoint_dir`` yields any checkpoint records at all
    (an empty or entirely-mismatched directory) -- a clear, actionable error
    instead of a silently-empty report.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `checkpoint_dir` | positional or keyword | `str \| Path` | `required` |
| `spec` | positional or keyword | `'Any'` | `required` |

#### Returns

`'Any'`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.rescore(...)
```
### result_store_summary

Qualified name: `macroforecast.pipeline.result_store.result_store_summary`

#### Signature

```python
macroforecast.pipeline.result_store_summary(store: str | Path) -> pd.DataFrame
```

#### Description

Summarise result-store manifests, one row per readable cell manifest.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `store` | positional or keyword | `str \| Path` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.result_store_summary(...)
```
### purge_result_store

Qualified name: `macroforecast.pipeline.result_store.purge_result_store`

#### Signature

```python
macroforecast.pipeline.purge_result_store(store: str | Path, *, before: str | datetime | None = None, version: str | None = None, digests: Sequence[str] | None = None) -> int
```

#### Description

Delete result-store cells matching the supplied filters and return a count.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `store` | positional or keyword | `str \| Path` | `required` |
| `before` | keyword only | `str \| datetime \| None` | `None` |
| `version` | keyword only | `str \| None` | `None` |
| `digests` | keyword only | `Sequence[str] \| None` | `None` |

#### Returns

`int`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.purge_result_store(...)
```
### run_arms

Qualified name: `macroforecast.pipeline.run.run_arms`

#### Signature

```python
macroforecast.pipeline.run_arms(spec: PipelineSpec) -> pd.DataFrame
```

#### Description

Execute every cell and concatenate into one master forecast frame.

(The name is retained for back-compat; the managed unit is a cell -- one arm
applied to one target over the window for a horizon-group -- not an arm.)
Columns include arm, model, contender, target, horizon, origin, date,
prediction, actual, target_transform, forecast_policy. Each cell runs its arm
with its own preprocessing/features/model against its target's resolved
(forecast_policy, target_transform).

The pipeline MANAGES atomic ``run()`` calls over (target, arm, horizon-group)
cells. When ``spec.n_jobs > 1`` the cells run across a process pool, one horizon
per cell; the result is numerically identical to the serial multi-horizon path.
Per-cell failures are isolated -- see :func:`run_pipeline` /
``PipelineReport.failed_cells`` for how they are surfaced.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `spec` | positional or keyword | `PipelineSpec` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.run_arms(...)
```
### run_pipeline

Qualified name: `macroforecast.pipeline.run.run_pipeline`

#### Signature

```python
macroforecast.pipeline.run_pipeline(spec: PipelineSpec)
```

#### Description

Execute the full pipeline: run arms, evaluate, and assemble a PipelineReport.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `spec` | positional or keyword | `PipelineSpec` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.run_pipeline(...)
```
### interpret_pipeline

Qualified name: `macroforecast.pipeline.interpret.interpret_pipeline`

#### Signature

```python
macroforecast.pipeline.interpret_pipeline(report: PipelineReport, *, methods: "'tuple[str, ...] | None'" = None, which_fit: str = "auto", arms: "'tuple[str, ...] | None'" = None, background: Any = None) -> dict[str, Any]
```

#### Description

Interpret the interpretable arms of a completed pipeline run.

``methods`` overrides each arm's ``InterpretSpec.methods`` when given. Returns a
nested dict ``{arm: {model[:target]: {method: table}}}`` stored on
``report.interpretation``. One arm failing to fit degrades to an error frame
and never aborts the others.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `report` | positional or keyword | `PipelineReport` | `required` |
| `methods` | keyword only | `'tuple[str, ...] \| None'` | `None` |
| `which_fit` | keyword only | `str` | `"auto"` |
| `arms` | keyword only | `'tuple[str, ...] \| None'` | `None` |
| `background` | keyword only | `Any` | `None` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.interpret_pipeline(...)
```
### PipelineReport

Qualified name: `macroforecast.pipeline.spec.PipelineReport`

#### Signature

```python
macroforecast.pipeline.PipelineReport(forecasts: "'Any'", accuracy: "'Any'", significance: "'Any'", mcs: "'Any'", provenance: Mapping[str, Any] = <factory>, leakage_audit: Mapping[str, Any] = <factory>, interpretation: Mapping[str, Any] | None = None, model_store: str = "trained_model", spec: "'Any'" = None, failed_cells: "'Sequence[Mapping[str, Any]]'" = <factory>, empty_cells: "'Sequence[Mapping[str, Any]]'" = <factory>, density: "'Any'" = None, calibration: "'Any'" = None) -> None
```

#### Description

Standard pipeline output (mutable: interpretation is filled in later).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `'Any'` | `required` |
| `accuracy` | positional or keyword | `'Any'` | `required` |
| `significance` | positional or keyword | `'Any'` | `required` |
| `mcs` | positional or keyword | `'Any'` | `required` |
| `provenance` | positional or keyword | `Mapping[str, Any]` | `<factory>` |
| `leakage_audit` | positional or keyword | `Mapping[str, Any]` | `<factory>` |
| `interpretation` | positional or keyword | `Mapping[str, Any] \| None` | `None` |
| `model_store` | positional or keyword | `str` | `"trained_model"` |
| `spec` | positional or keyword | `'Any'` | `None` |
| `failed_cells` | positional or keyword | `'Sequence[Mapping[str, Any]]'` | `<factory>` |
| `empty_cells` | positional or keyword | `'Sequence[Mapping[str, Any]]'` | `<factory>` |
| `density` | positional or keyword | `'Any'` | `None` |
| `calibration` | positional or keyword | `'Any'` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.PipelineReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `forecasts` | `'Any'` | `required` |
| `accuracy` | `'Any'` | `required` |
| `significance` | `'Any'` | `required` |
| `mcs` | `'Any'` | `required` |
| `provenance` | `Mapping[str, Any]` | `default_factory` |
| `leakage_audit` | `Mapping[str, Any]` | `default_factory` |
| `interpretation` | `Mapping[str, Any] \| None` | `None` |
| `model_store` | `str` | `"trained_model"` |
| `spec` | `'Any'` | `None` |
| `failed_cells` | `'Sequence[Mapping[str, Any]]'` | `default_factory` |
| `empty_cells` | `'Sequence[Mapping[str, Any]]'` | `default_factory` |
| `density` | `'Any'` | `None` |
| `calibration` | `'Any'` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_frame` | `to_frame(self) -> "'Any'"` | Return the master forecast frame. |
### evaluate

Qualified name: `macroforecast.pipeline.evaluate.evaluate`

#### Signature

```python
macroforecast.pipeline.evaluate(master: pd.DataFrame, spec: PipelineSpec) -> dict[str, pd.DataFrame]
```

#### Description

Run the full evaluation: combinations -> accuracy + significance + MCS
+ density + calibration.

``density``/``calibration`` are opt-in via ``EvalSpec.metrics``/``tests``
(see ``density_table``/``calibration_table``); a default ``EvalSpec`` never
computes them (empty frames), so ``forecasts``/``accuracy``/``significance``/
``mcs`` stay byte-identical to before these two keys existed.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `master` | positional or keyword | `pd.DataFrame` | `required` |
| `spec` | positional or keyword | `PipelineSpec` | `required` |

#### Returns

`dict[str, pd.DataFrame]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.evaluate(...)
```
### evaluate_cross_policy

Qualified name: `macroforecast.pipeline.evaluate.evaluate_cross_policy`

#### Signature

```python
macroforecast.pipeline.evaluate_cross_policy(forecasts: pd.DataFrame, *, benchmark: str, benchmark_policy: str, policy_column: str = "forecast_policy", separator: str = "::") -> pd.DataFrame
```

#### Description

Score every ``(arm, forecast_policy)`` contender against ONE benchmark fixed
to a single policy -- the common-denominator convention.

Use this when the benchmark you want lives under a different forecast policy
than the contenders. The GCLS (2021) appendix, for instance, scores both its
direct and its path-average tables against a single FM benchmark, the direct
FM. Running several policies for one target in a single spec pools the
policies' rows for the same arm, because :func:`accuracy_table` keys the
relative metrics on contender name within a ``(target, horizon)`` cell and
does not split on policy. This helper does the qualification for you: it makes
each ``(arm, forecast_policy)`` a distinct contender, scores all of them
against the single ``(benchmark, benchmark_policy)`` arm, and returns a tidy
accuracy table that keeps ``arm`` and ``forecast_policy`` as their own columns.

Parameters
forecasts:
    The master forecast frame (``report.forecasts``). Must carry the columns
    ``target, horizon, origin, prediction, actual, contender`` and
    ``policy_column``.
benchmark:
    The arm name of the benchmark (e.g. ``"FM"``).
benchmark_policy:
    The forecast policy whose copy of ``benchmark`` is THE denominator for
    every contender (e.g. ``"direct_average"`` for the direct FM).
policy_column:
    Column holding the per-row forecast policy. Default ``"forecast_policy"``.
separator:
    Joins arm and policy into the qualified contender key, then splits them
    back out. Must not appear in any arm name or policy value; the default
    ``"::"`` is safe for the underscore-bearing policy names
    (``direct_average``, ``path_average``).

Returns
The accuracy table with one row per ``(target, horizon, arm, forecast_policy)``
-- ``relative_mse`` / ``r2_oos`` / ``rmse`` computed pairwise against the fixed
benchmark -- plus ``arm`` and ``forecast_policy`` columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `pd.DataFrame` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `benchmark_policy` | keyword only | `str` | `required` |
| `policy_column` | keyword only | `str` | `"forecast_policy"` |
| `separator` | keyword only | `str` | `"::"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.evaluate_cross_policy(...)
```
### apply_combinations

Qualified name: `macroforecast.pipeline.evaluate.apply_combinations`

#### Signature

```python
macroforecast.pipeline.apply_combinations(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame
```

#### Description

Append cross-arm combination contenders to the master forecast frame.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `master` | positional or keyword | `pd.DataFrame` | `required` |
| `spec` | positional or keyword | `PipelineSpec` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.apply_combinations(...)
```
### Arm

Qualified name: `macroforecast.pipeline.spec.Arm`

#### Signature

```python
macroforecast.pipeline.Arm(name: str, model: Any, preprocessing: Any | None = None, preprocessing_policy: Any | None = None, features: Any | None = None, feature_policy: Any | None = None, params: Mapping[str, Any] | None = None, model_selection: Any | None = None, model_selection_metric: str = "mse", window: Any | None = None, interpret: InterpretSpec | tuple[str, ...] | None = None, is_benchmark: bool = False, nested_in_benchmark: bool = False, metadata: Mapping[str, Any] = <factory>) -> None
```

#### Description

A target-agnostic configuration: preprocessing + features + a single model.

An arm is NOT itself a cell. Applied to a target and a horizon it forms one
cell (executed by one ``run()`` call); in the evaluation it appears as exactly
one contender (one arm = one contender).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `model` | positional or keyword | `Any` | `required` |
| `preprocessing` | positional or keyword | `Any \| None` | `None` |
| `preprocessing_policy` | positional or keyword | `Any \| None` | `None` |
| `features` | positional or keyword | `Any \| None` | `None` |
| `feature_policy` | positional or keyword | `Any \| None` | `None` |
| `params` | positional or keyword | `Mapping[str, Any] \| None` | `None` |
| `model_selection` | positional or keyword | `Any \| None` | `None` |
| `model_selection_metric` | positional or keyword | `str` | `"mse"` |
| `window` | positional or keyword | `Any \| None` | `None` |
| `interpret` | positional or keyword | `InterpretSpec \| tuple[str, ...] \| None` | `None` |
| `is_benchmark` | positional or keyword | `bool` | `False` |
| `nested_in_benchmark` | positional or keyword | `bool` | `False` |
| `metadata` | positional or keyword | `Mapping[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.Arm(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `model` | `Any` | `required` |
| `preprocessing` | `Any \| None` | `None` |
| `preprocessing_policy` | `Any \| None` | `None` |
| `features` | `Any \| None` | `None` |
| `feature_policy` | `Any \| None` | `None` |
| `params` | `Mapping[str, Any] \| None` | `None` |
| `model_selection` | `Any \| None` | `None` |
| `model_selection_metric` | `str` | `"mse"` |
| `window` | `Any \| None` | `None` |
| `interpret` | `InterpretSpec \| tuple[str, ...] \| None` | `None` |
| `is_benchmark` | `bool` | `False` |
| `nested_in_benchmark` | `bool` | `False` |
| `metadata` | `Mapping[str, Any]` | `default_factory` |
### CombinationContender

Qualified name: `macroforecast.pipeline.spec.CombinationContender`

#### Signature

```python
macroforecast.pipeline.CombinationContender(name: str, method: str, over: tuple[str, ...] | str = "all", by: tuple[str, ...] = ('target', 'horizon'), params: Mapping[str, Any] | None = None, weight_window: int | None = None, shrink_to_equal: float | None = None) -> None
```

#### Description

A forecast combination that becomes an additional contender.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `method` | positional or keyword | `str` | `required` |
| `over` | positional or keyword | `tuple[str, ...] \| str` | `"all"` |
| `by` | positional or keyword | `tuple[str, ...]` | `("target", "horizon")` |
| `params` | positional or keyword | `Mapping[str, Any] \| None` | `None` |
| `weight_window` | positional or keyword | `int \| None` | `None` |
| `shrink_to_equal` | positional or keyword | `float \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.CombinationContender(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `method` | `str` | `required` |
| `over` | `tuple[str, ...] \| str` | `"all"` |
| `by` | `tuple[str, ...]` | `("target", "horizon")` |
| `params` | `Mapping[str, Any] \| None` | `None` |
| `weight_window` | `int \| None` | `None` |
| `shrink_to_equal` | `float \| None` | `None` |
### EvalSpec

Qualified name: `macroforecast.pipeline.spec.EvalSpec`

#### Signature

```python
macroforecast.pipeline.EvalSpec(benchmark: str, metrics: tuple[str | Callable[..., float], ...] = ('rmse', 'relative_mse', 'r2_oos'), tests: tuple[str, ...] = ('dm', 'cw', 'mcs'), by: tuple[str, ...] = ('target', 'horizon'), primary_axis: str = "contender", cw_for_nested: bool = True, mcs_alpha: float = 0.1, mcs_method: str = "iterative", multiple_testing: str | None = None, subsamples: Mapping[str, SubsampleWindow] | None = None, dm_kwargs: Mapping[str, Any] = <factory>, loss: Callable[[Any, Any], Any] | None = None, test_options: Mapping[str, Mapping[str, Any]] = <factory>, calibration_alpha: float = 0.05) -> None
```

#### Description

Automatic evaluation and significance-testing configuration.

``metrics`` entries are either a name resolved through the metric registry
(:func:`macroforecast.metrics.get_metric`, e.g. ``"mae"``, ``"mape"``) or a
callable ``metric(y_true, y_pred) -> float`` named by its ``__name__``.
``accuracy_table`` computes one column per listed metric, per contender, on
the same pairwise-vs-benchmark sample it always has. The three defaults
(``"rmse"``, ``"relative_mse"``, ``"r2_oos"``) keep their existing
benchmark-relative formulas regardless of how they are requested.

``loss`` is a per-observation loss ``loss(y_true, y_pred) -> ndarray`` used
by the Diebold-Mariano loss differential and the Model Confidence Set's loss
matrix; ``None`` (default) is squared error, the prior, only behavior. Since
the Clark-West adjustment is derived under quadratic loss, setting a custom
``loss`` makes ``significance_table`` skip CW (with a ``UserWarning``)
rather than compute it against the wrong loss.

``tests`` lists which significance tests actually run; unsupported names
raise at :func:`pipeline_spec` build time (see ``SUPPORTED_EVAL_TESTS``).
Pairwise contender-vs-benchmark tests are ``"dm"``, ``"cw"``, ``"gw"``,
``"enc_new"``, ``"enc_t"``, ``"gr"``, and ``"mz"``. ``"mz"`` is the
Mincer-Zarnowitz actual-on-forecast rationality regression. ``"pt"``, ``"hm"``, and
``"ag"`` are directional-accuracy tests for the contender's own sign
forecasts, evaluated on the same benchmark-aligned sample for consistency
with the pairwise tests. Joint multi-horizon pairwise tests are ``"uspa"``
and ``"aspa"``; they require at least two horizons and use ``"joint"`` as
their significance-table horizon sentinel. Full-set benchmark comparisons are ``"mcs"``,
``"spa"``, ``"rc"``, and ``"stepm"``; they populate ``PipelineReport.mcs``.
``"spa"``, ``"rc"``, and ``"stepm"`` require the ``arch`` extra
(``pip install "macroforecast[arch]"``).
``"berkowitz"``/``"pit_autocorr"``/``"coverage"`` are PIT-based calibration
diagnostics (Phase 1 density pipeline) -- they populate
``PipelineReport.calibration`` rather than ``significance``/``mcs`` and,
like every other test name, are opt-in only (absent from the default).
``test_options`` maps a requested test name to keyword options for that
test's underlying public callable. Option blocks are validated when
:func:`pipeline_spec` is built: the key must appear in ``tests`` and every
option name must be accepted by that test's callable.

Density/interval accuracy metrics -- ``"crps"``, ``"gaussian_nll"``,
``"log_score"``, ``"negative_log_score"``, ``"qlike"``, ``"pinball_loss"``,
``"coverage_rate"``, ``"interval_width"``, ``"interval_score"`` -- are
requested the SAME way as any other ``metrics`` entry; they land in
``PipelineReport.density`` instead of ``accuracy`` because they need a
``variance_prediction``/``quantile_predictions`` column rather than plain
``(y_true, y_pred)`` (see ``macroforecast.metrics.metric_kind``). Requesting
one on a forecast frame that carries no such column raises the same
actionable ``ValueError`` :func:`macroforecast.metrics.evaluate_forecasts`
already raises. Absent from the defaults, so a default-EvalSpec run never
computes them.

``calibration_alpha`` is the significance level for the calibration tests
above (Berkowitz LR test, PIT autocorrelation, and the nominal coverage
checked by the ``"coverage"`` test); it does not affect ``mcs_alpha``.

``subsamples`` optionally maps names to :class:`SubsampleWindow` values.
These are evaluation-window splits of an already-produced POOS forecast
frame: target-date rows are filtered before scoring and testing, without
refitting models or creating new forecast cells.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `benchmark` | positional or keyword | `str` | `required` |
| `metrics` | positional or keyword | `tuple[str \| Callable[..., float], ...]` | `("rmse", "relative_mse", "r2_oos")` |
| `tests` | positional or keyword | `tuple[str, ...]` | `("dm", "cw", "mcs")` |
| `by` | positional or keyword | `tuple[str, ...]` | `("target", "horizon")` |
| `primary_axis` | positional or keyword | `str` | `"contender"` |
| `cw_for_nested` | positional or keyword | `bool` | `True` |
| `mcs_alpha` | positional or keyword | `float` | `0.1` |
| `mcs_method` | positional or keyword | `str` | `"iterative"` |
| `multiple_testing` | positional or keyword | `str \| None` | `None` |
| `subsamples` | positional or keyword | `Mapping[str, SubsampleWindow] \| None` | `None` |
| `dm_kwargs` | positional or keyword | `Mapping[str, Any]` | `<factory>` |
| `loss` | positional or keyword | `Callable[[Any, Any], Any] \| None` | `None` |
| `test_options` | positional or keyword | `Mapping[str, Mapping[str, Any]]` | `<factory>` |
| `calibration_alpha` | positional or keyword | `float` | `0.05` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.EvalSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `benchmark` | `str` | `required` |
| `metrics` | `tuple[str \| Callable[..., float], ...]` | `("rmse", "relative_mse", "r2_oos")` |
| `tests` | `tuple[str, ...]` | `("dm", "cw", "mcs")` |
| `by` | `tuple[str, ...]` | `("target", "horizon")` |
| `primary_axis` | `str` | `"contender"` |
| `cw_for_nested` | `bool` | `True` |
| `mcs_alpha` | `float` | `0.1` |
| `mcs_method` | `str` | `"iterative"` |
| `multiple_testing` | `str \| None` | `None` |
| `subsamples` | `Mapping[str, SubsampleWindow] \| None` | `None` |
| `dm_kwargs` | `Mapping[str, Any]` | `default_factory` |
| `loss` | `Callable[[Any, Any], Any] \| None` | `None` |
| `test_options` | `Mapping[str, Mapping[str, Any]]` | `default_factory` |
| `calibration_alpha` | `float` | `0.05` |
### InterpretSpec

Qualified name: `macroforecast.pipeline.spec.InterpretSpec`

#### Signature

```python
macroforecast.pipeline.InterpretSpec(methods: tuple[str, ...] = (), which_fit: str = "auto", background: int | None = None, top_k: int | None = 20, on_targets: tuple[str, ...] | None = None, shap_kind: str = "auto") -> None
```

#### Description

ML interpretation request for an arm (deferred, multi-method).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `methods` | positional or keyword | `tuple[str, ...]` | `()` |
| `which_fit` | positional or keyword | `str` | `"auto"` |
| `background` | positional or keyword | `int \| None` | `None` |
| `top_k` | positional or keyword | `int \| None` | `20` |
| `on_targets` | positional or keyword | `tuple[str, ...] \| None` | `None` |
| `shap_kind` | positional or keyword | `str` | `"auto"` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.InterpretSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `methods` | `tuple[str, ...]` | `()` |
| `which_fit` | `str` | `"auto"` |
| `background` | `int \| None` | `None` |
| `top_k` | `int \| None` | `20` |
| `on_targets` | `tuple[str, ...] \| None` | `None` |
| `shap_kind` | `str` | `"auto"` |
### PipelineSpec

Qualified name: `macroforecast.pipeline.spec.PipelineSpec`

#### Signature

```python
macroforecast.pipeline.PipelineSpec(data: Any, targets: tuple[ResolvedTarget, ...], horizons: tuple[int, ...], window: Any, arms: tuple[Arm, ...], evaluation: EvalSpec, preprocessing: Any | None = None, preprocessing_policy: Any | None = None, combinations: tuple[CombinationContender, ...] = (), save_models: bool = True, model_store: str = "trained_model", checkpoint_dir: str | None = None, result_store: str | None = None, n_jobs: int = 1, model_threads: int = 1, preprocessing_cache_dir: str | Literal[False] | None = None, seed: int | None = 42, provenance: Mapping[str, Any] = <factory>, provenance_level: "Literal['full', 'basic']" = "full", policy_overrides: Mapping[tuple[str, str], str] = <factory>) -> None
```

#### Description

Validated, frozen configuration produced by :func:`pipeline_spec`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `targets` | positional or keyword | `tuple[ResolvedTarget, ...]` | `required` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `required` |
| `window` | positional or keyword | `Any` | `required` |
| `arms` | positional or keyword | `tuple[Arm, ...]` | `required` |
| `evaluation` | positional or keyword | `EvalSpec` | `required` |
| `preprocessing` | positional or keyword | `Any \| None` | `None` |
| `preprocessing_policy` | positional or keyword | `Any \| None` | `None` |
| `combinations` | positional or keyword | `tuple[CombinationContender, ...]` | `()` |
| `save_models` | positional or keyword | `bool` | `True` |
| `model_store` | positional or keyword | `str` | `"trained_model"` |
| `checkpoint_dir` | positional or keyword | `str \| None` | `None` |
| `result_store` | positional or keyword | `str \| None` | `None` |
| `n_jobs` | positional or keyword | `int` | `1` |
| `model_threads` | positional or keyword | `int` | `1` |
| `preprocessing_cache_dir` | positional or keyword | `str \| Literal[False] \| None` | `None` |
| `seed` | positional or keyword | `int \| None` | `42` |
| `provenance` | positional or keyword | `Mapping[str, Any]` | `<factory>` |
| `provenance_level` | positional or keyword | `Literal['full', 'basic']` | `"full"` |
| `policy_overrides` | positional or keyword | `Mapping[tuple[str, str], str]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.PipelineSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `data` | `Any` | `required` |
| `targets` | `tuple[ResolvedTarget, ...]` | `required` |
| `horizons` | `tuple[int, ...]` | `required` |
| `window` | `Any` | `required` |
| `arms` | `tuple[Arm, ...]` | `required` |
| `evaluation` | `EvalSpec` | `required` |
| `preprocessing` | `Any \| None` | `None` |
| `preprocessing_policy` | `Any \| None` | `None` |
| `combinations` | `tuple[CombinationContender, ...]` | `()` |
| `save_models` | `bool` | `True` |
| `model_store` | `str` | `"trained_model"` |
| `checkpoint_dir` | `str \| None` | `None` |
| `result_store` | `str \| None` | `None` |
| `n_jobs` | `int` | `1` |
| `model_threads` | `int` | `1` |
| `preprocessing_cache_dir` | `str \| Literal[False] \| None` | `None` |
| `seed` | `int \| None` | `42` |
| `provenance` | `Mapping[str, Any]` | `default_factory` |
| `provenance_level` | `Literal['full', 'basic']` | `"full"` |
| `policy_overrides` | `Mapping[tuple[str, str], str]` | `default_factory` |
### ResolvedTarget

Qualified name: `macroforecast.pipeline.spec.ResolvedTarget`

#### Signature

```python
macroforecast.pipeline.ResolvedTarget(name: str, policy: str, transform: str, tcode: int | None, annualize: bool) -> None
```

#### Description

A target with its forecast policy and transform resolved.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `policy` | positional or keyword | `str` | `required` |
| `transform` | positional or keyword | `str` | `required` |
| `tcode` | positional or keyword | `int \| None` | `required` |
| `annualize` | positional or keyword | `bool` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.ResolvedTarget(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `policy` | `str` | `required` |
| `transform` | `str` | `required` |
| `tcode` | `int \| None` | `required` |
| `annualize` | `bool` | `required` |
### SubsampleWindow

Qualified name: `macroforecast.pipeline.spec.SubsampleWindow`

#### Signature

```python
macroforecast.pipeline.SubsampleWindow(start: str | None = None, end: str | None = None, exclude: tuple[tuple[str, str], ...] = ()) -> None
```

#### Description

Evaluation-window filter applied to forecast target dates.

Bounds are inclusive date strings. ``exclude`` removes inclusive date ranges
after the optional start/end bounds are applied.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | positional or keyword | `str \| None` | `None` |
| `end` | positional or keyword | `str \| None` | `None` |
| `exclude` | positional or keyword | `tuple[tuple[str, str], ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.SubsampleWindow(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `start` | `str \| None` | `None` |
| `end` | `str \| None` | `None` |
| `exclude` | `tuple[tuple[str, str], ...]` | `()` |
### TargetSpec

Qualified name: `macroforecast.pipeline.spec.TargetSpec`

#### Signature

```python
macroforecast.pipeline.TargetSpec(name: str, transform: str | None = None, policy: str | None = None, annualize: bool = False, reduce_i2: bool = True) -> None
```

#### Description

A forecast target and how its forecast object is defined.

``name`` is the panel column to forecast. ``transform`` and ``policy`` may
be left as ``None`` so FRED transformation-code metadata chooses the
conventional forecast object. For example, a FRED-MD growth-rate target
resolves to a direct-average growth forecast rather than a raw level
forecast. ``annualize`` affects reporting scale only, while ``reduce_i2``
keeps the package's convention for I(2) series by forecasting the
first-difference object.

Returns
TargetSpec
    Immutable target declaration consumed by ``pipeline_spec(...)`` and
    resolved to ``ResolvedTarget`` during execution.

Example
>>> from macroforecast.pipeline import TargetSpec
>>> target = TargetSpec(name="INDPRO", annualize=True)

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `transform` | positional or keyword | `str \| None` | `None` |
| `policy` | positional or keyword | `str \| None` | `None` |
| `annualize` | positional or keyword | `bool` | `False` |
| `reduce_i2` | positional or keyword | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.pipeline.TargetSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `transform` | `str \| None` | `None` |
| `policy` | `str \| None` | `None` |
| `annualize` | `bool` | `False` |
| `reduce_i2` | `bool` | `True` |
### contender_names

Qualified name: `macroforecast.pipeline.spec.contender_names`

#### Signature

```python
macroforecast.pipeline.contender_names(arm: Arm) -> list[str]
```

#### Description

Display contender labels for an arm. A contender IS exactly an arm.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `arm` | positional or keyword | `Arm` | `required` |

#### Returns

`list[str]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.contender_names(...)
```
### is_vintage_aware

Qualified name: `macroforecast.pipeline.spec.is_vintage_aware`

#### Signature

```python
macroforecast.pipeline.is_vintage_aware(spec: PipelineSpec) -> bool
```

#### Description

Return whether a pipeline spec runs against per-origin vintage data.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `spec` | positional or keyword | `PipelineSpec` | `required` |

#### Returns

`bool`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.is_vintage_aware(...)
```
### model_arms

Qualified name: `macroforecast.pipeline.spec.model_arms`

#### Signature

```python
macroforecast.pipeline.model_arms(models: Sequence[Any] | Mapping[str, Any], *, names: Sequence[str] | None = None, preprocessing: Any | None = None, preprocessing_policy: Any | None = None, features: Any | None = None, feature_policy: Any | None = None, params: Mapping[str, Any] | None = None, model_selection: Any | None = None, model_selection_metric: str = "mse", interpret: InterpretSpec | tuple[str, ...] | None = None, nested_in_benchmark: bool | set[str] | Sequence[str] = False, metadata: Mapping[str, Any] | None = None) -> list[Arm]
```

#### Description

Build one :class:`Arm` per model for a pure model comparison.

A model comparison is "several Arms identical except ``model``". This helper
is the pipeline idiom for that: it returns a ``list[Arm]`` -- one arm (one
contender; each (target, horizon) of it is one cell) per model -- all
sharing the given preprocessing,
features, and evaluation config, differing only in ``model`` (and in the
per-model ``params``/``model_selection``/nesting when those are given as
mappings/sets).

``models`` is a sequence of single models (``str`` | ``Callable`` |
``ModelSpec``) or a ``Mapping[name -> model]``. Arm names default to the
model name (``str(model)`` / ``ModelSpec.name`` / ``callable.__name__``), or
the mapping keys, or the explicit ``names``. Names must be unique.

``params`` and ``model_selection`` are shared by every arm unless they are a
Mapping whose key set is exactly the arm names, in which case each entry is
applied to its arm (see :func:`_per_arm_or_shared`). A single shared dict of
hyperparameters is therefore unambiguous from a per-arm mapping.

``nested_in_benchmark`` is either a bool shared by all arms, or a set/sequence
of arm names that nest the benchmark.

The returned list is ready to pass to ``pipeline_spec(arms=...)``. Pure model
comparison shares all config; comparing feature cases still needs explicit
Arms (build them by hand, varying ``features``).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `models` | positional or keyword | `Sequence[Any] \| Mapping[str, Any]` | `required` |
| `names` | keyword only | `Sequence[str] \| None` | `None` |
| `preprocessing` | keyword only | `Any \| None` | `None` |
| `preprocessing_policy` | keyword only | `Any \| None` | `None` |
| `features` | keyword only | `Any \| None` | `None` |
| `feature_policy` | keyword only | `Any \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `model_selection` | keyword only | `Any \| None` | `None` |
| `model_selection_metric` | keyword only | `str` | `"mse"` |
| `interpret` | keyword only | `InterpretSpec \| tuple[str, ...] \| None` | `None` |
| `nested_in_benchmark` | keyword only | `bool \| set[str] \| Sequence[str]` | `False` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`list[Arm]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.model_arms(...)
```
### pipeline_spec

Qualified name: `macroforecast.pipeline.spec.pipeline_spec`

#### Signature

```python
macroforecast.pipeline.pipeline_spec(*, data: Any, targets: Sequence[str | TargetSpec], horizons: Sequence[int] | int, window: Any, arms: Sequence[Arm], evaluation: EvalSpec, combinations: Sequence[CombinationContender] = (), preprocessing: Any | None = None, preprocessing_policy: Any | None = None, tcode_target_map: Mapping[int, tuple[str, str]] | None = None, save_models: bool = True, model_store: str = "trained_model", checkpoint_dir: str | None = None, result_store: str | Path | None = None, n_jobs: int | str = 1, preprocessing_cache_dir: str | bool | None = None, seed: int | None = 42, provenance: Mapping[str, Any] | None = None, provenance_level: "Literal['full', 'basic']" = "full", on_unsupported_direct: "Literal['error', 'warn', 'reroute']" = "error") -> PipelineSpec
```

#### Description

Validate and build a :class:`PipelineSpec`.

``n_jobs`` is a positive int (explicit cell-worker count) or the literal
``"auto"``. ``"auto"`` inspects the core budget and the work structure
(``len(targets) * len(arms) * len(horizons)`` cells) via
:func:`auto_parallelism` and splits the cores between cell workers
(stored as the resolved ``PipelineSpec.n_jobs``) and per-cell model-internal
threads (stored as ``PipelineSpec.model_threads``).

``preprocessing_cache_dir`` is a path, ``None``, or ``False`` -- see the field
docstring on :class:`PipelineSpec` for the full three-state contract. In short:
a path pins a persistent shared on-disk preprocessing-fit cache; ``None``
(default) auto-manages a temporary one for the duration of the run when
``n_jobs>1`` (a no-op when ``n_jobs==1``); ``False`` explicitly opts out of any
disk-backed cache even when parallel. ``True`` is invalid and raises.

``provenance_level`` ("full" default, or "basic") controls how much
``PipelineReport.provenance`` a live ``run_pipeline``/``rescore`` call
attaches -- see the field docstring on :class:`PipelineSpec`. Note this is
independent of ``provenance=`` above (caller-supplied notes merged into
whichever shape results); "basic" does not drop caller-supplied notes, it
only omits the "environment"/"data"/"spec_echo" blocks.

``result_store`` is an optional directory for cross-run reuse of completed
forecast cells. When left at ``None`` (the default), the runner follows the
original execution path exactly.

``on_unsupported_direct`` controls what happens when a model that only
iterates its own dynamics is combined with ``direct`` or ``direct_average``:
``"error"`` (default) rejects the spec, ``"warn"`` preserves the old weak
benchmark behavior, and ``"reroute"`` runs only the affected arm-target
cells as ``recursive``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | keyword only | `Any` | `required` |
| `targets` | keyword only | `Sequence[str \| TargetSpec]` | `required` |
| `horizons` | keyword only | `Sequence[int] \| int` | `required` |
| `window` | keyword only | `Any` | `required` |
| `arms` | keyword only | `Sequence[Arm]` | `required` |
| `evaluation` | keyword only | `EvalSpec` | `required` |
| `combinations` | keyword only | `Sequence[CombinationContender]` | `()` |
| `preprocessing` | keyword only | `Any \| None` | `None` |
| `preprocessing_policy` | keyword only | `Any \| None` | `None` |
| `tcode_target_map` | keyword only | `Mapping[int, tuple[str, str]] \| None` | `None` |
| `save_models` | keyword only | `bool` | `True` |
| `model_store` | keyword only | `str` | `"trained_model"` |
| `checkpoint_dir` | keyword only | `str \| None` | `None` |
| `result_store` | keyword only | `str \| Path \| None` | `None` |
| `n_jobs` | keyword only | `int \| str` | `1` |
| `preprocessing_cache_dir` | keyword only | `str \| bool \| None` | `None` |
| `seed` | keyword only | `int \| None` | `42` |
| `provenance` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `provenance_level` | keyword only | `Literal['full', 'basic']` | `"full"` |
| `on_unsupported_direct` | keyword only | `Literal['error', 'warn', 'reroute']` | `"error"` |

#### Returns

`PipelineSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.pipeline_spec(...)
```
### resolve_target

Qualified name: `macroforecast.pipeline.spec.resolve_target`

#### Signature

```python
macroforecast.pipeline.resolve_target(target: str | TargetSpec, *, data: Any = None, tcode: int | None = None, tcode_map: Mapping[int, tuple[str, str]] | None = None, reduce_i2: bool = True) -> ResolvedTarget
```

#### Description

Resolve a target to its (forecast_policy, target_transform).

Explicit ``TargetSpec.transform``/``policy`` win; otherwise the t-code (passed
or read from ``data`` metadata) is mapped through ``tcode_map`` (defaults to
:data:`TCODE_TARGET_MAP`). Raises if neither an explicit transform nor a
t-code is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `target` | positional or keyword | `str \| TargetSpec` | `required` |
| `data` | keyword only | `Any` | `None` |
| `tcode` | keyword only | `int \| None` | `None` |
| `tcode_map` | keyword only | `Mapping[int, tuple[str, str]] \| None` | `None` |
| `reduce_i2` | keyword only | `bool` | `True` |

#### Returns

`ResolvedTarget`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.pipeline.resolve_target(...)
```
