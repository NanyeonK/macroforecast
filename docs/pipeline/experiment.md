# Experiment

`ForecastExperiment` is the main orchestrator of the pipeline layer. It lives in `macrocast.pipeline.experiment`. Given a transformed panel, a target series, a list of forecast horizons, and a model grid, it runs a pseudo-OOS evaluation loop and returns a `ResultSet`.

---

## ForecastExperiment

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `panel` | `pd.DataFrame` | required | Stationary-transformed predictor panel (T, N) with DatetimeIndex |
| `target` | `pd.Series` | required | Target series y_t with the same DatetimeIndex as panel |
| `horizons` | `list[int]` | required | Forecast horizons h. A separate model is trained per horizon. |
| `model_specs` | `list[ModelSpec]` | required | Model grid to evaluate |
| `feature_spec` | `FeatureSpec` | `None` | Feature construction config. Uses defaults if None. |
| `window` | `Window` | `Window.EXPANDING` | Outer evaluation window strategy |
| `rolling_size` | `int` | `None` | Training window length for rolling. Required when `window=Window.ROLLING`. |
| `oos_start` | timestamp or str | `None` | OOS start date. Defaults to the 80th percentile of the sample. |
| `oos_end` | timestamp or str | `None` | OOS end date. Defaults to last sample date minus max horizon. |
| `n_jobs` | `int` | `1` | Parallel workers via joblib. `-1` uses all available cores. |
| `experiment_id` | `str` | `None` | UUID auto-generated if None. |
| `output_dir` | `Path` or `str` | `None` | If provided, ResultSet written to parquet upon completion. |

---

## ModelSpec

`ModelSpec` is a dataclass that describes a single model configuration. One record per (model, horizon, date) triple is produced in the output.

| Field | Type | Description |
|-------|------|-------------|
| `model_cls` | `type` | Estimator class, a subclass of `MacrocastEstimator` or `SequenceEstimator` |
| `regularization` | `Regularization` | Regularization treatment component |
| `cv_scheme` | `CVSchemeType` | Hyperparameter-selection scheme |
| `loss_function` | `LossFunction` | Loss function optimised during training |
| `model_kwargs` | `dict` | Forwarded as keyword arguments to `model_cls(**model_kwargs)` |
| `model_id` | `str` | Human-readable identifier. Auto-generated from component values if None. |

`model_id` is auto-constructed as `"{nonlinearity}__{regularization}__{cv_scheme}__{loss_function}"` when left as None, e.g. `"krr__factors__KFoldCV(k=5)__l2"`.

---

## FeatureSpec

`FeatureSpec` controls how `FeatureBuilder` constructs Z_t for a given experiment run.

| Field | Default | Description |
|-------|---------|-------------|
| `use_factors` | `True` | Include PCA factors in Z_t |
| `n_factors` | `8` | Number of PCA factors |
| `n_lags` | `4` | Number of AR lags |
| `standardize_X` | `True` | Standardize predictor panel before PCA |
| `standardize_Z` | `False` | Standardize output feature matrix Z |
| `lookback` | `12` | LSTM look-back window in months. Ignored for cross-sectional models. |

---

## Pseudo-OOS Algorithm

For each evaluation date t* in [oos_start, oos_end] and each (model_spec, horizon h):

1. Construct the training window. Expanding: [t_start .. t*-h]. Rolling: [t*-h-rolling_size .. t*-h].
2. Fit `FeatureBuilder` on the training window only. PCA loadings are estimated from X_train; no future data enters.
3. Transform the single test row at t*-h to obtain Z_{t*-h} of shape (1, n_features).
4. Align y_train as y_{t+h} for the direct h-step formulation: each training target is the h-period-ahead realisation of the target series.
5. Fit the model on (Z_train, y_train_aligned).
6. Predict ŷ_{t*} = model.predict(Z_{t*-h}) and record (ŷ_{t*}, y_{t*}) in a `ForecastRecord`.

Multi-step forecasting uses the direct strategy only. A separate model is trained for each horizon h. Iterated forecasting is out of scope for version 1.

---

## Parallelism

`joblib` parallelises over (model_spec, horizon, date) triples when `n_jobs != 1`. The inner CV loop within each model call is already parallelised by sklearn's own `n_jobs=-1`. To avoid over-subscription, set `n_jobs=-1` at the experiment level and leave sklearn's inner parallelism at 1, or vice versa depending on the model count versus evaluation window length.

---

## Full Example

```python
import macrocast as mc
from macrocast.pipeline import (
    ForecastExperiment,
    ModelSpec,
    FeatureSpec,
    Regularization,
    CVScheme,
    LossFunction,
    Window,
    KRRModel,
)

# Load and transform data
md = mc.load_fred_md().trim(start="1970-01", end="2023-12").transform()
panel  = md.data
target = panel["INDPRO"]

# Define model grid
specs = [
    ModelSpec(
        model_cls=KRRModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(5),
        loss_function=LossFunction.L2,
    ),
]

# Configure and run
exp = ForecastExperiment(
    panel=panel,
    target=target,
    horizons=[1, 3, 12],
    model_specs=specs,
    feature_spec=FeatureSpec(n_factors=8, n_lags=4),
    window=Window.EXPANDING,
    oos_start="1990-01",
    n_jobs=-1,
    output_dir="~/.macrocast/results",
)

results = exp.run()
print(results)
# ResultSet(experiment_id='...', n_records=...)
```

---

## Output

`exp.run()` returns a `ResultSet`. If `output_dir` is specified, a parquet file is also written to `{output_dir}/{experiment_id}/results.parquet`. See [Results](results.md) for the full field reference and downstream usage.
