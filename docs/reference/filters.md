# macroforecast.filters

[Back to reference](index.md)

## Purpose

`macroforecast.filters` contains direct one-series filter and smoother
callables. These functions transform a single time series and return the
filtered components plus metadata. They do not create forecast targets, do not
build a model-ready feature matrix, and do not decide train/validation/test
windows.

Use `macroforecast.feature_engineering` when the same filters need to become
panel features such as `{column}_hp_cycle`, `{column}_hamilton_cycle`,
`{column}_savgol`, or `{column}_albama`.

## Public Functions

| Callable | Input | Output | Purpose |
| --- | --- | --- | --- |
| `hp_filter(y, lamb=129600.0, component="both")` | One numeric series. | `FilterResult` with `cycle` and/or `trend`. | Two-sided Hodrick-Prescott filter. |
| `hamilton_filter(y, h=8, p=4, fit_policy="expanding")` | One numeric series. | `FilterResult` with `cycle` and/or `trend`. | Hamilton regression filter with expanding or full-sample fit policy. |
| `savitzky_golay(y, window_length=5, polyorder=2)` | One numeric series. | `FilterResult` with `savgol`. | Centered Savitzky-Golay local polynomial smoother. |
| `wavelet_filter(y, n_levels=3, wavelet="db4")` | One numeric series. | `FilterResult` with `wA{level}` and `wD{level}` columns. | Existing causal rolling multi-resolution approximation. |
| `albama(y, mode="one_sided")` | One numeric series. | `AlbaMAResult` with `smoothed`, `weights`, and metadata. | Goulet Coulombe-Klieber adaptive learning-based moving average. |
| `AlbaMA(...)`, `AdaptiveMovingAverage(...)` | Estimator-style smoother settings. | Object with `fit`, `fit_transform`, and `result`. | Class wrappers around `albama()`. |

## Output Objects

### FilterResult

```python
macroforecast.filters.FilterResult(
    values: pandas.DataFrame,
    method: str,
    params: dict,
    metadata: dict,
    source: str | None = None,
)
```

| Field | Meaning |
| --- | --- |
| `values` | Filtered component frame indexed like the input series. |
| `method` | Canonical filter name, such as `hp_filter` or `hamilton_filter`. |
| `params` | Resolved parameter values. |
| `metadata` | Provenance, fit policy, backend, and formula notes. |
| `source` | Source series name when available. |

`FilterResult.component(name)` returns one component column from `values`.

### AlbaMAResult

`albama()` returns `AlbaMAResult`, not `FilterResult`, because the learned
observation-weight matrix is central to the method.

| Field | Meaning |
| --- | --- |
| `smoothed` | Adaptive moving-average series. |
| `weights` | Source-date by target-date observation-weight matrix. |
| `mode` | `"one_sided"` or `"two_sided"`. |
| `backend` | Current backend identifier. |
| `params` | Resolved tree-bagging settings. |
| `metadata` | Paper reference, R-code reference, and weight-extraction notes. |

## Flow

```python
import macroforecast as mf

hp = mf.filters.hp_filter(inflation, lamb=129600.0)
cycle = hp.component("cycle")

ham = mf.filters.hamilton_filter(inflation, h=24, p=12, fit_policy="expanding")
trend = ham.component("trend")

sg = mf.filters.savitzky_golay(inflation, window_length=13, polyorder=2)
smooth = sg.component("savgol")

albama = mf.filters.albama(inflation, mode="one_sided")
adaptive = albama.smoothed
weights = albama.weights
```

For panel feature construction:

```python
features = mf.feature_engineering.hp_filter_features(panel, columns=["CPIAUCSL"])
albama_features = mf.feature_engineering.adaptive_ma_rf_features(
    panel,
    columns=["CPIAUCSL"],
    sided="one",
)
```

## Leakage Boundary

| Filter | Default fit policy | Forecasting caution |
| --- | --- | --- |
| `hp_filter` | Full input, two-sided. | Not runner-safe on an unsplit panel; use only for retrospective analysis or already training-only data. |
| `hamilton_filter` | Expanding by default. | Can be causal when `fit_policy="expanding"`; `full_sample` is retrospective. |
| `savitzky_golay` | Full input, centered window. | Not runner-safe on an unsplit panel. |
| `wavelet_filter` | Causal rolling approximation. | Past-only by construction, but current implementation is not a true DWT backend. |
| `albama` | One-sided by default. | `mode="one_sided"` is real-time; `mode="two_sided"` is retrospective. |

## AlbaMA Reference

`albama()` implements the adaptive learning-based moving average from Goulet
Coulombe and Klieber (2025). The method fits a bagged tree ensemble to one
series with deterministic time as the only predictor:

```text
y_t = f(t) + error_t
```

Each tree partitions the time axis into terminal-node intervals. The tree
prediction is the within-leaf average, so the ensemble prediction is a learned
moving average. The learned weight matrix explains the implicit look-back
window at each date.

Reference:

> Goulet Coulombe, Philippe, and Karin Klieber. 2025. "An Adaptive Moving
> Average for Macroeconomic Monitoring." arXiv:2501.13222v1.
> <https://arxiv.org/abs/2501.13222>

R-code alignment:

| R source | Python mapping |
| --- | --- |
| `ranger(MAIN ~ Time_Trend, keep.inbag=TRUE)` | Manual `DecisionTreeRegressor` bagging with stored in-bag counts. |
| `predict(..., type="terminalNodes")` | `tree.apply(...)` terminal-node IDs. |
| `Albama_center` | `albama(..., mode="two_sided")`. |
| `Albama_right` recursive loop | `albama(..., mode="one_sided")`. |

The Python backend is method-aligned with the R code, but it does not promise
bit-level equality to `ranger` because tree split and randomization semantics
differ across backends.

## Function Details

### hp_filter

```python
macroforecast.filters.hp_filter(
    y,
    *,
    dates=None,
    lamb=129600.0,
    component="both",
    interpolate_missing=True,
    name=None,
) -> FilterResult
```

`component` is `cycle`, `trend`, or `both`. Missing values are linearly
interpolated before filtering when `interpolate_missing=True`.

### hamilton_filter

```python
macroforecast.filters.hamilton_filter(
    y,
    *,
    dates=None,
    h=8,
    p=4,
    component="both",
    fit_policy="expanding",
    min_train_size=None,
    missing="drop",
    name=None,
) -> FilterResult
```

The formula is:

```text
y[t+h] = alpha + beta_0 y[t] + ... + beta_{p-1} y[t-p+1] + error[t+h].
```

The trend is the fitted value and the cycle is the residual, both labeled at
`t+h`. `fit_policy="expanding"` estimates each row with earlier completed
Hamilton-regression rows. `fit_policy="full_sample"` reproduces the ordinary
in-sample filter style.

### savitzky_golay

```python
macroforecast.filters.savitzky_golay(
    y,
    *,
    dates=None,
    window_length=5,
    polyorder=2,
    derivative=0,
    interpolate_missing=True,
    name=None,
) -> FilterResult
```

This is a centered local-polynomial smoother through
`scipy.signal.savgol_filter`.

### wavelet_filter

```python
macroforecast.filters.wavelet_filter(
    y,
    *,
    dates=None,
    n_levels=3,
    wavelet="db4",
    name=None,
) -> FilterResult
```

The current implementation returns causal rolling approximation/detail columns
`wA{level}` and `wD{level}`. The `wavelet` argument is recorded for provenance;
it is not yet a true discrete-wavelet backend.

### albama

```python
macroforecast.filters.albama(
    y,
    *,
    dates=None,
    mode="one_sided",
    n_estimators=500,
    min_samples_leaf=6,
    sample_fraction=0.6,
    random_state=42,
    replace=True,
    inbag_rule="single",
    min_train_size=2,
    name=None,
) -> AlbaMAResult
```

`inbag_rule="single"` mirrors the R code condition
`inbag.counts[[tree]] == 1`. Use `macroforecast.feature_analysis` to summarize
the returned weights.
