# macroforecast.interpretation.dual

[Back to interpretation](interpretation.md)

`macroforecast.interpretation.dual` is the dedicated namespace for the dual
interpretation route in Goulet Coulombe, Goebel, and Klieber (2024), "Dual
Interpretation of Machine Learning Forecasts" (`arXiv:2412.13076`). Standard
variable-importance tools ask which predictor columns matter. Dual
interpretation asks which historical training observations matter for a
forecast.

The central identity is:

```text
yhat_new = sum_i w_i(new) y_i
```

The weights `w_i(new)` are observation weights, also called data-portfolio
weights in the paper/code. A positive weight means the model borrows from that
historical outcome. A negative weight means the model uses that observation by
contrast. A concentrated weight vector means the forecast relies on a small
number of episodes. A high short position or high gross leverage means the
forecast is extrapolative rather than a simple local average.

## Reference Sources

| Source | Used for |
| --- | --- |
| Goulet Coulombe, Goebel, and Klieber (2024), "Dual Interpretation of Machine Learning Forecasts" | Paper terminology and interpretation target. |
| `wiki/raw/paper_code/coulombe_site_github_20260530/dual_python/auxiliaries.py` | Ridge, kernel-ridge, and random-forest observation-weight formulas. |
| `wiki/raw/paper_code/coulombe_site_github_20260530/DualML_R/DualML.R` | Forecast concentration, forecast short position, forecast leverage, and forecast turnover definitions. |
| `wiki/raw/paper_code/coulombe_site_github_20260530/DualML_R/README.md` | Original model-route inventory: OLS, RF, LGB, RR, KRR, and NN. |

Implemented now: ridge/OLS, kernel ridge, and sklearn-style random forest.
Deferred routes: boosted-tree AXIL, LGB+/LGBA+ channel-specific weights, neural
embedding-ridge approximation, and classification log-odds decomposition.

## Public Functions

| Function | Input | Output | Purpose |
| --- | --- | --- | --- |
| `macroforecast.interpretation.dual.dual_interpretation()` | model, train features, train target, optional test features | `DualInterpretationResult` | Run the paper-aligned ridge/KRR/RF path and return all dual tables together. |
| `macroforecast.interpretation.dual.dual_from_forecast_result()` | completed `ForecastResult`, model, train features, train target, optional test features | `ForecastResult` or `DualInterpretationResult` | Build a dual sidecar for a completed runner result. |
| `macroforecast.interpretation.dual.observation_weights()` | model, `X_train`, optional `X_test` | long `DataFrame` | Compute historical observation/data-portfolio weights. |
| `macroforecast.interpretation.dual.observation_contributions()` | weights and `y_train` | long `DataFrame` | Multiply observation weights by historical outcomes. |
| `macroforecast.interpretation.dual.forecast_diagnostics()` | weights | `DataFrame` | Compute concentration, short position, leverage, gross leverage, and turnover. |
| `macroforecast.interpretation.dual.top_observations()` | weights or contributions | long `DataFrame` | Return the largest historical observations for each forecast. |
| `macroforecast.interpretation.dual.group_observation_weights()` | weights/contributions and a group mapping | `DataFrame` | Aggregate observation weights over user-defined regimes or episodes. |
| `DualInterpretationResult.to_tables()` | result object | dict of `DataFrame` | Expand the result for `macroforecast.output`. |

Backward-compatible aliases are still available:

| Alias | Preferred name |
| --- | --- |
| `outcome_contributions` | `observation_contributions` |
| `data_portfolio_diagnostics` | `forecast_diagnostics` |
| `top_episodes` | `top_observations` |
| `episode_group_weights` | `group_observation_weights` |

## Public Flow

```python
import macroforecast as mf

dual = mf.interpretation.dual.dual_interpretation(
    model,
    X_train,
    y_train,
    X_test,
    method="random_forest",
    top_n=10,
    groups={
        "gfc": gfc_train_dates,
        "covid": covid_train_dates,
    },
)

tables = dual.to_tables(prefix="inflation")
```

For completed forecast runs, attach the same result as a sidecar:

```python
result = mf.forecasting.run(feature_set, "ridge", window=window)

result = mf.interpretation.dual.dual_from_forecast_result(
    result,
    fit,
    X_train,
    y_train,
    X_test,
    method="ridge",
)

# Equivalent method form:
result = result.with_dual(fit, X_train, y_train, X_test, method="ridge")
```

`forecasting.run()` does not compute dual interpretation automatically. The
completed forecast table does not contain the exact fitted estimator,
training-feature matrix, training target, or forecast-row feature matrix. Those
objects must be passed explicitly to avoid silent look-ahead or stale-design
errors.

For a ridge/KRR route, `model` can be `None`:

```python
dual = mf.interpretation.dual.dual_interpretation(
    None,
    X_train,
    y_train,
    X_test,
    method="krr",
    kernel="laplace",
    sigma=1e-4,
    lambda_=0.1,
)
```

## dual_interpretation

```python
macroforecast.interpretation.dual.dual_interpretation(
    model,
    X_train,
    y_train,
    X_test=None,
    *,
    method="auto",
    lambda_=1e-8,
    kernel="linear",
    sigma=1.0,
    add_intercept=False,
    ridge_penalty_scale="n_train",
    normalize=False,
    center=False,
    include_base=False,
    top_n=10,
    top_sort_by="abs_weight",
    top_q=0.05,
    groups=None,
    include_contributions=True,
    include_diagnostics=True,
    include_top_observations=True,
    include_group_weights=None,
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | fitted model or `None` | required | Required for random-forest weights. Optional for ridge/KRR because weights are closed-form from `X_train` and `X_test`. |
| `X_train` | pandas `DataFrame` | required | Training feature matrix. Its index becomes `train_index`. |
| `y_train` | pandas `Series` or sequence | required | Training target aligned to `X_train`. If it is a `Series`, the index is aligned to `train_index`. |
| `X_test` | pandas `DataFrame` or `None` | `None` | Forecast-row feature matrix. If omitted, each training row is explained against the training panel. |
| `method` | string | `auto` | `auto`, `ridge`, `ols`, `krr`, `kernel_ridge`, `random_forest`, or `rf`. |
| `lambda_` | float | `1e-8` | Ridge/KRR regularization. |
| `kernel` | string | `linear` | KRR kernel: `linear`, `gaussian`, `rbf`, `laplace`, or `laplacian`. |
| `sigma` | float | `1.0` | Kernel bandwidth convention used by the reviewed code: `exp(-sigma * distance)`. |
| `add_intercept` | bool | `False` | Adds an unpenalized intercept for ridge/OLS. The paper code usually works with standardized no-intercept matrices. |
| `ridge_penalty_scale` | string | `n_train` | Ridge penalty convention. `n_train` uses `n_train * lambda_`; `none` uses `lambda_`. |
| `normalize` | bool | `False` | Re-normalize row weights to sum to one. Default is false because leverage and negative weights are meaningful diagnostics. |
| `center` | bool | `False` | Center `y_train` before contribution calculation. |
| `include_base` | bool | `False` | With `center=True`, add an explicit base-row contribution. |
| `top_n` | int | `10` | Number of top observations returned per forecast row. |
| `top_sort_by` | string | `abs_weight` | `abs_weight`, `weight`, `contribution`, or `abs_contribution`. |
| `top_q` | float | `0.05` | Share of observations used in concentration. Values above `1` are treated as `1`. |
| `groups` | mapping or `None` | `None` | Named historical episode groups, mapping group name to training-index labels. |
| `include_*` | bool | varies | Include or skip contribution, diagnostic, top-observation, and group tables. |

Output: `DualInterpretationResult`.

| Field | Type | Meaning |
| --- | --- | --- |
| `weights` | `DataFrame` | Observation/data-portfolio weights. |
| `contributions` | `DataFrame` or `None` | Observation-level forecast contributions. |
| `diagnostics` | `DataFrame` or `None` | Forecast concentration, short position, leverage, gross leverage, and turnover. |
| `top_observations` | `DataFrame` or `None` | Largest historical observations per forecast. |
| `group_weights` | `DataFrame` or `None` | Group-level observation weights and contributions. |
| `metadata` | dict | Paper route, implemented/deferred routes, and options used. |

## dual_from_forecast_result

```python
macroforecast.interpretation.dual.dual_from_forecast_result(
    result,
    model,
    X_train,
    y_train,
    X_test=None,
    *,
    attach=True,
    sidecar_name="dual",
    **dual_options,
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `result` | `ForecastResult` | required | Completed forecast runner output. |
| `model` | fitted model or `None` | required | Same model argument passed to `dual_interpretation(...)`. |
| `X_train`, `y_train`, `X_test` | pandas objects | required except `X_test` | Exact design matrices used for the dual explanation. |
| `attach` | bool | `True` | If true, return a copy of `ForecastResult` with the sidecar attached. If false, return the standalone `DualInterpretationResult`. |
| `sidecar_name` | str | `dual` | Name used in `ForecastResult.sidecars` and output artifact names. |
| `**dual_options` | keyword args | none | Forwarded to `dual_interpretation(...)`, such as `method`, `lambda_`, `kernel`, `groups`, and `top_n`. |

Output: with `attach=True`, a new `ForecastResult`; with `attach=False`, a
standalone `DualInterpretationResult`.

## observation_weights

```python
macroforecast.interpretation.dual.observation_weights(
    model,
    X_train,
    X_test=None,
    *,
    method="auto",
    lambda_=1e-8,
    kernel="linear",
    sigma=1.0,
    add_intercept=False,
    ridge_penalty_scale="n_train",
    normalize=False,
)
```

Implemented routes:

| Route | Formula / logic | Notes |
| --- | --- | --- |
| Ridge / OLS | `W = X_test (X_train' X_train + n lambda I)^-1 X_train'` by default | Set `ridge_penalty_scale="none"` for `lambda I`. `add_intercept=True` adds an unpenalized intercept. |
| Kernel ridge | `W = K_test (K_train + lambda I)^-1` | Kernels: `linear`, `gaussian`/`rbf`, `laplace`/`laplacian`. |
| Random forest | For each tree, assign test and train rows to leaves; train rows in the same leaf share weight; average across trees | For sklearn forests, bootstrap sample counts are used when recoverable. |

Output columns:

| Column | Meaning |
| --- | --- |
| `test_row`, `test_index` | Forecast-row position and index. |
| `train_row`, `train_index` | Historical observation position and index. |
| `weight`, `abs_weight` | Signed and absolute observation weight. |
| `channel` | Implemented route: `ridge`, `krr`, or `random_forest`. |

The dense matrix is attached as `attrs["weight_matrix"]` with shape
`(n_test, n_train)`.

## observation_contributions

```python
macroforecast.interpretation.dual.observation_contributions(
    weights,
    y_train,
    *,
    center=False,
    include_base=False,
)
```

Input: an observation-weight table and the aligned training target.

Output columns add:

| Column | Meaning |
| --- | --- |
| `train_y` | Realized historical outcome. |
| `centered_train_y` | `train_y - mean(y_train)` when `center=True`; otherwise `train_y`. |
| `contribution` | `weight * train_y` by default. |
| `prediction` | Sum of contributions for the forecast row. |
| `channel` | `episode`, or `base` when `center=True` and `include_base=True`. |

Default `center=False` preserves the exact identity
`prediction = weights @ y_train`. Centering is useful for plots but changes the
table into a base-plus-centered-contribution decomposition.

## forecast_diagnostics

```python
macroforecast.interpretation.dual.forecast_diagnostics(weights, *, top_q=0.05)
```

Output:

| Column | Paper/code meaning |
| --- | --- |
| `concentration` | Forecast concentration: sum of top absolute weights divided by total absolute weight. |
| `short_position` | Forecast short position: signed sum of negative weights. |
| `short_position_abs` | Absolute short-side exposure. |
| `leverage` | Signed weight sum. |
| `gross_leverage` | Sum of absolute weights. |
| `turnover` | Sum of absolute weight changes relative to the previous forecast row. |
| `top_q`, `top_k`, `n_train` | Diagnostic settings. |

Negative weights are not automatically errors. In this paper they identify
contrast-based use of historical observations. The caution is economic:
macroeconomic shocks are often asymmetric, so a mirror-image historical
analogy may be a weak explanation even if the model uses it.

## top_observations

```python
macroforecast.interpretation.dual.top_observations(
    weights,
    *,
    y_train=None,
    n=10,
    sort_by="abs_weight",
)
```

Input: observation weights or observation contributions. If `y_train` is
provided and the table lacks `contribution`, contributions are computed first.

Output: top historical observations per forecast row with a `rank` column.
Supported `sort_by` values: `abs_weight`, `weight`, `contribution`, and
`abs_contribution`.

## group_observation_weights

```python
macroforecast.interpretation.dual.group_observation_weights(
    weights,
    groups,
    *,
    y_train=None,
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `weights` | Observation-weight or contribution table. |
| `groups` | Mapping from group name to training-index labels. |
| `y_train` | Optional training target used to create contributions before grouping. |

Example:

```python
groups = {
    "gfc": pd.period_range("2007Q4", "2009Q2", freq="Q").to_timestamp("Q"),
    "covid": pd.period_range("2020Q1", "2021Q2", freq="Q").to_timestamp("Q"),
}

grouped = mf.interpretation.dual.group_observation_weights(
    dual.weights,
    groups,
    y_train=y_train,
)
```

Output columns: `test_row`, `test_index`, `episode_group`, `weight`,
`abs_weight`, `n_episodes`, and, when available, `contribution` and
`abs_contribution`.

## Output Integration

`DualInterpretationResult.to_tables(prefix="dual")` returns:

| Table key | Meaning |
| --- | --- |
| `dual_observation_weights` | Long observation-weight table. |
| `dual_observation_contributions` | Long contribution table, when requested. |
| `dual_forecast_diagnostics` | Concentration, short-position, leverage, gross-leverage, and turnover table. |
| `dual_top_observations` | Top historical observations per forecast row. |
| `dual_group_observation_weights` | Group-level weights/contributions, when groups are provided. |
| `dual_metadata` | Result metadata as key/value rows. |

The output module recognizes this result directly:

```python
bundle = mf.output.bundle_outputs(
    forecasts=result,
    interpretation={"dual": dual},
    metadata={"study": "inflation_dual"},
)

manifest = mf.output.write_artifacts(
    bundle,
    "results/inflation_dual",
    layout="grouped",
)
```

With `layout="grouped"`, dual tables are written under:

```text
interpretation/dual/
```

The same grouped path is used when a `ForecastResult` contains a dual sidecar:

```python
result = result.with_dual(fit, X_train, y_train, X_test, method="ridge")
mf.output.write_artifacts(result, "results/dual_run", layout="grouped")
```

This keeps DualML observation-based explanations separate from SHAP,
oShapley/PBSV, PDP/ICE/ALE, and other feature-based interpretation outputs.
