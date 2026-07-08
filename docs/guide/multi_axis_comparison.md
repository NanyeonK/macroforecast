# Multi-Axis Forecast Comparisons

[Back to user guide](index.md)

Model comparisons often vary more than one design choice at a time. A GCLS-style
study, for example, can vary nonlinearity, shrinkage, hyperparameter selection,
and loss-related estimator family across arms, then ask how much each design
axis is associated with forecast-error differences. In `macroforecast`, the
forecasting part stays a normal pipeline. The only extra contract is that each
`Arm` carries scalar `tags`, and the completed master forecast frame is passed to
`axis_contribution`.

Tags are written to the master frame as flat `tag_<key>` columns. A tag key must
be an identifier such as `NL`, `SH`, `CV`, or `LF`; values must be scalar strings,
numbers, or booleans. If a forecast frame already contains the same `tag_<key>`
column, the pipeline raises instead of overwriting it. Tags are descriptive
metadata, so changing tags does not change result-store cell digests.

## A Small End-to-End Example

```python
import numpy as np
import pandas as pd

import macroforecast as mf

idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
x = np.linspace(0.0, 1.0, len(idx))
panel = pd.DataFrame(
    {
        "y": 1.0 + 0.7 * x + 0.15 * np.sin(np.arange(len(idx)) / 4.0),
        "x1": x,
        "x2": np.cos(np.arange(len(idx)) / 6.0),
    },
    index=idx,
)
data = mf.data.custom_dataset(panel, transform_codes={"y": 1, "x1": 1, "x2": 1})

features = mf.feature_engineering.feature_spec(
    target="y",
    predictors=["x1", "x2"],
    lags=1,
    target_lags=(0, 1),
)

window = mf.window.spec(
    estimation=mf.window.estimation_expanding(min_size=48),
    val=mf.window.val_last_block(size=12),
    test=mf.window.test_origins(horizon=1, step=4),
)

arms = [
    mf.pipeline.Arm(
        "linear_base",
        model="ols",
        features=features,
        tags={"NL": 0, "SH": "none", "CV": "fixed", "LF": "quadratic"},
    ),
    mf.pipeline.Arm(
        "ridge_cv",
        model="ridge",
        features=features,
        model_selection=mf.model_selection.cv_path(param="alpha", values=[0.01, 0.1, 1.0]),
        tags={"NL": 0, "SH": "ridge", "CV": "poos", "LF": "quadratic"},
    ),
    mf.pipeline.Arm(
        "lasso_cv",
        model="lasso",
        features=features,
        model_selection=mf.model_selection.cv_path(param="alpha", values=[0.01, 0.1, 1.0]),
        tags={"NL": 0, "SH": "lasso", "CV": "poos", "LF": "quadratic"},
    ),
    mf.pipeline.Arm(
        "tree",
        model="random_forest",
        features=features,
        params={"n_estimators": 50, "max_depth": 3},
        tags={"NL": 1, "SH": "none", "CV": "fixed", "LF": "tree"},
    ),
]

spec = mf.pipeline.pipeline_spec(
    data=data,
    targets=["y"],
    horizons=[1],
    window=window,
    arms=arms,
    evaluation=mf.pipeline.EvalSpec(
        benchmark="linear_base",
        metrics=("rmse", "relative_mse", "r2_oos"),
    ),
    save_models=False,
)

report = mf.pipeline.run_pipeline(spec)

contrib = mf.axis_contribution(
    report.forecasts,
    features=["NL", "SH", "CV", "LF"],
    outcome="r2",
    reference="linear_base",
    fixed_effects=("target", "horizon", "date"),
    vcov="driscoll_kraay",
)

contrib[["feature", "level", "coef", "se", "p"]]
```

The contribution table is intentionally tidy: one row per estimated tag level or
interaction term. It can be sent directly to `DataFrame.to_latex()` or joined to
your own reporting labels. `vcov="driscoll_kraay"` is the GCLS-style inference
choice: it aggregates score contributions by forecast date and applies a
Bartlett kernel across dates, covering both same-date cross-sectional dependence
and serial dependence across forecast dates. The helper uses this estimator by
default; `vcov="cluster"` gives date-clustered CR0 standard errors, while
`vcov="hac"` keeps the legacy row-stacked Newey-West calculation for comparison.

## State Interactions

Some paper tables interact design tags with externally supplied state variables,
such as uncertainty or financial-condition indexes. `macroforecast` does not
ship those series. Pass a date-indexed `Series`, and the helper interacts it
with the treatment regressors it builds from the tag columns.

```python
macro_state = pd.Series(
    np.linspace(-1.0, 1.0, report.forecasts["date"].nunique()),
    index=sorted(report.forecasts["date"].unique()),
    name="macro_u",
)

state_contrib = mf.axis_contribution(
    report.forecasts,
    features=["NL", "SH"],
    outcome="squared_error",
    interactions={"macro_u": macro_state},
    fixed_effects=("target", "horizon", "date"),
    vcov="driscoll_kraay",
    hac_lags=1,
)
```

These regressions are descriptive error-attribution summaries. They are useful
for organizing a multi-axis forecast design, but the coefficients should not be
read as causal effects of a modeling choice.

## Reference

- [Pipeline reference](../reference/pipeline.md) -- `Arm`, `pipeline_spec`, and
  `run_pipeline`.
- [Public API reference](../reference/public_api.md) -- `macroforecast.analysis`
  and the top-level `axis_contribution` export.
