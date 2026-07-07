# Your Data, Your Model, One Table

[Back to User Guide](index.md)

[Getting Started](getting_started.md) runs a study on FRED-MD with two
built-in models. The reference pages for
[custom data](../reference/custom/custom_dataset.md) and
[custom models](../reference/custom/custom_model.md) each cover one stage in
isolation -- loading your own CSV, or wiring up your own estimator -- but stop
one step short of a scored comparison. This page connects every stage: your
own CSV, your own model, a horse race against a benchmark, and the one-line
table you would put in a paper. Every code block below was executed for real;
the output shown is genuine, not illustrative.

## What you will build

1. A small synthetic monthly panel, written to an actual CSV file.
2. Load it with `mf.data.load_custom_csv`.
3. Declare the forecast target -- and meet the real error the package raises
   when custom data has no FRED t-code to fall back on.
4. Wrap a plain `sklearn.linear_model.Ridge` as a `mf.models.custom_model`,
   next to an `"ar"` benchmark arm.
5. Run the pipeline over a window built with `mf.window.from_cutoffs`.
6. Turn the resulting `PipelineReport` into a referee-ready table with
   `mf.reporting.paper_accuracy_table(...).to_latex()`.

## 1. A small synthetic panel

The demo panel has one target (`demand_index`) and two predictors
(`orders_idx`, `sentiment_idx`) that the target is built from with a one-month
lag, so a model that can see the predictors has a genuine edge over a model
that only sees the target's own history:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 180
dates = pd.date_range("2005-01-31", periods=n, freq="ME")
orders_idx = rng.normal(0.0, 1.0, n)
sentiment_idx = rng.normal(0.0, 1.0, n)
demand_index = np.zeros(n)
for t in range(1, n):
    demand_index[t] = (
        0.05 * demand_index[t - 1]
        + 1.3 * orders_idx[t - 1]
        + 1.1 * sentiment_idx[t - 1]
        + rng.normal(0.0, 0.3)
    )
frame = pd.DataFrame(
    {
        "date": dates,
        "demand_index": demand_index,
        "orders_idx": orders_idx,
        "sentiment_idx": sentiment_idx,
    }
)
frame.to_csv("tutorial_panel.csv", index=False)
print(frame.head())
```

```text
        date  demand_index  orders_idx  sentiment_idx
0 2005-01-31      0.000000    0.001230      -0.441145
1 2005-02-28     -0.332604    0.298746      -0.507961
2 2005-03-31      0.374245   -0.274138       0.630083
3 2005-04-30      0.533016   -0.890592      -0.301868
4 2005-05-31     -1.446430   -0.454671      -0.151444
```

If your panel is already an in-memory `DataFrame` rather than a file on disk,
use `mf.data.custom_dataset(...)` instead of the CSV loader below -- both
return the same `DataBundle`. See
[custom_dataset](../reference/custom/custom_dataset.md) for the full contract
of either path.

## 2. Load it with `load_custom_csv`

```python
import macroforecast as mf

bundle = mf.data.load_custom_csv(
    "tutorial_panel.csv",
    date="date",
    dataset="tutorial_demo",
    frequency="monthly",
)
print(bundle.panel.shape)
print(bundle.metadata["dataset"], bundle.metadata["frequency"])
print(bundle.metadata["transform_codes"])
```

```text
(180, 3)
tutorial_demo monthly
{}
```

`transform_codes` is empty: nothing here told the loader which FRED-style
stationarity transform (McCracken-Ng t-code) each column uses, because this
data was never a FRED panel. That empty dict is exactly what makes the next
step raise.

## 3. Declare the target -- and meet a real error

FRED loaders resolve a `TargetSpec`'s forecast policy and transform from the
column's t-code automatically. Custom data has no t-code, so leaving
`TargetSpec(transform=...)` unset does not silently pick something reasonable
-- it raises:

```python
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

try:
    pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="demand_index")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start="2015-01-31", test_end="2019-06-30", horizon=1,
        ),
        arms=[Arm(name="AR", model="ar", is_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR"),
    )
except ValueError as exc:
    print(f"{type(exc).__name__}: {exc}")
```

```text
ValueError: target 'demand_index': no explicit transform and no usable t-code (provide TargetSpec(transform=...) or transform_codes metadata)
```

The fix is exactly what the message says -- give the target an explicit
transform. `demand_index` here is already a stationary synthetic series, so
`"level"` is the right choice (use `"change"`, `"growth"`, or `"log_growth"`
for a series that still needs differencing; see the {term}`t-code` glossary
entry for what each FRED t-code maps to):

```python
target_spec = TargetSpec(name="demand_index", transform="level", policy="direct")
```

## 4. Your own model

`mf.models.custom_model` wraps any `fit(X, y) -> fitted; fitted.predict(X)`
callable into a `ModelSpec` that the runner treats exactly like a built-in
model name. Here it wraps a plain ridge regression:

```python
from sklearn.linear_model import Ridge


def fit_ridge(X, y, *, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y.to_numpy().ravel())
    return model


ridge_spec = mf.models.custom_model(
    "ridge_demo", fit_ridge, default_params={"alpha": 1.0},
)
print(ridge_spec.name, ridge_spec.input_kind)
```

```text
ridge_demo supervised
```

Give it an explicit `FeatureSpec` that actually uses the panel -- an arm with
`features=None` does NOT mean "no predictors"; see the
[Models and Arms](concepts/models_and_arms.md) page and the default-feature-spec
`UserWarning` if you hit it on your own data:

```python
features = mf.feature_engineering.feature_spec(
    target="demand_index", predictors="all", lags=(0, 1),
)
```

## 5. Window, pipeline, and the horse race

`Arm(model="ar", is_benchmark=True)` is the benchmark -- an autoregression
using only `demand_index`'s own history -- run head to head against
`Arm(model=ridge_spec, features=features)`:

```python
window = mf.window.from_cutoffs(test_start="2015-01-31", test_end="2019-06-30", horizon=1)

spec = pipeline_spec(
    data=bundle,
    targets=[target_spec],
    horizons=[1, 3],
    window=window,
    arms=[
        Arm(name="AR", model="ar", is_benchmark=True),
        Arm(name="Ridge", model=ridge_spec, features=features),
    ],
    evaluation=EvalSpec(benchmark="AR", tests=("dm", "mcs")),
)
report = run_pipeline(spec)
print(report.accuracy[["target", "horizon", "contender", "rmse", "relative_mse", "r2_oos"]])
```

```text
      target  horizon contender     rmse  relative_mse    r2_oos
demand_index        1        AR 0.299713      1.000000  0.000000
demand_index        1     Ridge 0.295677      0.973249  0.026751
demand_index        3        AR 1.615274      1.000000  0.000000
demand_index        3     Ridge 1.627858      1.015641 -0.015641
```

At horizon 1, `Ridge` has the lower RMSE (`relative_mse` below 1) because it
can see `orders_idx`/`sentiment_idx` directly, exactly the predictors
`demand_index` was built from one month later. At horizon 3 the edge is gone
-- these two predictors carry no information three months ahead in this
synthetic design, and `Ridge` is (very slightly) worse than the AR benchmark
there. `report.significance` shows the horizon-1 edge does not clear
conventional significance in this small sample:

```python
print(report.significance)
```

```text
      target  horizon contender   dm_stat     dm_p
demand_index        1     Ridge -1.427501  0.159301
demand_index        3     Ridge  1.444998  0.154345
```

This is an honest, small-sample result, not a rigged one: a genuinely useful
predictor does not guarantee a statistically significant win over 54 test
origins. Real studies run longer windows, more horizons, and report the
Model Confidence Set alongside DM, exactly as `report.mcs` already does here:

```python
print(report.mcs)
```

```text
      target  horizon contender  in_mcs
demand_index        1        AR    True
demand_index        1     Ridge    True
demand_index        3        AR    True
demand_index        3     Ridge    True
```

## 6. One line to a referee-ready table

`report.accuracy`, `report.significance`, and `report.mcs` are three separate
long frames. `mf.reporting.paper_accuracy_table` joins them into the wide
models-by-horizons table a paper actually publishes -- rel-RMSE, DM
significance stars, and an MCS marker, one row per model, one column per
horizon:

```python
table = mf.reporting.paper_accuracy_table(report)
print(table.data)
print(table.to_latex(booktabs=True))
```

```text
         Model     h1     h3
AR (benchmark) 1.000† 1.000†
         Ridge 0.987† 1.008†
```

```text
\begin{table}[!htbp]
\centering
\caption{Forecast accuracy — demand\_index}
\begin{tabular}{lll}
\toprule
Model & h1 & h3 \\
\midrule
AR (benchmark) & 1.000† & 1.000† \\
Ridge & 0.987† & 1.008† \\
\bottomrule
\end{tabular}
\\[-0.2em]{\footnotesize Entries are rel-RMSE relative to the benchmark (AR); the benchmark's own value is 1.000 by construction.}
\\[-0.2em]{\footnotesize Significance markers: *** p<=0.01, ** p<=0.05, * p<=0.1. (Diebold-Mariano test vs. the benchmark).}
\\[-0.2em]{\footnotesize † denotes inclusion in the Model Confidence Set.}
\end{table}
```

Neither model earns a significance star at these thresholds in this
54-origin demo (matching `report.significance` above), and both stay in the
Model Confidence Set (the † marker on every cell) -- the table is telling the
truth about a small sample, exactly as it would on a real study with a
genuinely weak edge. `.to_html()` and `.to_markdown()` render the same
`ReportTable` for a notebook or a README instead of a paper.

## Where to go next

- [custom_dataset](../reference/custom/custom_dataset.md) and
  [custom_model](../reference/custom/custom_model.md) -- the full contract for
  each stage used above, including `custom_model_ensemble` and the file-loader
  variants.
- [paper_accuracy_table reference](../reference/reporting.md#paper_accuracy_table)
  -- every argument, including multi-target reports and dropping the
  benchmark row.
- [Getting Started](getting_started.md) and the
  [Replication Gallery](gallery.md) -- the same shape of study on FRED-MD with
  built-in models and richer feature engineering.
- [Models and Arms](concepts/models_and_arms.md) -- what an arm's implicit
  default feature spec actually resolves to, and when the runner warns about it.

## Reference

- [Data reference](../reference/data.md) — `load_custom_csv`, `load_custom_parquet`, `custom_dataset`.
- [Models reference](../reference/models.md) — `custom_model`, `ModelSpec`.
- [Pipeline reference](../reference/pipeline.md) — `Arm`, `TargetSpec`, `pipeline_spec`, `run_pipeline`, `PipelineReport`.
- [Reporting reference](../reference/reporting.md) — `paper_accuracy_table` and the other paper-facing table builders.
