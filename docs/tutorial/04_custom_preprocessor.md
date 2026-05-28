# Bring your own preprocessor

In this tutorial you will register a custom L2 preprocessor, verify that it
receives the data you expect, and integrate it into the benchmarking study
from {doc}`02_full_study`. By the end you will have a working
feature-selection preprocessor that filters columns by variance before any
model sees them.

The prerequisite is that you have completed {doc}`03_custom_model`. The
patterns here are the same — a decorated callable registered by name — but
the contract is slightly different because a preprocessor returns two
matrices instead of one scalar.

---

## Why a custom preprocessor?

L2 is macroforecast's cleaning boundary. It applies transforms,
outlier treatment, imputation, and frame-edge handling to the raw panel. After
L2 finishes, every downstream layer receives an identically processed dataset.

Sometimes the built-in L2 operations are not enough. You might want to:

- Drop columns whose in-sample variance falls below a threshold, so
  low-variance macroeconomic series do not dominate a ridge regression.
- Standardize each predictor block separately rather than the full panel.
- Apply a rolling z-score normalization that respects the expanding-window
  split at each forecast origin.

The `register_preprocessor` API lets you plug these operations into the L2
boundary without modifying the package. The runtime calls your function at
each forecast origin, after all built-in L2 operations have run, and passes
you the train and test splits as separate matrices. You return the
transformed pair. Everything else — provenance recording, manifest entries,
replication hashing — happens automatically.

---

## The preprocessor contract

Your function must accept exactly four arguments and return exactly two values:

```python
def my_preprocessor(
    X_train,   # pd.DataFrame: n_train rows x n_features cols
    y_train,   # pd.Series: n_train target values (read-only)
    X_test,    # pd.DataFrame: 1 row x n_features cols
    context,   # dict: runtime metadata
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...        # return (X_train_transformed, X_test_transformed)
```

Four rules mirror those for custom models:

1. Fit any statistics (e.g., column means, variances, PCA directions) only
   on `X_train`. Never touch `X_test` to estimate parameters.
2. Apply the same transformation to `X_test` using parameters estimated from
   `X_train`. This is the expanding-window contract.
3. Do not modify `y_train`. If you need target-scale transformations, use
   `register_target_transformer` instead.
4. Return DataFrames with the same index as the inputs. Column names may
   change if you drop or rename columns, but the row count must not change.

The `context` dictionary contains metadata fields that help you write
adaptive preprocessors. The fields available to a preprocessor are:

| Field | Type | Description |
|---|---|---|
| `feature_names` | `list[str]` | Column names of the L2 panel before your function runs. |
| `alignment` | `str` | Frequency alignment label, e.g. `"monthly"`. |
| `leakage_contract` | `str` | Leakage policy declared in the recipe. |
| `mode` | `str` | Either `"fit"` (training pass) or `"predict"` (test pass). When mode is `"predict"`, the runtime passes a 1-row `X_test`; your function still receives both matrices. |

---

## Step 1: write and register the preprocessor

Start with a variance-filter preprocessor. It drops columns whose in-sample
variance is below a threshold. This is a simple but realistic operation: FRED-MD
contains some series that are nearly constant over a short window, and those
series add noise without signal.

```python
import numpy as np
import pandas as pd
import macroforecast as mf

@mf.register_preprocessor("variance_filter")
def variance_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns with near-zero in-sample variance.

    The threshold is fixed at 1e-6. Columns below this threshold on X_train
    are dropped from both X_train and X_test using the same column mask,
    preserving the expanding-window contract.
    """
    threshold = 1e-6
    variances = X_train.var(ddof=1)
    keep_cols = variances[variances > threshold].index.tolist()

    if not keep_cols:
        # Safety fallback: if all columns are below threshold, keep them all
        # to avoid passing an empty matrix to downstream models.
        return X_train, X_test

    return X_train[keep_cols], X_test[keep_cols]

# Confirm registration
print(mf.list_custom_preprocessors())   # ('variance_filter',)
```

The function uses only `X_train` to compute variances, then applies the same
column mask to `X_test`. This guarantees that the test matrix never influences
which columns are kept.

---

## Step 2: verify the contract on synthetic data

Before putting the preprocessor into a recipe, verify that it behaves
correctly on a small synthetic dataset. This saves debugging time later.

```python
import numpy as np
import pandas as pd

np.random.seed(0)
n_train = 15
cols = ["varying_1", "varying_2", "constant_col"]

# Build a training matrix with one near-constant column
X_train_raw = pd.DataFrame(
    {
        "varying_1": np.random.randn(n_train),
        "varying_2": np.random.randn(n_train) * 2.0,
        "constant_col": np.full(n_train, 0.5),   # variance = 0
    }
)
X_test_raw = pd.DataFrame(
    {
        "varying_1": [0.12],
        "varying_2": [-0.34],
        "constant_col": [0.5],
    }
)
y_train_raw = pd.Series(np.random.randn(n_train), name="y")

context_mock = {
    "feature_names": cols,
    "alignment": "monthly",
    "leakage_contract": "strict",
    "mode": "fit",
}

X_train_out, X_test_out = variance_filter(
    X_train_raw, y_train_raw, X_test_raw, context_mock
)

print("Columns kept:", X_train_out.columns.tolist())
# ['varying_1', 'varying_2']  — constant_col was dropped

print("X_train shape:", X_train_out.shape)   # (15, 2)
print("X_test shape:", X_test_out.shape)     # (1, 2)

assert "constant_col" not in X_train_out.columns
assert list(X_train_out.columns) == list(X_test_out.columns)
print("Contract OK")
```

Two assertions confirm the two most important properties: the low-variance
column was removed, and both matrices have the same column layout after
filtering.

---

## Step 3: integrate into a recipe

Now put the preprocessor into a minimal benchmarking recipe. The recipe
uses a synthetic inline panel so you do not need FRED credentials.

The preprocessor is activated by setting `custom_preprocessor` in the L2
`leaf_config`. The runtime dispatches the registered callable by name at
each forecast origin, after all built-in L2 operations finish.

```python
import macroforecast as mf

# Registration must happen before mf.run()
@mf.register_preprocessor("variance_filter")
def variance_filter(X_train, y_train, X_test, context):
    threshold = 1e-6
    variances = X_train.var(ddof=1)
    keep_cols = variances[variances > threshold].index.tolist()
    if not keep_cols:
        return X_train, X_test
    return X_train[keep_cols], X_test[keep_cols]

recipe = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_policy: seeded_reproducible
  leaf_config:
    random_seed: 0

data:
  fixed_axes:
    panel_composition: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: gdp_growth
    target_horizons: [1]
    custom_panel_inline:
      date:
        [2015-01-01, 2015-02-01, 2015-03-01, 2015-04-01, 2015-05-01,
         2015-06-01, 2015-07-01, 2015-08-01, 2015-09-01, 2015-10-01,
         2015-11-01, 2015-12-01, 2016-01-01, 2016-02-01, 2016-03-01,
         2016-04-01, 2016-05-01, 2016-06-01, 2016-07-01, 2016-08-01]
      gdp_growth:
        [0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.9,
         0.8, 1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
      ip_index:
        [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5,
         105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5]
      const_series:
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
  leaf_config:
    custom_preprocessor: variance_filter

3_feature_engineering:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: 1}
      inputs: [src_X]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_main
      type: step
      op: fit
      params:
        model: ridge
        forecast_policy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      inputs: [src_X, src_y]
    - id: predict_model
      type: step
      op: predict
      inputs: [fit_main]
  sinks:
    l4_forecasts_v1: predict_model
    l4_model_artifacts_v1: fit_main
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
"""

result = mf.run(recipe)
print(f"Cells: {len(result.cells)}")   # 1
print("sink_hashes_match test run completed")
```

The `custom_preprocessor: variance_filter` entry under L2 `leaf_config` is
the only change relative to a standard recipe. The constant series
`const_series` will be dropped at each forecast origin before features reach
L3.

---

## A complete recipe example: adaptive column selection

The variance filter is the minimal case. Here is a more complete example:
a preprocessor that uses the `feature_names` context field to log which
columns survive, making it easier to audit what the pipeline actually used.

```python
import numpy as np
import pandas as pd
import macroforecast as mf

@mf.register_preprocessor("logged_variance_filter")
def logged_variance_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance filter that reports dropped columns via feature_names context."""
    threshold = 1e-6
    variances = X_train.var(ddof=1)
    keep_mask = variances > threshold
    keep_cols = variances[keep_mask].index.tolist()
    dropped = [c for c in X_train.columns if c not in keep_cols]

    if dropped:
        # context["feature_names"] lists all column names from the L2 panel.
        # Use it to build a human-readable report.
        known_names = context.get("feature_names", [])
        dropped_known = [c for c in dropped if c in known_names]
        # In production, write to a logger. Here we use print for clarity.
        if dropped_known:
            print(f"[logged_variance_filter] dropped {dropped_known}")

    if not keep_cols:
        return X_train, X_test
    return X_train[keep_cols], X_test[keep_cols]

print(mf.list_custom_preprocessors())
# ('variance_filter', 'logged_variance_filter')
```

The context field `feature_names` contains the column names as they exist in
the L2 panel at the time your function is called. Using it here lets you
cross-check that the column names you see in `X_train.columns` match the
names the runtime knows about.

---

## Debugging

Three failure modes appear most often when developing a custom preprocessor.

**Shape mismatch.** The runtime requires that you return exactly two
DataFrames. If you accidentally return a single DataFrame or a tuple of
arrays, you will see a `TypeError` at the first forecast origin. The fix is
to return `(X_train_out, X_test_out)` as a tuple explicitly.

**Column mismatch between train and test.** If your filter logic produces
different column sets for `X_train` and `X_test` — for example, by
computing statistics on `X_test` — the downstream L3 layer will raise a
shape error when it tries to compute lags from an inconsistently shaped
panel. Always apply the column mask estimated from `X_train` to both
outputs:

```python
# Correct: mask estimated from X_train, applied to both
keep_cols = variances_from_train[mask].index
return X_train[keep_cols], X_test[keep_cols]

# Wrong: computing a separate mask on X_test
keep_train = X_train.var() > threshold
keep_test = X_test.var() > threshold   # X_test has 1 row — variance is NaN
```

**Registration not visible to `mf.run()`.** The registry is per-process.
If you call `mf.run()` in a separate script without importing the module
where `@mf.register_preprocessor` runs, the recipe will fail with
`ValueError: unknown preprocessor 'variance_filter'`. Confirm the name is
registered before running:

```python
print(mf.list_custom_preprocessors())   # must include your name
```

Clear the registry between test runs to avoid stale registrations:

```python
mf.clear_custom_preprocessors()
print(mf.list_custom_preprocessors())   # ()
```

---

## What to do next

- See {doc}`../how_to/use_extension_points` for all five extension points:
  custom feature blocks, combiners, preprocessors, target transformers,
  and models.
- See {doc}`../explanation/layer_design` for the full rationale behind
  why the L2 boundary exists and what it prevents.
- See {doc}`03_custom_model` to compare the preprocessor contract with the
  model contract.
