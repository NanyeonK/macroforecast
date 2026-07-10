# Preprocessing

[Back to User Guide](../index.md)

`macroforecast.preprocessing` turns a canonical pandas panel from
`macroforecast.data` into a processed panel plus metadata. The preferred input is
a `DataBundle` or `DataSpec`; the output is a `PreprocessedData` object. The
default `reprocess()` path follows the public McCracken-Ng FRED-MD Matlab
workflow for FRED-MD/FRED-QD style panels.

For use inside a POOS runner, `preprocess_spec` stores the preprocessing choices
without executing them. The runner then applies the spec at each origin, refitting
stateful steps (outlier thresholds, EM factors, standardization scale) only on
the estimation-window rows available at that origin. This is the leak-free path.

## Key Callables

`mf.preprocessing.reprocess` applies the full preprocessing sequence to a panel
immediately (full-sample; for exploration and single-shot use).

`mf.preprocessing.preprocess_spec` stores preprocessing choices for runner-fitted
execution. Pass the returned `PreprocessSpec` to `forecasting.run` or to an
`Arm` in `pipeline_spec`.

```python
import macroforecast as mf

# Full-sample preprocessing (for exploration).
processed = mf.preprocessing.reprocess(
    data_spec,
    transform="official",   # apply McCracken-Ng t-codes
    outliers="iqr",         # IQR-based outlier replacement
    impute="em_factor",     # EM algorithm factor imputation
    standardize="zscore",
)

# Deferred preprocessing spec for runner-fitted execution (leak-free).
prep_spec = mf.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="em_factor",
    standardize="zscore",
)
# Pass prep_spec to Arm(..., preprocessing=prep_spec) or forecasting.run(...).
```

For deterministic missing-value handling, `impute="zero"` replaces remaining
missing cells with zero before standardization. The same option is available in
`preprocess_spec(...)`; under `policy="fit_window"` the zero-imputation state is
fit on the training window and then applied before the fitted standardization
state.

For predictor-only standardization through the current forecast origin, use
`preprocess_spec(..., standardize="zscore",
standardize_scope="origin_available_predictors")`. In origin-aware runner
transforms this fits predictor scaling on rows observable at the origin,
including the current prediction row, and applies that state to returned rows.
Rows after the origin never contribute to the fitted scaling state. Target
columns are excluded even when the realized future target row is present; the
known full-block target-y standardization convention is a look-ahead leak and is
not implemented as a package option.

## Executed walkthrough

Running the full-sample path on the loaded `data_spec` applies the t-code
transforms, flags outliers, imputes by EM factor, and standardizes:

```python
processed = mf.preprocessing.reprocess(
    data_spec,
    transform="official", outliers="iqr", impute="em_factor", standardize="zscore",
)
print(type(processed).__name__, processed.panel.shape)
print("NaN before:", int(bundle.panel.isna().sum().sum()),
      "| NaN after:", int(processed.panel.isna().sum().sum()))
print(processed.panel.iloc[:3, :4])
```

```text
PreprocessedData (694, 128)
NaN before: 942 | NaN after: 0
                 RPI   W875RX1  DPCERA3M086SBEA  CMRMTSPLx
date
1960-03-01 -0.166574 -0.325648         2.178270  -2.831788
1960-04-01  0.132535  0.205311         2.425007   0.704389
1960-05-01 -0.026582  0.006684        -4.410584  -3.134124
```

The output panel is stationary and standardized. The row count drops from 708 to
694 because the official transforms difference the early observations away, and
the EM-factor step fills the 942 missing entries, leaving none remaining. This
is the full-sample path for exploration; inside a runner, `preprocess_spec`
refits these same steps on each origin's estimation rows only.

## Custom Steps And Caches

Custom preprocessing steps run after the built-in transform/outlier/impute/
standardize/frame sequence. In the normal `origin_available` runner path, fitted
standardization is applied before custom steps at transform time, matching the
fit-time order, so scale-sensitive custom steps see the same units on train and
test rows.

Disk-backed preprocessing caches use content-derived identities. If a
`preprocess_spec(custom_steps=...)` contains a custom callable, set a stable
`func.__mf_digest__` string before using `preprocessing_cache_dir` or a
`PreprocessorStore`; update the digest when the callable behavior changes. Named
custom callables without a digest still run, but disk get/put is skipped and the
runner recomputes instead of risking stale reuse. Anonymous lambda custom steps
without `__mf_digest__` are rejected because they collide by qualified name.

Under `policy="fit_window"`, built-in outlier/imputation/standardization state is
fitted on the training window and applied to later rows, but custom steps are
re-executed on the apply window. Keep those custom steps row-local/stateless; a
custom step that computes statistics from all rows it receives can read
post-origin rows and leak future information.

## Reference

- [Preprocessing reference page](../../reference/preprocessing.md) — full function list including `plan`, `report`, `apply_transform_codes`, and individual step callables.
