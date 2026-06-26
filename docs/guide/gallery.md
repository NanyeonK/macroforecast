# Replication Gallery

[Back to User Guide](index.md)

The studies below show how published macro-forecasting designs are expressed
with the current `macroforecast` API. Each card links to a replication guide
or script set. These are not marketing examples. Each study separates three
objects:

1. the paper specification stated in the main text or appendix,
2. the closest reproducible `macroforecast` setting,
3. the remaining gap when the public paper does not expose a vintage, seed,
   software default, or full replication package.

::::{grid} 2

:::{grid-item-card} GCLS (2021) --- Macroeconomic data transformations matter
:link: ../replication/gcls_2021_replication
:link-type: doc

Goulet Coulombe, Leroux, Stevanovic, and Surprenant (2021). A single, honest
page for the replication: a verification summary (configuration faithfulness, the
two package bugs the run surfaced and fixed, and the residual
R-versus-scikit-learn random-forest divergence), the eight-step leak-free build
(FM, AR, and random-forest cases across FRED-MD targets), and the Appendix B
ground-truth tables the run is measured against.
:::

::::

## Pipeline Scripts

The scripts below are full pipeline runs used in the GCLS replication and
in the ML-Useful (2022) replication exercise. They serve as real-world usage
examples showing the complete `pipeline_spec` / `run_pipeline` workflow.

| Script | Description |
| --- | --- |
| `scripts/replication/gcls_2021_pipeline/run_pipeline_full.py` | Leak-free POOS pipeline for all GCLS (2021) targets with FM, AR, and RF feature cases. Supports `--smoke` mode for quick validation. |
| `scripts/replication/ml_useful_2022/run_full.py` | Full ML-Useful (2022) pipeline run. Sweeps targets, horizons, and model families as in the published exercise. |

## A complete pipeline, step by step

The example below is the full version of the study sketched in
[Getting Started](getting_started.md#a-full-study). Every step is annotated so
that each part of the specification is explained in place: global configuration,
data loading, preprocessing, the estimation window, feature engineering, the
competing arms, the targets, and the evaluation rule. It is a faithful template
for the replication scripts above.

```python
import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)

# 1. Configure global defaults (random seed, worker count).
mf.configure(random_seed=42, n_jobs=1)

# 2. Load FRED-MD (downloads if not cached; returns DataBundle).
bundle = mf.data.load_fred_md()

# 3. Declare preprocessing: official t-code transforms + EM imputation.
#    preprocess_spec stores the choices; the runner applies them per origin.
prep = mf.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="em_factor",
    standardize="zscore",
)

# 4. Build the estimation/val/test window.
#    from_cutoffs is the most common entry point: provide test_start,
#    estimation mode, and validation design.
window = mf.window.from_cutoffs(
    test_start="1985-01-01",
    test_end="2019-12-01",
    mode="expanding",
    val_method="last_block",
    horizon=1,
    step=1,
)

# 5. Declare feature engineering: MARX moving-average lags over the predictors.
#    predictors="all" uses every non-target series; the marx_step builds the
#    increasing-average lag ladder the RF/ML arms consume.
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=None,
    feature_steps=[
        mf.feature_engineering.marx_step(name="MARX_X", max_lag=12),
    ],
)

# 6. Declare arms: one configuration (preprocessing + features + model) each.
#    AR is the benchmark arm using target-only lags; RF adds MARX predictors.
arms = [
    Arm(
        name="AR",
        model="ar",
        features=mf.feature_engineering.feature_spec(
            target="INDPRO",
            predictors=[],
            lags=None,
            target_lags=range(1, 13),
        ),
        is_benchmark=True,
    ),
    Arm(
        name="RF",
        model="random_forest",
        preprocessing=prep,
        features=features,
    ),
]

# 7. Declare targets: resolve transform and forecast policy from t-code.
targets = [TargetSpec(name="INDPRO")]

# 8. Declare evaluation: benchmark arm, metrics, tests.
evaluation = EvalSpec(
    benchmark="AR",
    metrics=("rmse", "relative_mse", "r2_oos"),
    tests=("dm", "cw", "mcs"),
)

# 9. Build the validated, frozen pipeline spec.
spec = pipeline_spec(
    data=bundle,
    targets=targets,
    horizons=[1, 3, 6, 12],
    window=window,
    arms=arms,
    evaluation=evaluation,
)

# 10. Run: execute every (arm, target, horizon) cell and return PipelineReport.
report = run_pipeline(spec)

# 11. Inspect results.
print(report.accuracy)       # relative RMSE table by target/horizon/arm
print(report.significance)   # DM and CW p-values
print(report.mcs)            # Model Confidence Set membership
forecasts = report.forecasts  # full forecast DataFrame
```

```{toctree}
:hidden:
:maxdepth: 1

/replication/gcls_2021_replication
```
