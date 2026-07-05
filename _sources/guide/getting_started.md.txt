# Getting Started

[Back to User Guide](index.md)

`macroforecast` runs a pseudo-out-of-sample forecasting study from a single
declarative specification. You describe the data, the targets, the forecast
horizons, the evaluation window, the competing models, and the scoring rule
once, and `run_pipeline` executes every combination and scores it. This page
takes you from install to a first result, then to a full study.

## Installation

Install the package in editable mode:

```bash
pip install -e "."          # minimal
pip install -e ".[dev]"     # development extras
pip install -e ".[docs]"    # documentation tools
```

Python 3.10 or later is required. Torch is not installed by default and is only
needed for the neural-network model families. The project uses `uv` as the
environment manager in development, and all dependency groups are defined in
`pyproject.toml`.

## Key concepts

Five ideas cover almost everything. The [User Guide](index.md) stage pages
explains each in full and the [Glossary](glossary.md) defines every term.

**The spec.** A `pipeline_spec` bundles the data, targets, horizons, window,
arms, and evaluation rule into one frozen object. Running it returns a
`PipelineReport` with accuracy tables, significance tests, and raw forecasts.

**Arms.** An arm is one complete recipe of a preprocessing choice, a feature
set, and a single model. The report compares arms head to head, so swapping a
model or feature set means adding another arm rather than rewriting the pipeline.

**Targets.** A `TargetSpec` names the series you forecast and carries its own
transform and forecast policy. For FRED series these resolve from the official
transformation code, so `INDPRO` becomes a growth-rate forecast automatically.

**Windows.** A window defines the expanding or rolling estimation sample, an
optional validation block for model selection, the test points, and the refit
cadence. Everything is leak-aware, so no observation dated after the forecast
origin can enter training.

**Evaluation.** Each arm is scored against a benchmark arm. Beyond RMSE, the
report adds the Diebold-Mariano and Clark-West tests for pairwise significance
and the Model Confidence Set for the joint set of best-performing models.

## Quickstart

The snippets below move from the smallest possible run, to a full study, to
reproducing a published paper. Each one is self-contained.

### A single forecast

The minimal run pits an AR benchmark against a random forest on one target over
a short test span.

```python
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

# Load FRED-MD (downloads if not cached; returns DataBundle).
bundle = mf.data.load_fred_md()

# Two arms: an AR benchmark and a random forest.
arms = [
    Arm(name="AR", model="ar", is_benchmark=True),
    Arm(name="RF", model="random_forest"),
]

# A minimal expanding window over a short test span; one target.
window = mf.window.from_cutoffs(test_start="2010-01-01", test_end="2019-12-01")
targets = [TargetSpec(name="INDPRO")]

spec = pipeline_spec(
    data=bundle,
    targets=targets,
    horizons=[1],
    window=window,
    arms=arms,
    evaluation=EvalSpec(benchmark="AR"),
)
report = run_pipeline(spec)
print(report.accuracy)       # relative-accuracy table by target/horizon/arm
```

### A full study

A full study adds preprocessing, feature engineering, several horizons, and the
significance tests. The arms now differ in their feature sets, and the report
compares them across the whole horizon grid.

```python
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

mf.configure(random_seed=42, n_jobs=1)
bundle = mf.data.load_fred_md()

# Preprocessing: official t-code transforms, IQR outliers, EM-factor imputation.
prep = mf.preprocessing.preprocess_spec(
    transform="official", outliers="iqr", impute="em_factor", standardize="zscore",
)

# Expanding window with a last-block validation split for model selection.
window = mf.window.from_cutoffs(
    test_start="1985-01-01", test_end="2019-12-01",
    mode="expanding", val_method="last_block", horizon=1, step=1,
)

# Feature engineering: a MARX moving-average lag ladder over all predictors.
features = mf.feature_engineering.feature_spec(
    target="INDPRO", predictors="all", lags=None,
    feature_steps=[mf.feature_engineering.marx_step(name="MARX_X", max_lag=12)],
)

# AR benchmark uses target-only lags; RF adds the MARX predictors.
arms = [
    Arm(
        name="AR", model="ar", is_benchmark=True,
        features=mf.feature_engineering.feature_spec(
            target="INDPRO", predictors=[], lags=None, target_lags=range(1, 13),
        ),
    ),
    Arm(name="RF", model="random_forest", preprocessing=prep, features=features),
]

spec = pipeline_spec(
    data=bundle,
    targets=[TargetSpec(name="INDPRO")],
    horizons=[1, 3, 6, 12],
    window=window,
    arms=arms,
    evaluation=EvalSpec(
        benchmark="AR",
        metrics=("rmse", "relative_mse", "r2_oos"),
        tests=("dm", "cw", "mcs"),
    ),
)
report = run_pipeline(spec)

print(report.accuracy)       # relative-accuracy table by target/horizon/arm
print(report.significance)   # DM and CW p-values
print(report.mcs)            # Model Confidence Set membership
```

A fully annotated, step-by-step version of this pipeline lives in the
[Replication Gallery](gallery.md#a-complete-pipeline-step-by-step).

### Reproduce a published study

Replications ship as runnable scripts rather than a single call. Run one in
smoke mode to check your install, then see the [Replication Gallery](gallery.md)
for the full studies and the paper-versus-code comparison notes.

```bash
python -m scripts.replication.gcls_2021_pipeline.run_pipeline_full --smoke
```

## What Comes Next

The [User Guide](index.md) stage pages explain each step in detail.
The [Glossary](glossary.md) defines every term used above.
The [Models and Features](model_overview.md) page lists every feature step and registered model.
The [Replication Gallery](gallery.md) shows full published-paper replication examples.

## Reference

- [Data](../reference/data.md)
- [Preprocessing](../reference/preprocessing.md)
- [Window](../reference/window.md)
- [Feature Engineering](../reference/feature_engineering.md)
- [Pipeline](../reference/pipeline.md)
- [Forecasting](../reference/forecasting.md)
- [Evaluation](../reference/evaluation.md)
