# macroforecast

`macroforecast` is a pandas-first framework for reproducible macroeconomic
forecasting studies. It loads canonical macro panels, builds leak-aware
preprocessing and feature pipelines, fits competing model arms, evaluates them
with forecast-comparison tests, and renders research-ready outputs.

Hosted documentation: <https://macroforecast.readthedocs.io/>

## Install

From PyPI:

```bash
pip install "macroforecast"
pip install "macroforecast[all]"      # common optional model/reporting extras
```

From a clone:

```bash
git clone https://github.com/NanyeonK/macroforecast.git
cd macroforecast
pip install -e ".[dev]"
```

Useful extras: `parquet`, `xgboost`, `lightgbm`, `catboost`, `arch`, `plots`,
`macro_random_forest`, `interpretation`, `deep`, `docs`, `typecheck`,
`markdown`, `ci`, `dev`, and `all`.

## Quick Use

```python
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

mf.configure(random_seed=42, n_jobs=1)

bundle = mf.data.load_fred_md()
window = mf.window.from_cutoffs(test_start="2010-01-01", test_end="2019-12-01")

arms = [
    Arm(
        name="AR",
        model="ar",
        is_benchmark=True,
        features=mf.feature_engineering.feature_spec(
            target="INDPRO",
            predictors=[],
            lags=None,
            target_lags=(1, 2, 3),
        ),
    ),
    Arm(
        name="RF",
        model="random_forest",
        features=mf.feature_engineering.feature_spec(
            target="INDPRO",
            predictors=["UNRATE", "CPIAUCSL", "FEDFUNDS", "HOUST", "PAYEMS"],
            lags=(0, 1),
        ),
        model_selection={"random_forest": None},
    ),
]

spec = pipeline_spec(
    data=bundle,
    targets=[TargetSpec(name="INDPRO")],
    horizons=[1],
    window=window,
    arms=arms,
    evaluation=EvalSpec(benchmark="AR"),
)

report = run_pipeline(spec)
print(report.accuracy)
```

This is the shortest real study path. The
[Getting Started guide](https://macroforecast.readthedocs.io/en/latest/guide/getting_started.html)
walks through the same flow, and
[Your Data, Your Model, One Table](https://macroforecast.readthedocs.io/en/latest/guide/custom_data_tutorial.html)
shows the custom-data/custom-model path.

## What It Covers

- Data: FRED-MD, FRED-QD, FRED-SD, custom CSV/parquet panels, metadata, and
  real-time vintages.
- Preprocessing and features: official transform codes, outliers, EM
  imputation, standardization, lags, MARX, factors, filters, and feature
  selection.
- Models and arms: benchmarks, linear/regularized models, time-series models,
  tree/boosting models, neural models, volatility models, model ensembles, and
  custom `ModelSpec` hooks.
- Evaluation: RMSE and relative scores, Diebold-Mariano, Clark-West,
  Giacomini-White, MCS, SPA/uSPA/aSPA, and custom loss/test hooks.
- Output: result stores, provenance, checkpoints, artifact manifests, Markdown,
  HTML, and LaTeX reporting.

## Check A Wheel Install

This smoke check does not require a repository checkout:

```bash
python - <<'PY'
import macroforecast as mf

print("macroforecast", mf.__version__)
print(mf.models.list_model_specs()[["name", "family", "default_preset"]].head())
PY
```

## License

MIT
