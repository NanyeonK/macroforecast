# Custom Extension Surface

[Back to reference](../index.md)

`macroforecast` supports custom work by accepting ordinary Python callables at
the stage that owns the behavior. There is no separate custom registry and no
YAML wrapper requirement. A custom function must return the same object shape
that the next stage already expects.

Use this section when a built-in loader, cleaning step, feature transform,
model, search rule, diagnostic, or artifact is not enough.

## Pages

| Page | Use it for |
| --- | --- |
| [Custom Flow](custom.md) | End-to-end custom workflow and stage-local design rule. |
| [custom_dataset](custom_dataset.md) | In-memory panels, custom CSV/Parquet loaders, frequency and metadata contracts. |
| [custom_preprocess](custom_preprocess.md) | Direct preprocessing callables and runner-safe custom preprocessing steps. |
| [custom_features](custom_features.md) | Direct custom feature transforms and fitted feature-spec steps. |
| [custom_model](custom_model.md) | User estimators, fit return objects, model specs, model ensembles, and search spaces. |
| [custom_window_selection_forecasting](custom_window_selection_forecasting.md) | Custom sample policies, hyperparameter search algorithms, and forecast combinations. |
| [custom_evaluation_tests](custom_evaluation_tests.md) | Custom scalar metrics, custom forecast-comparison tests, and custom evaluation slices. |
| [custom_interpretation_analysis](custom_interpretation_analysis.md) | Custom model interpretation, feature diagnostics, and forecast diagnostics. |
| [custom_output](custom_output.md) | Saving custom tables, dictionaries, notes, diagnostics, and manifests. |

## Flow Contract

| Stage | Custom entry point | Must return |
| --- | --- | --- |
| Data | `mf.data.custom_dataset`, `mf.data.load_custom_csv`, `mf.data.load_custom_parquet` | `DataBundle` with canonical panel and metadata. |
| Preprocessing | `mf.preprocessing.custom_preprocess`, `mf.preprocessing.custom_preprocess_step` | `PreprocessedData` or a preprocessing step dictionary. |
| Feature engineering | `mf.feature_engineering.custom_features`, `mf.feature_engineering.custom_step` | Feature `DataFrame` or feature step dictionary. |
| Models | `mf.models.custom_model`, `mf.model_ensemble.custom_model_ensemble` | `ModelSpec`. |
| Windows and selection | `mf.window.custom_stage_policy`, `mf.model_selection.custom_search` | `StagePolicy` or `SearchSpec`. |
| Forecasting | `mf.forecasting.custom_combination` | `CombinationSpec`. |
| Evaluation and tests | callable metric, `mf.tests.custom_test`, custom aggregation mapping | Metric columns, `TestResult`, or `EvaluationReport` tables. |
| Interpretation and analysis | `mf.interpretation.custom_interpretation`, `mf.feature_analysis.custom_feature_diagnostic`, `mf.forecast_analysis.custom_forecast_diagnostic` | Schema-tagged `DataFrame`. |
| Output | `mf.output.write_artifacts({...})` | Files plus `ArtifactManifest`. |

## Navigation

```{toctree}
:maxdepth: 1

custom
custom_dataset
custom_preprocess
custom_features
custom_model
custom_window_selection_forecasting
custom_evaluation_tests
custom_interpretation_analysis
custom_output
```
