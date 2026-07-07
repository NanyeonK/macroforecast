# Workflow Contract

[Back to reference](index.md)

The current architecture is package-surface first. Users call Python functions and dataclass specs directly; the removed layered YAML/ops registry is not part of the live workflow.

## Module Ownership

| Module | Owns |
| --- | --- |
| `macroforecast.data` | Loads or wraps a canonical pandas panel plus metadata. |
| `macroforecast.preprocessing` | Builds reusable preprocessing specs and applies window-local transforms. |
| `macroforecast.window` | Defines estimation, validation, and test splits. |
| `macroforecast.feature_engineering` | Builds target columns and predictor matrices. |
| `macroforecast.models` | Fits individual model families and owns model specs. |
| `macroforecast.model_selection` | Tunes model-owned parameters. |
| `macroforecast.forecasting` | Runs one model/target/horizon forecast job. |
| `macroforecast.pipeline` | Runs and evaluates full multi-arm studies. |
| `macroforecast.metrics` and `macroforecast.tests` | Score forecasts and run forecast-comparison tests. |
| `macroforecast.output` and `macroforecast.reporting` | Collect artifacts and render tables. |

## Runner Shape

A full study is declared as `macroforecast.pipeline.pipeline_spec(...)` and executed by `macroforecast.pipeline.run_pipeline(...)`. A single-model run is declared directly through `macroforecast.forecasting.run(...)`.

The reference pages are generated from these importable modules. No generated page depends on the removed core ops registry or layer-spec module.
