# Custom Extension Overview

[Back to custom extensions](index.md)

Custom hooks are normal Python callables wrapped by small spec builders. The spec records metadata and defaults; the runner still owns splitting, fitting, scoring, and artifact collection.

| Hook | Builder | Returns |
| --- | --- | --- |
| Dataset | `mf.data.custom_dataset(...)`, `mf.data.load_custom_csv(...)` | `DataBundle` |
| Preprocessing | `mf.preprocessing.custom_preprocess_step(...)` | preprocessing step dict |
| Features | `mf.feature_engineering.custom_step(...)` | feature step dict |
| Model | `mf.models.custom_model(...)` | `ModelSpec` |
| Search | `mf.model_selection.custom_search(...)` | `SearchSpec` |
| Forecast combination | `mf.forecasting.custom_combination(...)` | `CombinationSpec` |
| Evaluation test | `mf.tests.custom_test(...)` | callable test wrapper |
| Interpretation | `mf.interpretation.custom_interpretation(...)` | interpretation callable wrapper |
| Output/reporting | `mf.output.write_artifacts(...)`, `mf.reporting.render_tables(...)` | artifacts or rendered tables |
