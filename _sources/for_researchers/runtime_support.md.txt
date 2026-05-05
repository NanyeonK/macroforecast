# Runtime Support Matrix

This page describes what the current core runtime executes directly, and what is currently schema-validated but delegated to future specialized runtimes.

macroforecast has two execution surfaces:

- `macroforecast.execution`: the legacy experiment engine used by older recipes and tests.
- `macroforecast.core.runtime`: the new layer-contract runtime used by L0-L8 and L1.5-L4.5 artifacts.

The support matrix below refers to `macroforecast.core.runtime.execute_minimal_forecast` unless stated otherwise.

## End-to-End Supported Path

The current core runtime executes one complete layer-contract path:

1. L1 loads a custom inline/CSV/Parquet panel, or official FRED-MD/FRED-QD from the raw adapters.
2. L2 applies selected transform, outlier, imputation, and frame-edge policies.
3. L3 executes deterministic feature DAG operations.
4. L4 fits expanding-window linear sklearn models and produces point forecasts.
5. L5 computes point metrics, benchmark-relative metrics, rankings, and report tables.
6. L1.5-L4.5 optionally materialize diagnostic JSON artifacts.
7. L6 optionally materializes statistical-test result dictionaries from realized forecast errors.
8. L7 optionally materializes basic importance artifacts from fitted models and L3 metadata.
9. L8 optionally writes a reproducible output directory with manifest, recipe, CSV, and JSON sidecars.

## Layer Support

| Layer | Runtime status | Supported now | Not yet full runtime |
|---|---|---|---|
| L0 Meta | Schema/resolution | Recipe metadata and seed pass-through where present | Full study registry integration |
| L1 Data | Partial runtime | Custom panel inline/records/CSV/Parquet; official FRED-MD/FRED-QD loader path | Official-plus-custom merge; every future dataset policy |
| L1.5 Data summary | Runtime artifact | Sample coverage, summary stats, missing/outlier audit, optional correlation as JSON metadata | Figure/table rendering |
| L2 Preprocessing | Partial runtime | No transform, t-code transform, IQR/z-score/winsorize outliers, mean/ffill/interpolation imputation, frame-edge policies | EM factor imputation and every advanced cleaning algorithm |
| L2.5 Pre/post diagnostics | Runtime artifact | Raw-vs-clean comparison, distribution shift, cleaning log, optional correlation shift | Figure/table rendering and multi-stage L2 hooks |
| L3 Features | Partial runtime | Source selection, lag, seasonal lag, moving average, MARX, concat, scale, log/diff/log_diff/pct_change, polynomial, interaction, season dummy, trend, direct target construction | Factor models, mixed-frequency disaggregation, feature selection runtime |
| L3.5 Feature diagnostics | Runtime artifact | Raw/clean/features comparison, feature summary, lineage summary, correlation, lag/factor/selection flags | Factor scree/loadings figures, selection stability figures |
| L4 Forecasting | Partial runtime | Expanding-window `ols`, `ridge`, `lasso`, `elastic_net` via sklearn; point forecasts; benchmark flag propagation | Tree/boosting/deep/VAR/MRF execution in the core runtime path |
| L4.5 Generator diagnostics | Runtime artifact | Forecast/model/training/fit/window summaries | Residual figures, tuning traces, ensemble weight figures |
| L5 Evaluation | Runtime | MSE/RMSE/MAE, relative MSE, OOS R2, relative MAE, MSE reduction, ranking | Density metrics and advanced decomposition computation |
| L6 Statistical tests | Minimal runtime | DM/GW-style pair tests, nested-lite comparisons, CPA paths, MCS-style inclusion, direction tests, residual diagnostics | Full bootstrap MCS/SPA/RC/StepM, exact HAC/West critical values, density/interval tests |
| L7 Interpretation | Minimal runtime | Linear coefficients, permutation-style importance, grouping, lineage attribution, transformation attribution table | SHAP backend, neural-gradient methods, VAR FEVD/IRF, rolling recompute, plot rendering |
| L8 Output | Runtime | Output directory, manifest JSON/JSONL, recipe JSON, forecasts CSV, metrics/ranking CSV, tests/importance/diagnostic JSON | Parquet/LaTeX/HTML rendering, compression, full replication API |

## Practical Meaning

Use the core runtime today when you need a reproducible, inspectable linear-model forecasting study with layer artifacts. It is appropriate for:

- smoke tests for a full recipe shape,
- custom-panel regression checks,
- benchmark-relative point forecast evaluation,
- lightweight statistical and importance artifacts,
- output-directory/provenance integration tests.

Use legacy execution APIs or future specialized execution paths for:

- large model horse races with tree/deep/VAR families,
- exact paper-grade bootstrap statistical tests,
- SHAP/NN/VAR interpretation backends,
- rendered diagnostics and publication exports.

## Minimal Core Runtime Example

```python
from macroforecast.core import execute_minimal_forecast

result = execute_minimal_forecast(open("examples/recipes/l4_minimal_ridge.yaml").read())

print(result.sink("l5_evaluation_v1").metrics_table)
```

If `8_output` is included in the recipe, inspect `result.sink("l8_artifacts_v1").output_directory` for exported files.
