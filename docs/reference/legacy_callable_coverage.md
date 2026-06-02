# Legacy Callable Coverage

[Back to reference](index.md)

This page tracks the old runtime/reference surface against the current
callable-first package. It is a migration audit, not a public recipe grammar.

Current verdict: the old statistical functionality is represented as callable
modules. The intentionally removed surface is YAML/runtime/registry plumbing
that will be rebuilt later as a wrapper around the callable API.

Last updated: 2026-06-01.

## Intentional Removals

These old surfaces are not gaps.

| Old surface | Current decision | Reason |
| --- | --- | --- |
| Old recipe schemas, validators, and parse helpers | Removed for now | Recipe/YAML support will be rebuilt as a thin wrapper over callables. |
| Old runtime dispatch nodes and materializers | Removed for now | Runner and direct callables now own execution. |
| Old compatibility shims and registries | Removed | Restoring compatibility shims would preserve the structure we are replacing. |
| `api/functions/*`, `functions/*` compatibility namespaces | Removed | Public API is now module-based: `data`, `preprocessing`, `feature_engineering`, etc. |
| `option_docs.py` generated option tables | Removed | Function-level docs now document input, output, defaults, and choices directly. |
| Recipe/gallery contract pages | Removed | Recipe docs will be rewritten after the callable API stabilizes. |
| Paper-method recipe constructors | Removed for now | Paper methods should be expressed as callable combinations or future recipe presets. |
| Holiday feature op | Intentionally omitted | Package targets monthly and quarterly macro panels; public-holiday calendars are not a core axis. |

## Covered Callable Surface

| Area | Current module | Coverage |
| --- | --- | --- |
| Global defaults | `macroforecast.meta` | Package-level config, seed/default management, context manager. |
| Data panels and metadata | `macroforecast.data` | Canonical pandas panel, metadata attachment, custom CSV/parquet loading, FRED-MD/QD/SD loaders, vintage listing, frequency inference/alignment, Chow-Lin disaggregation, same-period policy, regime metadata. |
| Data summary and comparison | `macroforecast.data_analysis` | Single-panel summary and before/after analysis. |
| Preprocessing | `macroforecast.preprocessing` | FRED-style transform codes, FRED-SD code suggestions, outlier handling, imputation, standardization, frame-edge handling, frequency alignment, reusable fit/transform specs. |
| Feature construction | `macroforecast.feature_engineering` | Direct, average, and path targets; lags; rolling means; MARX/MAF; mixed-frequency lags; time/season dummies; transforms; filters; polynomial/interactions; PCA, group PCA, sparse PCA, varimax, PLS, SIR, DFM; random projection/Nystroem; selection methods; reusable feature specs. |
| Feature analysis | `macroforecast.feature_analysis` | Overview, correlation, target correlation, factor diagnostics, lag diagnostics, MARX diagnostics, stage shift, selection stability/similarity, custom diagnostics. |
| Window and policies | `macroforecast.window` | Estimation/validation/test windows, expanding/rolling/fixed modes, blocked splits, origin generation, retrain/retune timing policies, stage panels. |
| Model selection | `macroforecast.model_selection` | Fixed/grid/random/Bayesian/genetic/custom search specs, distributions, parameter selection over supplied windows. |
| Models | `macroforecast.models` | Linear/penalized/group/adaptive, supervised factor models, PCR-style composites through factor models, SVR, tree, MARS, Macro Random Forest, MIDAS/mixed-frequency, AR/VAR/BVAR/FAR/FAVAR/DFM, volatility, torch-backed NN/LSTM/GRU/Transformer/HNN, model-owned search spaces, save/load helpers. |
| Model ensemble | `macroforecast.model_ensemble` | Fit-time member-model composition: bagging, subagging, random subspace, stacking, Super Learner, Booging, and model-ensemble-owned search spaces. |
| Forecast runner | `macroforecast.forecasting` | Runner composition across windows, preprocessing specs, feature specs, model specs, model ensembles, model selection, direct/recursive/path-average policies, forecast-output combinations. |
| Metrics | `macroforecast.metrics` | Point, relative, directional, interval, density, quantile, QLIKE, Theil, OOS R-squared, ranking. |
| Tests | `macroforecast.tests` | DM/GW/DMP/HN, nested/encompassing tests, CPA, exact MCS, SPA/Reality/StepM benchmark tests, blocked OOB reality check, density/interval tests, PIT diagnostics, residual diagnostics, custom tests. |
| Evaluation reports | `macroforecast.evaluation` | Multi-slice evaluation, OOS filtering, benchmark comparisons, regime scoring, error decomposition, aggregation and ranking. |
| Forecast analysis | `macroforecast.forecast_analysis` | Forecast overview, fitted-vs-actual, residual views, rolling loss, scale view, tuning traces, coefficient/parameter stability, DFM diagnostics, ensemble diagnostics, stage update trace. |
| Interpretation | `macroforecast.interpretation` | Native coefficients/importances, permutation/Strobl/LOFO, PDP/ALE/Friedman-H, SHAP variants, forecast decomposition, cumulative contribution, group/lineage/transformation attribution, OLS attention, MRF GTVP, VAR IRF/FEVD/historical decomposition, neural attribution, LSTM hidden states, custom interpretation. |
| Output | `macroforecast.output` | Output table generation, selection/naming/bundling, artifact writing, manifest, hashes, provenance, gzip/zip compression. |
| Reporting | `macroforecast.reporting` | Report-table formatting, accuracy/model-comparison/forecast-test presets, LaTeX/HTML/Markdown rendering, figure-ready data, report bundles. |
| Custom extensions | Per-module custom hooks | Custom datasets, preprocessing, feature steps, models, model ensembles, stage policies, combinations, diagnostics, interpretation, and artifacts. See `reference/custom/`; there is no separate `macroforecast.custom` module. |

## Remaining Real Gaps

These are not old runtime cleanup items. They are possible future
implementation work on top of the current callable API.

| Priority | Gap | Current status | Proposed handling |
| --- | --- | --- | --- |
| Medium | Replication/report package builder | `output` writes artifacts and `reporting` renders tables, but there is no one-call replication package assembler. | Add as future callable after output/reporting settle; likely `reporting` or a small package module, not `output`. |
| Low | External reference-verification expansions | `tests/reference/` now contains core anchors, but not every paper formula against external source code or known-DGP simulations. | Expand the reference suite incrementally as source-code fixtures or accepted known-DGP designs are added. |
| Low | Future YAML/recipe wrapper | Intentionally absent. | Rebuild later on top of the callable API, with no old runtime registry dependency. |

## No Current Action Needed

These areas looked like gaps in the old tree but are now covered or
intentionally excluded:

| Item | Status |
| --- | --- |
| Legacy diagnostic ops | Covered by `feature_analysis`, `forecast_analysis`, and `data_analysis`. |
| Old output ops | Covered by `output` plus the new `reporting` module. |
| Legacy MIDAS feature/model split | Covered by `mixed_frequency_lags()` and MIDAS model callables. |
| Legacy Chow-Lin op | Covered by `data.chow_lin_disaggregate()` and `data.align_frequency(..., quarterly_to_monthly="chow_lin")`. |
| Legacy blocked OOB reality check | Covered by `tests.blocked_oob_reality_check()`. |
| Full iterative MCS | Covered by exact `tests.model_confidence_set()`; `tests.iterative_model_confidence_set()` remains a descriptive alias. |
| Reporting presets | Covered by `reporting.accuracy_table()`, `reporting.model_comparison_table()`, and `reporting.forecast_test_table()`. |
| Holiday features | Intentionally excluded for monthly/quarterly macro workflows. |
