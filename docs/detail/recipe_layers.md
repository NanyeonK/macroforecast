# Recipe Layers

Recipes are the internal representation of a fully specified experiment path.
The public `Experiment` API can hide these layers, but every run must be
exportable as an auditable recipe.

## Canonical Layer Order

```text
0_meta
1_data_task
2_preprocessing
3_training
4_evaluation
5_output_provenance
6_stat_tests
7_importance
```

The directory names are legacy-compatible. The semantic ownership is:

| Layer | Name | Responsibility |
|---|---|---|
| 0 | Study design / execution grammar | Research design, experiment unit, sweep grammar, reproducibility, failure policy, compute mode. |
| 1 | FRED data frame | Dataset loading, source adapter, frequency, information set, target identity, sample period, official availability handling, official release-lag discipline. |
| 2 | Research preprocessing / feature representation | Researcher-chosen target/X transforms, scaling, imputation, outlier treatment, feature-block grammar, predictor family, feature builder bridge, PCA/factors, factor counts, deterministic features, custom preprocessors. |
| 3 | Forecast generator | Forecast generator family currently exposed as `model_family`, baseline generator role assignment currently exposed as `benchmark_family`, forecast type, forecast object, training windows, refit, model lag order, tuning, estimator training settings. |
| 4 | Evaluation protocol / metrics | Metric families, aggregation, rankings, reporting, regime-specific evaluation subsets. |
| 5 | Artifacts / provenance | Export format, saved objects, provenance depth, artifact granularity. |
| 6 | Statistical inference | Forecast comparison tests, dependence correction, nested/multiple-model tests, residual diagnostics, test scope. |
| 7 | Interpretation / importance | Importance method, model-native/model-agnostic methods, SHAP, PDP, grouped/temporal/local/stability importance. |

## Boundary Rules

- Layer 1 creates an FRED data frame. It should not choose a model, a
  benchmark, a hyperparameter search, or a researcher-specific transformation.
- Layer 2 changes the representation of the FRED frame for a study. It owns
  extra preprocessing after FRED-provided transforms and the construction of
  `Z_train`/`Z_pred` feature matrices.
- Layer 3 generates forecasts. Benchmarks and registered custom models belong
  here because they are forecast generators, not data definitions. A benchmark
  is a generator assigned the baseline role, not a separate model species. It
  consumes the Layer 2 representation handoff (`Z_train`, `y_train`, `Z_pred`,
  feature names, block metadata, and fit state); it does not own the feature
  representation grammar.
- Layer 4 scores forecasts. It should not fit models or transform data.
- Layer 6 performs statistical inference over already-computed forecast errors.

Existing recipe paths are accepted for compatibility during migration, but the
registry `layer` field is the canonical ownership contract.
