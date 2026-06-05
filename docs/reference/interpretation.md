# macroforecast.interpretation

[Back to reference](index.md)

`macroforecast.interpretation` owns post-fit interpretation helpers. These
functions consume fitted models and feature matrices. They do not fit models,
construct features, choose forecast windows, or run forecast-comparison tests.

Use the namespace form:

```python
import macroforecast as mf

fit = mf.models.ridge(X_train, y_train)
mf.interpretation.linear_coefficients(fit)
```

Every table returned by this module carries
`attrs["macroforecast_metadata_schema"]` with `kind`, `version`, `method`,
`model`, `n_features`, output `columns`, and function-specific metadata.

## Review Status

`interpretation` functions are not all the same kind of evidence. Use this
classification before treating a number as a paper table or structural claim.

| Class | Functions | Reference implementation | Status |
| --- | --- | --- | --- |
| Native/backend extraction | `linear_coefficients`, `tree_importance`, `model_native_linear_coef`, `model_native_tree_importance` | scikit-learn-style `coef_` / `feature_importances_` estimator conventions | Direct attribute extraction. Validity depends on the fitted estimator. |
| SHAP backend | `shap_values`, `shap_importance`, `shap_tree`, `shap_linear`, `shap_kernel`, `shap_deep` | Python `shap` package: `Explainer`, `TreeExplainer`, and `PermutationExplainer` | Backend call plus pandas reshaping. `shap_linear` and `shap_deep` are generic SHAP paths, not fixed `LinearExplainer` or `DeepExplainer` calls. |
| Forecast-accuracy Shapley | `oshapley_pipeline`, `oshapley_from_forecast_result`, `oshapley_output`, `forecast_shapley_output`, `oshapley_vi`, `pbsv`, `performance_based_shapley_value`, `model_accordance_score`, `ishapley_vi`, `shapley_variable_importance`, `performance_shapley_value` | Python `anatomy` package and Borup, Goulet Coulombe, Rapach, Montes Schutte, and Schwenk-Nebbe (2022) | User-facing callables use oShapley/PBSV/MAS names. The optional backend is the Python `anatomy` package. |
| Captum backend | `deep_lift` | Captum `DeepLift` | Direct Captum call. |
| Standard model-agnostic diagnostics | `permutation_importance`, `lofo_importance(fit_func=...)`, `partial_dependence`, `individual_conditional_expectation`, `ice_curves`, `accumulated_local_effect`, `friedman_h_interaction` | scikit-learn permutation/PDP/ICE, R `pdp::partial`, R `ALEPlot`, R `iml`/`hstats` H-statistic conventions | Implements standard diagnostic definitions directly. Interpretation is predictive/associational, not causal. |
| Approximate or fixed-model diagnostics | `permutation_importance_strobl`, `lofo_importance(fit_func=None)`, `cumulative_r2_contribution`, `rolling_recompute`, `bootstrap_jackknife`, non-linear `forecast_decomposition` fallback | Strobl conditional-permutation idea, LOFO refit idea, macroforecast fixed-model diagnostics | Metadata records approximation/fixed-model mode. Do not describe these as exact refit or additive decompositions. |
| VAR/macroeconomic interpretation | `generalized_irf`, `orthogonalised_irf`, `fevd`, `historical_decomposition` | statsmodels VAR result API when available; Pesaran-Shin GIRF formula; internal statsmodels-like adapter for macroforecast VAR | `fevd` now falls back to manual orthogonalized FEVD rather than IRF. `historical_decomposition` is reduced-form, not structural identification. |
| Neural manual attribution | `saliency_map`, `integrated_gradients`, `gradient_shap`, `lstm_hidden_state` | Captum-style torch-gradient methods | Manual torch autograd implementation, except `deep_lift`. Integrated gradients uses a straight-line Riemann approximation, not Captum's default Gauss-Legendre backend. |
| OLS as attention | `ols_attention_weights`, `ridge_attention_weights`, `ols_attention_embedding`, `ols_attention_equivalence`, `attention_weights`, `dual_decomposition` | Goulet Coulombe (2026), "Ordinary Least Squares as an Attention Mechanism" | Closed-form linear algebra. `ols_attention_weights` uses exact OLS by default; ridge variants are stabilizing extensions, not new forecasting models. |
| Dual interpretation | `dual`, `DualInterpretationResult`, `dual_interpretation`, `dual_from_forecast_result`, `observation_weights`, `observation_contributions`, `forecast_diagnostics`, `top_observations`, `group_observation_weights`, plus backward-compatible aliases `outcome_contributions`, `data_portfolio_diagnostics`, `top_episodes`, `episode_group_weights` | Goulet Coulombe, Goebel, and Klieber (2024), plus local `dual_python` / `DualML_R` reference code | Episode-based explanation: which historical observations the model uses. Use [Dual Interpretation](interpretation_dual.md) for the full contract. Ridge/KRR/RF routes are implemented directly; boosted-tree AXIL, NN embedding, and classification log-odds routes are documented as future extensions. |
| Aggregation/report plumbing | `group_aggregate`, `lineage_attribution`, `transformation_attribution`, `custom_interpretation` | No external fitting backend; table aggregation and user callables | Aggregates existing evidence. `transformation_attribution` is not component-level causality unless the input table was designed as a component-removal experiment. |

Every returned table includes this review classification in
`attrs["macroforecast_metadata_schema"]["reference"]`.

## Function Summary

| Function | Input | Output | Meaning |
| --- | --- | --- | --- |
| `linear_coefficients(model, sort=True)` | `ModelFit` or estimator with `coef_` | `DataFrame` | Native linear coefficient table. |
| `model_native_linear_coef(model, sort=True)` | `ModelFit` or estimator with `coef_` | `DataFrame` | Legacy-name alias for native linear coefficients. |
| `tree_importance(model, sort=True)` | `ModelFit` or estimator with `feature_importances_` | `DataFrame` | Native tree feature importance table. |
| `model_native_tree_importance(model, sort=True)` | `ModelFit` or estimator with `feature_importances_` | `DataFrame` | Legacy-name alias for native tree importance. |
| `permutation_importance(model, X, y, metric="mse", n_repeats=5, random_state=None)` | fitted predictor, feature frame, target vector | `DataFrame` | Loss degradation after permuting each feature. |
| `permutation_importance_strobl(model, X, y, metric="mse", n_repeats=5, n_bins=5, random_state=None)` | fitted predictor, feature frame, target vector | `DataFrame` | Conditional permutation within bins of correlated features. |
| `lofo_importance(model, X, y, fit_func=None, metric="mse")` | fitted predictor or refit callable, feature frame, target vector | `DataFrame` | Leave-one-feature-out or prediction-drop importance. |
| `partial_dependence(model, X, features, grid_size=20)` | fitted predictor and feature frame | `DataFrame` | One-way manual partial-dependence curves. |
| `individual_conditional_expectation(model, X, features, grid_size=20, center=False)` | fitted predictor and feature frame | long `DataFrame` | One-way individual conditional expectation curves. |
| `ice_curves(model, X, features, grid_size=20, center=False)` | fitted predictor and feature frame | long `DataFrame` | Alias for `individual_conditional_expectation`. |
| `accumulated_local_effect(model, X, feature, bins=10)` | fitted predictor and feature frame | `DataFrame` | First-order accumulated local effect curve. |
| `friedman_h_interaction(model, X, features=None, grid_size=10)` | fitted predictor and feature frame | `DataFrame` | Pairwise Friedman-Popescu H interaction statistics. |
| `shap_values(model, X, background=None, explainer="auto", check_additivity=True, **kwargs)` | fitted predictor and feature frame | long `DataFrame` | SHAP attribution values using optional `shap` backend. |
| `shap_importance(model, X, ...)` | fitted predictor and feature frame | `DataFrame` | Mean absolute SHAP feature importance. |
| `shap_tree`, `shap_linear`, `shap_kernel`, `shap_deep` | fitted predictor and feature frame | `DataFrame` | Convenience SHAP-importance wrappers. |
| `oshapley_pipeline(X, y, models, window=None, ...)` | aligned feature matrix, target, model specs | `ForecastShapleyResult` | Convenience path that precomputes the backend object and returns forecast explanation, oShapley-VI, and PBSV tables. |
| `oshapley_from_forecast_result(result, X, y, models, window=..., ...)` | `ForecastResult` plus explicit aligned feature matrix/target/model/window inputs | `ForecastResult` or `ForecastShapleyResult` | Build an oShapley/PBSV sidecar for a completed runner result. |
| `oshapley_output(value, output="oshapley", loss="rmse", sidecar_name=None, ...)` | `ForecastResult` sidecar, `ForecastShapleyResult`, or precomputed backend object | `DataFrame`, `dict`, or summary object | Select one oShapley/PBSV output table without manually opening the sidecar. |
| `forecast_shapley_output(...)` | same as `oshapley_output(...)` | same as `oshapley_output(...)` | Descriptive alias for `oshapley_output(...)`. |
| `oshapley_provider(X, y, models, window=None, ...)` | aligned feature matrix, target, model specs | `AnatomyModelProvider` | Build the provider required by the optional backend precompute loop. |
| `precompute_oshapley(X, y, models, window=None, ...)` | same as `oshapley_provider` | backend object | Build provider and run the optional backend precompute. |
| `window_to_oshapley_subsets(window, index, train_source="fit")` | `WindowSpec` and index | backend subsets | Convert macroforecast train/test origins to exact backend subsets. |
| `anatomy_output_transformer(output="forecast")` | raw forecast/loss output name or callable | `AnatomyModelOutputTransformer` | Build the optional backend transformer used only inside forecast-accuracy anatomy explanations. |
| `anatomy_explain(anatomy, model_groups=None, metric="forecast", output="long")` | precomputed backend object or saved path | `DataFrame` | Low-level backend wrapper for raw Shapley explanations. |
| `anatomy_model(model, feature_names=None)` | `ModelFit` or estimator with `predict` | `AnatomyModel` | Wrap a macroforecast fit as an anatomy prediction function. |
| `window_to_anatomy_subsets(window, index, train_source="fit")` | `WindowSpec` and index | `AnatomySubsets` | Convert macroforecast train/test origins to exact anatomy subsets. |
| `anatomy_provider(X, y, models, window=None, ...)` | aligned feature matrix, target, model specs | `AnatomyModelProvider` | Build the provider required by the upstream anatomy precompute loop. |
| `precompute_anatomy(X, y, models, window=None, ...)` | same as `anatomy_provider` | `Anatomy` | Build provider and run `Anatomy(...).precompute(...)`. |
| `anatomy_pipeline(X, y, models, window=None, ...)` | aligned feature matrix, target, model specs | `AnatomyPipelineResult` | Convenience path that precomputes anatomy and returns forecast explanation, oShapley-VI, and PBSV tables. |
| `anatomy_from_forecast_result(result, X, y, models, window=..., ...)` | `ForecastResult` plus explicit aligned feature matrix/target/model/window inputs | `ForecastResult` or `AnatomyPipelineResult` | Build an anatomy sidecar for a completed runner result. |
| `oshapley_vi(anatomy, ...)` | precomputed `anatomy.Anatomy` object or saved path | `DataFrame` | Backend oShapley-VI: mean absolute raw-forecast Shapley contribution. |
| `pbsv(anatomy, loss="rmse", ...)` | precomputed `anatomy.Anatomy` object or saved path | `DataFrame` | Backend PBSV_p loss decomposition. |
| `performance_based_shapley_value(...)` | same as `pbsv(...)` | `DataFrame` | Alias for backend PBSV_p. |
| `ishapley_vi(contributions, ...)` | in-sample contribution table | `DataFrame` | Explicit iShapley-VI_p table adapter for MAS input. |
| `shapley_variable_importance(contributions, ...)` | local contribution table | `DataFrame` | Mean absolute Shapley VI table, usable for in-sample or out-of-sample VI. |
| `model_accordance_score(is_vi, oos_pbsv, ...)` | VI table/Series and PBSV table/Series | one-row `DataFrame` | Backend MAS and optional MAS p-value. |
| `performance_shapley_value(contributions, y, loss="squared_error", ...)` | additive forecast-contribution table and aligned target | `DataFrame` | Package-native PBSV-style point-loss Shapley fallback from additive forecast contributions. |
| `forecast_decomposition(model, X, row=-1)` | fitted predictor and feature frame | `DataFrame` | Linear contribution of each feature to one prediction. |
| `cumulative_r2_contribution(model, X, y, feature_order=None)` | fitted predictor, feature frame, target vector | `DataFrame` | Sequential R-squared contribution under zero-filled inactive features. |
| `rolling_recompute(model, X, y, window=None, step=None, method="permutation_importance")` | fitted predictor, feature frame, target vector | `DataFrame` | Recompute importance over rolling evaluation windows. |
| `bootstrap_jackknife(model, X, y, fit_func=None, n_replications=50)` | fitted predictor or refit callable, feature frame, target vector | `DataFrame` | Bootstrap uncertainty summary for importance. |
| `group_aggregate(table, groups=None, ...)` | feature-importance table | `DataFrame` | Aggregate feature importance by user or inferred groups. |
| `lineage_attribution(table, lineage, level="pipeline_name")` | feature-importance table and lineage mapping | `DataFrame` | Aggregate importance by feature-engineering lineage metadata. |
| `transformation_attribution(evaluation, ..., lower_is_better=True, baseline="worst")` | evaluation score table | `DataFrame` | Attribute score improvements to mutually exclusive preprocessing/feature pipelines. |
| `attention_weights(X_train, X_test=None, ...)` | training and test feature frames | long `DataFrame` | OLS attention matrix weights. |
| `ols_attention_weights(X_train, X_test=None, ...)` | training and test feature frames | long `DataFrame` | Exact OLS-as-attention weights from Goulet Coulombe (2026). |
| `ridge_attention_weights(X_train, X_test=None, alpha=...)` | training and test feature frames | long `DataFrame` | Ridge-stabilized attention weights using the standard unpenalized-intercept ridge convention. |
| `ols_attention_embedding(X_train, X_test=None, ...)` | training and test feature frames | long `DataFrame` | Whitened train/test embeddings whose inner products reconstruct attention weights. |
| `ols_attention_equivalence(X_train, y_train, X_test=None, ...)` | train features/outcomes, optional test features, optional reference predictions | `DataFrame` | Audit that standard OLS/ridge predictions equal attention-weight predictions. |
| `dual_decomposition(X_train, y_train, X_test=None, ...)` | train features/outcomes and optional test features | long `DataFrame` | OLS prediction as weighted training-outcome contributions. |
| `dual.DualInterpretationResult` / `DualInterpretationResult` | output from `dual_interpretation(...)` | result object | Bundles observation weights, observation contributions, forecast diagnostics, top observations, group weights, and metadata. |
| `dual.dual_interpretation(...)` / `dual_interpretation(...)` | model, train/test feature frames, train target | `DualInterpretationResult` | Run the dedicated dual interpretation path. Full contract: [Dual Interpretation](interpretation_dual.md). |
| `dual.dual_from_forecast_result(...)` / `dual_from_forecast_result(...)` | completed `ForecastResult`, fitted model, explicit train/test feature frames, train target | `ForecastResult` or `DualInterpretationResult` | Build a dual sidecar for a completed runner result. |
| `dual.observation_weights(model, X_train, X_test=None, method="auto", ...)` / `observation_weights(...)` | fitted model when needed, train/test feature frames | long `DataFrame` | DualML-style historical-observation weights for ridge, KRR, and random forest. |
| `dual.observation_contributions(weights, y_train, center=False, include_base=False)` / `observation_contributions(...)` | observation-weight table and train target | long `DataFrame` | Convert observation weights into observation contributions whose rows sum to each forecast. |
| `dual.forecast_diagnostics(weights, top_q=0.05)` / `forecast_diagnostics(...)` | observation-weight table | `DataFrame` | Forecast concentration, short position, leverage, gross leverage, and turnover. |
| `dual.top_observations(weights, y_train=None, n=10, sort_by="abs_weight")` / `top_observations(...)` | observation weights or contributions | long `DataFrame` | Highest-weight historical observations per forecast date. |
| `dual.group_observation_weights(weights, groups, y_train=None)` / `group_observation_weights(...)` | observation weights and user episode groups | `DataFrame` | Aggregate weights/contributions over regimes such as recessions or inflation episodes. |
| `outcome_contributions`, `data_portfolio_diagnostics`, `top_episodes`, `episode_group_weights` | same as preferred dual names | `DataFrame` | Backward-compatible aliases. Prefer the paper-aligned names above in new code. |
| `mrf_gtvp(model, X=None)` | fitted `macro_random_forest` after prediction | long `DataFrame` | Time-varying coefficient path from MacroRandomForest `betas`. |
| `generalized_irf(model, n_periods=12, target=None)` | fitted VAR model | `DataFrame` | Pesaran-Shin generalized IRF importance. |
| `orthogonalised_irf(model, n_periods=12, target=None)` | fitted VAR model | `DataFrame` | Cholesky orthogonalised IRF importance. |
| `fevd(model, n_periods=12, target=None)` | fitted VAR model | `DataFrame` | Forecast error variance decomposition importance. |
| `historical_decomposition(model, max_lag=12, target=None)` | fitted VAR model | `DataFrame` | Reduced-form MA residual contribution summary. |
| `lasso_inclusion_frequency(model, X=None, y=None, fit_func=None, ...)` | lasso-style fit, optionally refit callable | `DataFrame` | Single-fit or bootstrap nonzero coefficient frequency. |
| `lstm_hidden_state(model, X)` | fitted torch LSTM/GRU model and feature frame | `DataFrame` | Hidden-unit activation importance. |
| `saliency_map`, `integrated_gradients`, `gradient_shap`, `deep_lift` | fitted torch model and feature frame | `DataFrame` | Gradient-based neural attribution. |
| `custom_interpretation(model, X, func, y=None, name=None, **params)` | fitted predictor, feature frame, user callable | `DataFrame` | User-defined interpretation table with metadata attrs. |

## Native Model Interpretation

### linear_coefficients

```python
macroforecast.interpretation.linear_coefficients(model, *, sort=True)
```

Input: a `ModelFit` or estimator exposing `coef_`.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `coefficient` | Signed coefficient. |
| `abs_coefficient` | Absolute coefficient used for sorting. |

### tree_importance

```python
macroforecast.interpretation.tree_importance(model, *, sort=True)
```

Input: a `ModelFit` or estimator exposing `feature_importances_`.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `importance` | Native estimator importance. |

## Model-Agnostic Importance

### permutation_importance

```python
macroforecast.interpretation.permutation_importance(
    model,
    X,
    y,
    *,
    metric="mse",
    n_repeats=5,
    random_state=None,
)
```

Input: fitted predictor, feature `DataFrame`, and aligned target vector.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Permuted feature. |
| `importance` | Mean loss increase after permutation. |
| `std` | Standard deviation across repeats. |
| `baseline_loss` | Loss before permutation. |
| `n_repeats` | Repeat count. |

Supported metric names: `mse`, `mae`. A custom callable can also be supplied.

### permutation_importance_strobl

```python
macroforecast.interpretation.permutation_importance_strobl(
    model,
    X,
    y,
    *,
    metric="mse",
    n_repeats=5,
    n_bins=5,
    random_state=None,
)
```

Input: fitted predictor, feature `DataFrame`, and aligned target vector.

Output columns: `feature`, `importance`, `std`, `baseline_loss`,
`n_repeats`, `conditioning_feature`, and `n_bins`.

This is a Strobl-style conditional-permutation approximation. For each
feature, the function finds the most correlated companion feature and permutes
inside quantile bins of that companion. The exact Strobl conditional
permutation literature can condition on richer structures; this callable uses
a single-companion quantile-bin rule and records
`exact_reference_implementation=False` in metadata.

### lofo_importance

```python
macroforecast.interpretation.lofo_importance(
    model,
    X,
    y,
    *,
    fit_func=None,
    metric="mse",
    sort=True,
)
```

Input: feature `DataFrame`, target vector, and either an already fitted model
or a refit callable. If `fit_func` is supplied, it must have signature
`fit_func(X_train, y_train) -> fitted_model`.

Output columns: `feature`, `importance`, `baseline_loss`, `heldout_loss`,
and `mode`.

`mode="refit"` means true leave-one-feature-out refitting.
`mode="prediction_drop"` means the fitted model is reused and the held-out
feature is set to zero. The second mode is a fixed-model diagnostic, not an
exact LOFO refit experiment.

## Effect Curves

### partial_dependence

```python
macroforecast.interpretation.partial_dependence(
    model,
    X,
    *,
    features,
    grid_size=20,
)
```

Input: fitted predictor, feature `DataFrame`, and one feature or list of
features.

Output columns: `feature`, `value`, `prediction`.

Grid strategy: `linear_min_max`. For each selected feature, macroforecast uses
`grid_size` equally spaced values between the observed minimum and maximum in
`X`. This is the same brute-force averaging definition as sklearn/R PDP, with
a simpler explicit grid rule.

### individual_conditional_expectation / ice_curves

```python
macroforecast.interpretation.individual_conditional_expectation(
    model,
    X,
    *,
    features,
    grid_size=20,
    center=False,
)

macroforecast.interpretation.ice_curves(
    model,
    X,
    *,
    features,
    grid_size=20,
    center=False,
)
```

Input: fitted predictor, feature `DataFrame`, and one feature or list of
features.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature varied along the grid. |
| `row` | Original row position. |
| `index` | Original pandas index value. |
| `value` | Grid value assigned to the feature. |
| `prediction` | Individual prediction for that row and grid value. |
| `centered_prediction` | Prediction minus the first grid prediction when `center=True`; otherwise `NaN`. |

This matches the brute-force ICE idea used by scikit-learn's partial
dependence with `kind="individual"` and R `pdp::partial(..., ice=TRUE)`.
`partial_dependence()` averages over rows. ICE keeps the row dimension.
The grid strategy is also `linear_min_max`.

### accumulated_local_effect

```python
macroforecast.interpretation.accumulated_local_effect(
    model,
    X,
    *,
    feature,
    bins=10,
)
```

Input: fitted predictor, feature `DataFrame`, and one feature name.

Output columns: `feature`, `bin`, `center`, `ale`, `local_effect`.

Binning strategy: empirical quantile bins. The returned `ale` curve is
centered to mean zero after accumulating local finite differences.

### friedman_h_interaction

```python
macroforecast.interpretation.friedman_h_interaction(
    model,
    X,
    *,
    features=None,
    grid_size=10,
)
```

Input: fitted predictor and feature `DataFrame`. `features=None` checks all
feature pairs.

Output columns: `feature_1`, `feature_2`, `h_statistic`,
`joint_variance`, `interaction_variance`, and `grid_size`.

The statistic is computed from manual one-way and two-way partial dependence.
It is an interaction diagnostic, not a forecast-accuracy test.

## SHAP Values

### shap_values

```python
macroforecast.interpretation.shap_values(
    model,
    X,
    *,
    background=None,
    explainer="auto",
    check_additivity=True,
    **kwargs,
)
```

Input: a fitted `ModelFit` or estimator plus a feature `DataFrame`. `background`
defaults to `X`. For stable out-of-sample interpretation, pass a training or
reference sample as `background`.

Output: long `DataFrame`, one row per `(observation, feature)`.

| Column | Meaning |
| --- | --- |
| `row` | Integer row position in `X`. |
| `index` | Original pandas index value. |
| `feature` | Feature name. |
| `feature_value` | Observed feature value. |
| `shap_value` | SHAP contribution. |
| `base_value` | SHAP base value when provided by the backend. |

Supported `explainer` values:

| Value | Backend behavior |
| --- | --- |
| `auto` | Uses `shap.Explainer` with the model prediction function and background frame. |
| `permutation` | Uses `shap.PermutationExplainer`. |
| `tree` | Uses `shap.TreeExplainer` on the native estimator. |

SHAP is optional:

```bash
pip install "macroforecast[interpretation]"
```

### shap_importance and SHAP wrappers

```python
macroforecast.interpretation.shap_importance(model, X, background=None, explainer="auto")
macroforecast.interpretation.shap_tree(model, X, background=None)
macroforecast.interpretation.shap_linear(model, X, background=None)
macroforecast.interpretation.shap_kernel(model, X, background=None)
macroforecast.interpretation.shap_deep(model, X, background=None)
```

Input: fitted predictor and feature `DataFrame`.

Output columns: `feature`, `importance`, `mean_shap`, and `std_shap`.
`importance` is mean absolute SHAP value.

`shap_tree` forces the tree explainer. `shap_kernel` uses the permutation SHAP
path. `shap_linear` and `shap_deep` use the generic explainer path because the
installed `shap` backend determines the exact implementation.

## Forecast Accuracy Anatomy

This section covers the Shapley-based forecast-accuracy anatomy in Borup,
Goulet Coulombe, Rapach, Montes Schutte, and Schwenk-Nebbe (2022), *The
Anatomy of Out-of-Sample Forecasting Accuracy*, Federal Reserve Bank of
Atlanta Working Paper 2022-16, DOI `10.29338/wp2022-16`, SSRN `4278745`.

The paper separates several related quantities:

| Quantity | Meaning |
| --- | --- |
| iShapley-VI | In-sample Shapley variable importance over a training/reference sample. The upstream `anatomy` package accepts this as a MAS input but does not compute a separate iShapley object from `Anatomy.explain(...)`; macroforecast standardizes user-supplied in-sample contribution tables with `ishapley_vi(...)`. |
| oShapley-VI | Out-of-sample Shapley variable importance over the forecast sample. In the upstream README this is computed by anatomizing raw forecasts and averaging absolute local contributions. |
| PBSV_p | Performance-based Shapley value for predictor `p`. This is not a separate `PBSVp` algorithm; the subscript `p` denotes the predictor. A negative value reduces a lower-is-better loss. |
| MAS | Model-adaptation score comparing variable-importance and performance-based rankings. |

Backend coverage in macroforecast:

| Method | Callable | Backend status |
| --- | --- | --- |
| Raw anatomy explanations | `anatomy_explain(..., metric="forecast")` | Direct `anatomy.Anatomy.explain(...)`. |
| oShapley-VI | `oshapley_vi(...)` | Direct backend raw-forecast explanation plus the upstream README aggregation rule. |
| PBSV_p | `pbsv(...)` / `performance_based_shapley_value(...)` | Direct backend loss transformer through `anatomy.Anatomy.explain(...)`. |
| MAS | `model_accordance_score(...)` | Direct `anatomy.MAS(...).compute(...)`. |
| iShapley-VI input | `ishapley_vi(...)` / `shapley_variable_importance(...)` | Table adapter. Use `ishapley_vi(...)` on an in-sample SHAP/anatomy contribution table; upstream `MAS` then consumes the resulting VI. |
| Additive fallback | `performance_shapley_value(...)` | macroforecast-native point-loss Shapley adapter for SHAP/decomposition tables; not the full `anatomy` provider. |

### Window support

The upstream Python `anatomy` package supports both single-forecast and
multi-window anatomy designs.

| Design | How it is represented upstream | macroforecast wrapper behavior |
| --- | --- | --- |
| Single forecast | Provider with one period, or `explanation_subset` selecting one forecast date. | `anatomy_explain(..., explanation_subset=[date])`, `pbsv(..., loss="squared_error")`, and `pbsv(..., loss="absolute_error")` return local rows for that forecast date. |
| Expanding window | `AnatomySubsets.generate(..., estimation_type=AnatomySubsets.EstimationType.EXPANDING, periods=..., gap=...)`. | macroforecast consumes the precomputed `Anatomy` and preserves the forecast-date index returned by the backend. |
| Rolling window | `AnatomySubsets.generate(..., estimation_type=AnatomySubsets.EstimationType.ROLLING, periods=..., gap=...)`. | Same as expanding: the window scheme is fixed when the upstream `Anatomy` object is precomputed. |
| Full OOS or subperiod summary | `Anatomy.explain(..., explanation_subset=None)` for all forecasts, or a subset index for a subperiod. Aggregating transformers such as RMSE/MAE/MSE return one global row per model set. | `pbsv(loss="rmse" | "mse" | "mae")` returns global or subperiod PBSV rows. |

Important boundary: `macroforecast.interpretation` does not decide the rolling
or expanding split. The `anatomy.Anatomy` object already contains the provider
loop, fitted models, forecast periods, and masking logic. For an end-to-end
macroforecast-native path, use `anatomy_provider(...)`,
`precompute_anatomy(...)`, or `anatomy_pipeline(...)` with a `WindowSpec`.

### oShapley/PBSV bridge utilities

The smooth user-facing path uses method names rather than backend package
names:

```python
result = mf.forecasting.run(feature_set, "ridge", window=window)

result = mf.interpretation.oshapley_from_forecast_result(
    result,
    feature_set.X,
    feature_set.y.iloc[:, 0],
    {"ridge": "ridge"},
    window=window,
    losses=("squared_error", "rmse"),
)

mf.output.write_artifacts(result, "results/oshapley_run")
```

Once the sidecar is attached, choose the exact output from
`macroforecast.interpretation`:

```python
vi = mf.interpretation.oshapley_output(result, output="oshapley")
loss = mf.interpretation.oshapley_output(result, output="pbsv", loss="rmse")
local = mf.interpretation.oshapley_output(result, output="forecast")
tables = mf.interpretation.oshapley_output(result, output="tables")
meta = mf.interpretation.oshapley_output(result, output="metadata")
```

Supported `output` values:

| Output | Meaning |
| --- | --- |
| `oshapley` | oShapley-VI table, equivalent aliases: `vi`, `variable_importance`, `importance`, `oshapley_vi`, `o_shapley_vi`. |
| `pbsv` | PBSV/loss-decomposition table for `loss`, equivalent aliases: `performance`, `loss`, `loss_decomposition`. |
| `forecast` | Raw forecast-explanation Shapley rows, equivalent aliases: `raw`, `prediction`, `explain`, `forecast_explanation`. |
| `tables` | Dictionary of all CSV/JSON-ready output tables. `output="all"` is an alias. Use `table="oshapley_variable_importance"` to pick one named table. |
| `metadata` | Flattened metadata table for the oShapley/PBSV run. |
| `summary` | JSON-ready dictionary with metadata, forecast explanations, oShapley-VI, and PBSV tables. |

If a `ForecastResult` has exactly one sidecar, the selector uses it. If
multiple sidecars are attached and none has the preferred names `oshapley`,
`forecast_shapley`, or `anatomy`, pass `sidecar_name=...`.

The `anatomy_*` functions below are backend aliases. Use them only when the
backend object itself is the object being handled.

### backend bridge utilities

```python
macroforecast.interpretation.window_to_anatomy_subsets(
    window,
    index,
    *,
    train_source="fit",
)
```

The user-facing alias is:

```python
macroforecast.interpretation.window_to_oshapley_subsets(...)
```

Input: a `WindowSpec`, method name, or `None`, plus the feature/target index.

Output: upstream `anatomy.AnatomySubsets`. Unlike
`AnatomySubsets.generate(...)`, this conversion follows the exact
macroforecast `WindowSpec` plan, including rolling/expanding/fixed estimation,
calendar or positional test steps, and retrain cadence. `train_source="fit"`
uses the model-fit sample at each origin; `train_source="estimation"` uses the
full estimation sample.

```python
macroforecast.interpretation.anatomy_provider(
    X,
    y,
    models,
    *,
    window=None,
    params=None,
    target_name=None,
    train_source="fit",
)
```

The user-facing alias is:

```python
macroforecast.interpretation.oshapley_provider(...)
```

Input: an already aligned forecast-row feature matrix `X` and target `y`.
`models` can be a model name, `ModelSpec`, sequence, or alias mapping. The
target must already represent the forecast object being explained, such as a
direct `y[t+h]` target or a precomputed growth target. Panel-to-feature
construction remains the job of `macroforecast.feature_engineering` or
`macroforecast.forecasting`.

Output: upstream `AnatomyModelProvider`. The provider fits the requested
macroforecast model at each anatomy period and wraps the fit with
`anatomy_model(...)`.

```python
macroforecast.interpretation.precompute_anatomy(
    X,
    y,
    models,
    *,
    window=None,
    n_iterations=32,
    n_jobs=1,
    save_path=None,
)
```

The user-facing alias is:

```python
macroforecast.interpretation.precompute_oshapley(...)
```

Output: precomputed upstream `anatomy.Anatomy` object.

```python
macroforecast.interpretation.anatomy_pipeline(
    X,
    y,
    models,
    *,
    window=None,
    losses=("rmse",),
    n_iterations=32,
    n_jobs=1,
)
```

The user-facing alias is:

```python
macroforecast.interpretation.oshapley_pipeline(...)
```

Output: `ForecastShapleyResult` (`AnatomyPipelineResult` backend alias) with:

| Attribute | Meaning |
| --- | --- |
| `anatomy` | Precomputed backend object. |
| `explanations["forecast"]` | Raw-forecast anatomy table from `anatomy_explain(...)`. |
| `variable_importance` | oShapley-VI table from `oshapley_vi(...)`. |
| `performance_values[loss]` | PBSV table for each requested loss. |
| `metadata` | Window/model/loss settings used to build the provider. |

`AnatomyPipelineResult.to_tables(prefix="anatomy")` returns named output tables
for raw forecast explanations, oShapley-VI, PBSV/loss decompositions, and
metadata. `macroforecast.output` uses this method when writing anatomy
sidecars attached to a `ForecastResult`.

Example:

```python
window = mf.window.from_cutoffs(
    test_start=X.index[120],
    estimation_min_size=120,
    horizon=1,
    step=1,
)

anat = mf.interpretation.anatomy_pipeline(
    X,
    y_direct,
    {"ridge": "ridge", "forest": "random_forest"},
    window=window,
    losses=("squared_error", "rmse"),
    n_iterations=64,
    n_jobs=4,
)

anat.variable_importance
anat.performance_values["rmse"]
```

```python
macroforecast.interpretation.oshapley_from_forecast_result(
    result,
    X,
    y,
    models,
    *,
    window,
    attach=True,
    sidecar_name="anatomy",
)
```

Input: a completed `ForecastResult` plus the aligned `X`, `y`, model specs, and
the exact `WindowSpec` used for the anatomy refit loop. The explicit `X/y` and
`window` are required because a forecast table alone cannot reconstruct the
feature matrix, train/test subsets, or model-provider path without risking
look-ahead mistakes.

Output: if `attach=True`, a copy of `ForecastResult` with an oShapley sidecar
attached under `sidecar_name`. If `attach=False`, returns the standalone
`ForecastShapleyResult`.

```python
result = mf.forecasting.run(feature_set, "ridge", window=window)

result = mf.interpretation.oshapley_from_forecast_result(
    result,
    feature_set.X,
    feature_set.y.iloc[:, 0],
    {"ridge": "ridge"},
    window=window,
    losses=("squared_error", "rmse"),
)

mf.output.write_artifacts(result, "results/anatomy_run")
```

Dual sidecars use the same completed-result pattern, but they do not refit the
forecasting pipeline. The user passes the fitted model and exact train/test
feature matrices:

```python
result = mf.forecasting.run(feature_set, "ridge", window=window)

result = mf.interpretation.dual_from_forecast_result(
    result,
    fit,
    X_train,
    y_train,
    X_test,
    method="ridge",
)

mf.output.write_artifacts(result, "results/dual_run", layout="grouped")
```

### anatomy_explain

```python
macroforecast.interpretation.anatomy_explain(
    anatomy,
    *,
    model_groups=None,
    transformer=None,
    metric="forecast",
    explanation_subset=None,
    output="long",
)
```

Input: a precomputed `anatomy.Anatomy` object or a path saved by the Python
`anatomy` package. The upstream package owns the expensive provider loop:
rolling or expanding forecast periods, train/test extraction, model retrieval
or fitting, background masking, model-combination handling, and Shapley
precomputation.

Output:

| Column | Meaning |
| --- | --- |
| `model_set` | Model or model-combination name from the anatomy object. |
| `index` | Forecast-date index for local explanations, or the global subset label for aggregated explanations. |
| `feature` | `base_contribution` or predictor name. |
| `contribution` | Shapley contribution returned by `anatomy.Anatomy.explain(...)`. |
| `is_base` | Whether the row is the base contribution. |

Set `output="wide"` to receive the backend's original wide table with
`base_contribution` plus feature columns.

Supported built-in `metric` values:

| Metric | Explained value |
| --- | --- |
| `forecast` | Raw forecast, matching the backend default transformer. |
| `squared_error` | Local point squared error `((y - y_hat) ** 2)`. |
| `absolute_error` | Local point absolute error `abs(y - y_hat)`. |
| `mse` | Aggregated mean squared error over the selected forecast subset. |
| `rmse` | Aggregated root mean squared error over the selected forecast subset. |
| `mae` | Aggregated mean absolute error over the selected forecast subset. |

You can pass a custom `transformer` directly. If it is callable,
macroforecast wraps it as `anatomy.AnatomyModelOutputTransformer`. The callable
must follow the upstream API and use a named `y_hat` argument, optionally with
`y`.

Example:

```python
table = mf.interpretation.anatomy_explain(
    "run_artifacts/anatomy.pkl",
    model_groups={"forest_plus_linear": ["rf", "ridge"]},
    metric="squared_error",
)
```

This is the preferred route when the goal is to reproduce the paper's full
forecast-anatomy design. It preserves the upstream provider's rolling or
expanding training windows and masking logic.

### oshapley_vi

```python
macroforecast.interpretation.oshapley_vi(
    anatomy,
    *,
    model_groups=None,
    explanation_subset=None,
    exclude_base=True,
)
```

Input: a precomputed `anatomy.Anatomy` object or saved path.

Output columns:

| Column | Meaning |
| --- | --- |
| `model_set` | Model or model-combination name. |
| `feature` | Predictor name. |
| `importance` | Mean absolute raw-forecast Shapley contribution. |
| `mean_contribution` | Signed mean contribution. |
| `std_contribution` | Standard deviation of local contributions. |
| `n_rows` | Number of local contribution rows. |
| `rank` | Rank within `model_set`. |

This matches the upstream README recipe:

```python
df_oshapley = anatomy.explain(
    model_sets=AnatomyModelCombination(groups=groups),
    transformer=AnatomyModelOutputTransformer(transform=lambda y_hat: y_hat),
)
vi = df_oshapley.loc["model_name"].abs().mean(axis=0)
```

macroforecast performs that sequence through the backend and returns a tidy
table.

### pbsv / performance_based_shapley_value

```python
macroforecast.interpretation.pbsv(
    anatomy,
    *,
    model_groups=None,
    loss="rmse",
    transformer=None,
    explanation_subset=None,
    output="long",
)
```

Input: a precomputed `anatomy.Anatomy` object or saved path.

Output: same shape as `anatomy_explain(...)`, with metadata kind `pbsv`.
`performance_based_shapley_value(...)` is an alias for `pbsv(...)`.

Built-in loss choices:

| Loss | Scope |
| --- | --- |
| `rmse` | Global/subperiod PBSV_p for RMSE. |
| `mse` | Global/subperiod PBSV_p for MSE. |
| `mae` | Global/subperiod PBSV_p for MAE. |
| `squared_error` | Local PBSV_p for each forecast row. |
| `absolute_error` | Local PBSV_p for each forecast row. |

For R-squared, gains, utility, custom benchmarks, or portfolio-style losses,
pass `transformer=` directly. The callable must follow the backend API:
`transform(y_hat, y)` or `transform(y_hat)`.

### ishapley_vi / shapley_variable_importance

```python
macroforecast.interpretation.ishapley_vi(
    contributions,
    *,
    contribution_col=None,
    feature_col="feature",
    group_col=None,
    exclude_base=True,
)
```

Input: an in-sample/training Shapley contribution table with a feature column
and a contribution column. This is the explicit macroforecast callable for
iShapley-VI_p.

Output: a mean-absolute in-sample Shapley VI table with metadata kind
`ishapley_vi`.

```python
macroforecast.interpretation.shapley_variable_importance(
    contributions,
    *,
    contribution_col=None,
    feature_col="feature",
    group_col=None,
    exclude_base=True,
    source="contribution_table",
)
```

Input: any local contribution table with a feature column and a contribution
column. The function infers common contribution names such as `contribution`,
`shap_value`, and `forecast_contribution`.

Output: a mean-absolute Shapley VI table. Use it in two places:

| Use | Example |
| --- | --- |
| iShapley-VI | Prefer `ishapley_vi(...)` on an in-sample/training contribution table, then pass to `model_accordance_score(...)`. |
| oShapley-VI | Run on an out-of-sample raw-forecast contribution table. `oshapley_vi(...)` does this automatically for backend anatomy objects. |

### model_accordance_score

```python
macroforecast.interpretation.model_accordance_score(
    is_vi,
    oos_pbsv,
    *,
    loss_type="lower_is_better",
    mas_type="importance_weighted",
    hypothesis_test=True,
    h0_alpha=0.5,
    n_samples=1000000,
    random_state=None,
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `is_vi` | Feature-indexed Series or VI table. This can be iShapley-VI or oShapley-VI. |
| `oos_pbsv` | Feature-indexed Series or PBSV table. |
| `loss_type` | `lower_is_better` for RMSE/MSE/MAE losses; `larger_is_better` for gains/scores. |
| `mas_type` | `importance_weighted` or `equal_weighted`, matching upstream `anatomy.MAS`. |

Output columns: `mas`, `mas_p_value`, `mas_type`, `loss_type`,
`hypothesis_test`, `h0_alpha`, `n_samples`, and `n_features`.

The function calls `anatomy.MAS(...).compute(...)` directly after converting
macroforecast tables to the feature-indexed Series expected by the backend.
When `random_state` is supplied, macroforecast temporarily seeds NumPy around
the backend hypothesis-test simulation and then restores the previous random
state.

### performance_shapley_value

```python
macroforecast.interpretation.performance_shapley_value(
    contributions,
    y,
    *,
    loss="squared_error",
    row_col=None,
    feature_col="feature",
    contribution_col=None,
    base_col="base_value",
    base_value=0.0,
    n_permutations=None,
    max_exact_features=8,
    random_state=0,
    return_local=False,
)
```

Input: an additive forecast-contribution table and an aligned target vector.
The contribution table can be the output of `shap_values(...)`,
`forecast_decomposition(...)`, or any custom table with:

| Required item | Default column inference |
| --- | --- |
| Row identity | `index` when it matches `y.index`, otherwise `row`. |
| Feature name | `feature`. |
| Additive forecast contribution | `shap_value`, `forecast_contribution`, `contribution`, or `value`. |
| Base forecast | `base_value`, otherwise scalar `base_value` argument. |

If the table has no `row` or `index` column, macroforecast treats it as a
single-forecast contribution table. This is useful for one-row output from
`forecast_decomposition(...)`.

For each forecast row, the function evaluates the cooperative game:

```text
v(S) = L(y_t, base_t + sum_{j in S} contribution_{t,j})
```

and assigns feature `j`:

```text
PBSV_{t,j} =
sum_{S subset N \ {j}}
  |S|! (p - |S| - 1)! / p!
  [v(S union {j}) - v(S)]
```

The base row is the empty-coalition loss `L(y_t, base_t)`. Therefore:

```text
base_loss_t + sum_j PBSV_{t,j} = full_loss_t
```

A negative `pbsv` means the feature reduced the selected loss for that
forecast. A positive `pbsv` means it increased the loss. This sign convention
is the useful paper-table convention for forecast accuracy: beneficial
variables are negative when the metric is a loss.

Output with `return_local=False`:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name or `__base__`. |
| `is_base` | Whether the row is the base loss. |
| `pbsv` | Mean PBSV across forecast rows. |
| `abs_pbsv` | Mean absolute PBSV. |
| `n_rows` | Number of forecast rows. |
| `baseline_loss` | Mean empty-coalition loss. |
| `full_loss` | Mean full-contribution loss. |
| `rank` | Display rank in the returned table. |

Output with `return_local=True` keeps one row per forecast and feature, with
`actual`, `base_prediction`, `full_prediction`, `baseline_loss`, `full_loss`,
`shapley_mode`, and `effective_permutations`.

Exact Shapley weights are used when the number of features in a forecast row is
at most `max_exact_features` and `n_permutations=None`. For larger rows, the
function samples random feature orderings. Metadata records
`max_efficiency_error`, which should be near zero when the contribution table
is additive and row alignment is correct.

Example from SHAP output:

```python
shap_table = mf.interpretation.shap_values(
    fit,
    X_test,
    background=X_train,
    explainer="tree",
)

pbsv = mf.interpretation.performance_shapley_value(
    shap_table,
    y_test,
    contribution_col="shap_value",
    loss="squared_error",
)
```

This adapter is useful when you already have additive local contributions and
want a PBSV-style loss view inside the macroforecast flow. It does not rebuild
the full Borup et al. rolling provider, background replacement, or iShapley /
oShapley machinery. Use `anatomy_explain(...)` for that full backend path.

## Forecast And Model Decomposition

### forecast_decomposition

```python
macroforecast.interpretation.forecast_decomposition(model, X, row=-1)
```

Input: fitted model and feature `DataFrame`. `row` can be an integer position
or an index label.

Output columns for linear models: `feature`, `feature_value`, `coefficient`,
`contribution`, and `abs_contribution`. For linear models, the contribution
sum equals the selected row's prediction up to numerical precision.

If the model has no coefficients but does expose tree importance, the function
returns a non-additive fallback. In that case `contribution` is `NaN`,
`status="tree_importance_fallback_not_additive"`, and metadata records
`prediction_additivity=False`. Do not report that fallback as a forecast
decomposition.

### cumulative_r2_contribution

```python
macroforecast.interpretation.cumulative_r2_contribution(
    model,
    X,
    y,
    *,
    feature_order=None,
)
```

Input: fitted predictor, feature `DataFrame`, and target vector.

Output columns: `step`, `feature`, `r2`, `incremental_r2`, and
`cumulative_features`.

Inactive features are set to zero. The default order is native coefficient
ranking when available, otherwise one-repeat permutation importance. This is a
fixed-model masking diagnostic; the model is not refit as features enter.

### rolling_recompute

```python
macroforecast.interpretation.rolling_recompute(
    model,
    X,
    y,
    *,
    window=None,
    step=None,
    method="permutation_importance",
    n_repeats=1,
    random_state=None,
)
```

Input: fitted predictor, feature `DataFrame`, and target vector.

Output: feature-importance rows with `window_id`, `window_start`, and
`window_end` columns added.

Supported methods:

| Method | Meaning |
| --- | --- |
| `permutation_importance` | Vanilla loss-degradation permutation within each rolling window. |
| `permutation_importance_strobl` | Conditional permutation within each rolling window. |

This function does not refit the model. It asks whether the already fitted
model relies on different variables in different evaluation windows. Metadata
records `refits_model=False`.

### bootstrap_jackknife

```python
macroforecast.interpretation.bootstrap_jackknife(
    model,
    X,
    y,
    *,
    fit_func=None,
    n_replications=50,
    random_state=None,
)
```

Input: fitted predictor, feature frame, target vector, and optionally a refit
callable. With `fit_func`, each bootstrap sample is refit and native
coefficient importance is summarized. Without it, each bootstrap sample uses
fixed-model permutation importance. The current implementation is bootstrap
with replacement; metadata records `jackknife=False`.

Output columns: `feature`, `importance`, `std`, `lower`, `upper`, and
`n_replications`.

### lasso_inclusion_frequency

```python
macroforecast.interpretation.lasso_inclusion_frequency(
    model,
    X=None,
    y=None,
    *,
    fit_func=None,
    n_bootstraps=50,
    random_state=None,
)
```

Input: a lasso-style fitted model. With `X`, `y`, and `fit_func`, the function
bootstraps the sample and refits to estimate nonzero coefficient frequency.
Without those arguments, it reports the single fitted model's nonzero
coefficient indicator.

Output columns include `feature`, `inclusion_frequency`, and `importance`.

## Group And Pipeline Attribution

### group_aggregate

```python
macroforecast.interpretation.group_aggregate(
    table,
    *,
    groups=None,
    group_column=None,
    value_column=None,
    aggregation="sum",
)
```

Input: a feature-importance table with a `feature` column. `groups` can be
either `{feature: group}` or `{group: [features...]}`.

Output columns: `group` and `importance`.

Supported aggregations: `sum`, `mean`, `max_abs`, and `signed_sum`.

### lineage_attribution

```python
macroforecast.interpretation.lineage_attribution(
    table,
    lineage,
    *,
    level="pipeline_name",
    value_column=None,
    aggregation="sum",
)
```

Input: a feature-importance table and a lineage mapping such as
`{"pc1": {"pipeline_name": "pca_block"}}`.

Output columns: the selected lineage level and `importance`.

### transformation_attribution

```python
macroforecast.interpretation.transformation_attribution(
    evaluation,
    *,
    pipeline_column=None,
    metric=None,
    method="shapley_over_pipelines",
    target_columns=("target", "horizon"),
    lower_is_better=True,
    baseline="worst",
)
```

Input: an evaluation table with a pipeline/model column and a loss metric
column. If not supplied, the function infers common names such as `model`,
`model_id`, `pipeline`, `mse`, `rmse`, and `mae`.

Output columns: grouping columns when present, `pipeline`, `loss`, `utility`,
`contribution`, `baseline`, `method`, `metric`, and `lower_is_better`.

For the default loss setting, utility is:

```text
utility_i = baseline_loss - loss_i
```

where `baseline="worst"` uses the worst observed pipeline in each
target/horizon group. `method="shapley_over_pipelines"` applies exact Shapley
weights to the utility game whose coalition value is average utility. This is
a coherent way to summarize mutually exclusive pipeline alternatives, but it
is not a causal component decomposition unless the evaluation table itself was
constructed from component-removal experiments.

Supported methods:

| Method | Meaning |
| --- | --- |
| `shapley_over_pipelines` | Exact Shapley contribution over average-utility coalitions. |
| `marginal_addition` | Pipeline utility relative to the selected baseline. |
| `leave_one_out_pipeline` | Average-utility change after removing one pipeline. |

## Dual Interpretation

This block covers the dual interpretation route in Goulet Coulombe, Goebel,
and Klieber (2024), "Dual Interpretation of Machine Learning Forecasts"
(`arXiv:2412.13076`). Variable-importance tools ask which predictors matter.
Dual interpretation asks which historical observations the model is using as
analogues for the current forecast.

The preferred public surface is the dedicated namespace
`macroforecast.interpretation.dual`. The short aliases in
`macroforecast.interpretation` remain available for compatibility. See
[Dual Interpretation](interpretation_dual.md) for the full input/output
contract, `DualInterpretationResult`, and grouped output layout.

The core object is a data-portfolio weight:

```text
yhat_new = sum_i w_i(new) y_i
```

Positive weights mean the model borrows from similar historical outcomes.
Negative weights mean the model uses an observation by contrast or symmetry.
Concentrated weights mean the model relies on a few episodes. High leverage
means the forecast is extrapolative rather than a simple average. High
turnover means the historical analogies change quickly over time.

The local reference code reviewed for this implementation is:

| Reference path | Used for |
| --- | --- |
| `wiki/raw/paper_code/coulombe_site_github_20260530/dual_python/auxiliaries.py` | Python RR/KRR/RF weight formulas and DualML diagnostics. |
| `wiki/raw/paper_code/coulombe_site_github_20260530/DualML_R/DualML.R` | R `FC_CR`, `FSP`, `FT` diagnostic definitions and RF leaf-weight logic. |
| `wiki/raw/paper_code/coulombe_site_github_20260530/DualML_R/README.md` | Model-route inventory: OLS, RF, LGB, RR, KRR, NN. |

Implemented routes:

| Route | Formula / logic | macroforecast callable |
| --- | --- | --- |
| Ridge / OLS | `W = X_test (X_train' X_train + n lambda I)^-1 X_train'` by default. Set `ridge_penalty_scale="none"` for `lambda I`. | `dual.observation_weights(..., method="ridge")` |
| Kernel ridge | `W = K_test (K_train + lambda I)^-1`; kernels: `linear`, `gaussian`/`rbf`, `laplace`/`laplacian`. | `dual.observation_weights(..., method="krr", kernel=...)` |
| Random forest | For each tree, assign test and train rows to leaves; train rows in the same leaf share weight; average across trees. For sklearn forests, bootstrap sample counts are used when recoverable. | `dual.observation_weights(forest, ..., method="random_forest")` |

Not yet implemented:

| Route | Status |
| --- | --- |
| Boosted trees / LightGBM AXIL | Needs a validated AXIL-style path and model-specific tree-channel storage. |
| LGB+ / LGBA+ channel-specific weights | Should wait until the LGB+ estimator stores separate linear/tree channels. |
| Neural networks | Needs penultimate-layer embeddings and ridge-on-embedding approximation error reporting. |
| Classification | Should default to log-odds-space contributions; probability-scale decomposition is order-dependent. |

### dual.observation_weights

```python
macroforecast.interpretation.dual.observation_weights(
    model,
    X_train,
    X_test=None,
    *,
    method="auto",
    lambda_=1e-8,
    kernel="linear",
    sigma=1.0,
    add_intercept=False,
    ridge_penalty_scale="n_train",
    normalize=False,
)
```

Input:

| Argument | Meaning |
| --- | --- |
| `model` | Required for `method="random_forest"`; can be `None` for ridge/KRR. |
| `X_train` | Training feature `DataFrame`. Its index becomes `train_index`. |
| `X_test` | Forecast-row feature `DataFrame`. If omitted, train rows are explained against themselves. |
| `method` | `auto`, `ridge`, `ols`, `krr`, `kernel_ridge`, `random_forest`, `rf`. |
| `lambda_` | Ridge/KRR regularization. Ridge defaults to the DualML convention `n_train * lambda_`; KRR uses `lambda_ I`. |
| `kernel` | `linear`, `gaussian`/`rbf`, or `laplace`/`laplacian`. |
| `sigma` | Kernel bandwidth convention used by the reference code: `exp(-sigma * distance)`. |
| `add_intercept` | Adds an unpenalized intercept for ridge/OLS. The paper/reference code usually assumes standardized no-intercept design. |
| `normalize` | Re-normalize row weights to sum to one. Default is `False` because negative weights and leverage are meaningful. |

Output columns:

| Column | Meaning |
| --- | --- |
| `test_row`, `test_index` | Forecast row position and index. |
| `train_row`, `train_index` | Historical observation position and index. |
| `weight`, `abs_weight` | Data-portfolio weight and absolute weight. |
| `channel` | Implemented route: `ridge`, `krr`, or `random_forest`. |

The full dense matrix is also attached at `attrs["weight_matrix"]` with shape
`(n_test, n_train)`.

Example:

```python
weights = mf.interpretation.observation_weights(
    None,
    X_train,
    X_test,
    method="krr",
    kernel="laplace",
    sigma=1e-4,
    lambda_=0.1,
)
```

### dual.observation_contributions

```python
macroforecast.interpretation.dual.observation_contributions(
    weights,
    y_train,
    *,
    center=False,
    include_base=False,
)
```

Input: an `observation_weights(...)` table and the aligned in-sample target.
If `y_train` is a `Series`, it is reindexed to `train_index` when possible.

Output columns add:

| Column | Meaning |
| --- | --- |
| `train_y` | Realized historical outcome. |
| `centered_train_y` | `train_y - mean(y_train)` when `center=True`; otherwise `train_y`. |
| `contribution` | `weight * train_y` by default, or `weight * centered_train_y` with optional base row. |
| `prediction` | Sum of contributions for the forecast row. |
| `channel` | `episode`, or `base` for the optional centered base row. |

Default `center=False` preserves the exact identity
`prediction = weights @ y_train`. Set `center=True` for plots that resemble
the cumulative centered paths in the DualML reference code.

### dual.forecast_diagnostics

```python
macroforecast.interpretation.dual.forecast_diagnostics(
    weights,
    *,
    top_q=0.05,
)
```

Output columns:

| Column | Meaning |
| --- | --- |
| `concentration` | Sum of the top `top_q` share of absolute weights divided by total absolute weight. This matches the R `FC_CR` definition. |
| `short_position` | Signed sum of negative weights. This matches the R `FSP` definition. |
| `short_position_abs` | Absolute value version of the short side. |
| `leverage` | Sum of signed weights. |
| `gross_leverage` | Sum of absolute weights. |
| `turnover` | Sum of absolute weight changes from the previous forecast row. The first row is `NaN`. |
| `top_q`, `top_k`, `n_train` | Diagnostic settings. |

Negative weights are not automatically errors. In this paper they can mean
symmetry-based inference or extrapolation. The caution is economic: macro
shocks are often asymmetric, so a mirror-image historical analogy can be
misleading.

### dual.top_observations

```python
macroforecast.interpretation.dual.top_observations(
    weights,
    *,
    y_train=None,
    n=10,
    sort_by="abs_weight",
)
```

Input: an observation-weight table or contribution table. If `y_train` is
supplied and the input has no `contribution` column, contributions are built
first.

Supported `sort_by` values: `abs_weight`, `weight`, `contribution`, and
`abs_contribution`.

Output: top historical observations per forecast row, with a `rank` column.
This is the most direct table for "which past episodes drove this forecast?"

### dual.group_observation_weights

```python
macroforecast.interpretation.dual.group_observation_weights(
    weights,
    groups,
    *,
    y_train=None,
)
```

Input: observation weights or contributions plus a mapping from group names to
training-index labels, for example:

```python
groups = {
    "volcker": pd.period_range("1979Q3", "1982Q4", freq="Q").to_timestamp("Q"),
    "gfc": pd.period_range("2007Q4", "2009Q2", freq="Q").to_timestamp("Q"),
    "covid": pd.period_range("2020Q1", "2021Q2", freq="Q").to_timestamp("Q"),
}
```

Output columns: `test_row`, `test_index`, `episode_group`, `weight`,
`abs_weight`, `n_episodes`, and, when available, `contribution` and
`abs_contribution`.

Use this when a paper table needs weights assigned to recessions, inflation
surges, tightening cycles, COVID episodes, or user-defined regimes.

## OLS As Attention

This block implements the algebra in Philippe Goulet Coulombe (2026),
"Ordinary Least Squares as an Attention Mechanism." It is not a separate
forecasting model. Standard OLS already produces the forecast. These functions
make the same forecast inspectable as a query-key-value style weighted average
of historical outcomes.

The paper identity is:

```text
beta_hat = (X_train' X_train)^-1 X_train' y_train
y_hat_test = X_test beta_hat
           = X_test (X_train' X_train)^-1 X_train' y_train
           = A y_train
```

where:

| Object | Meaning |
| --- | --- |
| `X_train` | training feature matrix, already preprocessed and feature-engineered |
| `X_test` | forecast-row feature matrix with the same columns |
| `y_train` | realized training outcomes |
| `A` | test-by-train attention matrix |
| `A[j, i]` | signed weight placed on training observation `i` when predicting test row `j` |

Positive weights borrow from a historical outcome. Negative weights use that
historical outcome by contrast. The row weights do not need to be non-negative,
so this is not softmax transformer attention. It is the linear OLS/ridge
attention operator implied by the regression geometry.

Recommended use:

```python
fit = mf.models.ols(X_train, y_train)

weights = mf.interpretation.ols_attention_weights(X_train, X_test)
audit = mf.interpretation.ols_attention_equivalence(
    X_train,
    y_train,
    X_test,
    reference_predictions=fit.predict(X_test),
)

embedding = mf.interpretation.ols_attention_embedding(X_train, X_test)
```

For high-dimensional macro panels, use the ridge version:

```python
fit = mf.models.ridge(X_train, y_train, alpha=0.1)
weights = mf.interpretation.ridge_attention_weights(X_train, X_test, alpha=0.1)
```

With `add_intercept=True`, the intercept column is included in the design but
is not penalized. This matches standard ridge practice in sklearn/R-style
linear regression and keeps the affine identity for intercept models.

### ols_attention_weights

```python
macroforecast.interpretation.ols_attention_weights(
    X_train,
    X_test=None,
    *,
    add_intercept=True,
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `X_train` | pandas `DataFrame` | required | Training feature matrix. Index labels become `train_index`. |
| `X_test` | pandas `DataFrame` or `None` | `None` | Forecast-row feature matrix. If omitted, training rows are explained against themselves. Missing columns are filled with zero after reindexing to `X_train.columns`. |
| `add_intercept` | bool | `True` | Add an intercept column before forming the attention matrix. |

Output: long pandas `DataFrame`.

| Column | Meaning |
| --- | --- |
| `test_row`, `test_index` | forecast-row position and label |
| `train_row`, `train_index` | training-row position and label |
| `weight` | attention weight `A[j, i]` |

The dense matrix `A` is attached at `attrs["attention_matrix"]`.

### ridge_attention_weights

```python
macroforecast.interpretation.ridge_attention_weights(
    X_train,
    X_test=None,
    *,
    alpha=1.0,
    add_intercept=True,
)
```

Input is the same as `ols_attention_weights`, plus:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `alpha` | float | `1.0` | Ridge penalty in `X_train'X_train + alpha I`. Must be non-negative. |

Output: same long attention-weight table. This function is useful when the
feature matrix is nearly singular or when the fitted forecasting model is
`mf.models.ridge(..., alpha=alpha)`.

### ols_attention_embedding

```python
macroforecast.interpretation.ols_attention_embedding(
    X_train,
    X_test=None,
    *,
    add_intercept=True,
    ridge=0.0,
    tol=1e-12,
)
```

This returns the whitened embedding view emphasized in the paper. In the
full-rank OLS case:

```text
(X_train'X_train)^-1 = U Lambda^-1 U'
F_train = X_train U Lambda^-1/2
F_test  = X_test  U Lambda^-1/2
A = F_test F_train'
```

For ridge or rank-deficient cases, macroforecast uses the symmetric
pseudoinverse precision matrix and keeps positive precision components above
`tol`.

Output: long pandas `DataFrame`.

| Column | Meaning |
| --- | --- |
| `sample` | `train` or `test` |
| `row`, `index` | row position and label |
| `component` | whitened embedding component |
| `value` | embedded coordinate |
| `precision_eigenvalue` | eigenvalue of the precision matrix used for that component |

Attached arrays:

| Attr key | Meaning |
| --- | --- |
| `train_embedding` | dense `F_train` matrix |
| `test_embedding` | dense `F_test` matrix |
| `attention_matrix` | reconstructed `F_test F_train'` matrix |
| `precision_matrix` | symmetric inverse or ridge inverse of the design Gram matrix |
| `precision_eigenvalues` | retained positive precision eigenvalues |

### ols_attention_equivalence

```python
macroforecast.interpretation.ols_attention_equivalence(
    X_train,
    y_train,
    X_test=None,
    *,
    reference_predictions=None,
    add_intercept=True,
    ridge=0.0,
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `X_train` | pandas `DataFrame` | required | Training feature matrix. |
| `y_train` | pandas `Series` or array | required | Training outcomes. If a `Series`, it is aligned to `X_train.index`. |
| `X_test` | pandas `DataFrame` or `None` | `None` | Forecast rows. |
| `reference_predictions` | sequence or `None` | `None` | Optional predictions from an already fitted model. If omitted, macroforecast computes the closed-form normal-equation prediction. |
| `add_intercept` | bool | `True` | Same convention as the weight functions. |
| `ridge` | float | `0.0` | Set to the ridge model's `alpha` when auditing a ridge forecast. |

Output: one row per test observation.

| Column | Meaning |
| --- | --- |
| `attention_prediction` | `A @ y_train` |
| `reference_prediction` | closed-form or user-supplied prediction |
| `equivalence_error` | signed difference |
| `abs_equivalence_error` | absolute difference |

Use this as the auditable paper check: if OLS/Ridge and attention are aligned,
`abs_equivalence_error` should be near machine precision.

### attention_weights

```python
macroforecast.interpretation.attention_weights(
    X_train,
    X_test=None,
    *,
    add_intercept=True,
    ridge=1e-8,
)
```

Input: training feature frame and optional test feature frame.

Output columns: `test_row`, `test_index`, `train_row`, `train_index`, and
`weight`. The full attention matrix is also attached at
`attrs["attention_matrix"]`.

The implemented formula is:

```text
Omega = X_test (X_train' X_train)^-1 X_train'
```

with a small ridge term for numerical stability. When `add_intercept=True`,
the ridge term is not applied to the intercept column, matching the standard
regression convention.

Prefer `ols_attention_weights()` for exact paper-style OLS and
`ridge_attention_weights()` for the ridge-stabilized variant. The older
`attention_weights()` helper remains available as the generic numerically
stabilized route.

### dual_decomposition

```python
macroforecast.interpretation.dual_decomposition(
    X_train,
    y_train,
    X_test=None,
    *,
    add_intercept=True,
    ridge=1e-8,
)
```

Input: training features, training target, and optional test features.

Output columns add `train_y` and `contribution` to the attention-weight table.
The per-test prediction summary is attached at `attrs["prediction_summary"]`.

## Macroeconomic Random Forest GTVP

### mrf_gtvp

```python
macroforecast.interpretation.mrf_gtvp(model, X=None)
```

Input: a fitted `mf.models.macro_random_forest(...)` result. The MRF reference
backend only creates time-varying coefficients after prediction, so call
`fit.predict(X_test)` first or pass `X` to `mrf_gtvp()` so it can trigger
prediction.

Output columns: `row`, `index`, `feature`, `coefficient`,
`abs_coefficient`, and `importance`.

`feature="__intercept__"` is the time-varying intercept. Other features are
the coefficient columns returned by the MacroRandomForest reference backend's
`betas` output. A compact feature-level summary is attached at
`attrs["summary"]`.

## VAR Interpretation

### generalized_irf

```python
macroforecast.interpretation.generalized_irf(model, n_periods=12, target=None)
```

Input: fitted VAR-style model, including `mf.models.var(...)`.

Output columns: `feature`, `importance`, `coefficient`, and `status`.

The function implements the Pesaran-Shin generalized impulse response:

```text
GIRF_h(j) = sigma_jj^(-1/2) A_h Sigma e_j
```

`importance` is the sum of absolute target responses across horizons.

### orthogonalised_irf

```python
macroforecast.interpretation.orthogonalised_irf(model, n_periods=12, target=None)
```

Input and output shape match `generalized_irf`, but responses come from the
Cholesky orthogonalised IRF path.

### fevd

```python
macroforecast.interpretation.fevd(model, n_periods=12, target=None)
```

Input: fitted VAR-style model.

Output columns: `feature`, `importance`, `coefficient`, and `status`.
`importance` is the summed forecast-error variance contribution over the
requested horizon range.

### historical_decomposition

```python
macroforecast.interpretation.historical_decomposition(model, max_lag=12, target=None)
```

Input: fitted VAR-style model.

Output columns: `feature`, `importance`, `mean_contribution`, and
`max_abs_contribution`. The implementation uses reduced-form MA coefficients
and fitted residuals to summarize historical shock contributions.

## Deep-Model Hidden States

### lstm_hidden_state

```python
macroforecast.interpretation.lstm_hidden_state(model, X)
```

Input: a fitted torch-backed LSTM/GRU model from `mf.models.lstm(...)` or
`mf.models.gru(...)`, plus a feature frame.

Output columns: `feature`, `importance`, and `coefficient`, where each feature
is a hidden unit. This requires the `deep` extra. Transformer models are
rejected because the recurrent hidden-state interpretation does not apply.

### Gradient attribution

```python
macroforecast.interpretation.saliency_map(model, X)
macroforecast.interpretation.gradient_attribution(model, X, method="saliency")
macroforecast.interpretation.integrated_gradients(model, X, baseline=None, n_steps=50)
macroforecast.interpretation.gradient_shap(model, X, baseline=None, n_samples=20)
macroforecast.interpretation.deep_lift(model, X, baseline=None)
```

Input: a fitted torch-backed model from `mf.models.nn(...)`, `mf.models.lstm(...)`,
`mf.models.gru(...)`, or `mf.models.transformer(...)`, plus a feature frame.

Output columns: `feature`, `importance`, `mean_attribution`,
`std_attribution`, and `method`.

`gradient_attribution` is the method-dispatching helper. `saliency_map`,
`integrated_gradients`, and `gradient_shap` are implemented directly with torch
autograd. `deep_lift` uses Captum and therefore requires the `deep` extra. For
recurrent models, attribution is averaged across sequence positions before
feature-level aggregation. If a baseline is supplied as a pandas `DataFrame`
for a standardized torch-backed model, macroforecast transforms the baseline
into the fitted model's input scale before attribution.

## Custom Interpretation

```python
macroforecast.interpretation.custom_interpretation(
    model,
    X,
    func,
    *,
    y=None,
    name=None,
    metadata=None,
    **params,
)
```

Input: a fitted `ModelFit` or predictor, feature `DataFrame`, optional target,
and a user callable.

Callable signature:

```python
func(model, X, *, y=None, metadata=None, **params)
```

Accepted callable outputs are `DataFrame`, `Series`, mapping, or a sequence
convertible to a `DataFrame`. The wrapper attaches:

| Attr | Meaning |
| --- | --- |
| `macroforecast_metadata_schema.kind` | Always `custom_interpretation`. |
| `macroforecast_metadata_schema.method` | `name` or callable name. |
| `macroforecast_metadata_schema.metadata.params` | User parameters passed to the callable. |
| `macroforecast_metadata_schema.metadata.user_metadata` | User-supplied metadata mapping. |

Example:

```python
def signed_mean_effect(model, X, *, y=None, metadata=None, scale=1.0):
    pred = model.predict(X)
    return {"signed_mean_prediction": float(pred.mean() * scale)}

custom = mf.interpretation.custom_interpretation(
    fit,
    X_test,
    signed_mean_effect,
    name="signed_mean_effect",
    scale=100.0,
)
```

## Examples

```python
fit = mf.models.ridge(X_train, y_train)

coef = mf.interpretation.linear_coefficients(fit)
perm = mf.interpretation.permutation_importance(
    fit,
    X_test,
    y_test,
    n_repeats=20,
    random_state=123,
)
pdp = mf.interpretation.partial_dependence(fit, X_test, features=["PAYEMS"])
ice = mf.interpretation.ice_curves(fit, X_test, features=["PAYEMS"])
ale = mf.interpretation.accumulated_local_effect(fit, X_test, feature="PAYEMS")
```

```python
tree = mf.models.random_forest(X_train, y_train, random_state=123)
shap_table = mf.interpretation.shap_values(
    tree,
    X_test,
    background=X_train,
    explainer="tree",
)
```

- `var_impulse_response` -- VAR impulse responses with Monte-Carlo bootstrap confidence bands (statsmodels errband_mc; vars::irf), as a tidy horizon/impulse/response table.
