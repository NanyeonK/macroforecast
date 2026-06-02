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
| Captum backend | `deep_lift` | Captum `DeepLift` | Direct Captum call. |
| Standard model-agnostic diagnostics | `permutation_importance`, `lofo_importance(fit_func=...)`, `partial_dependence`, `individual_conditional_expectation`, `ice_curves`, `accumulated_local_effect`, `friedman_h_interaction` | scikit-learn permutation/PDP/ICE, R `pdp::partial`, R `ALEPlot`, R `iml`/`hstats` H-statistic conventions | Implements standard diagnostic definitions directly. Interpretation is predictive/associational, not causal. |
| Approximate or fixed-model diagnostics | `permutation_importance_strobl`, `lofo_importance(fit_func=None)`, `cumulative_r2_contribution`, `rolling_recompute`, `bootstrap_jackknife`, non-linear `forecast_decomposition` fallback | Strobl conditional-permutation idea, LOFO refit idea, macroforecast fixed-model diagnostics | Metadata records approximation/fixed-model mode. Do not describe these as exact refit or additive decompositions. |
| VAR/macroeconomic interpretation | `generalized_irf`, `orthogonalised_irf`, `fevd`, `historical_decomposition` | statsmodels VAR result API when available; Pesaran-Shin GIRF formula; internal statsmodels-like adapter for macroforecast VAR | `fevd` now falls back to manual orthogonalized FEVD rather than IRF. `historical_decomposition` is reduced-form, not structural identification. |
| Neural manual attribution | `saliency_map`, `integrated_gradients`, `gradient_shap`, `lstm_hidden_state` | Captum-style torch-gradient methods | Manual torch autograd implementation, except `deep_lift`. Integrated gradients uses a straight-line Riemann approximation, not Captum's default Gauss-Legendre backend. |
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
| `forecast_decomposition(model, X, row=-1)` | fitted predictor and feature frame | `DataFrame` | Linear contribution of each feature to one prediction. |
| `cumulative_r2_contribution(model, X, y, feature_order=None)` | fitted predictor, feature frame, target vector | `DataFrame` | Sequential R-squared contribution under zero-filled inactive features. |
| `rolling_recompute(model, X, y, window=None, step=None, method="permutation_importance")` | fitted predictor, feature frame, target vector | `DataFrame` | Recompute importance over rolling evaluation windows. |
| `bootstrap_jackknife(model, X, y, fit_func=None, n_replications=50)` | fitted predictor or refit callable, feature frame, target vector | `DataFrame` | Bootstrap uncertainty summary for importance. |
| `group_aggregate(table, groups=None, ...)` | feature-importance table | `DataFrame` | Aggregate feature importance by user or inferred groups. |
| `lineage_attribution(table, lineage, level="pipeline_name")` | feature-importance table and lineage mapping | `DataFrame` | Aggregate importance by feature-engineering lineage metadata. |
| `transformation_attribution(evaluation, ..., lower_is_better=True, baseline="worst")` | evaluation score table | `DataFrame` | Attribute score improvements to mutually exclusive preprocessing/feature pipelines. |
| `attention_weights(X_train, X_test=None, ...)` | training and test feature frames | long `DataFrame` | OLS attention matrix weights. |
| `dual_decomposition(X_train, y_train, X_test=None, ...)` | train features/outcomes and optional test features | long `DataFrame` | OLS prediction as weighted training-outcome contributions. |
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

## Attention And Dual Views

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
