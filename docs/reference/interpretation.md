# macroforecast.interpretation

[Back to reference](index.md)

Model, forecast, and observation attribution helpers.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `AnatomyPipelineResult` | class | End-to-end ``anatomy`` backend output. |
| `DualInterpretationResult` | class | Paper-aligned dual interpretation output bundle. |
| `ForecastShapleyResult` | class | End-to-end ``anatomy`` backend output. |
| `accumulated_local_effect` | function | Compute a first-order accumulated local effect curve. |
| `accumulated_local_effect_2d` | function | Second-order (two-feature) accumulated local effect (Apley-Zhu 2020). |
| `anatomy_from_forecast_result` | function | Build anatomy outputs from a forecast result plus explicit X/y inputs. |
| `anatomy_model` | function | Wrap a macroforecast fit or estimator as ``anatomy.AnatomyModel``. |
| `anatomy_output_transformer` | function | Return an ``anatomy`` output transformer for forecast/loss explanations. |
| `anatomy_pipeline` | function | Run the complete ``anatomy`` provider, precompute, and summary path. |
| `anatomy_explain` | function | Explain a precomputed ``anatomy.Anatomy`` object. |
| `anatomy_provider` | function | Build an ``AnatomyModelProvider`` from aligned macroforecast X/y data. |
| `attention_weights` | function | OLS attention weights ``Omega = X_test (X_train'X_train)^-1 X_train'``. |
| `bootstrap_jackknife` | function | Bootstrap or jackknife-style uncertainty summary for importance. |
| `cumulative_r2_contribution` | function | Sequential contribution of features to in-sample prediction R-squared. |
| `custom_interpretation` | function | Run a user-supplied interpretation callable and attach metadata. |
| `data_portfolio_diagnostics` | function | Summarize DualML data-portfolio concentration, shorts, leverage, turnover. |
| `deep_lift` | function | DeepLift attribution through Captum for torch-backed models. |
| `dual` | module | No public docstring is available. |
| `dual_decomposition` | function | Represent OLS predictions as weighted sums of training outcomes. |
| `dual_from_forecast_result` | function | Build a dual interpretation sidecar for a completed forecast result. |
| `dual_interpretation` | function | Run the ridge/KRR/RF DualML interpretation path in one callable. |
| `episode_group_weights` | function | Aggregate historical-observation weights over named episode groups. |
| `fevd` | function | Forecast error variance decomposition importance for VAR models. |
| `forecast_decomposition` | function | Decompose one prediction into linear feature contributions. |
| `forecast_diagnostics` | function | Return concentration, short-position, leverage, and turnover diagnostics. |
| `forecast_shapley_output` | function | Select one oShapley/PBSV output from a result, sidecar, or backend object. |
| `friedman_h_interaction` | function | Compute pairwise Friedman-Popescu H interaction statistics. |
| `generalized_irf` | function | Pesaran-Shin generalized impulse response importance for VAR models. |
| `var_impulse_response` | function | Impulse-response functions with Monte-Carlo bootstrap confidence bands. |
| `gradient_attribution` | function | Gradient attribution for torch-backed models. |
| `gradient_shap` | function | Expected-gradients approximation to GradientSHAP. |
| `group_aggregate` | function | Aggregate feature-level importance into user or metadata groups. |
| `group_observation_weights` | function | Aggregate observation weights over named historical groups. |
| `historical_decomposition` | function | Reduced-form VAR historical contribution summary. |
| `ice_curves` | function | Alias for :func:`individual_conditional_expectation`. |
| `individual_conditional_expectation` | function | Compute one-way individual conditional expectation curves. |
| `integrated_gradients` | function | Integrated gradients for torch-backed models. |
| `ishapley_vi` | function | Aggregate in-sample Shapley contributions into iShapley-VI_p. |
| `lasso_inclusion_frequency` | function | Estimate coefficient nonzero frequency for lasso-style models. |
| `lineage_attribution` | function | Aggregate feature importance using feature-lineage metadata. |
| `linear_coefficients` | function | Return native coefficients for linear-style fitted models. |
| `lofo_importance` | function | Leave-one-feature-out importance. |
| `lstm_hidden_state` | function | LSTM/GRU hidden-unit activation importance for torch-backed models. |
| `model_native_linear_coef` | function | Alias for legacy L7 naming: native linear coefficients. |
| `model_native_tree_importance` | function | Alias for legacy L7 naming: native tree feature importance. |
| `model_accordance_score` | function | Compute the anatomy backend Model Accordance Score. |
| `mrf_gtvp` | function | Return Macroeconomic Random Forest GTVP coefficient paths. |
| `observation_contributions` | function | Convert observation weights into observation-level forecast contributions. |
| `observation_weights` | function | Compute DualML-style historical-observation weights for forecasts. |
| `ols_attention_embedding` | function | Return whitened train/test embeddings behind OLS-as-attention. |
| `ols_attention_equivalence` | function | Audit that closed-form predictions equal attention-weight predictions. |
| `ols_attention_weights` | function | Exact OLS-as-attention weights from Goulet Coulombe (2026). |
| `oshapley_vi` | function | Compute anatomy backend oShapley-VI from raw forecast explanations. |
| `oshapley_from_forecast_result` | function | Build an oShapley-VI/PBSV sidecar for a completed forecast result. |
| `oshapley_output` | function | Alias for :func:`forecast_shapley_output`. |
| `oshapley_pipeline` | function | Run the oShapley-VI/PBSV forecast-accuracy interpretation pipeline. |
| `oshapley_provider` | function | Build the provider used by oShapley-VI and PBSV precompute. |
| `orthogonalised_irf` | function | Cholesky orthogonalised impulse response importance for VAR models. |
| `outcome_contributions` | function | Convert observation weights into historical-outcome contributions. |
| `partial_dependence` | function | Compute one-way manual partial-dependence curves. |
| `pbsv` | function | Compute backend PBSV_p loss decomposition through ``anatomy``. |
| `permutation_importance` | function | Compute simple model-agnostic permutation importance. |
| `permutation_importance_strobl` | function | Conditional permutation importance following the Strobl idea. |
| `performance_based_shapley_value` | function | Alias for :func:`pbsv` when using the anatomy backend. |
| `performance_shapley_value` | function | Compute PBSV-style loss attribution from additive forecast contributions. |
| `precompute_anatomy` | function | Build, precompute, and optionally save an ``anatomy.Anatomy`` object. |
| `precompute_oshapley` | function | Precompute the backend object used by oShapley-VI and PBSV. |
| `ridge_attention_weights` | function | Ridge-stabilized OLS attention weights. |
| `rolling_recompute` | function | Recompute feature importance on rolling evaluation windows. |
| `saliency_map` | function | Vanilla input-gradient attribution for torch-backed models. |
| `shapley_variable_importance` | function | Aggregate local Shapley contributions into variable importance. |
| `shap_deep` | function | Deep-model SHAP-style global importance. |
| `shap_importance` | function | Summarize SHAP values as global mean absolute feature importance. |
| `shap_kernel` | function | Kernel/permutation SHAP-style global importance. |
| `shap_linear` | function | Linear SHAP-style global importance using ``shap.Explainer``. |
| `shap_tree` | function | Tree SHAP global importance using the optional ``shap`` backend. |
| `shap_values` | function | Return SHAP values in a long pandas table. |
| `tree_importance` | function | Return native tree importance for estimators exposing feature_importances_. |
| `transformation_attribution` | function | Attribute forecast score differences to preprocessing/feature pipelines. |
| `top_episodes` | function | Return the largest historical-episode weights for each forecast row. |
| `top_observations` | function | Return the largest historical observations per forecast row. |
| `window_to_anatomy_subsets` | function | Convert a macroforecast window into exact ``anatomy`` train/test subsets. |
| `window_to_oshapley_subsets` | function | Convert a macroforecast window for oShapley/PBSV backend precompute. |

## Data And Module Values

### `dual`

Kind: `module`

```python
dual = <module macroforecast.interpretation.dual>
```

## Callable And Class Reference

### AnatomyPipelineResult

Qualified name: `macroforecast.interpretation.anatomy.AnatomyPipelineResult`

#### Signature

```python
macroforecast.interpretation.AnatomyPipelineResult(anatomy: Any, explanations: dict[str, pd.DataFrame] = <factory>, variable_importance: pd.DataFrame | None = None, performance_values: dict[str, pd.DataFrame] = <factory>, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

End-to-end ``anatomy`` backend output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `anatomy` | positional or keyword | `Any` | `required` |
| `explanations` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `variable_importance` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `performance_values` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.interpretation.AnatomyPipelineResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `anatomy` | `Any` | `required` |
| `explanations` | `dict[str, pd.DataFrame]` | `default_factory` |
| `variable_importance` | `pd.DataFrame \| None` | `None` |
| `performance_values` | `dict[str, pd.DataFrame]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_tables` | `to_tables(self, *, prefix: str = "anatomy") -> dict[str, pd.DataFrame]` | Return JSON/CSV-ready anatomy output tables. |
### DualInterpretationResult

Qualified name: `macroforecast.interpretation.dual.DualInterpretationResult`

#### Signature

```python
macroforecast.interpretation.DualInterpretationResult(weights: pd.DataFrame, contributions: pd.DataFrame | None = None, diagnostics: pd.DataFrame | None = None, top_observations: pd.DataFrame | None = None, group_weights: pd.DataFrame | None = None, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Paper-aligned dual interpretation output bundle.

Goulet Coulombe, Goebel, and Klieber's DualML code reports observation
weights, observation contributions, and data-portfolio diagnostics as
connected objects. The result container keeps that relation explicit while
still exposing output-ready tables through ``to_tables``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `contributions` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `diagnostics` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `top_observations` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `group_weights` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.interpretation.DualInterpretationResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `weights` | `pd.DataFrame` | `required` |
| `contributions` | `pd.DataFrame \| None` | `None` |
| `diagnostics` | `pd.DataFrame \| None` | `None` |
| `top_observations` | `pd.DataFrame \| None` | `None` |
| `group_weights` | `pd.DataFrame \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_tables` | `to_tables(self, *, prefix: str = "dual") -> dict[str, pd.DataFrame]` | Return output-ready tables with paper-aligned names. |
### ForecastShapleyResult

Qualified name: `macroforecast.interpretation.anatomy.AnatomyPipelineResult`

#### Signature

```python
macroforecast.interpretation.ForecastShapleyResult(anatomy: Any, explanations: dict[str, pd.DataFrame] = <factory>, variable_importance: pd.DataFrame | None = None, performance_values: dict[str, pd.DataFrame] = <factory>, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

End-to-end ``anatomy`` backend output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `anatomy` | positional or keyword | `Any` | `required` |
| `explanations` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `variable_importance` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `performance_values` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.interpretation.ForecastShapleyResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `anatomy` | `Any` | `required` |
| `explanations` | `dict[str, pd.DataFrame]` | `default_factory` |
| `variable_importance` | `pd.DataFrame \| None` | `None` |
| `performance_values` | `dict[str, pd.DataFrame]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_tables` | `to_tables(self, *, prefix: str = "anatomy") -> dict[str, pd.DataFrame]` | Return JSON/CSV-ready anatomy output tables. |
### accumulated_local_effect

Qualified name: `macroforecast.interpretation.core.accumulated_local_effect`

#### Signature

```python
macroforecast.interpretation.accumulated_local_effect(model: Any, X: pd.DataFrame, *, feature: str, bins: int = 10) -> pd.DataFrame
```

#### Description

Compute a first-order accumulated local effect curve.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `feature` | keyword only | `str` | `required` |
| `bins` | keyword only | `int` | `10` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.accumulated_local_effect(...)
```
### accumulated_local_effect_2d

Qualified name: `macroforecast.interpretation.core.accumulated_local_effect_2d`

#### Signature

```python
macroforecast.interpretation.accumulated_local_effect_2d(model: Any, X: pd.DataFrame, *, features: tuple[str, str], bins: int = 10) -> pd.DataFrame
```

#### Description

Second-order (two-feature) accumulated local effect (Apley-Zhu 2020).

Computes the pure interaction ALE surface of two features: per 2D quantile
cell it averages the second-order finite difference of the prediction over
the cell corners, accumulates it over both axes, and removes the (weighted)
main effects so the surface is the interaction not explained by the
first-order ALEs. Returns a tidy table with the cell centres, the interaction
ALE, and the cell counts. A purely additive model yields ~0 everywhere.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `features` | keyword only | `tuple[str, str]` | `required` |
| `bins` | keyword only | `int` | `10` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.accumulated_local_effect_2d(...)
```
### anatomy_from_forecast_result

Qualified name: `macroforecast.interpretation.anatomy.anatomy_from_forecast_result`

#### Signature

```python
macroforecast.interpretation.anatomy_from_forecast_result(result: Any, X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], *, window: WindowSpec | str | None, attach: bool = True, sidecar_name: str = "anatomy", params: Mapping[str, Any] | None = None, model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, target_name: str | None = None, train_source: str = "fit", losses: Sequence[str] = ('rmse',), n_iterations: int = 32, n_jobs: int = 1, background_data_subsample: float = 1.0, save_path: str | Path | None = None) -> Any
```

#### Description

Build anatomy outputs from a forecast result plus explicit X/y inputs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `required` |
| `attach` | keyword only | `bool` | `True` |
| `sidecar_name` | keyword only | `str` | `"anatomy"` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `target_name` | keyword only | `str \| None` | `None` |
| `train_source` | keyword only | `str` | `"fit"` |
| `losses` | keyword only | `Sequence[str]` | `("rmse",)` |
| `n_iterations` | keyword only | `int` | `32` |
| `n_jobs` | keyword only | `int` | `1` |
| `background_data_subsample` | keyword only | `float` | `1.0` |
| `save_path` | keyword only | `str \| Path \| None` | `None` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_from_forecast_result(...)
```
### anatomy_model

Qualified name: `macroforecast.interpretation.anatomy.anatomy_model`

#### Signature

```python
macroforecast.interpretation.anatomy_model(model: Any, *, feature_names: Sequence[str] | None = None) -> Any
```

#### Description

Wrap a macroforecast fit or estimator as ``anatomy.AnatomyModel``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `feature_names` | keyword only | `Sequence[str] \| None` | `None` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_model(...)
```
### anatomy_output_transformer

Qualified name: `macroforecast.interpretation.anatomy.anatomy_output_transformer`

#### Signature

```python
macroforecast.interpretation.anatomy_output_transformer(output: MetricLike = "forecast") -> Any
```

#### Description

Return an ``anatomy`` output transformer for forecast/loss explanations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `output` | positional or keyword | `MetricLike` | `"forecast"` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_output_transformer(...)
```
### anatomy_pipeline

Qualified name: `macroforecast.interpretation.anatomy.anatomy_pipeline`

#### Signature

```python
macroforecast.interpretation.anatomy_pipeline(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], *, window: WindowSpec | str | None = None, params: Mapping[str, Any] | None = None, model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, target_name: str | None = None, train_source: str = "fit", losses: Sequence[str] = ('rmse',), n_iterations: int = 32, n_jobs: int = 1, background_data_subsample: float = 1.0, save_path: str | Path | None = None) -> AnatomyPipelineResult
```

#### Description

Run the complete ``anatomy`` provider, precompute, and summary path.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `target_name` | keyword only | `str \| None` | `None` |
| `train_source` | keyword only | `str` | `"fit"` |
| `losses` | keyword only | `Sequence[str]` | `("rmse",)` |
| `n_iterations` | keyword only | `int` | `32` |
| `n_jobs` | keyword only | `int` | `1` |
| `background_data_subsample` | keyword only | `float` | `1.0` |
| `save_path` | keyword only | `str \| Path \| None` | `None` |

#### Returns

`AnatomyPipelineResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_pipeline(...)
```
### anatomy_explain

Qualified name: `macroforecast.interpretation.core.anatomy_explain`

#### Signature

```python
macroforecast.interpretation.anatomy_explain(anatomy: Any, *, model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, transformer: Callable[..., Any] | Any | None = None, metric: str = "forecast", explanation_subset: pd.Index | Sequence[Any] | None = None, output: str = "long") -> pd.DataFrame
```

#### Description

Explain a precomputed ``anatomy.Anatomy`` object.

This is a thin backend wrapper around the Python ``anatomy`` package from
Borup et al., *The Anatomy of Out-of-Sample Forecasting Accuracy*.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `anatomy` | positional or keyword | `Any` | `required` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `transformer` | keyword only | `Callable[..., Any] \| Any \| None` | `None` |
| `metric` | keyword only | `str` | `"forecast"` |
| `explanation_subset` | keyword only | `pd.Index \| Sequence[Any] \| None` | `None` |
| `output` | keyword only | `str` | `"long"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_explain(...)
```
### anatomy_provider

Qualified name: `macroforecast.interpretation.anatomy.anatomy_provider`

#### Signature

```python
macroforecast.interpretation.anatomy_provider(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], *, window: WindowSpec | str | None = None, params: Mapping[str, Any] | None = None, target_name: str | None = None, train_source: str = "fit") -> Any
```

#### Description

Build an ``AnatomyModelProvider`` from aligned macroforecast X/y data.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target_name` | keyword only | `str \| None` | `None` |
| `train_source` | keyword only | `str` | `"fit"` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.anatomy_provider(...)
```
### attention_weights

Qualified name: `macroforecast.interpretation.core.attention_weights`

#### Signature

```python
macroforecast.interpretation.attention_weights(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, add_intercept: bool = True, ridge: float = 1e-08) -> pd.DataFrame
```

#### Description

OLS attention weights ``Omega = X_test (X_train'X_train)^-1 X_train'``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |
| `ridge` | keyword only | `float` | `1e-08` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.attention_weights(...)
```
### bootstrap_jackknife

Qualified name: `macroforecast.interpretation.core.bootstrap_jackknife`

#### Signature

```python
macroforecast.interpretation.bootstrap_jackknife(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None, n_replications: int = 50, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Bootstrap or jackknife-style uncertainty summary for importance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `fit_func` | keyword only | `Callable[[pd.DataFrame, pd.Series], Any] \| None` | `None` |
| `n_replications` | keyword only | `int` | `50` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.bootstrap_jackknife(...)
```
### cumulative_r2_contribution

Qualified name: `macroforecast.interpretation.core.cumulative_r2_contribution`

#### Signature

```python
macroforecast.interpretation.cumulative_r2_contribution(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, feature_order: Sequence[str] | None = None) -> pd.DataFrame
```

#### Description

Sequential contribution of features to in-sample prediction R-squared.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `feature_order` | keyword only | `Sequence[str] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.cumulative_r2_contribution(...)
```
### custom_interpretation

Qualified name: `macroforecast.interpretation.core.custom_interpretation`

#### Signature

```python
macroforecast.interpretation.custom_interpretation(model: Any, X: pd.DataFrame, func: Callable[..., Any], *, y: pd.Series | np.ndarray | None = None, name: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Run a user-supplied interpretation callable and attach metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `y` | keyword only | `pd.Series \| np.ndarray \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.custom_interpretation(...)
```
### data_portfolio_diagnostics

Qualified name: `macroforecast.interpretation.core.data_portfolio_diagnostics`

#### Signature

```python
macroforecast.interpretation.data_portfolio_diagnostics(weights: pd.DataFrame, *, top_q: float = 0.05) -> pd.DataFrame
```

#### Description

Summarize DualML data-portfolio concentration, shorts, leverage, turnover.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `top_q` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.data_portfolio_diagnostics(...)
```
### deep_lift

Qualified name: `macroforecast.interpretation.core.deep_lift`

#### Signature

```python
macroforecast.interpretation.deep_lift(model: Any, X: pd.DataFrame, *, baseline: float | pd.DataFrame | np.ndarray | None = None) -> pd.DataFrame
```

#### Description

DeepLift attribution through Captum for torch-backed models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `baseline` | keyword only | `float \| pd.DataFrame \| np.ndarray \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.deep_lift(...)
```
### dual_decomposition

Qualified name: `macroforecast.interpretation.core.dual_decomposition`

#### Signature

```python
macroforecast.interpretation.dual_decomposition(X_train: pd.DataFrame, y_train: pd.Series | np.ndarray, X_test: pd.DataFrame | None = None, *, add_intercept: bool = True, ridge: float = 1e-08) -> pd.DataFrame
```

#### Description

Represent OLS predictions as weighted sums of training outcomes.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |
| `ridge` | keyword only | `float` | `1e-08` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.dual_decomposition(...)
```
### dual_from_forecast_result

Qualified name: `macroforecast.interpretation.dual.dual_from_forecast_result`

#### Signature

```python
macroforecast.interpretation.dual_from_forecast_result(result: Any, model: Any | None, X_train: pd.DataFrame, y_train: pd.Series | Sequence[float], X_test: pd.DataFrame | None = None, *, attach: bool = True, sidecar_name: str = "dual", **kwargs: Any) -> Any
```

#### Description

Build a dual interpretation sidecar for a completed forecast result.

A forecast table cannot reconstruct the exact train/test feature matrices
used by the fitted model. The caller therefore passes the fitted model,
training features, training target, and forecast-row features explicitly.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `attach` | keyword only | `bool` | `True` |
| `sidecar_name` | keyword only | `str` | `"dual"` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.dual_from_forecast_result(...)
```
### dual_interpretation

Qualified name: `macroforecast.interpretation.dual.dual_interpretation`

#### Signature

```python
macroforecast.interpretation.dual_interpretation(model: Any | None, X_train: pd.DataFrame, y_train: pd.Series | Sequence[float], X_test: pd.DataFrame | None = None, *, method: str = "auto", lambda_: float = 1e-08, kernel: str = "linear", sigma: float = 1.0, add_intercept: bool = False, ridge_penalty_scale: str = "n_train", normalize: bool = False, center: bool = False, include_base: bool = False, top_n: int = 10, top_sort_by: str = "abs_weight", top_q: float = 0.05, groups: Mapping[str, Sequence[Any]] | None = None, include_contributions: bool = True, include_diagnostics: bool = True, include_top_observations: bool = True, include_group_weights: bool | None = None) -> DualInterpretationResult
```

#### Description

Run the ridge/KRR/RF DualML interpretation path in one callable.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `str` | `"auto"` |
| `lambda_` | keyword only | `float` | `1e-08` |
| `kernel` | keyword only | `str` | `"linear"` |
| `sigma` | keyword only | `float` | `1.0` |
| `add_intercept` | keyword only | `bool` | `False` |
| `ridge_penalty_scale` | keyword only | `str` | `"n_train"` |
| `normalize` | keyword only | `bool` | `False` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |
| `top_n` | keyword only | `int` | `10` |
| `top_sort_by` | keyword only | `str` | `"abs_weight"` |
| `top_q` | keyword only | `float` | `0.05` |
| `groups` | keyword only | `Mapping[str, Sequence[Any]] \| None` | `None` |
| `include_contributions` | keyword only | `bool` | `True` |
| `include_diagnostics` | keyword only | `bool` | `True` |
| `include_top_observations` | keyword only | `bool` | `True` |
| `include_group_weights` | keyword only | `bool \| None` | `None` |

#### Returns

`DualInterpretationResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.dual_interpretation(...)
```
### episode_group_weights

Qualified name: `macroforecast.interpretation.core.episode_group_weights`

#### Signature

```python
macroforecast.interpretation.episode_group_weights(weights: pd.DataFrame, groups: Mapping[str, Sequence[Any]], *, y_train: pd.Series | np.ndarray | None = None) -> pd.DataFrame
```

#### Description

Aggregate historical-observation weights over named episode groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `groups` | positional or keyword | `Mapping[str, Sequence[Any]]` | `required` |
| `y_train` | keyword only | `pd.Series \| np.ndarray \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.episode_group_weights(...)
```
### fevd

Qualified name: `macroforecast.interpretation.core.fevd`

#### Signature

```python
macroforecast.interpretation.fevd(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame
```

#### Description

Forecast error variance decomposition importance for VAR models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `n_periods` | keyword only | `int` | `12` |
| `target` | keyword only | `str \| int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.fevd(...)
```
### forecast_decomposition

Qualified name: `macroforecast.interpretation.core.forecast_decomposition`

#### Signature

```python
macroforecast.interpretation.forecast_decomposition(model: Any, X: pd.DataFrame, *, row: int | str | pd.Timestamp = -1, sort: bool = True) -> pd.DataFrame
```

#### Description

Decompose one prediction into linear feature contributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `row` | keyword only | `int \| str \| pd.Timestamp` | `-1` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.forecast_decomposition(...)
```
### forecast_diagnostics

Qualified name: `macroforecast.interpretation.dual.forecast_diagnostics`

#### Signature

```python
macroforecast.interpretation.forecast_diagnostics(weights: pd.DataFrame, *, top_q: float = 0.05) -> pd.DataFrame
```

#### Description

Return concentration, short-position, leverage, and turnover diagnostics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `top_q` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.forecast_diagnostics(...)
```
### forecast_shapley_output

Qualified name: `macroforecast.interpretation.anatomy.forecast_shapley_output`

#### Signature

```python
macroforecast.interpretation.forecast_shapley_output(value: Any, *, output: str = "oshapley", loss: str = "rmse", sidecar_name: str | None = None, prefix: str = "oshapley", model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, explanation_subset: pd.Index | Sequence[Any] | None = None, table: str | None = None) -> Any
```

#### Description

Select one oShapley/PBSV output from a result, sidecar, or backend object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |
| `output` | keyword only | `str` | `"oshapley"` |
| `loss` | keyword only | `str` | `"rmse"` |
| `sidecar_name` | keyword only | `str \| None` | `None` |
| `prefix` | keyword only | `str` | `"oshapley"` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `explanation_subset` | keyword only | `pd.Index \| Sequence[Any] \| None` | `None` |
| `table` | keyword only | `str \| None` | `None` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.forecast_shapley_output(...)
```
### friedman_h_interaction

Qualified name: `macroforecast.interpretation.core.friedman_h_interaction`

#### Signature

```python
macroforecast.interpretation.friedman_h_interaction(model: Any, X: pd.DataFrame, *, features: Sequence[str] | None = None, grid_size: int = 10) -> pd.DataFrame
```

#### Description

Compute pairwise Friedman-Popescu H interaction statistics.

The implementation uses manual one-way and two-way partial dependence on a
regular grid. Values are bounded to ``[0, inf)`` by construction; larger
values indicate stronger interaction relative to the pair's joint partial
dependence variation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `features` | keyword only | `Sequence[str] \| None` | `None` |
| `grid_size` | keyword only | `int` | `10` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.friedman_h_interaction(...)
```
### generalized_irf

Qualified name: `macroforecast.interpretation.core.generalized_irf`

#### Signature

```python
macroforecast.interpretation.generalized_irf(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame
```

#### Description

Pesaran-Shin generalized impulse response importance for VAR models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `n_periods` | keyword only | `int` | `12` |
| `target` | keyword only | `str \| int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.generalized_irf(...)
```
### var_impulse_response

Qualified name: `macroforecast.interpretation.core.var_impulse_response`

#### Signature

```python
macroforecast.interpretation.var_impulse_response(panel: Any, *, n_lag: int = 1, periods: int = 10, orthogonalized: bool = True, signif: float = 0.05, repl: int = 1000, seed: int | None = None, trend: str = "c") -> pd.DataFrame
```

#### Description

Impulse-response functions with Monte-Carlo bootstrap confidence bands.

Fits a VAR(``n_lag``) on ``panel`` and returns a tidy table with, for each
horizon/impulse/response triple, the (orthogonalised by default) impulse
response and its ``1 - signif`` Monte-Carlo error band (statsmodels
``IRAnalysis.errband_mc``; R vars::irf with bootstrap CI). Macro IRFs are
essentially always reported with such bands.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `periods` | keyword only | `int` | `10` |
| `orthogonalized` | keyword only | `bool` | `True` |
| `signif` | keyword only | `float` | `0.05` |
| `repl` | keyword only | `int` | `1000` |
| `seed` | keyword only | `int \| None` | `None` |
| `trend` | keyword only | `str` | `"c"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.var_impulse_response(...)
```
### gradient_attribution

Qualified name: `macroforecast.interpretation.core.gradient_attribution`

#### Signature

```python
macroforecast.interpretation.gradient_attribution(model: Any, X: pd.DataFrame, *, method: str = "saliency_map", baseline: float | pd.DataFrame | np.ndarray | None = None, n_steps: int = 50, n_samples: int = 20, noise_scale: float = 0.0, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Gradient attribution for torch-backed models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"saliency_map"` |
| `baseline` | keyword only | `float \| pd.DataFrame \| np.ndarray \| None` | `None` |
| `n_steps` | keyword only | `int` | `50` |
| `n_samples` | keyword only | `int` | `20` |
| `noise_scale` | keyword only | `float` | `0.0` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.gradient_attribution(...)
```
### gradient_shap

Qualified name: `macroforecast.interpretation.core.gradient_shap`

#### Signature

```python
macroforecast.interpretation.gradient_shap(model: Any, X: pd.DataFrame, *, baseline: float | pd.DataFrame | np.ndarray | None = None, n_samples: int = 20, noise_scale: float = 0.0, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Expected-gradients approximation to GradientSHAP.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `baseline` | keyword only | `float \| pd.DataFrame \| np.ndarray \| None` | `None` |
| `n_samples` | keyword only | `int` | `20` |
| `noise_scale` | keyword only | `float` | `0.0` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.gradient_shap(...)
```
### group_aggregate

Qualified name: `macroforecast.interpretation.core.group_aggregate`

#### Signature

```python
macroforecast.interpretation.group_aggregate(table: pd.DataFrame, *, groups: Mapping[str, str | Sequence[str]] | None = None, group_column: str | None = None, value_column: str | None = None, aggregation: str = "sum") -> pd.DataFrame
```

#### Description

Aggregate feature-level importance into user or metadata groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `table` | positional or keyword | `pd.DataFrame` | `required` |
| `groups` | keyword only | `Mapping[str, str \| Sequence[str]] \| None` | `None` |
| `group_column` | keyword only | `str \| None` | `None` |
| `value_column` | keyword only | `str \| None` | `None` |
| `aggregation` | keyword only | `str` | `"sum"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.group_aggregate(...)
```
### group_observation_weights

Qualified name: `macroforecast.interpretation.dual.group_observation_weights`

#### Signature

```python
macroforecast.interpretation.group_observation_weights(weights: pd.DataFrame, groups: Mapping[str, Sequence[Any]], *, y_train: pd.Series | Sequence[float] | None = None) -> pd.DataFrame
```

#### Description

Aggregate observation weights over named historical groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `groups` | positional or keyword | `Mapping[str, Sequence[Any]]` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.group_observation_weights(...)
```
### historical_decomposition

Qualified name: `macroforecast.interpretation.core.historical_decomposition`

#### Signature

```python
macroforecast.interpretation.historical_decomposition(model: Any, *, max_lag: int = 12, target: str | int | None = None) -> pd.DataFrame
```

#### Description

Reduced-form VAR historical contribution summary.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `max_lag` | keyword only | `int` | `12` |
| `target` | keyword only | `str \| int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.historical_decomposition(...)
```
### ice_curves

Qualified name: `macroforecast.interpretation.core.ice_curves`

#### Signature

```python
macroforecast.interpretation.ice_curves(model: Any, X: pd.DataFrame, *, features: Iterable[str] | str, grid_size: int = 20, center: bool = False) -> pd.DataFrame
```

#### Description

Alias for :func:`individual_conditional_expectation`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `features` | keyword only | `Iterable[str] \| str` | `required` |
| `grid_size` | keyword only | `int` | `20` |
| `center` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ice_curves(...)
```
### individual_conditional_expectation

Qualified name: `macroforecast.interpretation.core.individual_conditional_expectation`

#### Signature

```python
macroforecast.interpretation.individual_conditional_expectation(model: Any, X: pd.DataFrame, *, features: Iterable[str] | str, grid_size: int = 20, center: bool = False) -> pd.DataFrame
```

#### Description

Compute one-way individual conditional expectation curves.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `features` | keyword only | `Iterable[str] \| str` | `required` |
| `grid_size` | keyword only | `int` | `20` |
| `center` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.individual_conditional_expectation(...)
```
### integrated_gradients

Qualified name: `macroforecast.interpretation.core.integrated_gradients`

#### Signature

```python
macroforecast.interpretation.integrated_gradients(model: Any, X: pd.DataFrame, *, baseline: float | pd.DataFrame | np.ndarray | None = None, n_steps: int = 50) -> pd.DataFrame
```

#### Description

Integrated gradients for torch-backed models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `baseline` | keyword only | `float \| pd.DataFrame \| np.ndarray \| None` | `None` |
| `n_steps` | keyword only | `int` | `50` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.integrated_gradients(...)
```
### ishapley_vi

Qualified name: `macroforecast.interpretation.core.ishapley_vi`

#### Signature

```python
macroforecast.interpretation.ishapley_vi(contributions: pd.DataFrame | pd.Series, *, contribution_col: str | None = None, feature_col: str = "feature", group_col: str | None = None, exclude_base: bool = True) -> pd.DataFrame
```

#### Description

Aggregate in-sample Shapley contributions into iShapley-VI_p.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `contributions` | positional or keyword | `pd.DataFrame \| pd.Series` | `required` |
| `contribution_col` | keyword only | `str \| None` | `None` |
| `feature_col` | keyword only | `str` | `"feature"` |
| `group_col` | keyword only | `str \| None` | `None` |
| `exclude_base` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ishapley_vi(...)
```
### lasso_inclusion_frequency

Qualified name: `macroforecast.interpretation.core.lasso_inclusion_frequency`

#### Signature

```python
macroforecast.interpretation.lasso_inclusion_frequency(model: Any, X: pd.DataFrame | None = None, y: pd.Series | np.ndarray | None = None, *, fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None, n_bootstraps: int = 50, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Estimate coefficient nonzero frequency for lasso-style models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `y` | positional or keyword | `pd.Series \| np.ndarray \| None` | `None` |
| `fit_func` | keyword only | `Callable[[pd.DataFrame, pd.Series], Any] \| None` | `None` |
| `n_bootstraps` | keyword only | `int` | `50` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.lasso_inclusion_frequency(...)
```
### lineage_attribution

Qualified name: `macroforecast.interpretation.core.lineage_attribution`

#### Signature

```python
macroforecast.interpretation.lineage_attribution(table: pd.DataFrame, lineage: Mapping[str, Any], *, level: str = "pipeline_name", value_column: str | None = None, aggregation: str = "sum") -> pd.DataFrame
```

#### Description

Aggregate feature importance using feature-lineage metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `table` | positional or keyword | `pd.DataFrame` | `required` |
| `lineage` | positional or keyword | `Mapping[str, Any]` | `required` |
| `level` | keyword only | `str` | `"pipeline_name"` |
| `value_column` | keyword only | `str \| None` | `None` |
| `aggregation` | keyword only | `str` | `"sum"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.lineage_attribution(...)
```
### linear_coefficients

Qualified name: `macroforecast.interpretation.core.linear_coefficients`

#### Signature

```python
macroforecast.interpretation.linear_coefficients(model: Any, *, sort: bool = True) -> pd.DataFrame
```

#### Description

Return native coefficients for linear-style fitted models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.linear_coefficients(...)
```
### lofo_importance

Qualified name: `macroforecast.interpretation.core.lofo_importance`

#### Signature

```python
macroforecast.interpretation.lofo_importance(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None, metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse", sort: bool = True) -> pd.DataFrame
```

#### Description

Leave-one-feature-out importance.

If ``fit_func`` is supplied, the model is refit without each feature. If it
is omitted, the already fitted model is evaluated after setting the held-out
feature to zero. The latter is a prediction-drop diagnostic, not a refit
LOFO experiment, and the returned metadata records that mode explicitly.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `fit_func` | keyword only | `Callable[[pd.DataFrame, pd.Series], Any] \| None` | `None` |
| `metric` | keyword only | `Callable[[np.ndarray, np.ndarray], float] \| str` | `"mse"` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.lofo_importance(...)
```
### lstm_hidden_state

Qualified name: `macroforecast.interpretation.core.lstm_hidden_state`

#### Signature

```python
macroforecast.interpretation.lstm_hidden_state(model: Any, X: pd.DataFrame) -> pd.DataFrame
```

#### Description

LSTM/GRU hidden-unit activation importance for torch-backed models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.lstm_hidden_state(...)
```
### model_native_linear_coef

Qualified name: `macroforecast.interpretation.core.model_native_linear_coef`

#### Signature

```python
macroforecast.interpretation.model_native_linear_coef(model: Any, *, sort: bool = True) -> pd.DataFrame
```

#### Description

Alias for legacy L7 naming: native linear coefficients.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.model_native_linear_coef(...)
```
### model_native_tree_importance

Qualified name: `macroforecast.interpretation.core.model_native_tree_importance`

#### Signature

```python
macroforecast.interpretation.model_native_tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame
```

#### Description

Alias for legacy L7 naming: native tree feature importance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.model_native_tree_importance(...)
```
### model_accordance_score

Qualified name: `macroforecast.interpretation.core.model_accordance_score`

#### Signature

```python
macroforecast.interpretation.model_accordance_score(is_vi: pd.DataFrame | pd.Series | Mapping[str, float], oos_pbsv: pd.DataFrame | pd.Series | Mapping[str, float], *, loss_type: str = "lower_is_better", mas_type: str = "importance_weighted", hypothesis_test: bool = True, h0_alpha: float = 0.5, n_samples: int = 1000000, vi_value_col: str | None = None, pbsv_value_col: str | None = None, feature_col: str = "feature", random_state: int | None = None) -> pd.DataFrame
```

#### Description

Compute the anatomy backend Model Accordance Score.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `is_vi` | positional or keyword | `pd.DataFrame \| pd.Series \| Mapping[str, float]` | `required` |
| `oos_pbsv` | positional or keyword | `pd.DataFrame \| pd.Series \| Mapping[str, float]` | `required` |
| `loss_type` | keyword only | `str` | `"lower_is_better"` |
| `mas_type` | keyword only | `str` | `"importance_weighted"` |
| `hypothesis_test` | keyword only | `bool` | `True` |
| `h0_alpha` | keyword only | `float` | `0.5` |
| `n_samples` | keyword only | `int` | `1000000` |
| `vi_value_col` | keyword only | `str \| None` | `None` |
| `pbsv_value_col` | keyword only | `str \| None` | `None` |
| `feature_col` | keyword only | `str` | `"feature"` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.model_accordance_score(...)
```
### mrf_gtvp

Qualified name: `macroforecast.interpretation.core.mrf_gtvp`

#### Signature

```python
macroforecast.interpretation.mrf_gtvp(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame
```

#### Description

Return Macroeconomic Random Forest GTVP coefficient paths.

The vendored MacroRandomForest backend emits ``betas`` after prediction.
This callable exposes those paths directly instead of reducing them to a
static forest importance score.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.mrf_gtvp(...)
```
### observation_contributions

Qualified name: `macroforecast.interpretation.dual.observation_contributions`

#### Signature

```python
macroforecast.interpretation.observation_contributions(weights: pd.DataFrame, y_train: pd.Series | Sequence[float], *, center: bool = False, include_base: bool = False) -> pd.DataFrame
```

#### Description

Convert observation weights into observation-level forecast contributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.observation_contributions(...)
```
### observation_weights

Qualified name: `macroforecast.interpretation.core.observation_weights`

#### Signature

```python
macroforecast.interpretation.observation_weights(model: Any | None, X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, method: str = "auto", lambda_: float = 1e-08, kernel: str = "linear", sigma: float = 1.0, add_intercept: bool = False, ridge_penalty_scale: str = "n_train", normalize: bool = False) -> pd.DataFrame
```

#### Description

Compute DualML-style historical-observation weights for forecasts.

This is the episode-based interpretation from Goulet Coulombe, Goebel, and
Klieber (2024). It explains a forecast through training observations rather
than through predictor variables. The implemented paper-aligned routes are:
ridge/OLS, kernel ridge, and sklearn-style random forests.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `str` | `"auto"` |
| `lambda_` | keyword only | `float` | `1e-08` |
| `kernel` | keyword only | `str` | `"linear"` |
| `sigma` | keyword only | `float` | `1.0` |
| `add_intercept` | keyword only | `bool` | `False` |
| `ridge_penalty_scale` | keyword only | `str` | `"n_train"` |
| `normalize` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.observation_weights(...)
```
### ols_attention_embedding

Qualified name: `macroforecast.interpretation.core.ols_attention_embedding`

#### Signature

```python
macroforecast.interpretation.ols_attention_embedding(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, add_intercept: bool = True, ridge: float = 0.0, tol: float = 1e-12) -> pd.DataFrame
```

#### Description

Return whitened train/test embeddings behind OLS-as-attention.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |
| `ridge` | keyword only | `float` | `0.0` |
| `tol` | keyword only | `float` | `1e-12` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ols_attention_embedding(...)
```
### ols_attention_equivalence

Qualified name: `macroforecast.interpretation.core.ols_attention_equivalence`

#### Signature

```python
macroforecast.interpretation.ols_attention_equivalence(X_train: pd.DataFrame, y_train: pd.Series | np.ndarray, X_test: pd.DataFrame | None = None, *, reference_predictions: pd.Series | Sequence[float] | np.ndarray | None = None, add_intercept: bool = True, ridge: float = 0.0) -> pd.DataFrame
```

#### Description

Audit that closed-form predictions equal attention-weight predictions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `reference_predictions` | keyword only | `pd.Series \| Sequence[float] \| np.ndarray \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |
| `ridge` | keyword only | `float` | `0.0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ols_attention_equivalence(...)
```
### ols_attention_weights

Qualified name: `macroforecast.interpretation.core.ols_attention_weights`

#### Signature

```python
macroforecast.interpretation.ols_attention_weights(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, add_intercept: bool = True) -> pd.DataFrame
```

#### Description

Exact OLS-as-attention weights from Goulet Coulombe (2026).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ols_attention_weights(...)
```
### oshapley_vi

Qualified name: `macroforecast.interpretation.core.oshapley_vi`

#### Signature

```python
macroforecast.interpretation.oshapley_vi(anatomy: Any, *, model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, explanation_subset: pd.Index | Sequence[Any] | None = None, exclude_base: bool = True) -> pd.DataFrame
```

#### Description

Compute anatomy backend oShapley-VI from raw forecast explanations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `anatomy` | positional or keyword | `Any` | `required` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `explanation_subset` | keyword only | `pd.Index \| Sequence[Any] \| None` | `None` |
| `exclude_base` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.oshapley_vi(...)
```
### oshapley_from_forecast_result

Qualified name: `macroforecast.interpretation.anatomy.oshapley_from_forecast_result`

#### Signature

```python
macroforecast.interpretation.oshapley_from_forecast_result(result: Any, X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], *, sidecar_name: str = "oshapley", **kwargs: Any) -> Any
```

#### Description

Build an oShapley-VI/PBSV sidecar for a completed forecast result.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `sidecar_name` | keyword only | `str` | `"oshapley"` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.oshapley_from_forecast_result(...)
```
### oshapley_output

Qualified name: `macroforecast.interpretation.anatomy.oshapley_output`

#### Signature

```python
macroforecast.interpretation.oshapley_output(value: Any, **kwargs: Any) -> Any
```

#### Description

Alias for :func:`forecast_shapley_output`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.oshapley_output(...)
```
### oshapley_pipeline

Qualified name: `macroforecast.interpretation.anatomy.oshapley_pipeline`

#### Signature

```python
macroforecast.interpretation.oshapley_pipeline(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], **kwargs: Any) -> ForecastShapleyResult
```

#### Description

Run the oShapley-VI/PBSV forecast-accuracy interpretation pipeline.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`ForecastShapleyResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.oshapley_pipeline(...)
```
### oshapley_provider

Qualified name: `macroforecast.interpretation.anatomy.oshapley_provider`

#### Signature

```python
macroforecast.interpretation.oshapley_provider(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], **kwargs: Any) -> Any
```

#### Description

Build the provider used by oShapley-VI and PBSV precompute.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.oshapley_provider(...)
```
### orthogonalised_irf

Qualified name: `macroforecast.interpretation.core.orthogonalised_irf`

#### Signature

```python
macroforecast.interpretation.orthogonalised_irf(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame
```

#### Description

Cholesky orthogonalised impulse response importance for VAR models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `n_periods` | keyword only | `int` | `12` |
| `target` | keyword only | `str \| int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.orthogonalised_irf(...)
```
### outcome_contributions

Qualified name: `macroforecast.interpretation.core.outcome_contributions`

#### Signature

```python
macroforecast.interpretation.outcome_contributions(weights: pd.DataFrame, y_train: pd.Series | np.ndarray, *, center: bool = False, include_base: bool = False) -> pd.DataFrame
```

#### Description

Convert observation weights into historical-outcome contributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.outcome_contributions(...)
```
### partial_dependence

Qualified name: `macroforecast.interpretation.core.partial_dependence`

#### Signature

```python
macroforecast.interpretation.partial_dependence(model: Any, X: pd.DataFrame, *, features: Iterable[str] | str, grid_size: int = 20) -> pd.DataFrame
```

#### Description

Compute one-way manual partial-dependence curves.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `features` | keyword only | `Iterable[str] \| str` | `required` |
| `grid_size` | keyword only | `int` | `20` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.partial_dependence(...)
```
### pbsv

Qualified name: `macroforecast.interpretation.core.pbsv`

#### Signature

```python
macroforecast.interpretation.pbsv(anatomy: Any, *, model_groups: Mapping[str, Sequence[str] | Mapping[str, float]] | None = None, loss: str = "rmse", transformer: Callable[..., Any] | Any | None = None, explanation_subset: pd.Index | Sequence[Any] | None = None, output: str = "long") -> pd.DataFrame
```

#### Description

Compute backend PBSV_p loss decomposition through ``anatomy``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `anatomy` | positional or keyword | `Any` | `required` |
| `model_groups` | keyword only | `Mapping[str, Sequence[str] \| Mapping[str, float]] \| None` | `None` |
| `loss` | keyword only | `str` | `"rmse"` |
| `transformer` | keyword only | `Callable[..., Any] \| Any \| None` | `None` |
| `explanation_subset` | keyword only | `pd.Index \| Sequence[Any] \| None` | `None` |
| `output` | keyword only | `str` | `"long"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.pbsv(...)
```
### permutation_importance

Qualified name: `macroforecast.interpretation.core.permutation_importance`

#### Signature

```python
macroforecast.interpretation.permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse", n_repeats: int = 5, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Compute simple model-agnostic permutation importance.

Importance is the degradation in the loss metric after permuting one
feature. For score metrics where higher is better, pass a callable that
already returns a loss-like value if positive degradation is desired.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `metric` | keyword only | `Callable[[np.ndarray, np.ndarray], float] \| str` | `"mse"` |
| `n_repeats` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.permutation_importance(...)
```
### permutation_importance_strobl

Qualified name: `macroforecast.interpretation.core.permutation_importance_strobl`

#### Signature

```python
macroforecast.interpretation.permutation_importance_strobl(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse", n_repeats: int = 5, n_bins: int = 5, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Conditional permutation importance following the Strobl idea.

Each feature is permuted within bins of its most correlated companion
feature. This keeps the permutation closer to the observed conditional
distribution than a marginal shuffle, which is the relevant distinction
when macro predictors are strongly collinear.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `metric` | keyword only | `Callable[[np.ndarray, np.ndarray], float] \| str` | `"mse"` |
| `n_repeats` | keyword only | `int` | `5` |
| `n_bins` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.permutation_importance_strobl(...)
```
### performance_based_shapley_value

Qualified name: `macroforecast.interpretation.core.performance_based_shapley_value`

#### Signature

```python
macroforecast.interpretation.performance_based_shapley_value(*args: Any, **kwargs: Any) -> pd.DataFrame
```

#### Description

Alias for :func:`pbsv` when using the anatomy backend.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.performance_based_shapley_value(...)
```
### performance_shapley_value

Qualified name: `macroforecast.interpretation.core.performance_shapley_value`

#### Signature

```python
macroforecast.interpretation.performance_shapley_value(contributions: pd.DataFrame, y: pd.Series | Sequence[float], *, loss: str = "squared_error", row_col: str | None = None, feature_col: str = "feature", contribution_col: str | None = None, base_col: str = "base_value", base_value: float = 0.0, n_permutations: int | None = None, max_exact_features: int = 8, random_state: int | None = 0, return_local: bool = False) -> pd.DataFrame
```

#### Description

Compute PBSV-style loss attribution from additive forecast contributions.

Negative feature contributions reduce the selected point loss; positive
contributions increase it. Full Borup et al. rolling/expanding refit anatomy
should use ``anatomy_explain`` with a precomputed ``anatomy`` object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `contributions` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `row_col` | keyword only | `str \| None` | `None` |
| `feature_col` | keyword only | `str` | `"feature"` |
| `contribution_col` | keyword only | `str \| None` | `None` |
| `base_col` | keyword only | `str` | `"base_value"` |
| `base_value` | keyword only | `float` | `0.0` |
| `n_permutations` | keyword only | `int \| None` | `None` |
| `max_exact_features` | keyword only | `int` | `8` |
| `random_state` | keyword only | `int \| None` | `0` |
| `return_local` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.performance_shapley_value(...)
```
### precompute_anatomy

Qualified name: `macroforecast.interpretation.anatomy.precompute_anatomy`

#### Signature

```python
macroforecast.interpretation.precompute_anatomy(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], *, window: WindowSpec | str | None = None, params: Mapping[str, Any] | None = None, target_name: str | None = None, train_source: str = "fit", n_iterations: int = 32, n_jobs: int = 1, background_data_subsample: float = 1.0, save_path: str | Path | None = None) -> Any
```

#### Description

Build, precompute, and optionally save an ``anatomy.Anatomy`` object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target_name` | keyword only | `str \| None` | `None` |
| `train_source` | keyword only | `str` | `"fit"` |
| `n_iterations` | keyword only | `int` | `32` |
| `n_jobs` | keyword only | `int` | `1` |
| `background_data_subsample` | keyword only | `float` | `1.0` |
| `save_path` | keyword only | `str \| Path \| None` | `None` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.precompute_anatomy(...)
```
### precompute_oshapley

Qualified name: `macroforecast.interpretation.anatomy.precompute_oshapley`

#### Signature

```python
macroforecast.interpretation.precompute_oshapley(X: Any, y: Any, models: str | Callable[..., Any] | ModelSpec | Sequence[str | Callable[..., Any] | ModelSpec] | Mapping[str, str | Callable[..., Any] | ModelSpec], **kwargs: Any) -> Any
```

#### Description

Precompute the backend object used by oShapley-VI and PBSV.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `models` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec \| Sequence[str \| Callable[..., Any] \| ModelSpec] \| Mapping[str, str \| Callable[..., Any] \| ModelSpec]` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.precompute_oshapley(...)
```
### ridge_attention_weights

Qualified name: `macroforecast.interpretation.core.ridge_attention_weights`

#### Signature

```python
macroforecast.interpretation.ridge_attention_weights(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, alpha: float = 1.0, add_intercept: bool = True) -> pd.DataFrame
```

#### Description

Ridge-stabilized OLS attention weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `alpha` | keyword only | `float` | `1.0` |
| `add_intercept` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.ridge_attention_weights(...)
```
### rolling_recompute

Qualified name: `macroforecast.interpretation.core.rolling_recompute`

#### Signature

```python
macroforecast.interpretation.rolling_recompute(model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, *, window: int | None = None, step: int | None = None, method: str = "permutation_importance", n_repeats: int = 1, random_state: int | None = None) -> pd.DataFrame
```

#### Description

Recompute feature importance on rolling evaluation windows.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.Series \| np.ndarray` | `required` |
| `window` | keyword only | `int \| None` | `None` |
| `step` | keyword only | `int \| None` | `None` |
| `method` | keyword only | `str` | `"permutation_importance"` |
| `n_repeats` | keyword only | `int` | `1` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.rolling_recompute(...)
```
### saliency_map

Qualified name: `macroforecast.interpretation.core.saliency_map`

#### Signature

```python
macroforecast.interpretation.saliency_map(model: Any, X: pd.DataFrame) -> pd.DataFrame
```

#### Description

Vanilla input-gradient attribution for torch-backed models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.saliency_map(...)
```
### shapley_variable_importance

Qualified name: `macroforecast.interpretation.core.shapley_variable_importance`

#### Signature

```python
macroforecast.interpretation.shapley_variable_importance(contributions: pd.DataFrame | pd.Series, *, contribution_col: str | None = None, feature_col: str = "feature", group_col: str | None = None, exclude_base: bool = True, source: str = "contribution_table") -> pd.DataFrame
```

#### Description

Aggregate local Shapley contributions into variable importance.

This matches the anatomy README's oShapley-VI summary rule: average the
absolute raw-forecast Shapley contribution by predictor. The same table
adapter can standardize user-supplied in-sample Shapley VI before MAS.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `contributions` | positional or keyword | `pd.DataFrame \| pd.Series` | `required` |
| `contribution_col` | keyword only | `str \| None` | `None` |
| `feature_col` | keyword only | `str` | `"feature"` |
| `group_col` | keyword only | `str \| None` | `None` |
| `exclude_base` | keyword only | `bool` | `True` |
| `source` | keyword only | `str` | `"contribution_table"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shapley_variable_importance(...)
```
### shap_deep

Qualified name: `macroforecast.interpretation.core.shap_deep`

#### Signature

```python
macroforecast.interpretation.shap_deep(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame
```

#### Description

Deep-model SHAP-style global importance.

This callable uses the generic SHAP explainer path because deep backends
vary by installed torch/shap version. Gradient-specific methods are exposed
separately and require ``captum``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_deep(...)
```
### shap_importance

Qualified name: `macroforecast.interpretation.core.shap_importance`

#### Signature

```python
macroforecast.interpretation.shap_importance(model: Any, X: pd.DataFrame, *, background: pd.DataFrame | None = None, explainer: str = "auto", check_additivity: bool = True, **kwargs: Any) -> pd.DataFrame
```

#### Description

Summarize SHAP values as global mean absolute feature importance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `background` | keyword only | `pd.DataFrame \| None` | `None` |
| `explainer` | keyword only | `str` | `"auto"` |
| `check_additivity` | keyword only | `bool` | `True` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_importance(...)
```
### shap_kernel

Qualified name: `macroforecast.interpretation.core.shap_kernel`

#### Signature

```python
macroforecast.interpretation.shap_kernel(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame
```

#### Description

Kernel/permutation SHAP-style global importance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_kernel(...)
```
### shap_linear

Qualified name: `macroforecast.interpretation.core.shap_linear`

#### Signature

```python
macroforecast.interpretation.shap_linear(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame
```

#### Description

Linear SHAP-style global importance using ``shap.Explainer``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_linear(...)
```
### shap_tree

Qualified name: `macroforecast.interpretation.core.shap_tree`

#### Signature

```python
macroforecast.interpretation.shap_tree(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame
```

#### Description

Tree SHAP global importance using the optional ``shap`` backend.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_tree(...)
```
### shap_values

Qualified name: `macroforecast.interpretation.core.shap_values`

#### Signature

```python
macroforecast.interpretation.shap_values(model: Any, X: pd.DataFrame, *, background: pd.DataFrame | None = None, explainer: str = "auto", check_additivity: bool = True, **kwargs: Any) -> pd.DataFrame
```

#### Description

Return SHAP values in a long pandas table.

SHAP is an optional backend. Install ``macroforecast[interpretation]`` to
use this helper.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `background` | keyword only | `pd.DataFrame \| None` | `None` |
| `explainer` | keyword only | `str` | `"auto"` |
| `check_additivity` | keyword only | `bool` | `True` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.shap_values(...)
```
### tree_importance

Qualified name: `macroforecast.interpretation.core.tree_importance`

#### Signature

```python
macroforecast.interpretation.tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame
```

#### Description

Return native tree importance for estimators exposing feature_importances_.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `sort` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.tree_importance(...)
```
### transformation_attribution

Qualified name: `macroforecast.interpretation.core.transformation_attribution`

#### Signature

```python
macroforecast.interpretation.transformation_attribution(evaluation: pd.DataFrame, *, pipeline_column: str | None = None, metric: str | None = None, method: str = "shapley_over_pipelines", target_columns: Sequence[str] = ('target', 'horizon'), lower_is_better: bool = True, baseline: str | float = "worst") -> pd.DataFrame
```

#### Description

Attribute forecast score differences to preprocessing/feature pipelines.

This helper works on mutually exclusive pipeline/model rows. It is not a
component-level causal decomposition unless the input table was designed as
a component-removal experiment. For loss metrics, the default converts loss
to improvement relative to the worst observed pipeline in each group.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `evaluation` | positional or keyword | `pd.DataFrame` | `required` |
| `pipeline_column` | keyword only | `str \| None` | `None` |
| `metric` | keyword only | `str \| None` | `None` |
| `method` | keyword only | `str` | `"shapley_over_pipelines"` |
| `target_columns` | keyword only | `Sequence[str]` | `("target", "horizon")` |
| `lower_is_better` | keyword only | `bool` | `True` |
| `baseline` | keyword only | `str \| float` | `"worst"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.transformation_attribution(...)
```
### top_episodes

Qualified name: `macroforecast.interpretation.core.top_episodes`

#### Signature

```python
macroforecast.interpretation.top_episodes(weights: pd.DataFrame, *, y_train: pd.Series | np.ndarray | None = None, n: int = 10, sort_by: str = "abs_weight") -> pd.DataFrame
```

#### Description

Return the largest historical-episode weights for each forecast row.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | keyword only | `pd.Series \| np.ndarray \| None` | `None` |
| `n` | keyword only | `int` | `10` |
| `sort_by` | keyword only | `str` | `"abs_weight"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.top_episodes(...)
```
### top_observations

Qualified name: `macroforecast.interpretation.dual.top_observations`

#### Signature

```python
macroforecast.interpretation.top_observations(weights: pd.DataFrame, *, y_train: pd.Series | Sequence[float] | None = None, n: int = 10, sort_by: str = "abs_weight") -> pd.DataFrame
```

#### Description

Return the largest historical observations per forecast row.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |
| `n` | keyword only | `int` | `10` |
| `sort_by` | keyword only | `str` | `"abs_weight"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.top_observations(...)
```
### window_to_anatomy_subsets

Qualified name: `macroforecast.interpretation.anatomy.window_to_anatomy_subsets`

#### Signature

```python
macroforecast.interpretation.window_to_anatomy_subsets(window: WindowSpec | str | None, index: pd.Index | Sequence[Any], *, train_source: str = "fit") -> Any
```

#### Description

Convert a macroforecast window into exact ``anatomy`` train/test subsets.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `window` | positional or keyword | `WindowSpec \| str \| None` | `required` |
| `index` | positional or keyword | `pd.Index \| Sequence[Any]` | `required` |
| `train_source` | keyword only | `str` | `"fit"` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.window_to_anatomy_subsets(...)
```
### window_to_oshapley_subsets

Qualified name: `macroforecast.interpretation.anatomy.window_to_oshapley_subsets`

#### Signature

```python
macroforecast.interpretation.window_to_oshapley_subsets(window: WindowSpec | str | None, index: pd.Index | Sequence[Any], *, train_source: str = "fit") -> Any
```

#### Description

Convert a macroforecast window for oShapley/PBSV backend precompute.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `window` | positional or keyword | `WindowSpec \| str \| None` | `required` |
| `index` | positional or keyword | `pd.Index \| Sequence[Any]` | `required` |
| `train_source` | keyword only | `str` | `"fit"` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.window_to_oshapley_subsets(...)
```
