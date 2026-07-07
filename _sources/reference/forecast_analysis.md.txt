# macroforecast.forecast_analysis

[Back to reference](index.md)

Forecast diagnostics for fitted values, residuals, tuning traces, and forecast paths.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `ForecastDiagnosticReport` | class | Container returned by :func:`diagnose_forecasts`. |
| `coefficient_trace` | function | Read saved fit sidecars and return coefficient paths over origins. |
| `custom_forecast_diagnostic` | function | Run a user-supplied forecast diagnostic and attach macroforecast metadata. |
| `diagnose_forecasts` | function | Run the standard forecast diagnostics on a ForecastResult or table. |
| `dfm_factor_stability` | function | Summarize filtered DFM factors saved in fit diagnostics. |
| `dfm_idiosyncratic_acf` | function | Return ACF diagnostics for DFM idiosyncratic residual series. |
| `ensemble_member_contribution` | function | Return member-level weighted forecast contributions for combinations. |
| `ensemble_weight_concentration` | function | Return concentration metrics for identifiable forecast-combination weights. |
| `ensemble_weights_over_time` | function | Reconstruct combination weights when the method has identifiable weights. |
| `first_vs_last_forecast` | function | Compare the first and last forecast row inside each group. |
| `fitted_vs_actual` | function | Return forecast rows with residual and error columns. |
| `forecast_overview` | function | Return compact shape, model, horizon, and metadata coverage counts. |
| `forecast_scale_view` | function | Return transformed and back-transformed forecast rows when possible. |
| `hyperparameter_path` | function | Return long-form selected hyperparameters by origin. |
| `parameter_stability` | function | Summarize coefficient drift and sign changes over forecast origins. |
| `residual_autocorrelation` | function | Return residual autocorrelation by model/horizon group. |
| `residual_qq` | function | Return residual QQ table against a fitted normal reference. |
| `residual_report` | function | Return grouped out-of-sample residual diagnostics. |
| `rolling_loss` | function | Return rolling out-of-sample loss by model and horizon. |
| `rolling_training_loss` | function | Return a rolling trace of saved in-sample training metrics. |
| `select_forecast_origins` | function | Return a forecast table filtered to a requested origin view. |
| `stage_update_trace` | function | Return stage update records saved by the forecasting runner. |
| `training_loss_trace` | function | Read saved model sidecars and return in-sample fit metrics by origin. |
| `tuning_objective_trace` | function | Return best validation objective over origins for tuned models. |
| `tuning_score_distribution` | function | Summarize the distribution of selected validation scores over origins. |
| `tuning_trace` | function | Return one row per forecast row carrying parameter-selection metadata. |

## Callable And Class Reference

### ForecastDiagnosticReport

Qualified name: `macroforecast.forecast_analysis.core.ForecastDiagnosticReport`

#### Signature

```python
macroforecast.forecast_analysis.ForecastDiagnosticReport(overview: dict[str, Any], fitted: pd.DataFrame | None = None, residuals: pd.DataFrame | None = None, residual_acf: pd.DataFrame | None = None, residual_qq: pd.DataFrame | None = None, rolling_loss: pd.DataFrame | None = None, forecast_scale: pd.DataFrame | None = None, coefficients: pd.DataFrame | None = None, parameter_stability: pd.DataFrame | None = None, training_loss: pd.DataFrame | None = None, rolling_training_loss: pd.DataFrame | None = None, first_vs_last: pd.DataFrame | None = None, tuning: pd.DataFrame | None = None, tuning_objective: pd.DataFrame | None = None, hyperparameters: pd.DataFrame | None = None, tuning_scores: pd.DataFrame | None = None, ensemble_weights: pd.DataFrame | None = None, ensemble_concentration: pd.DataFrame | None = None, member_contribution: pd.DataFrame | None = None, dfm_idiosyncratic_acf: pd.DataFrame | None = None, dfm_factor_stability: pd.DataFrame | None = None, stage_updates: pd.DataFrame | None = None, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Container returned by :func:`diagnose_forecasts`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `overview` | positional or keyword | `dict[str, Any]` | `required` |
| `fitted` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `residuals` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `residual_acf` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `residual_qq` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `rolling_loss` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `forecast_scale` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `coefficients` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `parameter_stability` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `training_loss` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `rolling_training_loss` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `first_vs_last` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `tuning` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `tuning_objective` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `hyperparameters` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `tuning_scores` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `ensemble_weights` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `ensemble_concentration` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `member_contribution` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `dfm_idiosyncratic_acf` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `dfm_factor_stability` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `stage_updates` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.forecast_analysis.ForecastDiagnosticReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `overview` | `dict[str, Any]` | `required` |
| `fitted` | `pd.DataFrame \| None` | `None` |
| `residuals` | `pd.DataFrame \| None` | `None` |
| `residual_acf` | `pd.DataFrame \| None` | `None` |
| `residual_qq` | `pd.DataFrame \| None` | `None` |
| `rolling_loss` | `pd.DataFrame \| None` | `None` |
| `forecast_scale` | `pd.DataFrame \| None` | `None` |
| `coefficients` | `pd.DataFrame \| None` | `None` |
| `parameter_stability` | `pd.DataFrame \| None` | `None` |
| `training_loss` | `pd.DataFrame \| None` | `None` |
| `rolling_training_loss` | `pd.DataFrame \| None` | `None` |
| `first_vs_last` | `pd.DataFrame \| None` | `None` |
| `tuning` | `pd.DataFrame \| None` | `None` |
| `tuning_objective` | `pd.DataFrame \| None` | `None` |
| `hyperparameters` | `pd.DataFrame \| None` | `None` |
| `tuning_scores` | `pd.DataFrame \| None` | `None` |
| `ensemble_weights` | `pd.DataFrame \| None` | `None` |
| `ensemble_concentration` | `pd.DataFrame \| None` | `None` |
| `member_contribution` | `pd.DataFrame \| None` | `None` |
| `dfm_idiosyncratic_acf` | `pd.DataFrame \| None` | `None` |
| `dfm_factor_stability` | `pd.DataFrame \| None` | `None` |
| `stage_updates` | `pd.DataFrame \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### coefficient_trace

Qualified name: `macroforecast.forecast_analysis.core.coefficient_trace`

#### Signature

```python
macroforecast.forecast_analysis.coefficient_trace(forecasts: Any, *, include_intercept: bool = True, load_pickle: bool = False, models: Iterable[str] | None = None) -> pd.DataFrame
```

#### Description

Read saved fit sidecars and return coefficient paths over origins.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `include_intercept` | keyword only | `bool` | `True` |
| `load_pickle` | keyword only | `bool` | `False` |
| `models` | keyword only | `Iterable[str] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.coefficient_trace(...)
```
### custom_forecast_diagnostic

Qualified name: `macroforecast.forecast_analysis.core.custom_forecast_diagnostic`

#### Signature

```python
macroforecast.forecast_analysis.custom_forecast_diagnostic(forecasts: Any, func: Callable[..., Any], *, name: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Run a user-supplied forecast diagnostic and attach macroforecast metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `name` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.custom_forecast_diagnostic(...)
```
### diagnose_forecasts

Qualified name: `macroforecast.forecast_analysis.core.diagnose_forecasts`

#### Signature

```python
macroforecast.forecast_analysis.diagnose_forecasts(forecasts: Any, *, include_fitted: bool = True, include_residuals: bool = True, include_residual_acf: bool = False, include_residual_qq: bool = False, include_rolling_loss: bool = True, rolling_window: int = 12, rolling_metric: LossMetric = "rmse", include_forecast_scale: bool = False, levels: Any | None = None, scale_view: ScaleView = "both_overlay", back_transform: Callable[..., Any] | None = None, include_training_loss: bool = False, include_rolling_training_loss: bool = False, training_loss_metric: str = "rmse", include_first_vs_last: bool = False, include_dfm_idiosyncratic_acf: bool = False, include_dfm_factor_stability: bool = False, include_coefficients: bool = True, include_parameter_stability: bool = True, include_tuning: bool = True, include_tuning_objective: bool = True, include_hyperparameters: bool = True, include_tuning_scores: bool = True, include_ensemble_weights: bool = True, include_ensemble_concentration: bool = True, include_member_contribution: bool = False, include_stage_updates: bool = True, include_combined: bool = True) -> ForecastDiagnosticReport
```

#### Description

Run the standard forecast diagnostics on a ForecastResult or table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `include_fitted` | keyword only | `bool` | `True` |
| `include_residuals` | keyword only | `bool` | `True` |
| `include_residual_acf` | keyword only | `bool` | `False` |
| `include_residual_qq` | keyword only | `bool` | `False` |
| `include_rolling_loss` | keyword only | `bool` | `True` |
| `rolling_window` | keyword only | `int` | `12` |
| `rolling_metric` | keyword only | `LossMetric` | `"rmse"` |
| `include_forecast_scale` | keyword only | `bool` | `False` |
| `levels` | keyword only | `Any \| None` | `None` |
| `scale_view` | keyword only | `ScaleView` | `"both_overlay"` |
| `back_transform` | keyword only | `Callable[..., Any] \| None` | `None` |
| `include_training_loss` | keyword only | `bool` | `False` |
| `include_rolling_training_loss` | keyword only | `bool` | `False` |
| `training_loss_metric` | keyword only | `str` | `"rmse"` |
| `include_first_vs_last` | keyword only | `bool` | `False` |
| `include_dfm_idiosyncratic_acf` | keyword only | `bool` | `False` |
| `include_dfm_factor_stability` | keyword only | `bool` | `False` |
| `include_coefficients` | keyword only | `bool` | `True` |
| `include_parameter_stability` | keyword only | `bool` | `True` |
| `include_tuning` | keyword only | `bool` | `True` |
| `include_tuning_objective` | keyword only | `bool` | `True` |
| `include_hyperparameters` | keyword only | `bool` | `True` |
| `include_tuning_scores` | keyword only | `bool` | `True` |
| `include_ensemble_weights` | keyword only | `bool` | `True` |
| `include_ensemble_concentration` | keyword only | `bool` | `True` |
| `include_member_contribution` | keyword only | `bool` | `False` |
| `include_stage_updates` | keyword only | `bool` | `True` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`ForecastDiagnosticReport`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.diagnose_forecasts(...)
```
### dfm_factor_stability

Qualified name: `macroforecast.forecast_analysis.core.dfm_factor_stability`

#### Signature

```python
macroforecast.forecast_analysis.dfm_factor_stability(source: Any, *, load_pickle: bool = False) -> pd.DataFrame
```

#### Description

Summarize filtered DFM factors saved in fit diagnostics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `Any` | `required` |
| `load_pickle` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.dfm_factor_stability(...)
```
### dfm_idiosyncratic_acf

Qualified name: `macroforecast.forecast_analysis.core.dfm_idiosyncratic_acf`

#### Signature

```python
macroforecast.forecast_analysis.dfm_idiosyncratic_acf(source: Any, *, max_lag: int = 12, load_pickle: bool = False) -> pd.DataFrame
```

#### Description

Return ACF diagnostics for DFM idiosyncratic residual series.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `Any` | `required` |
| `max_lag` | keyword only | `int` | `12` |
| `load_pickle` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.dfm_idiosyncratic_acf(...)
```
### ensemble_member_contribution

Qualified name: `macroforecast.forecast_analysis.core.ensemble_member_contribution`

#### Signature

```python
macroforecast.forecast_analysis.ensemble_member_contribution(forecasts: Any) -> pd.DataFrame
```

#### Description

Return member-level weighted forecast contributions for combinations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.ensemble_member_contribution(...)
```
### ensemble_weight_concentration

Qualified name: `macroforecast.forecast_analysis.core.ensemble_weight_concentration`

#### Signature

```python
macroforecast.forecast_analysis.ensemble_weight_concentration(forecasts: Any) -> pd.DataFrame
```

#### Description

Return concentration metrics for identifiable forecast-combination weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.ensemble_weight_concentration(...)
```
### ensemble_weights_over_time

Qualified name: `macroforecast.forecast_analysis.core.ensemble_weights_over_time`

#### Signature

```python
macroforecast.forecast_analysis.ensemble_weights_over_time(forecasts: Any, *, unsupported: UnsupportedWeights = "skip") -> pd.DataFrame
```

#### Description

Reconstruct combination weights when the method has identifiable weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `unsupported` | keyword only | `UnsupportedWeights` | `"skip"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.ensemble_weights_over_time(...)
```
### first_vs_last_forecast

Qualified name: `macroforecast.forecast_analysis.core.first_vs_last_forecast`

#### Signature

```python
macroforecast.forecast_analysis.first_vs_last_forecast(forecasts: Any, *, group_by: Sequence[str] = ('model', 'horizon'), include_combined: bool = True) -> pd.DataFrame
```

#### Description

Compare the first and last forecast row inside each group.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.first_vs_last_forecast(...)
```
### fitted_vs_actual

Qualified name: `macroforecast.forecast_analysis.core.fitted_vs_actual`

#### Signature

```python
macroforecast.forecast_analysis.fitted_vs_actual(forecasts: Any, *, include_combined: bool = True, drop_missing_actual: bool = True) -> pd.DataFrame
```

#### Description

Return forecast rows with residual and error columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `include_combined` | keyword only | `bool` | `True` |
| `drop_missing_actual` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.fitted_vs_actual(...)
```
### forecast_overview

Qualified name: `macroforecast.forecast_analysis.core.forecast_overview`

#### Signature

```python
macroforecast.forecast_analysis.forecast_overview(forecasts: Any) -> dict[str, Any]
```

#### Description

Return compact shape, model, horizon, and metadata coverage counts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.forecast_overview(...)
```
### forecast_scale_view

Qualified name: `macroforecast.forecast_analysis.core.forecast_scale_view`

#### Signature

```python
macroforecast.forecast_analysis.forecast_scale_view(forecasts: Any, *, levels: Any | None = None, target: str | None = None, transform: str | None = None, view: ScaleView = "both_overlay", back_transform: Callable[..., Any] | None = None, include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return transformed and back-transformed forecast rows when possible.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `levels` | keyword only | `Any \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `transform` | keyword only | `str \| None` | `None` |
| `view` | keyword only | `ScaleView` | `"both_overlay"` |
| `back_transform` | keyword only | `Callable[..., Any] \| None` | `None` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.forecast_scale_view(...)
```
### hyperparameter_path

Qualified name: `macroforecast.forecast_analysis.core.hyperparameter_path`

#### Signature

```python
macroforecast.forecast_analysis.hyperparameter_path(forecasts: Any) -> pd.DataFrame
```

#### Description

Return long-form selected hyperparameters by origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.hyperparameter_path(...)
```
### parameter_stability

Qualified name: `macroforecast.forecast_analysis.core.parameter_stability`

#### Signature

```python
macroforecast.forecast_analysis.parameter_stability(forecasts: Any, *, include_intercept: bool = True, load_pickle: bool = False, group_by: Sequence[str] = ('model', 'horizon', 'feature'), models: Iterable[str] | None = None) -> pd.DataFrame
```

#### Description

Summarize coefficient drift and sign changes over forecast origins.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `include_intercept` | keyword only | `bool` | `True` |
| `load_pickle` | keyword only | `bool` | `False` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon", "feature")` |
| `models` | keyword only | `Iterable[str] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.parameter_stability(...)
```
### residual_autocorrelation

Qualified name: `macroforecast.forecast_analysis.core.residual_autocorrelation`

#### Signature

```python
macroforecast.forecast_analysis.residual_autocorrelation(forecasts: Any, *, max_lag: int = 12, group_by: Sequence[str] = ('model', 'horizon'), include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return residual autocorrelation by model/horizon group.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `max_lag` | keyword only | `int` | `12` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.residual_autocorrelation(...)
```
### residual_qq

Qualified name: `macroforecast.forecast_analysis.core.residual_qq`

#### Signature

```python
macroforecast.forecast_analysis.residual_qq(forecasts: Any, *, n_quantiles: int = 21, group_by: Sequence[str] = ('model', 'horizon'), include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return residual QQ table against a fitted normal reference.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `n_quantiles` | keyword only | `int` | `21` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.residual_qq(...)
```
### residual_report

Qualified name: `macroforecast.forecast_analysis.core.residual_report`

#### Signature

```python
macroforecast.forecast_analysis.residual_report(forecasts: Any, *, group_by: Sequence[str] = ('model', 'horizon'), include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return grouped out-of-sample residual diagnostics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.residual_report(...)
```
### rolling_loss

Qualified name: `macroforecast.forecast_analysis.core.rolling_loss`

#### Signature

```python
macroforecast.forecast_analysis.rolling_loss(forecasts: Any, *, metric: LossMetric = "rmse", window: int = 12, min_periods: int | None = None, group_by: Sequence[str] = ('model', 'horizon'), include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return rolling out-of-sample loss by model and horizon.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `metric` | keyword only | `LossMetric` | `"rmse"` |
| `window` | keyword only | `int` | `12` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.rolling_loss(...)
```
### rolling_training_loss

Qualified name: `macroforecast.forecast_analysis.core.rolling_training_loss`

#### Signature

```python
macroforecast.forecast_analysis.rolling_training_loss(forecasts_or_trace: Any, *, metric: str = "rmse", window: int = 12, min_periods: int | None = None, group_by: Sequence[str] = ('model', 'horizon'), load_pickle: bool = False) -> pd.DataFrame
```

#### Description

Return a rolling trace of saved in-sample training metrics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts_or_trace` | positional or keyword | `Any` | `required` |
| `metric` | keyword only | `str` | `"rmse"` |
| `window` | keyword only | `int` | `12` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `load_pickle` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.rolling_training_loss(...)
```
### select_forecast_origins

Qualified name: `macroforecast.forecast_analysis.core.select_forecast_origins`

#### Signature

```python
macroforecast.forecast_analysis.select_forecast_origins(forecasts: Any, *, view: OriginView = "all_origins", every_n: int = 12, include_last: bool = True, include_combined: bool = True) -> pd.DataFrame
```

#### Description

Return a forecast table filtered to a requested origin view.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `view` | keyword only | `OriginView` | `"all_origins"` |
| `every_n` | keyword only | `int` | `12` |
| `include_last` | keyword only | `bool` | `True` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.select_forecast_origins(...)
```
### stage_update_trace

Qualified name: `macroforecast.forecast_analysis.core.stage_update_trace`

#### Signature

```python
macroforecast.forecast_analysis.stage_update_trace(forecasts: Any) -> pd.DataFrame
```

#### Description

Return stage update records saved by the forecasting runner.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.stage_update_trace(...)
```
### training_loss_trace

Qualified name: `macroforecast.forecast_analysis.core.training_loss_trace`

#### Signature

```python
macroforecast.forecast_analysis.training_loss_trace(forecasts: Any, *, load_pickle: bool = False) -> pd.DataFrame
```

#### Description

Read saved model sidecars and return in-sample fit metrics by origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `load_pickle` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.training_loss_trace(...)
```
### tuning_objective_trace

Qualified name: `macroforecast.forecast_analysis.core.tuning_objective_trace`

#### Signature

```python
macroforecast.forecast_analysis.tuning_objective_trace(forecasts: Any) -> pd.DataFrame
```

#### Description

Return best validation objective over origins for tuned models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.tuning_objective_trace(...)
```
### tuning_score_distribution

Qualified name: `macroforecast.forecast_analysis.core.tuning_score_distribution`

#### Signature

```python
macroforecast.forecast_analysis.tuning_score_distribution(forecasts: Any, *, group_by: Sequence[str] = ('model', 'horizon', 'method')) -> pd.DataFrame
```

#### Description

Summarize the distribution of selected validation scores over origins.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `group_by` | keyword only | `Sequence[str]` | `("model", "horizon", "method")` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.tuning_score_distribution(...)
```
### tuning_trace

Qualified name: `macroforecast.forecast_analysis.core.tuning_trace`

#### Signature

```python
macroforecast.forecast_analysis.tuning_trace(forecasts: Any) -> pd.DataFrame
```

#### Description

Return one row per forecast row carrying parameter-selection metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.tuning_trace(...)
```
