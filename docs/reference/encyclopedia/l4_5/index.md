# Layer L4.5  --  Generator diagnostics

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l4_5`
- Category: `diagnostic`
- Sub-layers: 6
- Axes: 13
- Options across axes: 42

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `L4_5_A_in_sample_fit` | In-sample fit | always | [fit_view](axes/fit_view.md), [fit_per_origin](axes/fit_per_origin.md) |
| `L4_5_B_forecast_scale_view` | Forecast scale view | always | [forecast_scale_view](axes/forecast_scale_view.md), [back_transform_method](axes/back_transform_method.md) |
| `L4_5_C_window_stability` | Window stability | always | [window_view](axes/window_view.md), [coef_view_models](axes/coef_view_models.md) |
| `L4_5_D_tuning_history` | Tuning history | always | [tuning_view](axes/tuning_view.md) |
| `L4_5_E_ensemble_diagnostics` | Ensemble diagnostics | always | [ensemble_view](axes/ensemble_view.md), [weights_over_time_method](axes/weights_over_time_method.md) |
| `L4_5_Z_export` | Diagnostic export | always | [diagnostic_format](axes/diagnostic_format.md), [attach_to_manifest](axes/attach_to_manifest.md), [figure_dpi](axes/figure_dpi.md), [latex_export](axes/latex_export.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/fit_view
axes/fit_per_origin
axes/forecast_scale_view
axes/back_transform_method
axes/window_view
axes/coef_view_models
axes/tuning_view
axes/ensemble_view
axes/weights_over_time_method
axes/diagnostic_format
axes/attach_to_manifest
axes/figure_dpi
axes/latex_export
```
