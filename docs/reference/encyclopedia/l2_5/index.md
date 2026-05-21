# Layer L2.5  --  Pre vs post preprocessing

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l2_5`
- Category: `diagnostic`
- Sub-layers: 5
- Axes: 12
- Options across axes: 42

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `L2_5_A_comparison_axis` | Comparison axis | always | [comparison_pair](axes/comparison_pair.md), [comparison_output_form](axes/comparison_output_form.md) |
| `L2_5_B_distribution_shift` | Distribution shift | always | [distribution_metric](axes/distribution_metric.md), [distribution_view](axes/distribution_view.md) |
| `L2_5_C_correlation_shift` | Correlation shift | always | [correlation_shift](axes/correlation_shift.md), [correlation_method](axes/correlation_method.md) |
| `L2_5_D_cleaning_effect_summary` | Cleaning effect summary | always | [cleaning_summary_view](axes/cleaning_summary_view.md), [t_code_application_log](axes/t_code_application_log.md) |
| `L2_5_Z_export` | Diagnostic export | always | [diagnostic_format](axes/diagnostic_format.md), [attach_to_manifest](axes/attach_to_manifest.md), [figure_dpi](axes/figure_dpi.md), [latex_export](axes/latex_export.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/comparison_pair
axes/comparison_output_form
axes/distribution_metric
axes/distribution_view
axes/correlation_shift
axes/correlation_method
axes/cleaning_summary_view
axes/t_code_application_log
axes/diagnostic_format
axes/attach_to_manifest
axes/figure_dpi
axes/latex_export
```
