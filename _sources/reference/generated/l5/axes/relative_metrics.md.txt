# `relative_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``relative_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `['relative_mse', 'r2_oos']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `mse_reduction`  --  operational

``MSE_benchmark - MSE_model`` (absolute MSE reduction) -- positive means the candidate beats the benchmark.

See [mse_reduction function page](../relative_metrics/mse_reduction.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mse_reduction``.

### `r2_oos`  --  operational

Out-of-sample R² (Campbell-Thompson 2008) -- ``1 - SSE_model / SSE_benchmark``.

See [r2_oos function page](../relative_metrics/r2_oos.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.r2_oos``.

### `relative_mae`  --  operational

Forecast MAE divided by the L4 ``is_benchmark`` model's MAE.

See [relative_mae function page](../relative_metrics/relative_mae.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.relative_mae``.

### `relative_mse`  --  operational

Forecast MSE divided by the L4 ``is_benchmark`` model's MSE.

See [relative_mse function page](../relative_metrics/relative_mse.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.relative_mse``.
