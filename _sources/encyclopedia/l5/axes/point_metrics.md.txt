# `point_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``point_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `['mse', 'mae']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 7 option(s)
- Future: 0 option(s)

## Options

### `mae`  --  operational

Mean absolute error -- ``(1/N) Σ |y_t - ŷ_t|``.

See [mae function page](../point_metrics/mae.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mae``.

### `mape`  --  operational

Mean absolute percentage error -- ``(100/N) Σ |y_t - ŷ_t| / |y_t|``.

See [mape function page](../point_metrics/mape.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mape``.

### `medae`  --  operational

Median absolute error -- ``median |y_t - ŷ_t|``.

See [medae function page](../point_metrics/medae.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.medae``.

### `mse`  --  operational

Mean squared error -- ``(1/N) Σ (y_t - ŷ_t)²``.

See [mse function page](../point_metrics/mse.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mse``.

### `rmse`  --  operational

Root mean squared error -- ``√MSE``.

See [rmse function page](../point_metrics/rmse.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.rmse``.

### `theil_u1`  --  operational

Theil's U1 inequality coefficient -- bounded in ``[0, 1]``.

See [theil_u1 function page](../point_metrics/theil_u1.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.theil_u1``.

### `theil_u2`  --  operational

Theil's U2 inequality coefficient -- ratio of forecast MSE to no-change MSE.

See [theil_u2 function page](../point_metrics/theil_u2.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.theil_u2``.
