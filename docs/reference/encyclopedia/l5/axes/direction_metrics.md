# `direction_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``direction_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `[]`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `pesaran_timmermann_metric`  --  operational

Pesaran-Timmermann (1992) directional-accuracy statistic.

See [pesaran_timmermann_metric function page](../direction_metrics/pesaran_timmermann_metric.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pesaran_timmermann_metric``.

### `success_ratio`  --  operational

Hit-rate of correct directional forecasts -- ``(1/N) Σ 1{sign(ŷ_t) = sign(y_t)}``.

See [success_ratio function page](../direction_metrics/success_ratio.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.success_ratio``.
