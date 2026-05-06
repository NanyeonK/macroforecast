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

Directional-accuracy metric ``pesaran_timmermann_metric``. Adjusts the success ratio for the joint probability of agreement under independence (so a constant-sign forecast no longer scores high). Asymptotically standard normal under the null of no directional skill; the L6.F test computes the corresponding p-value.

**When to use**

Formal directional-accuracy reporting (paired with the L6 PT test).

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Pesaran & Timmermann (1992) 'A simple nonparametric test of predictive performance', JBES 10(4): 461-465.

**Related options**: [`success_ratio`](#success-ratio)

_Last reviewed 2026-05-05 by macroforecast author._

### `success_ratio`  --  operational

Hit-rate of correct directional forecasts -- ``(1/N) Σ 1{sign(ŷ_t) = sign(y_t)}``.

Directional-accuracy metric ``success_ratio``. Naive directional accuracy, bounded in ``[0, 1]``. Does not adjust for the unconditional direction frequency, so a constant 'always positive' forecast can score 0.7 on a growth target. For statistical significance, pair with ``pesaran_timmermann_metric`` and the L6.F PT test.

**When to use**

Quick directional-accuracy reporting; reporting the raw hit-rate alongside the PT statistic.

**When NOT to use**

Standalone significance testing -- needs PT correction for unconditional direction frequency.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Pesaran & Timmermann (1992) 'A simple nonparametric test of predictive performance', JBES 10(4): 461-465.

**Related options**: [`pesaran_timmermann_metric`](#pesaran-timmermann-metric)

_Last reviewed 2026-05-05 by macroforecast author._
