# `stability_metric`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``stability_metric`` on sub-layer ``L3_5_E_selected_features_post_selection`` (layer ``l3_5``).

## Sub-layer

**L3_5_E_selected_features_post_selection**

## Axis metadata

- Default: `'jaccard'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `jaccard`  --  operational

Jaccard similarity over selection sets across origins.

L3.5.E stability metric ``jaccard``.

This option configures the ``stability_metric`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Default stability metric; ``|A ∩ B| / |A ∪ B|`` is intuitive and bounded in [0, 1].

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`kuncheva`](#kuncheva)

_Last reviewed 2026-05-05 by macroforecast author._

### `kuncheva`  --  operational

Kuncheva-corrected stability index.

L3.5.E stability metric ``kuncheva``.

This option configures the ``stability_metric`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Larger feature panels where Jaccard is biased toward 0; explicitly corrects for chance agreement.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Kuncheva (2007) 'A stability index for feature selection', AIA proceedings.

**Related options**: [`jaccard`](#jaccard)

_Last reviewed 2026-05-05 by macroforecast author._
