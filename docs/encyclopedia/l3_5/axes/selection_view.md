# `selection_view`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``selection_view`` on sub-layer ``L3_5_E_selected_features_post_selection`` (layer ``l3_5``).

## Sub-layer

**L3_5_E_selected_features_post_selection**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `multi`  --  operational

Render every selection view.

L3.5.E selection view ``multi``.

This option configures the ``selection_view`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Default rich audit. Activates the ``multi`` branch on L3.5.selection_view; combine with related options on the same sub-layer for a comprehensive diagnostic.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`selected_list`](#selected-list), [`selection_count_per_origin`](#selection-count-per-origin), [`selection_stability`](#selection-stability)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

_(no schema description for `none`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l3_5.py`` are welcome.

### `selected_list`  --  operational

List of selected features per OOS origin.

L3.5.E selection view ``selected_list``.

This option configures the ``selection_view`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Cheapest readout; the raw record of feature-selection decisions.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`selection_count_per_origin`](#selection-count-per-origin), [`selection_stability`](#selection-stability), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `selection_count_per_origin`  --  operational

Count of selected features per OOS origin.

L3.5.E selection view ``selection_count_per_origin``.

This option configures the ``selection_view`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Detecting selection volatility; large variation across origins flags an unstable selection process.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`selected_list`](#selected-list), [`selection_stability`](#selection-stability), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `selection_stability`  --  operational

Jaccard / Kuncheva-style stability across origins.

L3.5.E selection view ``selection_stability``.

This option configures the ``selection_view`` axis on the ``L3_5_E_selected_features_post_selection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_E_selected_features_post_selection/`` alongside the other selected views.

**When to use**

Quantifying selection robustness; high stability is a positive indicator for the feature-selection method.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`selected_list`](#selected-list), [`selection_count_per_origin`](#selection-count-per-origin), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
