# `horizon_set`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``horizon_set`` on sub-layer ``l1_f`` (layer ``l1``).

## Sub-layer

**l1_f**

## Axis metadata

- Default: `'derived'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `standard_md`  --  operational

Standard FRED-MD horizons: {1, 3, 6, 9, 12, 18, 24} months.

The canonical multi-horizon set used in the McCracken-Ng / Stock-Watson tradition for monthly forecasting. Models are fit per-horizon (when ``forecast_strategy = direct``) and metrics report per-(model, horizon) rows.

**When to use**

Default for monthly studies. Comparable to published monthly benchmarks.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`standard_qd`](#standard-qd), [`single`](#single), [`custom_list`](#custom-list), [`range_up_to_h`](#range-up-to-h)

_Last reviewed 2026-05-04 by macroforecast author._

### `standard_qd`  --  operational

Standard FRED-QD horizons: {1, 2, 4, 8} quarters.

Quarterly counterpart of ``standard_md``.

Configures the ``horizon_set`` axis on ``l1_f`` (layer ``l1``); the ``standard_qd`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default for quarterly (FRED-QD) studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`standard_md`](#standard-md), [`single`](#single), [`custom_list`](#custom-list)

_Last reviewed 2026-05-04 by macroforecast author._

### `single`  --  operational

A single horizon (defaults to h=1).

Forecasts only one horizon per cell. Sets ``leaf_config.target_horizons = [N]`` to override the default of 1. Faster than multi-horizon studies and clearer metrics tables when the study's question is single-horizon.

**When to use**

One-shot studies (h=1 nowcasting, h=12 long-horizon ablation).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`standard_md`](#standard-md), [`standard_qd`](#standard-qd), [`custom_list`](#custom-list)

_Last reviewed 2026-05-04 by macroforecast author._

### `custom_list`  --  operational

User-supplied horizon list (any non-empty integer set).

Requires ``leaf_config.target_horizons: [int...]``. Useful for non-standard horizon comparisons (e.g., {1, 2, 3, 6, 12} or {6, 12, 24, 36}).

**When to use**

Replication of papers with non-standard horizon sets; ablation studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`standard_md`](#standard-md), [`standard_qd`](#standard-qd), [`range_up_to_h`](#range-up-to-h)

_Last reviewed 2026-05-04 by macroforecast author._

### `range_up_to_h`  --  operational

Every horizon from 1 to leaf_config.max_horizon (inclusive).

Equivalent to ``custom_list`` with ``[1, 2, ..., max_horizon]``. Useful for direct-h forecasting where the study wants dense horizon coverage (e.g., 1-12 months).

**When to use**

Dense horizon studies with direct-h forecasting.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`custom_list`](#custom-list), [`standard_md`](#standard-md)

_Last reviewed 2026-05-04 by macroforecast author._
