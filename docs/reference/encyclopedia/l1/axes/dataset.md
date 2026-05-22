# `dataset`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``dataset`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'fred_md'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `fred_md`  --  operational

FRED-MD: 130+ monthly US macro series (1959-).

The McCracken & Ng (2016) Monthly Database for Macroeconomic Research. Curated set of ~130 macroeconomic and financial series with stable transformation codes, group tags, and a single vintage per month.

Default for monthly forecasting work; pairs with ``horizon_set: standard_md`` (h ∈ {1, 3, 6, 9, 12, 18, 24}) and ``frequency: monthly``.

**When to use**

Monthly inflation, employment, industrial-production, and term-structure forecasting.

**References**

* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`custom_source_policy`](#custom-source-policy), [`frequency`](#frequency), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._

### `fred_qd`  --  operational

FRED-QD: 250+ quarterly US macro series (1959-).

The McCracken & Ng (2020) Quarterly Database for Macroeconomic Research. Larger variable count than FRED-MD; quarterly cadence matches GDP / NIPA-style targets.

Default for quarterly forecasting; pairs with ``horizon_set: standard_qd`` (h ∈ {1, 2, 4, 8}) and ``frequency: quarterly``.

**When to use**

GDP, consumption, investment, productivity nowcasting / forecasting.

**References**

* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`custom_source_policy`](#custom-source-policy), [`frequency`](#frequency), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._

### `fred_sd`  --  operational

FRED-SD: state-level US series with geographic axes.

State-level macro panel covering ~50 states + DC. Activates the L1.D geography axes (target_geography_scope / predictor_geography_scope) and the L7 ``us_state_choropleth`` figure type for spatial interpretation.

FRED-SD ships with mixed monthly + quarterly frequencies; the L2.A frequency-alignment rules (issue #202) handle the mixed case.

**When to use**

State-level employment / payroll / housing forecasting; geographic-importance studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`custom_source_policy`](#custom-source-policy), [`frequency`](#frequency), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._

### `fred_md+fred_sd`  --  operational

Joint FRED-MD + FRED-SD panel.

Concatenates the FRED-MD national series with FRED-SD state-level series on the date index. Useful when a study needs both national context (FRED-MD) and state-level granularity (FRED-SD) -- e.g., a state-level employment forecast conditioned on national CPI.

**When to use**

Studies where state-level targets need national-aggregate predictors.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`custom_source_policy`](#custom-source-policy), [`frequency`](#frequency), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._

### `fred_qd+fred_sd`  --  operational

Joint FRED-QD + FRED-SD panel (quarterly + state-level mixed).

Concatenates FRED-QD with FRED-SD. Triggers the L2.A frequency-alignment rules because FRED-QD is quarterly while much of FRED-SD is monthly.

**When to use**

Quarterly state-level studies (rare; use only when the target is quarterly state-level).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`custom_source_policy`](#custom-source-policy), [`frequency`](#frequency), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._
