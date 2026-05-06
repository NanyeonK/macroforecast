# `fred_sd_variable_group`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``fred_sd_variable_group`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'all_sd_variables'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 12 option(s)
- Future: 0 option(s)

## Options

### `all_sd_variables`  --  operational

All FRED-SD state-level variable categories.

FRED-SD variable category: Default. Includes every variable category in the FRED-SD groups manifest. Use as the broadest possible predictor block; subset via sd_variable_selection if specific filtering is needed.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Default; broadest predictor block.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector), [`income`](#income)

_Last reviewed 2026-05-05 by macroforecast author._

### `labor_market_core`  --  operational

Core labour-market series (employment, unemployment, hours).

FRED-SD variable category: Includes nonfarm employment, unemployment rate, labour-force participation, and average hours. Standard labour-market battery used in most state-level macroeconomic studies.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Labour-market focused studies; Sahm-rule recession analysis at state level.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`employment_sector`](#employment-sector), [`income`](#income)

_Last reviewed 2026-05-05 by macroforecast author._

### `employment_sector`  --  operational

Sectoral employment series (NAICS supersector breakdowns).

FRED-SD variable category: Sectoral employment counts (manufacturing, construction, services, government, etc.). Useful when industry mix explains target variation.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Industry-level employment studies; structural-transformation analysis.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`income`](#income)

_Last reviewed 2026-05-05 by macroforecast author._

### `gsp_output`  --  operational

Gross state product / output series.

FRED-SD variable category: BEA gross state product (GSP), the state-level analogue of national GDP. Released quarterly with publication lag; main aggregate state-level output measure.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Aggregate output studies; state-level GDP forecasting.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `housing`  --  operational

State housing series (permits, prices, starts).

FRED-SD variable category: Building permits, housing starts, house-price indices. Leading indicator of state economic activity; central to any housing-cycle analysis.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Housing-cycle studies; foreclosure / mortgage-market analysis.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `trade`  --  operational

Trade / commerce series.

FRED-SD variable category: Retail sales, wholesale trade, port activity. State-level trade-flow indicators where available.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Trade-flow studies; port-region economic analysis.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `income`  --  operational

Personal income / earnings series.

FRED-SD variable category: Includes per-capita personal income, total state income, and components (wages, transfers, dividends). Slow-moving but persistent predictor of state economic activity.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Consumer / household income studies; transfer-payment analysis.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `direct_analog_high_confidence`  --  operational

Variables with direct national analog (high-confidence cross-frequency join).

FRED-SD variable category: FRED-SD variables that map directly onto a known FRED-MD / -QD national series at the same definition. The cleanest subset for cross-frequency studies that need national-state correspondence.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Cross-frequency studies needing direct national-state mapping.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `provisional_analog_medium`  --  operational

Variables with provisional national analog (medium-confidence join).

FRED-SD variable category: FRED-SD variables that *approximately* map onto a national series but with some definition mismatch (coverage gap, methodology change, etc.). Use with caution; the join is provisional.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Sensitivity analyses on the analog mapping.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `semantic_review_outputs`  --  operational

Outputs of the FRED-SD semantic review process.

FRED-SD variable category: Variables flagged through the FRED-SD semantic-review pipeline (audit-trail diagnostics produced by the FRED-SD construction process). Mostly used for diagnostic provenance, not as predictors.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Audit-trail diagnostics for the FRED-SD construction process.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `no_reliable_analog`  --  operational

Variables without a reliable national analog.

FRED-SD variable category: FRED-SD-only series that have no clean correspondence to a FRED-MD / -QD national variable. Useful for state-only studies that exclude national benchmarks.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

State-only studies; spatial-econometric panels that ignore national aggregates.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._

### `custom_sd_variable_group`  --  operational

User-supplied variable list (leaf_config.custom_sd_variables).

FRED-SD variable category: Bespoke variable selections -- e.g. 'manufacturing + trade only' or 'a specific BLS series list'. Reads the explicit variable list from ``leaf_config.custom_sd_variables``.

Restricts the predictor block to series tagged with this category in the FRED-SD groups manifest. Combine with ``fred_sd_state_group`` to control geography and with ``sd_variable_selection`` to restrict further within this category.

**When to use**

Bespoke variable selections not captured by built-in groupings.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`all_sd_variables`](#all-sd-variables), [`labor_market_core`](#labor-market-core), [`employment_sector`](#employment-sector)

_Last reviewed 2026-05-05 by macroforecast author._
