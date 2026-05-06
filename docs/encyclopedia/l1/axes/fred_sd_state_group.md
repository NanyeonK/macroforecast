# `fred_sd_state_group`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``fred_sd_state_group`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'all_states'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 16 option(s)
- Future: 0 option(s)

## Options

### `all_states`  --  operational

All 50 states + DC (51 jurisdictions).

FRED-SD state grouping: Default. Includes every US state and the District of Columbia. Use as the broadest possible FRED-SD panel; subset thereafter via state_selection if specific filtering is needed.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Default; comprehensive 51-jurisdiction panel.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest), [`census_region_south`](#census-region-south)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_region_northeast`  --  operational

Census Northeast Region (9 states): CT, ME, MA, NH, NJ, NY, PA, RI, VT.

FRED-SD state grouping: Census Bureau's Region 1. Combines New England (CT, ME, MA, NH, RI, VT) and Mid-Atlantic (NJ, NY, PA) divisions. Heavily-populated, services-dominated regional economy.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Northeastern regional studies; comparing services-heavy economies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_midwest`](#census-region-midwest), [`census_region_south`](#census-region-south)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_region_midwest`  --  operational

Census Midwest Region (12 states): IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI.

FRED-SD state grouping: Census Bureau's Region 2. Combines East North Central (IL, IN, MI, OH, WI) and West North Central (IA, KS, MN, MO, NE, ND, SD) divisions. Manufacturing-heavy 'Rust Belt' + agricultural Plains economies.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Manufacturing-belt and Plains regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_south`](#census-region-south)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_region_south`  --  operational

Census South Region (16 states + DC): AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV.

FRED-SD state grouping: Census Bureau's Region 3. Combines South Atlantic (DE, DC, FL, GA, MD, NC, SC, VA, WV), East South Central (AL, KY, MS, TN), and West South Central (AR, LA, OK, TX) divisions. Largest Census region by population; mix of energy (TX, LA, OK) and Sun Belt service economies.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Southern regional studies; Sun Belt vs Rust Belt comparisons.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_region_west`  --  operational

Census West Region (13 states): AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY.

FRED-SD state grouping: Census Bureau's Region 4. Combines Mountain (AZ, CO, ID, MT, NV, NM, UT, WY) and Pacific (AK, CA, HI, OR, WA) divisions. Tech-heavy Pacific Coast + commodity / tourism Mountain economies.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Pacific Coast tech and Mountain West commodity studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_new_england`  --  operational

Census New England Division (6 states): CT, ME, MA, NH, RI, VT.

FRED-SD state grouping: Census Bureau's Division 1. Tight-knit historical region with finance / education / biotech concentration.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Finance / education / biotech regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_middle_atlantic`  --  operational

Census Middle Atlantic Division (3 states): NJ, NY, PA.

FRED-SD state grouping: Census Bureau's Division 2. Hosts the New York metropolitan financial centre; largest population Census division.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Financial-centre regional studies (NY metro).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_east_north_central`  --  operational

Census East North Central Division (5 states): IL, IN, MI, OH, WI.

FRED-SD state grouping: Census Bureau's Division 3. Great Lakes manufacturing belt; the historical 'Industrial Heartland' of the US.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Manufacturing / Rust Belt regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_west_north_central`  --  operational

Census West North Central Division (7 states): IA, KS, MN, MO, NE, ND, SD.

FRED-SD state grouping: Census Bureau's Division 4. Agricultural Great Plains with grain / livestock concentration.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Agricultural / commodity regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_south_atlantic`  --  operational

Census South Atlantic Division (8 states + DC): DE, DC, FL, GA, MD, NC, SC, VA, WV.

FRED-SD state grouping: Census Bureau's Division 5. Atlantic Seaboard from Delaware to Florida; mix of government (DC, VA), tech (NC, MD), and Sun Belt service economies (FL, GA).

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Atlantic Seaboard regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_east_south_central`  --  operational

Census East South Central Division (4 states): AL, KY, MS, TN.

FRED-SD state grouping: Census Bureau's Division 6. Tennessee Valley region; automotive-supplier and traditional manufacturing concentration.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Tennessee Valley / Auto-Alley regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_west_south_central`  --  operational

Census West South Central Division (4 states): AR, LA, OK, TX.

FRED-SD state grouping: Census Bureau's Division 7. Energy-dominated regional economy (TX, LA, OK oil & gas).

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Energy-sector regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_mountain`  --  operational

Census Mountain Division (8 states): AZ, CO, ID, MT, NV, NM, UT, WY.

FRED-SD state grouping: Census Bureau's Division 8. Mountain West; mining, tourism (NV, CO, UT), and tech-corridor (CO, UT) economies.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Mountain West regional studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `census_division_pacific`  --  operational

Census Pacific Division (5 states): AK, CA, HI, OR, WA.

FRED-SD state grouping: Census Bureau's Division 9. Pacific Coast tech concentration (CA, WA, OR) + non-contiguous states (AK, HI).

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Pacific Coast tech and non-contiguous-state studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `contiguous_48_plus_dc`  --  operational

Contiguous 48 states + DC (excludes AK, HI).

FRED-SD state grouping: Drops Alaska and Hawaii from the all-states panel. Useful when the analysis assumes a contiguous geographic structure (e.g. spatial econometrics with adjacency weights).

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Continental US studies; spatial econometrics with adjacency matrices.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._

### `custom_state_group`  --  operational

User-supplied state list (leaf_config.custom_state_list).

FRED-SD state grouping: Bespoke regional groupings -- e.g. 'oil-producing states' (TX, OK, ND, NM, LA), 'eurozone-equivalent BEA regions', or 'states with right-to-work laws'. Reads the explicit state list from ``leaf_config.custom_state_list``.

This option selects which state-level series enter the predictor / target panels. The grouping does not affect national-aggregate variables; combine with ``predictor_geography_scope`` to control whether predictors follow the target's geographic scope or use a different state set.

**When to use**

Bespoke regional groupings not captured by Census definitions.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.
* US Census Bureau (2020) 'Geographic Levels: Regions and Divisions', US Census Bureau Geography Division. <https://www.census.gov/programs-surveys/economic-census/guidance-geographies/levels.html>

**Related options**: [`all_states`](#all-states), [`census_region_northeast`](#census-region-northeast), [`census_region_midwest`](#census-region-midwest)

_Last reviewed 2026-05-05 by macroforecast author._
