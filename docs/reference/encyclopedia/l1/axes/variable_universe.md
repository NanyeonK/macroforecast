# `variable_universe`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``variable_universe`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'all_variables'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `all_variables`  --  operational

Use every series in the chosen dataset.

FRED-MD/QD ships ~130 / ~250 series respectively. ``all_variables`` uses every one of them as predictors (target excluded). Standard for high-dimensional forecasting comparisons (PCR, lasso, factor models).

**When to use**

Default. Any high-dimensional benchmark following McCracken-Ng.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`missing_availability`](#missing-availability), [`official_transform_policy`](#official-transform-policy)

_Last reviewed 2026-05-04 by macroforecast author._

### `core_variables`  --  operational

Restrict to McCracken-Ng's curated 'core' subset (~30 series).

Smaller predictor set covering output, prices, money/credit, interest rates, and labor. Useful when a study wants a low-dimensional benchmark or replicates a paper that used the core set explicitly.

**When to use**

Low-dimensional benchmark; comparison against published 'core' panel results.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`missing_availability`](#missing-availability), [`official_transform_policy`](#official-transform-policy)

_Last reviewed 2026-05-04 by macroforecast author._

### `category_variables`  --  operational

Restrict to one McCracken-Ng category (e.g., 'output_and_income').

Uses one of the 8 (FRED-MD) / 14 (FRED-QD) category groupings as the predictor set. Requires ``leaf_config.variable_category`` naming the chosen category.

**When to use**

Within-category importance studies; testing whether one block alone is sufficient.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`missing_availability`](#missing-availability), [`official_transform_policy`](#official-transform-policy)

_Last reviewed 2026-05-04 by macroforecast author._

### `target_specific_variables`  --  operational

Use a custom predictor list keyed to the target.

Requires ``leaf_config.target_specific_columns: {target: [predictors...]}``. Different targets see different predictor sets. Useful when domain knowledge says only certain series are relevant for a given target (e.g., housing-target studies use housing-block predictors).

**When to use**

Domain-specific studies where each target has a known predictor block.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`missing_availability`](#missing-availability), [`official_transform_policy`](#official-transform-policy)

_Last reviewed 2026-05-04 by macroforecast author._

### `explicit_variable_list`  --  operational

Use exactly the columns listed in leaf_config.variable_universe_columns.

Most flexible option. The recipe author supplies the full predictor column list in leaf_config; macroforecast filters the panel to that list verbatim. No grouping logic, no category lookup.

**When to use**

Replication scripts that need an exact predictor set; ablations.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`missing_availability`](#missing-availability), [`official_transform_policy`](#official-transform-policy)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `variable_universe_columns` | `list[str]` | — | Required when variable_universe=explicit_variable_list; must be non-empty. | Explicit list of column names from the data source to use as the predictor universe. Validator rejects missing or empty list. |

_Last reviewed 2026-05-04 by macroforecast author._
