# `custom_source_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``custom_source_policy`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'official_only'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `official_only`  --  operational

Use the McCracken-Ng curated FRED-MD/QD/SD vintages.

Loads the bundled FRED snapshot via macroforecast's raw adapter -- no network access at runtime, no per-user data file. Vintages are pinned in ``macroforecast/raw/datasets/`` so two users on the same package version see identical raw inputs.

This is the canonical recipe path: every published replication script, every example in the gallery, and every CI check uses ``official_only`` so cross-user comparability is bit-exact.

**When to use**

Reproducing or extending published macro forecasting work; running benchmarks where readers need to repeat the study from the recipe alone; default for any FRED-based analysis.

**When NOT to use**

Forecasting on non-FRED panels (firm-level data, country-specific series); needs a vintage newer than the bundled snapshot.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)
* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`custom_panel_only`](#custom-panel-only), [`official_plus_custom`](#official-plus-custom), [`dataset`](#dataset), [`vintage_policy`](#vintage-policy)

**Examples**

*FRED-MD baseline*

```yaml
1_data:
  fixed_axes:
    custom_source_policy: official_only
    dataset: fred_md
  leaf_config:
    target: CPIAUCSL

```

_Last reviewed 2026-05-04 by macroforecast author._

### `custom_panel_only`  --  operational

Load a single user-supplied panel (CSV / Parquet / inline dict).

Bypasses the FRED adapter entirely. The user provides:

* an inline ``custom_panel_inline`` dict (small synthetic panels), or
* a ``custom_source_path`` pointing to a CSV / Parquet file.

The L1 runtime applies no schema-level validation beyond 'has a date column and at least the requested target series'. Variable metadata that the McCracken-Ng panel ships (group tags, t-codes, release dates) is unavailable, so axes that depend on it -- ``official_transform_policy``, ``fred_sd_state_group``, etc. -- are inactive.

**When to use**

Forecasting on proprietary firm panels, country-specific series, or any data not in FRED. Also the standard path for unit tests and tutorial recipes that ship deterministic synthetic data.

**When NOT to use**

When McCracken-Ng's curation (t-codes, group tags) is part of the study design -- ``official_only`` or ``official_plus_custom`` preserves it.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`official_only`](#official-only), [`official_plus_custom`](#official-plus-custom)

**Examples**

*Inline panel for a unit test*

```yaml
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
  leaf_config:
    target: y
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01]
      y:    [1.0, 2.0]
      x1:   [0.5, 1.0]

```

_Last reviewed 2026-05-04 by macroforecast author._

### `official_plus_custom`  --  operational

Merge the official FRED panel with a user-supplied auxiliary panel.

Loads the FRED vintage (per ``dataset``) and joins a user CSV / Parquet on the date index. Requires ``custom_source_path`` plus ``custom_merge_rule`` (one of ``inner_join`` / ``left_join`` / ``outer_join``) so the merge contract is explicit.

This is the canonical extension path for studies that want McCracken-Ng predictors plus a few additional series (e.g., proprietary survey indicators, alternative-data nowcast inputs).

**When to use**

Augmenting FRED-based studies with a small number of additional predictors that are not in the official panel.

**When NOT to use**

Pure custom panels (use ``custom_panel_only``); pure official panels (use ``official_only``); mixing two FRED vintages (the merge rule expects one FRED + one custom).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`official_only`](#official-only), [`custom_panel_only`](#custom-panel-only)

**Examples**

*FRED-MD plus a single proprietary series*

```yaml
1_data:
  fixed_axes:
    custom_source_policy: official_plus_custom
    dataset: fred_md
  leaf_config:
    target: CPIAUCSL
    custom_source_path: data/proprietary_indicator.parquet
    custom_merge_rule: left_join

```

_Last reviewed 2026-05-04 by macroforecast author._
