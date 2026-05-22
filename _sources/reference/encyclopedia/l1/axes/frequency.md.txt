# `frequency`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``frequency`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'derived'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `monthly`  --  operational

Monthly observation frequency.

Pinned monthly frequency. Sets the canonical sampling cadence to one calendar month per observation, so horizon h=1 means one-month-ahead and ``standard_md`` horizons h ∈ {1, 3, 6, 9, 12, 18, 24} are interpreted in months.

Compatible with ``dataset=fred_md`` and ``dataset=fred_md+fred_sd``. When ``frequency`` is unset, the default ``'derived'`` sentinel resolves to ``monthly`` for FRED-MD datasets via ``_derived_frequency()`` at L1 normalization -- setting it explicitly is redundant for FRED-MD but required for custom panels that carry monthly observations.

**When to use**

Monthly macro forecasting (industrial production, payrolls, CPI, etc.); custom panels with monthly observations; explicit override of the FRED-MD default for documentation clarity.

**References**

* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`quarterly`](#quarterly), [`dataset`](#dataset), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-16 by macroforecast author._

### `quarterly`  --  operational

Quarterly observation frequency.

Pinned quarterly frequency. Sets the canonical sampling cadence to one calendar quarter per observation, so horizon h=1 means one-quarter-ahead and ``standard_qd`` horizons h ∈ {1, 2, 4, 8} are interpreted in quarters.

Compatible with ``dataset=fred_qd`` and ``dataset=fred_qd+fred_sd``. The ``'derived'`` default resolves to ``quarterly`` when ``dataset=fred_qd`` via ``_derived_frequency()``. Setting it explicitly is required for custom panels that carry quarterly observations.

**When to use**

Quarterly macro forecasting (GDP, productivity, NIPA-style targets); quarterly custom panels; FRED-QD-based studies.

**References**

* McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', Federal Reserve Bank of St. Louis Review.

**Related options**: [`monthly`](#monthly), [`dataset`](#dataset), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-16 by macroforecast author._
