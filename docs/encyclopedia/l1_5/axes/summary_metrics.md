# `summary_metrics`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``summary_metrics`` on sub-layer ``L1_5_B_univariate_summary`` (layer ``l1_5``).

## Sub-layer

**L1_5_B_univariate_summary**

## Axis metadata

- Default: `['mean', 'sd', 'min', 'max', 'n_missing']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 8 option(s)
- Future: 0 option(s)

## Options

### `kurtosis`  --  operational

Sample excess kurtosis per series (fourth standardised moment, normal = 0).

Adds ``kurtosis`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Heavy-tail diagnostic; large values motivate winsorisation at L2.C or robust losses at L5.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`mean`](#mean), [`sd`](#sd), [`min`](#min), [`max`](#max), [`skew`](#skew)

_Last reviewed 2026-05-05 by macroforecast author._

### `max`  --  operational

Sample maximum per series.

Adds ``max`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Detecting outlier records prior to L2.C handling; suspicious upper bounds.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`mean`](#mean), [`sd`](#sd), [`min`](#min), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `mean`  --  operational

Sample mean per series.

Adds ``mean`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

First-moment summary for level series; cross-series comparison of central tendency.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`sd`](#sd), [`min`](#min), [`max`](#max), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `min`  --  operational

Sample minimum per series.

Adds ``min`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Detecting clipping artifacts (e.g. a 0 sentinel) or suspicious lower bounds.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`mean`](#mean), [`sd`](#sd), [`max`](#max), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `n_missing`  --  operational

Count of NaN entries per series.

Adds ``n_missing`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Quantifying imputation load before L2.D runs; high counts may justify dropping the series.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`mean`](#mean), [`sd`](#sd), [`min`](#min), [`max`](#max), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `n_obs`  --  operational

Number of non-NaN observations per series.

Adds ``n_obs`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Pair with ``n_missing`` to spot heavily-missing predictors that L2.D will need to impute.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_missing`](#n-missing), [`mean`](#mean), [`sd`](#sd), [`min`](#min), [`max`](#max), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `sd`  --  operational

Sample standard deviation per series.

Adds ``sd`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Second-moment scale; informs whether L3 ``scale`` standardisation is necessary.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`mean`](#mean), [`min`](#min), [`max`](#max), [`skew`](#skew), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._

### `skew`  --  operational

Sample skewness per series (third standardised moment).

Adds ``skew`` to the per-series summary table emitted by L1.5.B. ``summary_metrics`` is a multi-select axis -- listing several metrics produces a wide-form table with one row per series and one column per chosen metric.

**When to use**

Identifying asymmetric distributions that may justify a log transform at L2.B.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`n_obs`](#n-obs), [`n_missing`](#n-missing), [`mean`](#mean), [`sd`](#sd), [`min`](#min), [`max`](#max), [`kurtosis`](#kurtosis)

_Last reviewed 2026-05-05 by macroforecast author._
