# `equal_predictive_test`

[Back to L6](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``equal_predictive_test`` on sub-layer ``L6_A_equal_predictive`` (layer ``l6``).

## Sub-layer

**L6_A_equal_predictive**

## Axis metadata

- Default: `'dm_diebold_mariano'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Assumptions

The statistical validity of the equal-predictive-ability tests on this axis depends on the following conditions. Violations do not prevent the runtime from producing a result, but they affect inferential validity.

### DM and DMP (covariance stationarity)

The Diebold-Mariano test and the DMP multi-horizon variant assume that the loss differential series
`d_t = L(e_{a,t}) - L(e_{b,t})` is **covariance-stationary** (weakly stationary). Specifically:

- The mean of `d_t` must be constant over time.
- The autocovariance of `d_t` must depend only on the lag, not on calendar time.

If forecast performance shifts over the evaluation window (e.g., structural breaks, regime changes,
or expanding evaluation windows with changing sample composition), stationarity may not hold.
In such cases, prefer the Giacomini-White test (`gw_giacomini_white`), which is valid under
non-stationary loss differentials via the conditional predictive ability framework.

The Newey-West HAC standard error (default `dependence_correction: newey_west`) corrects for
serial correlation and heteroskedasticity in `d_t`, but does not correct for non-stationarity.
The HLN small-sample correction adjusts the test statistic for finite-sample bias at horizon `h > 1`
but similarly does not address non-stationarity.

### GW (rolling window requirement)

The Giacomini-White test requires that forecasts are produced from a **rolling** or **fixed** estimation
window of bounded size. It is not valid when the estimation window grows without bound (expanding
window), because the asymptotic theory requires that the estimation error vanishes at a controlled
rate. If `training_start_rule: expanding` is used, GW results should be interpreted cautiously.

### MCS bootstrap (stationarity of loss matrix)

When `multiple_model_test: mcs_hansen` is used (L6_D_multiple_model sub-layer), the stationary
bootstrap requires that the loss matrix rows are **weakly stationary**. The `bootstrap_block_length`
parameter (`auto` by default) is set to approximately `T^{1/3}` following Hansen, Lunde & Nason
(2011). If the evaluation window is short (T < 50), bootstrap inference may be unreliable regardless
of block length. In such cases, increase `bootstrap_n_replications` and treat MCS p-values as
approximate.

## Options

### `dm_diebold_mariano`  --  operational

Diebold-Mariano (1995) equal-predictive-ability test with Newey-West HAC SE.

Pairwise test of equal expected loss between two forecasts. Implements DM with HLN small-sample correction (Harvey-Leybourne-Newbold 1997) and a configurable HAC kernel (``newey_west`` default, ``andrews`` / ``parzen`` available). Two-sided alternative tests equality of MSE / MAE losses.

**When to use**

Pairwise comparison of two non-nested forecasts.

**When NOT to use**

Nested-model comparisons -- use Clark-West (L6.B) instead.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Diebold & Mariano (1995) 'Comparing Predictive Accuracy', JBES 13(3): 253-263.
* Harvey, Leybourne & Newbold (1997) 'Testing the equality of prediction mean squared errors', IJF 13(2): 281-291.

**Related options**: [`gw_giacomini_white`](#gw-giacomini-white), [`dmp_multi_horizon`](#dmp-multi-horizon), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `gw_giacomini_white`  --  operational

Giacomini-White (2006) conditional equal-predictive-ability test.

Generalises DM to test conditional predictive ability given a vector of predictors. Robust to non-stationary performance differentials and works with rolling / expanding-window forecasts.

**When to use**

Conditional / regime-dependent forecast comparisons.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Giacomini & White (2006) 'Tests of Conditional Predictive Ability', Econometrica 74(6): 1545-1578.

**Related options**: [`dm_diebold_mariano`](#dm-diebold-mariano), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `dmp_multi_horizon`  --  operational

Diebold-Mariano-Pesaran joint multi-horizon test.

HAC-adjusted stacked DM test that evaluates equality of predictive ability across all forecast horizons simultaneously. v0.3 implementation following Pesaran-Timmermann.

**When to use**

Joint significance across multiple horizons (avoids per-horizon p-value adjustment).

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Pesaran & Timmermann (2007) 'Selection of estimation window in the presence of breaks', JoE 137(1): 134-161.

**Related options**: [`dm_diebold_mariano`](#dm-diebold-mariano)

_Last reviewed 2026-05-05 by macroforecast author._

### `harvey_newbold_encompassing`  --  operational

Harvey-Leybourne-Newbold (1998) forecast-encompassing test.

Tests the null that forecast f_1 encompasses f_2 -- i.e. the optimal linear combination of the two forecasts puts zero weight on f_2's error. Constructs ``d_t = e_a (e_a - e_b)`` from the per-period forecast errors and tests its mean against zero with a Newey-West HAC long-run variance and an HLN small-sample correction at horizon h>1. Asymmetric by construction (f_1 encompasses f_2 ≠ f_2 encompasses f_1).

**When to use**

Deciding whether one forecast contains all the information of another.

**When NOT to use**

Symmetric equal-MSE comparison -- use ``dm_diebold_mariano`` instead.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Harvey, Leybourne & Newbold (1998) 'Tests for Forecast Encompassing', JBES 16(2): 254-259.

**Related options**: [`dm_diebold_mariano`](#dm-diebold-mariano), [`gw_giacomini_white`](#gw-giacomini-white), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Run DM + GW + DMP and stack the results.

Multi-test convenience option; emits a single output table with one row per test. Useful as a robustness check.

**When to use**

Comprehensive equal-predictive-ability audits.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'

**Related options**: [`dm_diebold_mariano`](#dm-diebold-mariano), [`gw_giacomini_white`](#gw-giacomini-white), [`dmp_multi_horizon`](#dmp-multi-horizon), [`harvey_newbold_encompassing`](#harvey-newbold-encompassing)

_Last reviewed 2026-05-05 by macroforecast author._
