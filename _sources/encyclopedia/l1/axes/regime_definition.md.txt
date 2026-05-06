# `regime_definition`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``regime_definition`` on sub-layer ``l1_g`` (layer ``l1``).

## Sub-layer

**l1_g**

## Axis metadata

- Default: `'none'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 6 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

No regime structure -- the cell-loop sees a single time-invariant world.

Default. The L1 regime metadata sink is empty; downstream layers that conditionally reference regime info (L4 regime_wrapper, L5 by_regime decomposition, L6.C cpa conditioning) are inactive.

**When to use**

Default for any study without an explicit regime structure.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`external_nber`](#external-nber), [`external_user_provided`](#external-user-provided), [`estimated_markov_switching`](#estimated-markov-switching)

_Last reviewed 2026-05-04 by macroforecast author._

### `external_nber`  --  operational

NBER recession dates loaded from the bundled USREC series.

Loads the NBER official recession indicator (USREC) and emits a two-state regime label ('expansion' / 'recession') per observation. No estimation -- the labels come directly from NBER's published business-cycle dates.

Pairs with L4 ``regime_wrapper = separate_fit`` for regime-conditional forecasting and with L5 ``by_regime`` decomposition for state-dependent metrics.

**When to use**

Standard recession-conditioning studies; comparing models' performance during recessions vs expansions.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* National Bureau of Economic Research, 'US Business Cycle Expansions and Contractions'. <https://www.nber.org/research/business-cycle-dating>

**Related options**: [`none`](#none), [`external_user_provided`](#external-user-provided)

**Examples**

*NBER-conditioned ridge*

```yaml
1_data:
  fixed_axes:
    regime_definition: external_nber

```

_Last reviewed 2026-05-04 by macroforecast author._

### `external_user_provided`  --  operational

User-supplied regime label series.

Requires ``leaf_config.regime_indicator_path`` (CSV / Parquet with date + label columns) **or** ``leaf_config.regime_dates_list`` (inline date ranges per label). The L1 runtime aligns the labels to the panel index without estimation.

Useful for custom regime definitions: monetary-policy-stance labels, fiscal-stance labels, geographic-dummy regimes, etc.

**When to use**

Custom regime studies (monetary stance, fiscal stance, alternative recession-dating chronologies).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`none`](#none), [`external_nber`](#external-nber)

_Last reviewed 2026-05-04 by macroforecast author._

### `estimated_markov_switching`  --  operational

Hamilton (1989) Markov-switching regime estimated from the target.

Fits ``statsmodels.tsa.regime_switching.MarkovRegression`` with ``leaf_config.n_regimes`` (default 2) regimes and switching variance. Returns smoothed posterior probabilities, the argmax label per observation, and the estimated transition matrix.

The estimation is per-origin (no leakage) via ``regime_estimation_temporal_rule``: ``expanding_window_per_origin`` (default) or ``rolling_window_per_origin`` / ``block_recompute``.

**When to use**

Inflation-regime studies; recession-prediction; any study that wants endogenous regime estimation.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* Hamilton (1989) 'A new approach to the economic analysis of nonstationary time series and the business cycle', Econometrica 57(2). (doi:10.2307/1912559)

**Related options**: [`estimated_threshold`](#estimated-threshold), [`estimated_structural_break`](#estimated-structural-break), [`regime_estimation_temporal_rule`](#regime-estimation-temporal-rule), [`n_regimes`](#n-regimes)

_Last reviewed 2026-05-04 by macroforecast author._

### `estimated_threshold`  --  operational

Tong (1990) SETAR threshold model -- regime by threshold variable.

Estimates ``n_regimes`` regimes by grid-searching the threshold on a chosen ``threshold_variable`` (defaults to lagged target). The regime per observation is determined by which side of the threshold the variable falls on.

Tong's original SETAR uses AR(p) per regime; macroforecast's runtime fits AR(``threshold_ar_p``) per partition and selects the threshold minimising joint SSR.

**When to use**

Self-exciting threshold dynamics (e.g., interest-rate stance regimes); non-linear AR studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* Tong (1990) 'Non-linear Time Series: A Dynamical System Approach', Oxford University Press.

**Related options**: [`estimated_markov_switching`](#estimated-markov-switching), [`estimated_structural_break`](#estimated-structural-break)

_Last reviewed 2026-05-04 by macroforecast author._

### `estimated_structural_break`  --  operational

Bai-Perron (1998) global LSE break detection.

Detects up to ``leaf_config.max_breaks`` structural breaks via the Bai (1997) dynamic-programming exact recursion + BIC selection. Each segment between breaks becomes a regime.

**When to use**

Studies hypothesising structural change (e.g., pre/post Volcker, pre/post Great Moderation).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* Bai & Perron (1998) 'Estimating and testing linear models with multiple structural changes', Econometrica 66(1). (doi:10.2307/2998540)

**Related options**: [`estimated_markov_switching`](#estimated-markov-switching), [`estimated_threshold`](#estimated-threshold), [`max_breaks`](#max-breaks)

_Last reviewed 2026-05-04 by macroforecast author._
