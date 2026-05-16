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

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `regime_indicator_path` | `str | Path` | — | Exactly one of {regime_indicator_path, regime_dates_list} must be set when regime_definition=external_user_provided. | Filesystem path to a CSV or Parquet file with columns 'date' (ISO date strings) and 'regime_label' (str or int). Aligned to the panel date index by the L1 runtime. |
| `regime_dates_list` | `list[dict]` | — | Exactly one of {regime_indicator_path, regime_dates_list} must be set when regime_definition=external_user_provided. Each dict must have keys: start (ISO date), end (ISO date), label (str). | Inline regime-date specification as a list of {start, end, label} dicts. Dates are interpreted as inclusive bounds. Gaps between ranges receive label None. |
| `n_regimes` | `int` | `2` | Positive integer. Informational only for external_user_provided. | Declared number of distinct regime labels in the user-supplied series. Used by downstream metrics and L5 by_regime decomposition. |

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

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `n_regimes` | `int` | `2` | Positive integer >= 2. Passed to statsmodels MarkovRegression as k_regimes. | Number of Markov-switching regimes. Defaults to 2 (expansion/recession). Higher values increase estimation cost quadratically. |

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

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `threshold_variable` | `str` | — | Required when regime_definition=estimated_threshold. Must be a column name present in the panel or 'lagged_target'. | Series used to determine regime membership. Each observation is assigned to a regime based on whether threshold_variable exceeds the estimated threshold value. |
| `n_thresholds` | `int` | `1` | Optional. Positive integer; n_thresholds+1 regimes are created. Must be >= 1. | Number of threshold breakpoints to estimate. 1 threshold -> 2 regimes (low/high). 2 thresholds -> 3 regimes (low/mid/high), etc. |

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

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `max_breaks` | `int` | `5` | Optional. Positive integer; upper bound on break count searched by the Bai-Perron dynamic program. | Maximum number of structural breaks to detect. BIC selects the final break count <= max_breaks. Larger values increase computation time. |
| `break_ic_criterion` | `str` | `'bic'` | Optional. One of {'aic', 'bic'}. Controls model-selection criterion for break-count choice. | Information criterion used to select the number of breaks. 'bic' (default) penalises extra breaks more heavily; 'aic' is less conservative. |

_Last reviewed 2026-05-04 by macroforecast author._
