# `multiple_model_test`

[Back to L6](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``multiple_model_test`` on sub-layer ``L6_D_multiple_model`` (layer ``l6``).

## Sub-layer

**L6_D_multiple_model**

## Axis metadata

- Default: `'mcs_hansen'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `mcs_hansen`  --  operational

Hansen-Lunde-Nason Model Confidence Set (2011).

Default multiple-comparison test. Returns the set of models that contain the best at confidence level 1 - α via stationary-bootstrap (Politis-White 2004) iterated elimination. v0.25 uses the auto-tuned block length.

**When to use**

Identifying the small set of equally-best models out of many candidates.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Hansen, Lunde & Nason (2011) 'The Model Confidence Set', Econometrica 79(2): 453-497.

**Related options**: [`spa_hansen`](#spa-hansen), [`reality_check_white`](#reality-check-white), [`step_m_romano_wolf`](#step-m-romano-wolf)

_Last reviewed 2026-05-05 by macroforecast author._

### `spa_hansen`  --  operational

Hansen Superior Predictive Ability test (2005).

Tests whether any candidate beats the benchmark; studentises losses and uses a centred-bootstrap p-value. Compared to RC, less sensitive to poor models.

**When to use**

Testing whether the best candidate beats a fixed benchmark.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Hansen (2005) 'A Test for Superior Predictive Ability', JBES 23(4): 365-380.

**Related options**: [`mcs_hansen`](#mcs-hansen), [`reality_check_white`](#reality-check-white)

_Last reviewed 2026-05-05 by macroforecast author._

### `reality_check_white`  --  operational

White's Reality Check (2000).

Tests whether the best of N candidates beats a fixed benchmark. Original multiple-comparison test; SPA improves by studentising.

**When to use**

Foundational reality-check; compatibility with older studies.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* White (2000) 'A Reality Check for Data Snooping', Econometrica 68(5): 1097-1126.

**Related options**: [`spa_hansen`](#spa-hansen)

_Last reviewed 2026-05-05 by macroforecast author._

### `step_m_romano_wolf`  --  operational

Romano-Wolf StepM (2005) multiple-testing procedure.

Step-down procedure that controls FWER asymptotically. Returns ranked subset of candidates that beat the benchmark at level α.

**When to use**

Identifying which specific models in a large pool beat the benchmark.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Romano & Wolf (2005) 'Stepwise Multiple Testing as Formalized Data Snooping', Econometrica 73(4): 1237-1282.

**Related options**: [`mcs_hansen`](#mcs-hansen), [`spa_hansen`](#spa-hansen)

_Last reviewed 2026-05-05 by macroforecast author._
