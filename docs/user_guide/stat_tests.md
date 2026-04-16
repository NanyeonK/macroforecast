# Statistical Tests

macrocast supports 20 statistical tests for forecast comparison. This guide helps you choose the right test.

## Decision flowchart

1. **Comparing 2 models?** Go to "Pairwise tests"
2. **Comparing 3+ models?** Go to "Multiple comparison"
3. **Checking model health?** Go to "Diagnostics"
4. **Testing direction accuracy?** Go to "Directional tests"

## Pairwise tests: Equal predictive ability

Use these when comparing two non-nested models (e.g., Ridge vs RandomForest).

### `dm` — Diebold-Mariano test
- **When to use:** Standard choice for comparing two forecasting models
- **H0:** Equal predictive ability (same expected loss)
- **Statistic:** Mean loss differential / HAC std error -> N(0,1)
- **Key reference:** Diebold & Mariano (1995)

### `dm_hln` — Harvey-Leybourne-Newbold correction
- **When to use:** Small samples (< 100 OOS observations)
- **Improvement over dm:** Finite-sample bias correction
- **Key reference:** Harvey, Leybourne & Newbold (1997)

### `dm_modified` — Modified DM for multi-step horizons
- **When to use:** Horizons h > 1 where forecast errors are serially correlated
- **Improvement over dm:** Accounts for MA(h-1) structure in multi-step errors

## Pairwise tests: Nested models

Use these when one model nests the other (e.g., AR(1) nested in AR + macro predictors).

### `cw` — Clark-West test
- **When to use:** When the benchmark is nested within the model (e.g., AR benchmark vs Ridge with many predictors)
- **Why not DM:** DM is undersized for nested models; CW corrects for the noise in estimated parameters
- **Key reference:** Clark & West (2007)

### `enc_new` — Encompassing test (ENC-NEW)
- **When to use:** Testing if one model's forecasts encompass (contain all information from) another's
- **Key reference:** Clark & McCracken (2001)

### `mse_f` — MSE-F test
- **When to use:** Equal MSPE testing for nested models with fixed estimation scheme
- **Key reference:** McCracken (2007)

### `mse_t` — MSE-T test
- **When to use:** Same as MSE-F but with different critical value derivation

## Conditional predictive ability

Use these when you suspect forecast ability varies over time.

### `cpa` — Giacomini-White Conditional Predictive Ability
- **When to use:** Testing if relative forecast ability is state-dependent (e.g., better in recessions)
- **Allows:** Conditioning on observable state variables
- **Key reference:** Giacomini & White (2006)

### `rossi` — Rossi-Sekhposyan Forecast Stability
- **When to use:** Testing if forecast ability has changed over the evaluation sample
- **Detects:** Structural breaks in relative performance
- **Key reference:** Rossi & Sekhposyan (2016)

### `rolling_dm` — Rolling Diebold-Mariano
- **When to use:** Tracking how DM statistic evolves over rolling windows
- **Output:** Time series of DM statistics (not just one number)

## Multiple comparison

Use these when comparing 3 or more models simultaneously.

### `mcs` — Model Confidence Set
- **When to use:** Finding the set of "best" models at a given confidence level
- **Output:** Confidence set (models not significantly worse than the best) + eliminated models
- **Key advantage:** Controls for data snooping when testing many models
- **Key reference:** Hansen, Lunde & Nason (2011)

### `reality_check` — White's Reality Check
- **When to use:** Testing if the best model beats a benchmark, accounting for multiple testing
- **Key reference:** White (2000)

### `spa` — Hansen's Superior Predictive Ability
- **When to use:** More powerful version of Reality Check
- **Key reference:** Hansen (2005)

## Residual diagnostics

Use these to check model health.

### `mincer_zarnowitz` — Mincer-Zarnowitz regression
- **Tests:** Forecast optimality (unbiasedness and efficiency)
- **Regress:** y_true on constant + y_pred. H0: intercept=0, slope=1.

### `ljung_box` — Ljung-Box test on forecast errors
- **Tests:** Serial correlation in errors. If significant, model misses temporal patterns.

### `arch_lm` — ARCH-LM test on forecast errors
- **Tests:** Heteroskedasticity (time-varying error variance). If significant, consider regime-dependent models.

### `bias_test` — Simple bias test
- **Tests:** Whether mean forecast error is significantly different from zero.

### `diagnostics_full` — Full diagnostic bundle
- **Runs all four above** (MZ, Ljung-Box, ARCH-LM, bias) in one call
- **Artifact:** Single JSON with all diagnostic results

## Directional tests

### `pesaran_timmermann` — Pesaran-Timmermann test
- **Tests:** Directional accuracy: does the model predict the sign of changes correctly?
- **Key reference:** Pesaran & Timmermann (1992)

### `binomial_hit` — Binomial hit test
- **Tests:** Whether directional accuracy exceeds 50% (coin flip)

## Dependence corrections

Forecast errors at horizons h > 1 are serially correlated. Choose an appropriate correction:

| Correction | When to use |
|-----------|-------------|
| `none` | h = 1 (no serial correlation) |
| `nw_hac` | h > 1, fixed bandwidth Newey-West HAC |
| `nw_hac_auto` | h > 1, automatic bandwidth selection |
| `block_bootstrap` | Nonstandard error distributions, small samples |

## Quick reference table

| Test | Type | Use case | Models |
|------|------|----------|--------|
| `dm` | Pairwise | Standard 2-model comparison | Non-nested |
| `dm_hln` | Pairwise | Small sample | Non-nested |
| `dm_modified` | Pairwise | Multi-step horizon | Non-nested |
| `cw` | Pairwise | Nested model test | Nested (AR vs AR+X) |
| `enc_new` | Pairwise | Encompassing | Nested |
| `mse_f` | Pairwise | Equal MSPE | Nested |
| `mse_t` | Pairwise | Equal MSPE | Nested |
| `cpa` | Conditional | State-dependent ability | Any |
| `rossi` | Conditional | Forecast stability | Any |
| `rolling_dm` | Conditional | Time-varying DM | Any |
| `mcs` | Multiple | Best model set | 3+ models |
| `reality_check` | Multiple | Beat benchmark? | 3+ models |
| `spa` | Multiple | Superior ability | 3+ models |
| `diagnostics_full` | Diagnostic | Model health | Single model |

**See also:**
- [Mathematical Background: Statistical Tests](../math/stat_tests.md) — formal definitions and formulas
- [Example: Statistical Test Gallery](../examples/stat_test_gallery.md) — runnable code
- [User Guide: Models](models.md) — model families to compare
