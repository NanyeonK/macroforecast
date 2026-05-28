# `stationarity_test`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``stationarity_test`` on sub-layer ``L1_5_C_stationarity_tests`` (layer ``l1_5``).

## Sub-layer

**L1_5_C_stationarity_tests**

## Axis metadata

- Default: `'none'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `adf`  --  operational

Augmented Dickey-Fuller (Said-Dickey 1984) unit-root test.

Standard unit-root test based on the autoregressive specification ``Δy_t = α + β t + γ y_{t-1} + Σ δ_j Δy_{t-j} + ε_t``. Null hypothesis: ``γ = 0`` (unit root). Lag length auto-selected by BIC. statsmodels ``adfuller`` backend; emits per-series test statistic, lag, and MacKinnon (1996) p-value.

**When to use**

Default unit-root test; lowest power but widely cited.

**When NOT to use**

Series with strong autocorrelation in residuals -- ADF over-rejects; pair with PP or KPSS for triangulation.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Said & Dickey (1984) 'Testing for unit roots in autoregressive-moving average models of unknown order', Biometrika 71(3): 599-607.

**Related options**: [`kpss`](#kpss), [`pp`](#pp), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `kpss`  --  operational

Kwiatkowski-Phillips-Schmidt-Shin (1992) stationarity test.

Complementary to ADF: null hypothesis is *stationarity* (reject = unit root). Useful for breaking ties when ADF and KPSS disagree -- the variable's stationarity status is then ambiguous and probably benefits from a transformation. statsmodels ``kpss`` backend.

**When to use**

Triangulating ADF results; running both is the gold-standard pre-cleaning audit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Kwiatkowski, Phillips, Schmidt & Shin (1992) 'Testing the null hypothesis of stationarity against the alternative of a unit root', JoE 54(1-3): 159-178.

**Related options**: [`adf`](#adf), [`pp`](#pp), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Run ADF + KPSS + PP and stack the results into one table.

Triangulated stationarity verdict for every series. When all three reject (ADF, PP) / fail to reject (KPSS) the same direction, you have a clean stationarity verdict; conflicting results flag series for closer inspection.

**When to use**

Recommended default; gold-standard pre-cleaning audit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`adf`](#adf), [`kpss`](#kpss), [`pp`](#pp)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip stationarity tests entirely.

Useful when the panel is known stationary by construction (returns, log-changes, growth rates) and the test overhead provides no information.

**When to use**

Already-stationary panels (returns / log-changes); CI smoke runs.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`adf`](#adf), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `pp`  --  operational

Phillips-Perron (1988) unit-root test with native MacKinnon p-values.

Like ADF but corrects for serial correlation and heteroscedasticity in the residuals via a non-parametric Newey-West HAC adjustment rather than ADF's parametric lag augmentation. v0.25 ships a native macroforecast implementation -- no ``arch`` dependency required.

**When to use**

ADF alternative when residual autocorrelation is suspected.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Phillips & Perron (1988) 'Testing for a unit root in time series regression', Biometrika 75(2): 335-346.

**Related options**: [`adf`](#adf), [`kpss`](#kpss), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
