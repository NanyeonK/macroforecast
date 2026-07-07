# macroforecast.tests

[Back to reference](index.md)

Forecast-comparison tests, residual tests, density diagnostics, and model-confidence-set procedures.

Guide context: [../guide/concepts/evaluation.md](../guide/concepts/evaluation.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `TestResult` | class | Forecast comparison test result. |
| `anatolyev_gerko_test` | function | Anatolyev-Gerko excess profitability directional accuracy test. |
| `clark_west_test` | function | Clark-West nested forecast comparison test. |
| `conditional_predictive_ability_test` | function | Giacomini-Rossi rolling fluctuation test or package recursive extension. |
| `custom_test` | function | Run a user-supplied forecast test and coerce it to ``TestResult``. |
| `cw_test` | function | Alias for :func:`clark_west_test`. |
| `density_interval_tests` | function | Density and interval diagnostics for PIT values. |
| `directional_accuracy_test` | function | Pesaran-Timmermann or Henriksson-Merton directional accuracy test. |
| `dm_test` | function | Diebold-Mariano equal predictive ability test. |
| `jarque_bera_test` | function | Jarque-Bera test of normality for a single series. |
| `giacomini_white_test` | function | Giacomini-White (2006) conditional predictive ability (Wald) test. |
| `var_serial_test` | function | Multivariate residual serial-correlation test for a VAR (vars::serial.test). |
| `var_normality_test` | function | Multivariate normality test for VAR residuals (vars::normality.test). |
| `var_arch_test` | function | Multivariate ARCH-LM test for VAR residuals (vars::arch.test, Lutkepohl). |
| `granger_causality` | function | Granger causality test in a VAR (R vars::causality / statsmodels). |
| `instantaneous_causality` | function | Instantaneous (contemporaneous) causality test in a VAR (vars::causality). |
| `dmp_test` | function | Diebold-Mariano-Pesaran joint multi-horizon test on stacked losses. |
| `dynamic_quantile_test` | function | Engle-Manganelli dynamic quantile test for VaR forecasts. |
| `equal_predictive_tests` | function | Run multiple equal-predictive-ability tests and stack results. |
| `enc_new_test` | function | ENC-NEW nested forecast encompassing test. |
| `enc_t_test` | function | ENC-T nested forecast encompassing test. |
| `gw_test` | function | Legacy GW-compatible DM-style equal predictive ability callable. |
| `harvey_newbold_test` | function | Legacy forecast-encompassing covariance t approximation. |
| `henriksson_merton_test` | function | Henriksson-Merton directional accuracy test. |
| `hn_test` | function | Alias for :func:`harvey_newbold_test`. |
| `interval_coverage_test` | function | Kupiec, Christoffersen, and duration diagnostics for forecast intervals. |
| `blocked_oob_reality_check` | function | Legacy block-bootstrap benchmark-superiority screen. |
| `iterative_model_confidence_set` | function | Descriptive alias for :func:`model_confidence_set`. |
| `mincer_zarnowitz_test` | function | Mincer-Zarnowitz forecast-rationality regression. |
| `model_confidence_set` | function | Exact Hansen-Lunde-Nason model confidence set. |
| `multi_horizon_spa_test` | function | Quaedvlieg (2021) multi-horizon SPA test for one pair of models. |
| `nested_tests` | function | Run multiple nested-model forecast tests and stack results. |
| `pesaran_timmermann_test` | function | Pesaran-Timmermann directional accuracy test. |
| `pit_autocorrelation_test` | function | Normal approximation test for serial dependence in PIT values. |
| `pit_histogram` | function | Return PIT histogram counts against a uniform reference. |
| `reality_check_test` | function | White reality check against a benchmark via ``arch.bootstrap``. |
| `residual_diagnostics` | function | Run residual diagnostic tests and return one row per test. |
| `shortfall_de_test` | function | Du-Escanciano expected shortfall tests on PIT values. |
| `stepm_test` | function | Stepwise multiple-comparison test against a benchmark via ``arch.bootstrap``. |
| `superior_predictive_ability_test` | function | White-Hansen superior predictive ability test via ``arch.bootstrap``. |

## Callable And Class Reference

### TestResult

Qualified name: `macroforecast.tests.TestResult`

#### Signature

```python
macroforecast.tests.TestResult(statistic: float | None, p_value: float | None, decision: bool, alternative: str, correction_policy: str | None = None, n_obs: int | None = None, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Forecast comparison test result.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `statistic` | positional or keyword | `float \| None` | `required` |
| `p_value` | positional or keyword | `float \| None` | `required` |
| `decision` | positional or keyword | `bool` | `required` |
| `alternative` | positional or keyword | `str` | `required` |
| `correction_policy` | positional or keyword | `str \| None` | `None` |
| `n_obs` | positional or keyword | `int \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.tests.TestResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `statistic` | `float \| None` | `required` |
| `p_value` | `float \| None` | `required` |
| `decision` | `bool` | `required` |
| `alternative` | `str` | `required` |
| `correction_policy` | `str \| None` | `None` |
| `n_obs` | `int \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `summary` | `summary(self) -> str` | No public docstring is available. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_json` | `to_json(self, path: str \| Path \| None = None, *, indent: int \| None = 2) -> str` | Return JSON text, and optionally write it to ``path``. |
### anatolyev_gerko_test

Qualified name: `macroforecast.tests.anatolyev_gerko_test`

#### Signature

```python
macroforecast.tests.anatolyev_gerko_test(*args: Any, **kwargs: Any) -> TestResult
```

#### Description

Anatolyev-Gerko excess profitability directional accuracy test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.anatolyev_gerko_test(...)
```
### clark_west_test

Qualified name: `macroforecast.tests.clark_west_test`

#### Signature

```python
macroforecast.tests.clark_west_test(loss_small: Any, loss_large: Any, forecast_small: Any, forecast_large: Any, *, horizon: int = 1, cw_adjustment: bool = True, kernel: str = "newey_west", alpha: float = 0.05) -> TestResult
```

#### Description

Clark-West nested forecast comparison test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_small` | positional or keyword | `Any` | `required` |
| `loss_large` | positional or keyword | `Any` | `required` |
| `forecast_small` | positional or keyword | `Any` | `required` |
| `forecast_large` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `cw_adjustment` | keyword only | `bool` | `True` |
| `kernel` | keyword only | `str` | `"newey_west"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.clark_west_test(...)
```
### conditional_predictive_ability_test

Qualified name: `macroforecast.tests.conditional_predictive_ability_test`

#### Signature

```python
macroforecast.tests.conditional_predictive_ability_test(loss_a: Any, loss_b: Any, *, method: str = "giacomini_rossi", window_ratio: float = 0.5, dmv_fullsample: bool = True, lag_truncate: int = 0, alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Giacomini-Rossi rolling fluctuation test or package recursive extension.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any` | `required` |
| `method` | keyword only | `str` | `"giacomini_rossi"` |
| `window_ratio` | keyword only | `float` | `0.5` |
| `dmv_fullsample` | keyword only | `bool` | `True` |
| `lag_truncate` | keyword only | `int` | `0` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.conditional_predictive_ability_test(...)
```
### custom_test

Qualified name: `macroforecast.tests.custom_test`

#### Signature

```python
macroforecast.tests.custom_test(name: str, func: Callable[..., Any], *args: Any, alternative: str = "two_sided", alpha: float = 0.05, correction_policy: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> TestResult
```

#### Description

Run a user-supplied forecast test and coerce it to ``TestResult``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `args` | var positional | `Any` | `required` |
| `alternative` | keyword only | `str` | `"two_sided"` |
| `alpha` | keyword only | `float` | `0.05` |
| `correction_policy` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.custom_test(...)
```
### cw_test

Qualified name: `macroforecast.tests.cw_test`

#### Signature

```python
macroforecast.tests.cw_test(*args: Any, **kwargs: Any) -> TestResult
```

#### Description

Alias for :func:`clark_west_test`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.cw_test(...)
```
### density_interval_tests

Qualified name: `macroforecast.tests.density_interval_tests`

#### Signature

```python
macroforecast.tests.density_interval_tests(pit: Any, *, alpha: float = 0.05, n_bins: int = 10, pit_lag: int = 1) -> dict[str, Any]
```

#### Description

Density and interval diagnostics for PIT values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `pit` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_bins` | keyword only | `int` | `10` |
| `pit_lag` | keyword only | `int` | `1` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.density_interval_tests(...)
```
### directional_accuracy_test

Qualified name: `macroforecast.tests.directional_accuracy_test`

#### Signature

```python
macroforecast.tests.directional_accuracy_test(y_true: Any, y_pred: Any, *, threshold: float = 0.0, method: str = "pesaran_timmermann", alpha: float = 0.05) -> TestResult
```

#### Description

Pesaran-Timmermann or Henriksson-Merton directional accuracy test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `threshold` | keyword only | `float` | `0.0` |
| `method` | keyword only | `str` | `"pesaran_timmermann"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.directional_accuracy_test(...)
```
### dm_test

Qualified name: `macroforecast.tests.dm_test`

#### Signature

```python
macroforecast.tests.dm_test(loss_a: Any, loss_b: Any, *, horizon: int = 1, correction: str = "hln", kernel: str = "acf", input_type: str = "loss", power: float = 2.0, alternative: str = "two_sided", alpha: float = 0.05) -> TestResult
```

#### Description

Diebold-Mariano equal predictive ability test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `correction` | keyword only | `str` | `"hln"` |
| `kernel` | keyword only | `str` | `"acf"` |
| `input_type` | keyword only | `str` | `"loss"` |
| `power` | keyword only | `float` | `2.0` |
| `alternative` | keyword only | `str` | `"two_sided"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.dm_test(...)
```
### jarque_bera_test

Qualified name: `macroforecast.tests.jarque_bera_test`

#### Signature

```python
macroforecast.tests.jarque_bera_test(series: Any, *, alpha: float = 0.05) -> TestResult
```

#### Description

Jarque-Bera test of normality for a single series.

Uses population (1/n) skewness and excess-kurtosis moments, JB ~ chi2(2),
matching ``tseries::jarque.bera.test``. The decision rejects normality when
p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.jarque_bera_test(...)
```
### giacomini_white_test

Qualified name: `macroforecast.tests.giacomini_white_test`

#### Signature

```python
macroforecast.tests.giacomini_white_test(loss_a: Any, loss_b: Any, *, horizon: int = 1, instruments: Any | None = None, alpha: float = 0.05, small_sample: bool = True) -> TestResult
```

#### Description

Giacomini-White (2006) conditional predictive ability (Wald) test.

Tests H0: ``E[h_{t-1} * dL_t] = 0`` where ``dL_t = loss_a - loss_b`` is the
loss differential and ``h_{t-1}`` is a test-function instrument available at
the forecast origin (default ``[1, dL_{t-h}]``). ``instruments`` may be
supplied as an array aligned to ``dL_t``. ``Omega`` is a HAC estimator of
``R_t = h_{t-1} * dL_t`` built from lags ``0..horizon-1`` -- the known
dependence order of an h-step-ahead loss differential, the same order
``dm_test`` uses.

``small_sample=True`` (default -- see CHANGELOG, p-values for horizon > 1
change relative to prior releases): WP-A1's Monte Carlo size validation
(``tests/mc/test_giacomini_white_size.py``, following on WP-V3) found the
*original* Bartlett-tapered-HAC + chi2(q) construction genuinely oversized
for horizon > 1 (2-2.5x nominal at h=4; confirmed NOT a small-n artifact,
it does not vanish out to n=100,000). Root cause, isolated by direct
comparison of the estimated HAC covariance against its true population
value: the linear Bartlett taper ``1 - lag/h`` applied to lags 1..h-1
systematically discards a large, non-vanishing fraction of the *known*
(finite-order, exactly h-1-dependent) autocovariance of an h-step loss
differential -- e.g. at h=4 the taper's population expectation is only
~69% of the true long-run variance (matches the closed-form taper-weight
calculation exactly). ``dm_test`` never had this problem because it
already uses an UNTAPERED ("acf") sum over the same lags (matching R's
``forecast::dm.test``) -- appropriate because the true autocovariance is
*exactly* zero beyond lag h-1 here, so tapering (whose purpose is
guaranteeing positive semi-definiteness for general, not-known-finite-
order processes) only throws away real signal.

The corrected estimator: (a) sums UNTAPERED sample autocovariances over
lags ``0..bandwidth`` (``bandwidth = horizon - 1``), matching ``dm_test``;
(b) falls back to a smaller bandwidth (down to 0) if that untapered sum is
not positive semi-definite -- untapered sums lose Newey-West's automatic
PSD guarantee, so this mirrors ``_long_run_variance``'s own existing
non-positive-variance fallback; (c) references the Wald statistic against
``F(q, ESS - q)`` (Hotelling-style, statistic scaled by ``q``) rather than
chi2(q) whenever a HAC lag was actually used, with
``ESS = n / (1 + 2 * bandwidth_used)`` the standard effective-sample-size
correction for serially dependent data -- this mops up the residual
(much smaller than the taper bias, but still real at small n) finite-
sample over-rejection of a Wald test built on an estimated multi-
dimensional covariance. At horizon=1 (bandwidth=0, already well-
calibrated per WP-V3/WP-A1 MC results) this reduces exactly to the
original chi2(q) reference -- verified by MC to introduce no regression.

``small_sample=False`` restores the pre-WP-A1 behavior exactly: Bartlett-
tapered HAC + chi2(q) reference, for users who need bit-identical
backward-compatible p-values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `instruments` | keyword only | `Any \| None` | `None` |
| `alpha` | keyword only | `float` | `0.05` |
| `small_sample` | keyword only | `bool` | `True` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.giacomini_white_test(...)
```
### var_serial_test

Qualified name: `macroforecast.tests.var_serial_test`

#### Signature

```python
macroforecast.tests.var_serial_test(panel: Any, *, n_lag: int = 1, test_lags: int | None = None, trend: str = "c", adjusted: bool = False, alpha: float = 0.05) -> TestResult
```

#### Description

Multivariate residual serial-correlation test for a VAR (vars::serial.test).

Lutkepohl Portmanteau / LM test of no autocorrelation in the VAR residual
vector up to ``test_lags`` lags (statsmodels VARResults.test_whiteness).
Rejects no-serial-correlation when p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `test_lags` | keyword only | `int \| None` | `None` |
| `trend` | keyword only | `str` | `"c"` |
| `adjusted` | keyword only | `bool` | `False` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.var_serial_test(...)
```
### var_normality_test

Qualified name: `macroforecast.tests.var_normality_test`

#### Signature

```python
macroforecast.tests.var_normality_test(panel: Any, *, n_lag: int = 1, trend: str = "c", alpha: float = 0.05) -> TestResult
```

#### Description

Multivariate normality test for VAR residuals (vars::normality.test).

Doornik-Hansen / Lutkepohl joint test of skewness and kurtosis on the
standardised VAR residuals (statsmodels VARResults.test_normality). Rejects
multivariate normality when p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `trend` | keyword only | `str` | `"c"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.var_normality_test(...)
```
### var_arch_test

Qualified name: `macroforecast.tests.var_arch_test`

#### Signature

```python
macroforecast.tests.var_arch_test(panel: Any, *, n_lag: int = 1, arch_lags: int = 5, trend: str = "c", alpha: float = 0.05) -> TestResult
```

#### Description

Multivariate ARCH-LM test for VAR residuals (vars::arch.test, Lutkepohl).

Regresses the vech of the residual outer products on ``arch_lags`` of its own
lags and forms the multivariate ARCH-LM statistic
``VARCH_LM = T * N * R2_m`` with ``R2_m = 1 - tr(Omega Omega0^-1)/N`` and
``N = K(K+1)/2``, chi-squared with ``arch_lags * N^2`` df. Rejects no
multivariate ARCH when p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `arch_lags` | keyword only | `int` | `5` |
| `trend` | keyword only | `str` | `"c"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.var_arch_test(...)
```
### granger_causality

Qualified name: `macroforecast.tests.granger_causality`

#### Signature

```python
macroforecast.tests.granger_causality(panel: Any, *, caused: str, causing: str | Sequence[str], n_lag: int = 1, kind: str = "f", trend: str = "c", alpha: float = 0.05) -> TestResult
```

#### Description

Granger causality test in a VAR (R vars::causality / statsmodels).

Tests whether ``causing`` Granger-causes ``caused`` in a VAR(``n_lag``) fit on
``panel``. ``kind='f'`` uses the F statistic, ``'wald'`` the chi-squared Wald.
The decision rejects non-causality (i.e. ``causing`` does Granger-cause
``caused``) when p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `caused` | keyword only | `str` | `required` |
| `causing` | keyword only | `str \| Sequence[str]` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `kind` | keyword only | `str` | `"f"` |
| `trend` | keyword only | `str` | `"c"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.granger_causality(...)
```
### instantaneous_causality

Qualified name: `macroforecast.tests.instantaneous_causality`

#### Signature

```python
macroforecast.tests.instantaneous_causality(panel: Any, *, caused: str, causing: str | Sequence[str] | None = None, n_lag: int = 1, trend: str = "c", alpha: float = 0.05) -> TestResult
```

#### Description

Instantaneous (contemporaneous) causality test in a VAR (vars::causality).

Tests for contemporaneous correlation between the residuals of ``caused`` and
the other variables (or ``causing`` if given). Rejects no-instantaneous-
causality when p < alpha.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `caused` | keyword only | `str` | `required` |
| `causing` | keyword only | `str \| Sequence[str] \| None` | `None` |
| `n_lag` | keyword only | `int` | `1` |
| `trend` | keyword only | `str` | `"c"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.instantaneous_causality(...)
```
### dmp_test

Qualified name: `macroforecast.tests.dmp_test`

#### Signature

```python
macroforecast.tests.dmp_test(loss_differences: Any, *, kernel: str = "newey_west", alpha: float = 0.05) -> TestResult
```

#### Description

Diebold-Mariano-Pesaran joint multi-horizon test on stacked losses.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_differences` | positional or keyword | `Any` | `required` |
| `kernel` | keyword only | `str` | `"newey_west"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.dmp_test(...)
```
### dynamic_quantile_test

Qualified name: `macroforecast.tests.dynamic_quantile_test`

#### Signature

```python
macroforecast.tests.dynamic_quantile_test(y_true: Any, var: Any, *, alpha: float = 0.05, lag: int = 1, lag_hit: int = 1, lag_var: int = 1) -> TestResult
```

#### Description

Engle-Manganelli dynamic quantile test for VaR forecasts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `var` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `lag` | keyword only | `int` | `1` |
| `lag_hit` | keyword only | `int` | `1` |
| `lag_var` | keyword only | `int` | `1` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.dynamic_quantile_test(...)
```
### equal_predictive_tests

Qualified name: `macroforecast.tests.equal_predictive_tests`

#### Signature

```python
macroforecast.tests.equal_predictive_tests(loss_a: Any, loss_b: Any, *, tests: Sequence[str] = ('dm', 'gw', 'dmp'), error_a: Any | None = None, error_b: Any | None = None, horizon: int = 1, correction: str = "hln", kernel: str = "acf", alpha: float = 0.05) -> pd.DataFrame
```

#### Description

Run multiple equal-predictive-ability tests and stack results.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any` | `required` |
| `tests` | keyword only | `Sequence[str]` | `("dm", "gw", "dmp")` |
| `error_a` | keyword only | `Any \| None` | `None` |
| `error_b` | keyword only | `Any \| None` | `None` |
| `horizon` | keyword only | `int` | `1` |
| `correction` | keyword only | `str` | `"hln"` |
| `kernel` | keyword only | `str` | `"acf"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.equal_predictive_tests(...)
```
### enc_new_test

Qualified name: `macroforecast.tests.enc_new_test`

#### Signature

```python
macroforecast.tests.enc_new_test(error_small: Any, error_large: Any, *, critical_value: float | None = None, alpha: float = 0.05) -> TestResult
```

#### Description

ENC-NEW nested forecast encompassing test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `error_small` | positional or keyword | `Any` | `required` |
| `error_large` | positional or keyword | `Any` | `required` |
| `critical_value` | keyword only | `float \| None` | `None` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.enc_new_test(...)
```
### enc_t_test

Qualified name: `macroforecast.tests.enc_t_test`

#### Signature

```python
macroforecast.tests.enc_t_test(error_small: Any, error_large: Any, *, horizon: int = 1, kernel: str = "newey_west", critical_value: float | None = None, normal_approximation: bool = False, alpha: float = 0.05) -> TestResult
```

#### Description

ENC-T nested forecast encompassing test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `error_small` | positional or keyword | `Any` | `required` |
| `error_large` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `kernel` | keyword only | `str` | `"newey_west"` |
| `critical_value` | keyword only | `float \| None` | `None` |
| `normal_approximation` | keyword only | `bool` | `False` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.enc_t_test(...)
```
### gw_test

Qualified name: `macroforecast.tests.gw_test`

#### Signature

```python
macroforecast.tests.gw_test(loss_a: Any, loss_b: Any, *, horizon: int = 1, correction: str = "hln", kernel: str = "acf", input_type: str = "loss", power: float = 2.0, alternative: str = "two_sided", alpha: float = 0.05) -> TestResult
```

#### Description

Legacy GW-compatible DM-style equal predictive ability callable.

This callable keeps the public ``gw_test`` surface but computes the same HAC
loss-differential statistic as :func:`dm_test`. For the conditional
Giacomini-White Wald test, use :func:`giacomini_white_test`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `correction` | keyword only | `str` | `"hln"` |
| `kernel` | keyword only | `str` | `"acf"` |
| `input_type` | keyword only | `str` | `"loss"` |
| `power` | keyword only | `float` | `2.0` |
| `alternative` | keyword only | `str` | `"two_sided"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.gw_test(...)
```
### harvey_newbold_test

Qualified name: `macroforecast.tests.harvey_newbold_test`

#### Signature

```python
macroforecast.tests.harvey_newbold_test(error_a: Any, error_b: Any, *, horizon: int = 1, kernel: str = "newey_west", small_sample: bool = True, alpha: float = 0.05) -> TestResult
```

#### Description

Legacy forecast-encompassing covariance t approximation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `error_a` | positional or keyword | `Any` | `required` |
| `error_b` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `kernel` | keyword only | `str` | `"newey_west"` |
| `small_sample` | keyword only | `bool` | `True` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.harvey_newbold_test(...)
```
### henriksson_merton_test

Qualified name: `macroforecast.tests.henriksson_merton_test`

#### Signature

```python
macroforecast.tests.henriksson_merton_test(*args: Any, **kwargs: Any) -> TestResult
```

#### Description

Henriksson-Merton directional accuracy test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.henriksson_merton_test(...)
```
### hn_test

Qualified name: `macroforecast.tests.hn_test`

#### Signature

```python
macroforecast.tests.hn_test(*args: Any, **kwargs: Any) -> TestResult
```

#### Description

Alias for :func:`harvey_newbold_test`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.hn_test(...)
```
### interval_coverage_test

Qualified name: `macroforecast.tests.interval_coverage_test`

#### Signature

```python
macroforecast.tests.interval_coverage_test(y_true: Any, lower: Any, upper: Any, *, alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Kupiec, Christoffersen, and duration diagnostics for forecast intervals.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `lower` | positional or keyword | `Any` | `required` |
| `upper` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.interval_coverage_test(...)
```
### blocked_oob_reality_check

Qualified name: `macroforecast.tests.blocked_oob_reality_check`

#### Signature

```python
macroforecast.tests.blocked_oob_reality_check(loss_panel: pd.DataFrame, *, benchmark: str, loss: str = "squared_error", alpha: float = 0.05, n_boot: int = 1000, block_length: int | str = 4, bootstrap_method: str = "fixed_block_bootstrap", random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> pd.DataFrame
```

#### Description

Legacy block-bootstrap benchmark-superiority screen.

The input can be either a long per-origin loss panel with model and loss
columns, or a wide loss matrix whose columns are model names. Positive
``mean_diff`` means the candidate has lower average loss than the
benchmark. This callable is not the exact White Reality Check; use
``reality_check_test``/``superior_predictive_ability_test``/``stepm_test``
for the arch-backed multiple-comparison procedures.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `4` |
| `bootstrap_method` | keyword only | `str` | `"fixed_block_bootstrap"` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.blocked_oob_reality_check(...)
```
### iterative_model_confidence_set

Qualified name: `macroforecast.tests.iterative_model_confidence_set`

#### Signature

```python
macroforecast.tests.iterative_model_confidence_set(loss_panel: pd.DataFrame, *, loss: str = "squared_error", alpha: float = 0.1, n_boot: int = 1000, block_length: int | str = "auto", bootstrap_method: str = "mcs_fixed_block", statistic: str = "max", random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> dict[str, Any]
```

#### Description

Descriptive alias for :func:`model_confidence_set`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.1` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `"auto"` |
| `bootstrap_method` | keyword only | `str` | `"mcs_fixed_block"` |
| `statistic` | keyword only | `str` | `"max"` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.iterative_model_confidence_set(...)
```
### mincer_zarnowitz_test

Qualified name: `macroforecast.tests.mincer_zarnowitz_test`

#### Signature

```python
macroforecast.tests.mincer_zarnowitz_test(y_true: Any, y_pred: Any, *, hac_lags: int = 0, alpha: float = 0.05) -> TestResult
```

#### Description

Mincer-Zarnowitz forecast-rationality regression.

Regresses actual values on a constant and the forecast, then tests the joint
null ``intercept = 0`` and ``slope = 1`` using a HAC covariance matrix.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `hac_lags` | keyword only | `int` | `0` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.mincer_zarnowitz_test(...)
```
### model_confidence_set

Qualified name: `macroforecast.tests.model_confidence_set`

#### Signature

```python
macroforecast.tests.model_confidence_set(loss_panel: pd.DataFrame, *, loss: str = "squared_error", alpha: float = 0.1, n_boot: int = 1000, block_length: int | str = "auto", bootstrap_method: str = "mcs_fixed_block", statistic: str = "max", random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> dict[str, Any]
```

#### Description

Exact Hansen-Lunde-Nason model confidence set.

This is the canonical MCS callable. It follows the R ``MCS`` package's
``MCSprocedure`` structure: pairwise loss differences are bootstrapped,
either ``Tmax`` or ``TR`` is evaluated, one model is removed each step, and
included/excluded sets are determined from the cumulative MCS p-values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.1` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `"auto"` |
| `bootstrap_method` | keyword only | `str` | `"mcs_fixed_block"` |
| `statistic` | keyword only | `str` | `"max"` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.model_confidence_set(...)
```
### multi_horizon_spa_test

Qualified name: `macroforecast.tests.multi_horizon_spa_test`

#### Signature

```python
macroforecast.tests.multi_horizon_spa_test(loss_a: Any, loss_b: Any | None = None, *, statistic: str = "uspa", weights: Sequence[float] | None = None, alpha: float = 0.05, n_boot: int = 999, block_length: int = 3, hac_bandwidth: int | str = "auto", random_state: int = 0, alternative: str = "greater") -> TestResult
```

#### Description

Quaedvlieg (2021) multi-horizon SPA test for one pair of models.

The input is a loss-differential panel with one column per horizon. Pass a
single panel as ``loss_a`` to use it directly, or pass two aligned loss
panels and the differential is ``loss_a - loss_b``. With the two-panel
contract, positive means model ``loss_b`` has lower loss than model
``loss_a``. ``statistic="uspa"`` tests uniform superior predictive ability
with the minimum horizon-specific studentized statistic; ``"aspa"`` tests
average superior predictive ability with the studentized weighted average.

Implementation notes
Quaedvlieg's Algorithm 1 uses a moving-block bootstrap for studentized
statistics and, in the simulation section, sets block length to 3 and
``B=999``. Those are the defaults here. The paper specifies a Quadratic
Spectral HAC estimator for the original statistic but does not pin the HAC
bandwidth in Algorithm 1; ``hac_bandwidth="auto"`` uses the same automatic
bandwidth convention as this module's other HAC helpers, while an integer
pins it. Bootstrap statistics use the paper's natural block-mean variance
estimator.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_a` | positional or keyword | `Any` | `required` |
| `loss_b` | positional or keyword | `Any \| None` | `None` |
| `statistic` | keyword only | `str` | `"uspa"` |
| `weights` | keyword only | `Sequence[float] \| None` | `None` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_boot` | keyword only | `int` | `999` |
| `block_length` | keyword only | `int` | `3` |
| `hac_bandwidth` | keyword only | `int \| str` | `"auto"` |
| `random_state` | keyword only | `int` | `0` |
| `alternative` | keyword only | `str` | `"greater"` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.multi_horizon_spa_test(...)
```
### nested_tests

Qualified name: `macroforecast.tests.nested_tests`

#### Signature

```python
macroforecast.tests.nested_tests(loss_small: Any, loss_large: Any, *, forecast_small: Any | None = None, forecast_large: Any | None = None, error_small: Any | None = None, error_large: Any | None = None, tests: Sequence[str] = ('clark_west', 'enc_new', 'enc_t'), horizon: int = 1, kernel: str = "newey_west", enc_critical_value: float | None = None, enc_normal_approximation: bool = False, alpha: float = 0.05) -> pd.DataFrame
```

#### Description

Run multiple nested-model forecast tests and stack results.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_small` | positional or keyword | `Any` | `required` |
| `loss_large` | positional or keyword | `Any` | `required` |
| `forecast_small` | keyword only | `Any \| None` | `None` |
| `forecast_large` | keyword only | `Any \| None` | `None` |
| `error_small` | keyword only | `Any \| None` | `None` |
| `error_large` | keyword only | `Any \| None` | `None` |
| `tests` | keyword only | `Sequence[str]` | `("clark_west", "enc_new", "enc_t")` |
| `horizon` | keyword only | `int` | `1` |
| `kernel` | keyword only | `str` | `"newey_west"` |
| `enc_critical_value` | keyword only | `float \| None` | `None` |
| `enc_normal_approximation` | keyword only | `bool` | `False` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.nested_tests(...)
```
### pesaran_timmermann_test

Qualified name: `macroforecast.tests.pesaran_timmermann_test`

#### Signature

```python
macroforecast.tests.pesaran_timmermann_test(*args: Any, **kwargs: Any) -> TestResult
```

#### Description

Pesaran-Timmermann directional accuracy test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.pesaran_timmermann_test(...)
```
### pit_autocorrelation_test

Qualified name: `macroforecast.tests.pit_autocorrelation_test`

#### Signature

```python
macroforecast.tests.pit_autocorrelation_test(pit: Any, *, lag: int = 1, alpha: float = 0.05) -> TestResult
```

#### Description

Normal approximation test for serial dependence in PIT values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `pit` | positional or keyword | `Any` | `required` |
| `lag` | keyword only | `int` | `1` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`TestResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.pit_autocorrelation_test(...)
```
### pit_histogram

Qualified name: `macroforecast.tests.pit_histogram`

#### Signature

```python
macroforecast.tests.pit_histogram(pit: Any, *, n_bins: int = 10) -> pd.DataFrame
```

#### Description

Return PIT histogram counts against a uniform reference.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `pit` | positional or keyword | `Any` | `required` |
| `n_bins` | keyword only | `int` | `10` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.pit_histogram(...)
```
### reality_check_test

Qualified name: `macroforecast.tests.reality_check_test`

#### Signature

```python
macroforecast.tests.reality_check_test(loss_panel: pd.DataFrame, *, benchmark: str, loss: str = "squared_error", alpha: float = 0.05, n_boot: int = 1000, block_length: int | str = "auto", bootstrap_method: str = "stationary_bootstrap", p_value_type: str = "consistent", studentize: bool = True, nested: bool = False, random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> dict[str, Any]
```

#### Description

White reality check against a benchmark via ``arch.bootstrap``.

Size caveat: this shares the arch-backed SPA path's dependent-loss
over-rejection disclosure. The return payload includes
``metadata.size_caveat``; prefer ``model_confidence_set`` under dependent
losses.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `"auto"` |
| `bootstrap_method` | keyword only | `str` | `"stationary_bootstrap"` |
| `p_value_type` | keyword only | `str` | `"consistent"` |
| `studentize` | keyword only | `bool` | `True` |
| `nested` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.reality_check_test(...)
```
### residual_diagnostics

Qualified name: `macroforecast.tests.residual_diagnostics`

#### Signature

```python
macroforecast.tests.residual_diagnostics(residuals: Any, *, tests: Sequence[str] = ('ljung_box_q', 'arch_lm', 'jarque_bera_normality', 'durbin_watson'), lag: int = 10, alpha: float = 0.05, model_df: int = 0, exog: Any | None = None, demean_arch: bool = False) -> pd.DataFrame
```

#### Description

Run residual diagnostic tests and return one row per test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `residuals` | positional or keyword | `Any` | `required` |
| `tests` | keyword only | `Sequence[str]` | `("ljung_box_q", "arch_lm", "jarque_bera_normality", "durbin_w...` |
| `lag` | keyword only | `int` | `10` |
| `alpha` | keyword only | `float` | `0.05` |
| `model_df` | keyword only | `int` | `0` |
| `exog` | keyword only | `Any \| None` | `None` |
| `demean_arch` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.residual_diagnostics(...)
```
### shortfall_de_test

Qualified name: `macroforecast.tests.shortfall_de_test`

#### Signature

```python
macroforecast.tests.shortfall_de_test(pit: Any, *, alpha: float = 0.05, lags: int = 1, boot: bool = False, n_boot: int = 2000, random_state: int = 0) -> dict[str, Any]
```

#### Description

Du-Escanciano expected shortfall tests on PIT values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `pit` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `lags` | keyword only | `int` | `1` |
| `boot` | keyword only | `bool` | `False` |
| `n_boot` | keyword only | `int` | `2000` |
| `random_state` | keyword only | `int` | `0` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.shortfall_de_test(...)
```
### stepm_test

Qualified name: `macroforecast.tests.stepm_test`

#### Signature

```python
macroforecast.tests.stepm_test(loss_panel: pd.DataFrame, *, benchmark: str, loss: str = "squared_error", alpha: float = 0.05, n_boot: int = 1000, block_length: int | str = "auto", bootstrap_method: str = "stationary_bootstrap", studentize: bool = True, nested: bool = False, random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> dict[str, Any]
```

#### Description

Stepwise multiple-comparison test against a benchmark via ``arch.bootstrap``.

Size caveat: dependent-null Monte Carlo diagnostics mirror the arch-backed
SPA/RC over-rejection disclosure. The return payload includes
``metadata.size_caveat``; prefer ``model_confidence_set`` under dependent
losses.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `"auto"` |
| `bootstrap_method` | keyword only | `str` | `"stationary_bootstrap"` |
| `studentize` | keyword only | `bool` | `True` |
| `nested` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.stepm_test(...)
```
### superior_predictive_ability_test

Qualified name: `macroforecast.tests.superior_predictive_ability_test`

#### Signature

```python
macroforecast.tests.superior_predictive_ability_test(loss_panel: pd.DataFrame, *, benchmark: str, loss: str = "squared_error", alpha: float = 0.05, n_boot: int = 1000, block_length: int | str = "auto", bootstrap_method: str = "stationary_bootstrap", p_value_type: str = "consistent", studentize: bool = True, nested: bool = False, random_state: int = 0, target: str = "target", horizon: str = "horizon", origin: str = "origin", model: str = "model_id") -> dict[str, Any]
```

#### Description

White-Hansen superior predictive ability test via ``arch.bootstrap``.

Size caveat: dependent-null Monte Carlo diagnostics currently show roughly
1.5-2x nominal over-rejection when losses are serially correlated. The return
payload includes ``metadata.size_caveat``; prefer ``multi_horizon_spa_test``
(``uspa``/``aspa``) or ``model_confidence_set`` under dependent losses.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `loss_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `alpha` | keyword only | `float` | `0.05` |
| `n_boot` | keyword only | `int` | `1000` |
| `block_length` | keyword only | `int \| str` | `"auto"` |
| `bootstrap_method` | keyword only | `str` | `"stationary_bootstrap"` |
| `p_value_type` | keyword only | `str` | `"consistent"` |
| `studentize` | keyword only | `bool` | `True` |
| `nested` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int` | `0` |
| `target` | keyword only | `str` | `"target"` |
| `horizon` | keyword only | `str` | `"horizon"` |
| `origin` | keyword only | `str` | `"origin"` |
| `model` | keyword only | `str` | `"model_id"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.tests.superior_predictive_ability_test(...)
```
