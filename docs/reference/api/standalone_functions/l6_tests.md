# Standalone functions: L6 forecast evaluation tests (7 ops)

L6 test callables take arrays of forecast losses or errors and return a frozen result dataclass. Every L6 result type exposes `.stat` (float), `.pvalue` (float), `.decision` (str: `'reject'` or `'fail to reject'`), and `.summary()` (formatted string).

## Equal predictive ability tests (4 ops)

#### `dm_test(loss_a: np.ndarray, loss_b: np.ndarray, *, horizon: int = 1, correction: "Literal[hln, none]" = hln, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> DMTestResult`

Diebold-Mariano (1995) equal predictive ability test with optional HLN small-sample correction.

Returns `DMTestResult`: `.alternative`, `.correction_method`, `.decision`, `.hln_correction`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_a = (rng.standard_normal(100) * 0.5)**2
loss_b = (rng.standard_normal(100) * 0.8)**2
result = mf.functions.dm_test(loss_a, loss_b, horizon=1)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/equal_predictive_test/dm_diebold_mariano.md)

#### `dmp_test(loss_differentials: list[np.ndarray] | np.ndarray, *, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> DMPTestResult`

Panel Diebold-Mariano test (stacked loss differentials).

Returns `DMPTestResult`: `.alternative`, `.correction_method`, `.decision`, `.horizon`, `.n_obs_stacked`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_diffs = [rng.standard_normal(100), rng.standard_normal(100)]
result = mf.functions.dmp_test(loss_diffs)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/equal_predictive_test/dmp_multi_horizon.md)

#### `gw_test(loss_a: np.ndarray, loss_b: np.ndarray, *, horizon: int = 1, correction: "Literal[hln, none]" = hln, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> GWTestResult`

Giacomini-White (2006) conditional predictive ability test.

Returns `GWTestResult`: `.alternative`, `.correction_method`, `.decision`, `.hln_correction`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_a = (rng.standard_normal(100) * 0.5)**2
loss_b = (rng.standard_normal(100) * 0.8)**2
result = mf.functions.gw_test(loss_a, loss_b, horizon=1)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/equal_predictive_test/gw_giacomini_white.md)

#### `hn_test(e_a: np.ndarray, e_b: np.ndarray, *, horizon: int = 1, kernel: "Literal[newey_west, andrews, parzen]" = newey_west, small_sample: bool = True) -> HNTestResult`

Harvey-Newbold (1998) forecast encompassing test.

Returns `HNTestResult`: `.alternative`, `.correction_method`, `.decision`, `.encompassing`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
e_a = rng.standard_normal(100) * 0.5
e_b = rng.standard_normal(100) * 0.8
result = mf.functions.hn_test(e_a, e_b, horizon=1)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/equal_predictive_test/harvey_newbold_encompassing.md)

## Nested model and encompassing tests (3 ops)

#### `cw_test(loss_small: np.ndarray, loss_large: np.ndarray, f_small: np.ndarray, f_large: np.ndarray, *, horizon: int = 1, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> CWTestResult`

Clark-West (2006/2007) nested-model predictive ability test (4 positional args).

Returns `CWTestResult`: `.alternative`, `.correction_method`, `.cw_adjustment`, `.decision`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_small = (rng.standard_normal(100))**2
loss_large = (rng.standard_normal(100))**2
f_small = rng.standard_normal(100)
f_large = rng.standard_normal(100)
result = mf.functions.cw_test(loss_small, loss_large, f_small, f_large)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/nested_test/clark_west.md)

#### `enc_new_test(loss_small: np.ndarray, loss_large: np.ndarray, *, horizon: int = 1, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> EncNewTestResult`

ENC-NEW test (Clark and McCracken 2001) for nested-model encompassing.

Returns `EncNewTestResult`: `.alternative`, `.correction_method`, `.decision`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_small = (rng.standard_normal(100))**2
loss_large = (rng.standard_normal(100))**2
result = mf.functions.enc_new_test(loss_small, loss_large, horizon=1)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/nested_test/enc_new.md)

#### `enc_t_test(loss_small: np.ndarray, loss_large: np.ndarray, *, horizon: int = 1, kernel: "Literal[newey_west, andrews, parzen]" = newey_west) -> EncTTestResult`

ENC-t test (Clark and McCracken 2001) for nested-model encompassing.

Returns `EncTTestResult`: `.alternative`, `.correction_method`, `.decision`, `.horizon`, `.n_obs`, `.pvalue`, `.stat`, `.summary()`.

```python
rng = np.random.default_rng(42)
loss_small = (rng.standard_normal(100))**2
loss_large = (rng.standard_normal(100))**2
result = mf.functions.enc_t_test(loss_small, loss_large, horizon=1)
print(result.stat, result.pvalue, result.decision)
```

[Encyclopedia](../encyclopedia/l6/nested_test/enc_t.md)

## Return type reference

All test result dataclasses expose `.stat` (float), `.pvalue` (float), `.decision` (str), and `.summary()`.

## Quick example

```python
import macroforecast as mf
import numpy as np

rng = np.random.default_rng(42)
e1 = rng.standard_normal(100) * 0.5
e2 = rng.standard_normal(100) * 0.8
loss_a, loss_b = e1**2, e2**2

result = mf.functions.dm_test(loss_a, loss_b)
print(result.stat, result.pvalue, result.decision)
print(result.summary())
```
