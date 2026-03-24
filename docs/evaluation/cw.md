# Clark-West Test

`macrocast.evaluation.cw` implements the Clark-West (2007) test for equal predictive accuracy
in **nested** models. The standard Diebold-Mariano test is size-distorted in nested settings
because the population loss differential is non-positive under H0. Clark and West (2007) propose
an adjustment that restores correct size.

**Reference:** Clark, T.E. and West, K.D. (2007). Approximately Normal Tests for Equal Predictive
Accuracy in Nested Models. *Journal of Econometrics*, 138(1), 291–311.

---

## Test Statistic

Let `e1_t = y_t - f1_t` (benchmark errors) and `e2_t = y_t - f2_t` (larger model errors).
The adjusted loss differential is:

```
d_t = e1_t² - [e2_t² - (f1_t - f2_t)²]
```

The `(f1_t - f2_t)²` term corrects for the noise from estimating the extra parameters.
The CW statistic is the t-ratio of `d_bar` with HAC standard errors.

Under H0, `CW ~ N(0, 1)`. The test is **one-sided**: reject H0 when `CW > z_alpha`.

---

## Usage

```python
from macrocast.evaluation.cw import cw_test

result = cw_test(y_true, y_hat_benchmark, y_hat_model, horizon=3)
print(result.statistic, result.p_value)
```

See full API reference: [`macrocast.evaluation.cw`](../api/evaluation.md#macrocastevaluationcw)
