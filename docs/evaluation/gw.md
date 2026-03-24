# Giacomini-White Test

`macrocast.evaluation.gw` implements the Giacomini-White (2006) test for **conditional**
predictive ability. Unlike DM (unconditional), GW tests whether one model is better
conditional on a set of instruments — e.g., whether model A outperforms model B specifically
during recessions or high-uncertainty periods.

Using `h_t = 1` (a constant instrument) reduces GW to the standard DM test.

**Reference:** Giacomini, R. and White, H. (2006). Tests of Conditional Predictive Ability.
*Econometrica*, 74(6), 1545–1578.

---

## Test Statistic

Let `d_t = L(e1_t) - L(e2_t)` be the per-period loss differential and `h_t` an instrument
vector. Define `Z_t = h_t ⊗ d_t`. The GW statistic is:

```
GW = T * Z_bar' * S_hat⁻¹ * Z_bar
```

where `S_hat` is a HAC estimate of the long-run covariance of `Z_t`. Under H0,
`GW ~ χ²(q)` where `q = dim(h_t)`.

---

## Usage

```python
import numpy as np
from macrocast.evaluation.gw import gw_test

# Unconditional (equivalent to DM)
result = gw_test(y_true, y_hat1, y_hat2, instruments=np.ones((T, 1)))

# Conditional on NBER recession indicator
result = gw_test(y_true, y_hat1, y_hat2, instruments=recession_indicator.reshape(-1, 1))
print(result.statistic, result.p_value)
```

See full API reference: [`macrocast.evaluation.gw`](../api/evaluation.md#macrocastevaluationgw)
