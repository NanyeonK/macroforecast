# How to compute generalized impulse response functions

The goal is to compute generalized impulse response functions from a
fitted VAR using the Pesaran and Shin (1998) order-invariant
formulation. We fit the VAR via the public class `VAR` from
`macroforecast.models` and pass the fitted result to the `GIRF` class
from `macroforecast.interpretation`. The generalized IRF avoids the
dependency of orthogonalized IRFs on the Cholesky ordering, which is one
of the central limitations of structural VAR analysis.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.models import VAR
from macroforecast.interpretation import GIRF

rng = np.random.RandomState(0)
T = 200

# Simple bivariate system with cross-dependence.
e = rng.randn(T, 2) * 0.3
y1 = np.zeros(T)
y2 = np.zeros(T)
for t in range(1, T):
    y1[t] = 0.6 * y1[t-1] + 0.2 * y2[t-1] + e[t, 0]
    y2[t] = 0.3 * y1[t-1] + 0.5 * y2[t-1] + e[t, 1]

idx = pd.date_range("2010-01-01", periods=T, freq="ME")
panel = pd.DataFrame({"y1": y1, "y2": y2}, index=idx)
```

## Fit the VAR and compute the GIRF

```python
# Fit the public VAR class (wraps statsmodels VAR).
var_model = VAR()
var_model.fit(panel.iloc[:-10], panel["y1"].iloc[:-10])

girf = GIRF()
importance_df = girf.compute(var_model, n_periods=12)
print(importance_df)
```

The `GIRF.compute` method accepts either a `FitResult` from the public
`VAR` class (which exposes `._model`) or a raw fitted statsmodels VAR
results object. The default `n_periods=12` accumulates the response over
twelve horizons.

## Output

The call `compute` returns a `pd.DataFrame` with columns
`["feature", "importance", "method"]`. The `importance` column is the
L1 norm of the GIRF response across horizons `0..n_periods` for each
input variable. At horizon `h` the formula is

    GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma * e_j

where `A_h` is the reduced-form MA coefficient matrix from `irf.irfs`,
`Sigma` is the residual covariance, and `e_j` is the unit vector at
position `j`. The function falls back to permutation importance when the
input is not a statsmodels VAR.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `importance` column matches permutation importance | Input was not a statsmodels VAR result | Confirm the model was fit via `mf.models.VAR` or another VAR-producing function |
| `n_periods` ignored | Passed as a positional arg in the wrong position | Pass `n_periods=12` as a keyword argument |
| Importance values look orthogonalized (depend on column order) | Confused GIRF with Cholesky IRF | The GIRF formula uses `irf.irfs`, not `orth_irfs`, so column ordering should not affect results |

## See also

- {doc}`bayesian_var_minnesota`
- Paper: Pesaran and Shin (1998), Economics Letters 58(1).
