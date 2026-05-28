# How to fit a Bayesian VAR with the Minnesota prior

The goal is to fit a Bayesian VAR with the Litterman (1986) Minnesota
prior using the public class `BVARMinnesota` from
`macroforecast.models`. The model implements the closed-form posterior
mean `β̂ = (V⁻¹ + X'X)⁻¹ (V⁻¹ m + X'y)` with a random-walk anchor on
the first own-lag-1 column. The parent class `BVAR` exposes the more
general interface for users who want to choose between the Minnesota and
Normal-Inverse-Wishart priors directly.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.models import BVAR, BVARMinnesota

rng = np.random.RandomState(0)
T = 200
trend = np.linspace(0, 1, T)
y_series = trend + 0.3 * rng.randn(T).cumsum() / np.sqrt(T)

# BVAR fit expects X with lagged columns named y_lagK so the random-walk
# anchor activates on y_lag1.
y = pd.Series(y_series, name="y")
X = pd.DataFrame({
    "y_lag1": y.shift(1),
    "y_lag2": y.shift(2),
}).dropna()
y_aligned = y.loc[X.index]
```

## Fit and predict

```python
model = BVARMinnesota(
    p=2,
    lambda1=0.2,
    lambda_decay=1.0,
    lambda_cross=0.5,
    b_AR=1.0,
    random_state=0,
)
model.fit(X, y_aligned)

predictions = model.predict(X.tail(10))
```

The hyperparameter `lambda1` is the overall shrinkage tightness,
`lambda_decay` controls how quickly the prior shrinks higher-order lags,
and `lambda_cross` regularizes cross-equation terms more strongly than
own-lag terms. The random-walk anchor `b_AR=1.0` recovers the I(1)
default, while setting `b_AR=0.9` matches the VARCTIC Appx-A.3
calibration.

## Validating the prior constraint

`BVARMinnesota` is a thin subclass that pins `prior='minnesota'`.
Attempting to override the prior on the subclass raises `ValueError`,
which guards against silent prior substitution.

```python
try:
    BVARMinnesota(prior="bvar_normal_inverse_wishart", p=2)
except ValueError as exc:
    print("rejected:", exc)
```

For the NIW alternative, use the parent class directly.

```python
niw = BVAR(prior="bvar_normal_inverse_wishart", p=2, random_state=0)
```

## Output

The call `fit` returns `self`. The call `predict(X) -> np.ndarray`
produces point forecasts for each row of `X`. The posterior parameters
live on `model._params` once `fit` has completed.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: BVARMinnesota requires prior='minnesota'` | Tried to override the prior on the subclass | Use the `BVAR` parent class directly and set `prior='bvar_normal_inverse_wishart'` |
| Posterior mean very close to OLS | `lambda1` set too large, so the prior is loose | Reduce `lambda1` toward 0.1 for tighter shrinkage |
| Forecasts look like a random walk regardless of data | `b_AR=1.0` with very small `lambda1` (tight prior on the unit-root anchor) | Either loosen the prior or set `b_AR=0.9` |
| Random-walk anchor not active | Lag columns not named `y_lag1`, `y_lag2`, ... | Follow the column naming convention so `_classify_columns` detects the anchor |

## See also

- {doc}`irf_pesaran_shin_girf`
- Paper: Litterman (1986), Journal of Business and Economic Statistics 4(1).
