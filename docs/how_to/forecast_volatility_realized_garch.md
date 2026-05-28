# How to forecast volatility with Realized GARCH

The goal is to forecast one-step-ahead conditional variance using the
Hansen-Huang-Shek (2012) Realized GARCH model with a realized-variance
side channel. We use the public class `RealizedGARCH` from
`macroforecast.models`, which exposes a sklearn-style `fit` and
`predict_variance` interface. The model does not require the optional
`arch` package because the joint MLE is implemented in scipy.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.models import RealizedGARCH

# Synthetic daily returns with a stylized realized-variance series.
rng = np.random.RandomState(0)
T = 500
true_h = 0.5 + 0.3 * np.abs(rng.randn(T)).cumsum() / T
returns = rng.randn(T) * np.sqrt(true_h)
realized_var = true_h * np.exp(0.1 * rng.randn(T))

index = pd.date_range("2018-01-01", periods=T, freq="B")
y = pd.Series(returns, index=index, name="ret")
X = pd.DataFrame({"rv": realized_var}, index=index)
```

## Fit and predict

```python
model = RealizedGARCH(realized_variance="rv", n_starts=8, random_state=0)
model.fit(X, y)

one_step_variance = model.predict_variance(h_steps=1)
ten_step_variance = model.predict_variance(h_steps=10)
```

The `realized_variance` argument names the column in `X` that carries
the realized-variance side channel. The constructor defaults to
`n_starts=8` multi-starts and `random_state=0`. If every start returns a
non-finite objective the model raises `RuntimeError`, a safety
guard against silent convergence to a degenerate parameter vector.

## Output

`predict_variance(h_steps)` returns a length-`h_steps` numpy array of
conditional-variance forecasts. The companion `predict(X)` returns the
constant conditional mean broadcast over `len(X)`. The fitted parameter
dictionary is exposed via `model._params` for diagnostic inspection.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: ... requires >= 30 observations` | Sample below minimum | Provide at least 30 non-missing returns |
| Variance forecast is constant across `h_steps` | `realized_variance` column missing in X, model fell back to the `r^2` proxy | Pass the realized-variance column name and confirm it is present in `X.columns` |
| `RuntimeError: all multi-starts produced non-finite objective` | Pathological data such as a vector of zeros | Inspect `y` and `X['rv']` for degeneracy and rescale returns if needed |
| `ValueError: dist=... is not supported` | Passed `dist='t'` or another non-default distribution | Only `dist='normal'` is implemented in the current release |

## See also

- {doc}`bayesian_var_minnesota`
- Paper: Hansen, Huang, and Shek (2012), Journal of Applied Econometrics 27(6).
