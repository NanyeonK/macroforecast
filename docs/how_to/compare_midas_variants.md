# How to compare MIDAS variants side by side

The goal is to compare four MIDAS variants on a common mixed-frequency
dataset. We import `MidasAlmon`, `MidasBeta`, `MidasStep`, and
`UnrestrictedMidas` from `macroforecast.models` and fit all four to the
same low-frequency target with the same high-frequency predictor lags.
The four estimators differ only in how they parametrize the lag-weight
function, so the comparison isolates the parsimony-flexibility
tradeoff.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.models import (
    MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
)

rng = np.random.RandomState(0)
T = 200
m = 3  # quarterly target with monthly predictor

# Generate a high-frequency predictor and aggregate by decreasing weights.
true_weights = np.array([0.4, 0.3, 0.2, 0.1])
n_lags = 4
X_high = rng.randn(T, n_lags)
y = X_high @ true_weights + 0.3 * rng.randn(T)

X = pd.DataFrame(X_high, columns=[f"x_lag{k}" for k in range(n_lags)])
y = pd.Series(y, name="y")
```

## Fit all four variants

```python
models = {
    "almon":        MidasAlmon(freq_ratio=m, n_lags_high=n_lags,
                               polynomial_order=2, n_starts=5, random_state=0),
    "beta":         MidasBeta(freq_ratio=m, n_lags_high=n_lags,
                              n_starts=5, random_state=0),
    "step":         MidasStep(freq_ratio=m, n_lags_high=n_lags,
                              n_steps=2),
    "unrestricted": UnrestrictedMidas(freq_ratio=m, n_lags_high=n_lags,
                                      include_y_lag=False, random_state=0),
}

results = {}
for name, model in models.items():
    model.fit(X.iloc[:-20], y.iloc[:-20])
    pred = model.predict(X.iloc[-20:])
    results[name] = pred

table = pd.DataFrame(results, index=X.iloc[-20:].index)
```

## Read the comparison

Construct a one-column-per-variant prediction frame and compare against
the held-out tail.

```python
errors = table.sub(y.iloc[-20:], axis=0)
rmse = (errors ** 2).mean().pow(0.5)
print(rmse.sort_values())
```

We typically find that `MidasStep` and `MidasAlmon` perform similarly on
smooth weight patterns, `MidasBeta` excels when the true weights are
sharply peaked, and `UnrestrictedMidas` performs best when the sample is
large enough to support free coefficients. Parsimony helps in small
samples and hurts in large ones, which is the standard MIDAS message
from Foroni, Marcellino, and Schumacher (2015).

## Variant tradeoffs at a glance

| Variant | Parameters | Strength | Weakness |
|---|---|---|---|
| `MidasAlmon` | `polynomial_order + 1` | Smooth flexible weights | Local optima at low `n_starts` |
| `MidasBeta` | 2 weight params | Captures sharply peaked patterns | Sensitive to numerical optimization |
| `MidasStep` | `n_steps` block weights | Fast and stable | Coarse if `n_steps` small |
| `UnrestrictedMidas` | `n_lags_high` free coefs | Best when sample is large | Overfits in small samples |

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| All four predictions are essentially identical | `n_lags_high` is small enough that the parametric and free variants coincide | Increase `n_lags_high` or shorten the polynomial order |
| `MidasAlmon` or `MidasBeta` converges to a local optimum | NLS multi-start count too low | Increase `n_starts` to 10 or higher |
| `UnrestrictedMidas` overfits | Sample too small for free coefficients | Pass `n_lags_high='bic'` for BIC-driven lag selection |
| `MidasStep` is too coarse | `n_steps` too small | Increase `n_steps` toward `freq_ratio` for finer step blocks |

## See also

- {doc}`tune_hyperparameters`
- {doc}`sweep_over_models`
- Papers: Ghysels, Sinko, and Valkanov (2007), and Foroni, Marcellino, and Schumacher (2015).
