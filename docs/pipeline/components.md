# Components

The `macrocast.pipeline.components` module defines the four treatment dimensions of the decomposition framework. Each dimension is an enum or enum-like object — not a string flag — so that combinations can be iterated systematically and used as dict keys.

---

## Nonlinearity

The `Nonlinearity` enum identifies the functional form of the model.

| Value | Description |
|-------|-------------|
| `LINEAR` | All models linear in parameters (AR, ARDI, Ridge, LASSO, etc.) |
| `KRR` | Kernel Ridge Regression with RBF kernel |
| `SVR_RBF` | SVR with RBF kernel — loss function comparison with KRR |
| `SVR_LINEAR` | Linear SVR — loss function comparison, no kernel nonlinearity |
| `RANDOM_FOREST` | Random Forest |
| `XGBOOST` | Gradient Boosted Trees |
| `NEURAL_NET` | Feedforward NN, ReLU activations, 1-2 hidden layers |
| `LSTM` | Sequence model implementing SequenceEstimator |

Linear regularized models that run in macrocastR all share `Nonlinearity.LINEAR`.

---

## Regularization

The `Regularization` enum identifies the penalisation or dimension-reduction strategy.

| Value | Description | Side |
|-------|-------------|------|
| `NONE` | OLS, no penalty | Python |
| `RIDGE` | L2 penalty | R (glmnet α=0) |
| `LASSO` | L1 penalty | R (glmnet α=1) |
| `ADAPTIVE_LASSO` | Weighted L1, penalty factors from OLS coefficients | R (glmnet penalty.factor) |
| `GROUP_LASSO` | Group L1 using FRED variable groups | R (grpreg) |
| `ELASTIC_NET` | L1+L2 mixture | R (glmnet α=0.5) |
| `FACTORS` | PCA diffusion index (ARDI), used by both sides | R + Python |
| `TVP_RIDGE` | Time-varying parameters via recursive Ridge | R |
| `BOOGING` | Bootstrap aggregating with pruning | R |

ARDI uses `Regularization.FACTORS` on the R side with lag-augmented predictors. On the Python side, `Regularization.FACTORS` triggers `FeatureBuilder` to construct Z_t from PCA factors plus AR lags.

---

## CVScheme

`CVScheme` is a factory that produces frozen dataclasses usable as dict keys. Three schemes are provided:

- **`CVScheme.BIC`** — Bayesian Information Criterion. Selects AR lag order for the benchmark and the Ridge penalty for linear models. Applicable to linear models only. Computationally cheap.
- **`CVScheme.POOS`** — Pseudo-OOS expanding one-step-ahead CV. Performs forward-chaining cross-validation within each training window.
- **`CVScheme.KFOLD(k=5)`** — K-fold cross-validation. Preferred for nonlinear models per Coulombe et al. (2022) because it exploits the full time series for HP selection. `k` defaults to 5.

```python
from macrocast.pipeline import CVScheme

bic   = CVScheme.BIC
poos  = CVScheme.POOS
kfold = CVScheme.KFOLD(k=5)
```

CVScheme objects are frozen dataclasses and can be used as dictionary keys or in sets. This allows factorial experiment grids to be constructed via `itertools.product`.

**Important distinction:** CVScheme governs the inner hyperparameter-selection loop. The outer pseudo-OOS evaluation loop is controlled by the `Window` enum, defined below.

---

## LossFunction

The `LossFunction` enum identifies the training objective.

| Value | Description |
|-------|-------------|
| `L2` | Mean squared error (default; empirically preferred per Coulombe et al. 2022) |
| `EPSILON_INSENSITIVE` | SVR-type epsilon-insensitive loss; ignores errors smaller than epsilon |

The KRR (`Nonlinearity.KRR`, `LossFunction.L2`) and SVR-RBF (`Nonlinearity.SVR_RBF`, `LossFunction.EPSILON_INSENSITIVE`) pairing is the key identification strategy for the loss function treatment effect. Both models use the RBF kernel on the same feature matrix, so any difference in OOS performance is attributable solely to the loss function.

---

## Window

The `Window` enum controls the outer evaluation loop strategy.

| Value | Description |
|-------|-------------|
| `EXPANDING` | Use all available history back to sample start (default, matches Coulombe et al. 2022) |
| `ROLLING` | Discard distant history; use a fixed-length window back from each evaluation date |

`Window.ROLLING` requires the `rolling_size` parameter in `ForecastExperiment`. It is useful when structural breaks make early data uninformative, at the cost of reduced effective sample size.

---

## Full Example

```python
from macrocast.pipeline import (
    Nonlinearity,
    Regularization,
    CVScheme,
    LossFunction,
    Window,
)

# Inspect enum members
print(list(Nonlinearity))
print(list(Regularization))
print(list(LossFunction))
print(list(Window))

# Build a CVScheme
kfold = CVScheme.KFOLD(k=5)
print(kfold)          # KFoldCV(k=5)
print(hash(kfold))    # frozen dataclass, hashable

# All four components for a KRR model spec
nonlinearity  = Nonlinearity.KRR
regularization = Regularization.FACTORS
cv_scheme     = CVScheme.KFOLD(k=5)
loss_function = LossFunction.L2
window        = Window.EXPANDING
```
