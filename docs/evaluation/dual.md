# Dual Observation Weights

`macrocast.evaluation.dual` computes the dual weight representation of nonlinear forecasts, following Coulombe, Goulet-Coulombe, and Kichian (2024). Any forecast of the form `ŷ_t = f(X_test[t])` can be written as a weighted average of the training targets:

```
ŷ_t = sum_{s=1}^{T_train} w_{ts} * y_s
```

where the weight matrix `W` has shape `(T_test, T_train)`. The rows of `W` reveal which historical episodes the model draws on when forming each forecast, providing an interpretable bridge between nonlinear methods and classic nearest-neighbor intuition.

**Reference:** Coulombe, P. G., Goulet-Coulombe, C., and Kichian, M. (2024). Dual Observation Weights for Interpretable Machine Learning Forecasts. Working paper.

---

## KRR: Kernel Ridge Regression

For KRR with kernel matrix `K` and regularization `alpha`, the dual weights have the closed form:

```
w_t = k_t @ (K + alpha * I)^{-1}
```

where `k_t` is the vector of kernel evaluations between test point `t` and all training points.

### krr_dual_weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | array (T_train, N) | required | Training features |
| `X_test` | array (T_test, N) | required | Test features |
| `alpha` | float | required | KRR regularization parameter |
| `gamma` | float | required | RBF kernel bandwidth |
| `kernel` | str | `"rbf"` | Kernel type; currently only `"rbf"` is supported |

Returns `ndarray` of shape `(T_test, T_train)`.

---

## Tree Models

For ensemble tree methods (Random Forest, XGBoost), the dual weight between test point `t` and training point `s` is the fraction of trees in which `s` and `t` fall in the same leaf node.

### tree_dual_weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | fitted sklearn-compatible estimator | required | A fitted tree ensemble (e.g., `RandomForestRegressor`) |
| `X_train` | array (T_train, N) | required | Training features |
| `X_test` | array (T_test, N) | required | Test features |

Returns `ndarray` of shape `(T_test, T_train)`.

---

## Neural Networks

For neural networks, the dual weights are computed via a linear approximation using the penultimate layer activations `phi`:

```
w_t = phi_t @ (Phi @ Phi')^{-1} @ phi_s / T_train
```

where `Phi` is the `(T_train, D)` matrix of training activations and `phi_t` is the `(D,)` activation for test point `t`.

### nn_dual_weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phi_train` | array (T_train, D) | required | Penultimate layer activations for training data |
| `phi_test` | array (T_test, D) | required | Penultimate layer activations for test data |
| `ridge_alpha` | float | `1e-3` | Ridge regularization for the linear dual approximation |

Returns `ndarray` of shape `(T_test, T_train)`.

---

## Derived Quantities

### effective_history_length

The effective number of training observations contributing to each forecast, computed as the inverse sum of squares of the weight row:

```
EHL_t = 1 / sum_s w_{ts}^2
```

A high EHL means the model draws on a long, diffuse slice of history. A low EHL means it concentrates on a few close analogies.

| Parameter | Type | Description |
|-----------|------|-------------|
| `W` | array (T_test, T_train) | Dual weight matrix |

Returns `ndarray` of shape `(T_test,)`.

### top_analogies

For each test date, returns the `k` training dates with the highest weight.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `W` | array (T_test, T_train) | required | Dual weight matrix |
| `dates_train` | array-like (T_train,) | required | Training dates corresponding to rows of the training set |
| `k` | int | `5` | Number of top analogies to return per test date |

Returns `list[list[tuple]]`, where each inner list contains `k` tuples of `(date, weight)` sorted by descending weight.

---

## Example

```python
from macrocast.evaluation import krr_dual_weights, effective_history_length, top_analogies

# W shape: (T_test, T_train)
W = krr_dual_weights(X_train, X_test, alpha=0.01, gamma=0.1)

# Effective history length for each forecast date
eff_len = effective_history_length(W)
print("Avg effective history:", eff_len.mean())

# Top-5 analogies for each test date
analogies = top_analogies(W, dates_train=train_dates, k=5)
print("Analogies for first test date:", analogies[0])
```
