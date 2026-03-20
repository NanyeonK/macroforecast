# PBSV and oShapley Variable Importance

`macrocast.evaluation.pbsv` implements two tools for decomposing OOS forecast gains by predictor group, following Coulombe, Boldea, Renneson, Spierdijk, and Spierdijk (2022), "Anatomy of Out-of-Sample Gains."

**Reference:** Coulombe, P. G., Boldea, O., Renneson, J., Spierdijk, L., and Spierdijk, S. (2022). Anatomy of Out-of-Sample Gains. Working paper.

---

## oShapley Variable Importance

The oShapley-VI (Equations 16 in the reference) quantifies the average marginal contribution of each predictor group to OOS accuracy, using the Shapley value from cooperative game theory. The value function assigns OOS-R² to each coalition (subset) of predictor groups. The Shapley value for group `n` is the weighted average of its marginal contribution across all coalitions that do not include it.

The implementation uses exact enumeration over `2^N` subsets, which is computationally feasible for `N <= 15` predictor groups (`2^15 = 32,768` subset evaluations). For larger `N`, approximation methods are needed.

### oshapley_vi

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | list[list[int]] | required | List of N group definitions; each element is a list of column indices belonging to that group |
| `forecast_fn` | callable | required | Function `f(X_train, y_train, X_test) -> array(T_test,)` implementing the forecasting model |
| `X_train` | array (T_train, P) | required | Training features |
| `y_train` | array (T_train,) | required | Training targets |
| `X_test` | array (T_test, P) | required | Test features |
| `y_test` | array (T_test,) | required | Test targets |
| `loss_fn` | callable or None | None | Custom loss function; defaults to MSFE if None |

Returns `dict[int, float]` mapping each group index to its Shapley value.

---

## PBSV: Predictor Block Shapley Values

PBSV (Equations 19–25) extends oShapley-VI to produce a time-varying decomposition. Rather than a single aggregate value per group, PBSV produces a `(T_test, N_groups)` matrix where each entry quantifies the contribution of that group to that particular forecast.

### compute_pbsv

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | list[list[int]] | required | List of N group definitions; each element is a list of column indices |
| `forecast_fn` | callable | required | Function `f(X_train, y_train, X_test) -> array(T_test,)` |
| `X_train` | array (T_train, P) | required | Training features |
| `y_train` | array (T_train,) | required | Training targets |
| `X_test` | array (T_test, P) | required | Test features |
| `loss_fn` | callable or None | None | Custom loss function; defaults to MSFE if None |

Returns `ndarray` of shape `(T_test, N_groups)`.

---

## Model Accordance Score

The model accordance score measures how closely a model's dual observation weights align with a given predictor group membership structure. Higher values indicate that the model assigns heavier weight to training observations that are informative for the predictor groups active at the test date.

### model_accordance_score

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `W` | array (T_test, T_train) | required | Dual weight matrix from `dual` module |
| `groups` | list[list[int]] | required | Predictor group definitions |
| `X_train` | array (T_train, P) | required | Training features |
| `X_test` | array (T_test, P) | required | Test features |

Returns `ndarray` of shape `(T_test,)`.

---

## Computational Cost

Exact Shapley enumeration requires `2^N` calls to `forecast_fn`, where N is the number of predictor groups. The FRED variable group structure in macrocast uses 7 groups (output_income, labor, housing, prices, money, interest_rates, stock_market), giving `2^7 = 128` evaluations per call. For N = 15, this rises to 32,768 evaluations. The `forecast_fn` is called once per subset, so the total cost scales with the cost of fitting and predicting the underlying model.

---

## Example

```python
from macrocast.evaluation import oshapley_vi, compute_pbsv

# Groups: list of column index lists (one per FRED group)
groups = [
    list(range(0, 20)),   # output_income
    list(range(20, 40)),  # labor
    # ... other groups
]

def forecast_fn(X_train, y_train, X_test):
    # any estimator
    from sklearn.linear_model import Ridge
    m = Ridge().fit(X_train, y_train)
    return m.predict(X_test)

vi = oshapley_vi(groups, forecast_fn, X_train, y_train, X_test, y_test)
print(vi)  # {0: 0.045, 1: 0.031, ...}

pbsv_matrix = compute_pbsv(groups, forecast_fn, X_train, y_train, X_test)
print(pbsv_matrix.shape)  # (T_test, N_groups)
```
