# Treatment Effect Decomposition

`macrocast.evaluation.decomposition` implements the OLS-based decomposition of out-of-sample forecast gains following Coulombe, Leroux, Stevanovic, and Surprenant (2022, JBES), Equation 11. The decomposition attributes average OOS-R² values across the model grid to four binary treatment indicators corresponding to the four design dimensions of the pipeline.

---

## Regression Equation

For each model m in the experiment grid, the OOS-R² value is regressed on four indicator dummies:

```
OOS-R²_m = alpha
           + beta_1 * 1[nonlinearity = nonlinear]
           + beta_2 * 1[regularization = data-rich]
           + beta_3 * 1[cv_scheme = K-fold]
           + beta_4 * 1[loss_function = L2]
           + epsilon_m
```

The intercept `alpha` captures the average OOS-R² of the linear AR baseline cell. Each `beta_k` is the average marginal contribution of switching that treatment from the reference category to the treatment category, averaging over all other dimensions. HC3 heteroskedasticity-consistent standard errors are used throughout; the small-sample correction is important because the number of model cells is typically modest (16 to 64 across the full grid).

---

## decompose_treatment_effects

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_df` | DataFrame | required | Tidy result DataFrame from `ResultSet.to_dataframe()` |
| `benchmark_model_id` | str | `"linear__none__bic__l2"` | Model ID of the AR benchmark used to compute OOS-R² |
| `horizon` | int or None | None | If provided, filter to this horizon before decomposing |

Returns a `DecompositionResult`.

---

## DecompositionResult

| Field | Type | Description |
|-------|------|-------------|
| `coef` | dict[str, float] | OLS coefficient for each component |
| `se` | dict[str, float] | HC3 standard error for each component |
| `t_stat` | dict[str, float] | t-statistic for each component |
| `r_squared` | float | R² of the decomposition regression |
| `n_models` | int | Number of model cells included in the regression |
| `summary_df` | DataFrame | Tidy summary with columns: `component`, `coef`, `se_hc3`, `t_stat` |

---

## Note on HC3 Standard Errors

HC3 (heteroskedasticity-consistent, small-sample correction) is the default. The design matrix `X'X` is often near-singular when the number of model cells is small relative to the four indicator dimensions. HC3 down-weights high-leverage observations and is preferred over HC0 in this setting (MacKinnon and White 1985). With 16 model cells and 4 regressors, degrees of freedom are limited; interpret t-statistics with care.

---

## Example

```python
from macrocast.evaluation import decompose_treatment_effects

decomp = decompose_treatment_effects(
    result_df=df,
    benchmark_model_id="linear__none__bic__l2",
    horizon=1,
)
print(decomp.r_squared)
print(decomp.summary_df)
#      component      coef  se_hc3  t_stat
# 0    intercept   -0.021    ...     ...
# 1  d_nonlinear    0.052    ...     ...
# 2  d_data_rich    0.081    ...     ...
# 3      d_kfold    0.019    ...     ...
# 4        d_l2    0.011    ...     ...
```

---

## Reference

Coulombe, P. G., Leroux, M., Stevanovic, D., and Surprenant, S. (2022). How is machine learning useful for macroeconomic forecasting? *Journal of Applied Econometrics*, 37(5), 920–964.
