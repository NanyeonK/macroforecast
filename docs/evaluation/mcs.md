# Model Confidence Set

`macrocast.evaluation.mcs` implements the Model Confidence Set (MCS) of Hansen, Lunde, and Nason (2011). The MCS is the set of models that contains the best model with a pre-specified confidence level `1 - alpha`, given the data. It is a sequential testing procedure that eliminates models one at a time based on pairwise loss differentials.

**Reference:** Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The Model Confidence Set. *Econometrica*, 79(2), 453–497.

---

## Algorithm

The MCS algorithm proceeds as follows:

1. Compute pairwise loss differentials `d_{ij,t} = L(e_{i,t}) - L(e_{j,t})` for all model pairs (i, j).
2. Compute the MCS test statistic `T_max = max_{i,j} t_{ij}`, where `t_{ij}` is the studentized mean of `d_{ij,t}` using Newey-West HAC standard errors.
3. Obtain the null distribution of `T_max` via stationary block bootstrap (Politis and Romano 1994).
4. If the null hypothesis of equal predictive ability is rejected at level `alpha`, eliminate the model with the largest average loss differential from the candidate set.
5. Repeat steps 1–4 on the reduced candidate set until the null is no longer rejected. The surviving models form the MCS.

---

## mcs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_df` | DataFrame | required | Tidy loss DataFrame with one row per (model, date) |
| `alpha` | float | `0.10` | Significance level for the elimination test |
| `block_size` | int | `12` | Block length for the stationary block bootstrap |
| `n_bootstrap` | int | `1000` | Number of bootstrap replications |
| `loss_col` | str | `"squared_error"` | Column name for the per-period loss values |
| `model_col` | str | `"model_id"` | Column name for the model identifier |
| `date_col` | str | `"forecast_date"` | Column name for the forecast date |
| `seed` | int | `42` | Random seed for reproducibility |

Returns an `MCSResult`.

---

## MCSResult

| Field | Type | Description |
|-------|------|-------------|
| `included` | list[str] | Model IDs in the MCS |
| `excluded` | list[str] | Model IDs eliminated during the procedure |
| `p_values` | dict[str, float] | P-value for each model; models surviving to the end receive `p_value=1.0` |
| `mcs_alpha` | float | The `alpha` level used |

---

## Note on block_size

For monthly data, `block_size=12` (one calendar year of blocks) is recommended. This choice captures the serial correlation in forecast errors that typically persists at the annual frequency for macroeconomic variables. For quarterly data, `block_size=4` is a natural default.

---

## Example

```python
from macrocast.evaluation import mcs

df["squared_error"] = (df["y_true"] - df["y_hat"]) ** 2
result = mcs(
    df[["model_id", "forecast_date", "squared_error"]],
    alpha=0.10,
    block_size=12,
    n_bootstrap=1000,
)
print("In MCS:", result.included)
print("Excluded:", result.excluded)
print("P-values:", result.p_values)
```
