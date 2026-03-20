# Regime-Conditional Evaluation

`macrocast.evaluation.regime` evaluates forecast accuracy conditional on economic regimes, following the approach in Coulombe, Leroux, Stevanovic, and Surprenant (2022), Table 5. The OOS period is partitioned into regimes defined by quantile bins of a continuous regime indicator (e.g., VXO for uncertainty) or by the 0/1 values of a binary indicator (e.g., USREC for NBER recession dates). Relative MSFE is then computed separately within each regime.

---

## regime_conditional_msfe

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_df` | DataFrame | required | Tidy result DataFrame from `ResultSet.to_dataframe()` |
| `regime_series` | pd.Series | required | Series indexed by date; values define the regime variable |
| `n_quantiles` | int | `3` | Number of quantile bins for continuous indicators; ignored for binary indicators |
| `regime_labels` | list[str] or None | None | Labels for the regime bins; defaults to `["Q1", "Q2", ..., "Qn"]` |
| `benchmark_model_id` | str | `"linear__none__bic__l2"` | Model ID of the AR benchmark |
| `horizon` | int or None | None | If provided, filter to this horizon before splitting |
| `date_col` | str | `"forecast_date"` | Column name for the forecast date in `result_df` |
| `model_col` | str | `"model_id"` | Column name for the model identifier in `result_df` |

Returns a `RegimeResult`.

---

## RegimeResult

| Field | Type | Description |
|-------|------|-------------|
| `regime_labels` | list[str] | Ordered list of regime names |
| `msfe_by_regime` | dict[str, dict[str, float]] | Raw MSFE per model per regime: `{model_id: {regime: msfe}}` |
| `relative_msfe_by_regime` | dict[str, dict[str, float]] | Relative MSFE (vs benchmark) per model per regime |
| `n_obs_by_regime` | dict[str, int] | Number of forecast dates in each regime |
| `summary_df` | DataFrame | Tidy summary with columns: `model_id`, `regime`, `msfe`, `relative_msfe` |

---

## Binary vs Quantile Splitting

When `regime_series` contains only values 0 and 1, the function automatically splits into two regimes regardless of `n_quantiles`. The default labels in this case are `["expansion", "recession"]`, which align with the NBER USREC convention. For any other binary indicator, supply explicit `regime_labels`.

For continuous indicators (e.g., VXO, term spread), the function assigns each date to one of `n_quantiles` bins based on the unconditional quantiles of the regime series over the full OOS window.

---

## Example

```python
from macrocast.evaluation import regime_conditional_msfe

# Load VXO from FRED-MD (already in the MacroFrame)
vxo = md_t.data["VXOCLSX"]

result = regime_conditional_msfe(
    result_df=df,
    regime_series=vxo,
    n_quantiles=3,
    regime_labels=["low_uncertainty", "medium_uncertainty", "high_uncertainty"],
    benchmark_model_id="linear__none__bic__l2",
    horizon=1,
)
print(result.summary_df)
```
