# Volatility and GARCH

[Back to Models and Features](../model_overview.md)

Volatility models forecast conditional variance rather than the level, using the GARCH family and its asymmetric and realized extensions.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `egarch` | EGARCH volatility model. | volatility | `arch` | no | default | 4 |
| `garch11` | GARCH volatility model. | volatility | `arch` | no | default | 3 |
| `gjr_garch` | GJR-GARCH asymmetric volatility model. | volatility | `arch` | no | default | 4 |
| `realized_garch` | Compact realized GARCH volatility model. | volatility | none | no | default | 1 |
| `tgarch` | Threshold GARCH (TGARCH/Zakoian) volatility model. | volatility | `arch` | no | default | 4 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
