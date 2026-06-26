# Mixed frequency

[Back to Models and Features](../model_overview.md)

Mixed-frequency models combine series sampled at different frequencies, for example monthly predictors for a quarterly target, through MIDAS and dynamic factor designs.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `dfm_mixed_mariano_murasawa` | Mixed-frequency dynamic factor model using Mariano-Murasawa quarterly aggregation. | panel | none | no | pass a native mixed monthly/quarterly panel from macroforecast.data.combine(..., frequency='native'), keep quarterly flow variables on their observed quarterly dates; the model applies Mariano-Murasawa aggregation | 2 |
| `dfm_unrestricted_midas` | Composite DynamicFactorMQ factors plus unrestricted MIDAS forecast head. | panel | none | no | pass a native mixed monthly/quarterly panel with column-level frequency metadata, use feature_engineering.mixed_frequency_lags directly when you need full manual control | 5 |
| `midas_almon` | Fixed-shape MIDAS over lag groups using midasr::nealmon-style normalized exponential Almon weights. | supervised | none | no | default | 1 |
| `midas_beta` | Fixed-shape MIDAS over lag groups using midasr::nbetaMT-style beta weights. | supervised | none | no | default | 2 |
| `midas_step` | Fixed-shape MIDAS over lag groups using normalized midasr::polystep-style step weights. | supervised | none | no | default | 2 |
| `restricted_midas` | midasr::midas_r-style nonlinear restricted MIDAS over explicit lag columns. | supervised | none | no | default | 0 |
| `unrestricted_midas` | Unrestricted MIDAS over explicit lag columns. | supervised | none | no | default | 1 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
