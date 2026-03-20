# Pipeline Layer

Layer 2 of macrocast implements the four-component decomposition framework of Coulombe, Leroux, Stevanovic, and Surprenant (2022, JBES). It takes a stationary-transformed MacroFrame and a model grid, runs a pseudo-OOS evaluation loop, and returns a ResultSet.

---

## Architecture

The pipeline layer moves data through four stages:

```
MacroFrame  →  FeatureBuilder  →  ForecastExperiment  →  ResultSet
                                                              ↓
                                                     Evaluation Layer
```

`FeatureBuilder` constructs the predictor matrix Z_t from the transformed panel. `ForecastExperiment` iterates over all (model, horizon, date) triples in the pseudo-OOS window and collects `ForecastRecord` objects into a `ResultSet`. The `ResultSet` passes downstream to the evaluation layer for MSFE computation, relative MSFE ratios, decomposition regressions, and MCS.

---

## Modules

| Module | Key exports |
|--------|-------------|
| `macrocast.pipeline.components` | `Nonlinearity`, `Regularization`, `CVScheme`, `LossFunction`, `Window` |
| `macrocast.pipeline.estimator` | `MacrocastEstimator`, `SequenceEstimator` |
| `macrocast.pipeline.features` | `FeatureBuilder` |
| `macrocast.pipeline.models` | `KRRModel`, `SVRRBFModel`, `SVRLinearModel`, `RFModel`, `XGBoostModel`, `NNModel`, `LSTMModel` |
| `macrocast.pipeline.experiment` | `ForecastExperiment`, `ModelSpec`, `FeatureSpec` |
| `macrocast.pipeline.results` | `ForecastRecord`, `ResultSet` |

---

## Four-Component Decomposition Framework

The framework follows the design of Coulombe et al. (2022). Each forecast model is characterised by four treatment dimensions:

1. **Nonlinearity** — whether the model is linear in parameters or employs a nonlinear function class (kernel, tree, neural network).
2. **Regularization** — the penalisation or dimension-reduction strategy applied to the predictor matrix.
3. **CV scheme** — the method used to select hyperparameters during training (BIC, pseudo-OOS CV, K-fold CV).
4. **Loss function** — the objective minimised during estimation (L2, epsilon-insensitive).

The experiment runs all factorial combinations defined in the model grid. OOS-R² gains relative to the AR benchmark are then regressed on component indicator dummies to isolate the marginal contribution of each treatment effect. This decomposition is the primary output of the paper.

---

## Python / R Split

Linear regularized models — Ridge, LASSO, Adaptive LASSO, Group LASSO, Elastic Net, and ARDI — run in the `macrocastR` companion package and exchange results via parquet files under `~/.macrocast/results/{experiment_id}/`. Nonlinear models (KRR, SVR, RF, XGBoost, NN, LSTM) run in Python. The `ResultSet` class reads parquet files from both sides into a single unified DataFrame for evaluation.

Group LASSO uses FRED variable groups (output_income, labor, housing, prices, money, interest_rates, stock_market) as the group structure, following McCracken and Ng (2016).

---

## Sub-pages

- [Components](components.md) — enum definitions for all four treatment dimensions
- [Features](features.md) — FeatureBuilder: PCA diffusion index and AR lag construction
- [Models](models.md) — Python model zoo and MacrocastEstimator interface
- [Experiment](experiment.md) — ForecastExperiment orchestrator and ModelSpec/FeatureSpec
- [Results](results.md) — ForecastRecord and ResultSet data structures
