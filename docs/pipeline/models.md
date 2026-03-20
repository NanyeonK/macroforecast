# Models

The Python model zoo contains seven nonlinear estimators. Linear regularized models (Ridge, LASSO, Adaptive LASSO, Group LASSO, Elastic Net, ARDI) run in the `macrocastR` companion package and share results via parquet. All Python models implement the `MacrocastEstimator` or `SequenceEstimator` abstract base class and declare a `nonlinearity_type` class attribute.

---

## Model Summary

| Model class | Nonlinearity | Underlying library | Key HP grid |
|-------------|-------------|-------------------|-------------|
| `KRRModel` | `KRR` | sklearn KernelRidge | alpha, gamma (log-spaced) |
| `SVRRBFModel` | `SVR_RBF` | sklearn SVR(kernel='rbf') | C, gamma, epsilon |
| `SVRLinearModel` | `SVR_LINEAR` | sklearn SVR(kernel='linear') | C, epsilon |
| `RFModel` | `RANDOM_FOREST` | sklearn RandomForestRegressor | n_estimators, max_features |
| `XGBoostModel` | `XGBOOST` | xgboost XGBRegressor | n_estimators, max_depth, learning_rate |
| `NNModel` | `NEURAL_NET` | PyTorch | hidden_size, n_layers, lr, n_epochs |
| `LSTMModel` | `LSTM` | PyTorch (SequenceEstimator) | hidden_size, n_layers, lr, n_epochs |

---

## Python / R Split

FACTORS/ARDI regularization and all linear penalized models run in macrocastR and deposit parquet files under `~/.macrocast/results/{experiment_id}/`. Group LASSO uses FRED variable groups (output_income, labor, housing, prices, money, interest_rates, stock_market) as the group structure, following McCracken and Ng (2016). `ResultSet` reads both Python and R parquet outputs into a single unified frame.

---

## KRR vs SVR-RBF Pairing

`KRRModel` and `SVRRBFModel` share the same RBF kernel and the same predictor matrix Z_t. The sole difference is the training objective: L2 loss for KRR and epsilon-insensitive loss for SVR-RBF. Any difference in OOS performance between the two identifies the `LossFunction` treatment effect in the CLSS 2022 decomposition.

---

## MacrocastEstimator ABC

Any custom estimator that implements `.fit(X, y)` and `.predict(X)` and declares a `nonlinearity_type` class attribute can be plugged into `ForecastExperiment` via `ModelSpec`. The ABC enforces the interface.

```python
from macrocast.pipeline import MacrocastEstimator, Nonlinearity
import numpy as np

class MeanModel(MacrocastEstimator):
    nonlinearity_type = Nonlinearity.LINEAR

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanModel":
        self._mean = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean)
```

---

## SequenceEstimator ABC

`LSTMModel` implements `SequenceEstimator`, which extends `MacrocastEstimator` with a three-dimensional input contract. The feature matrix passed to `.fit` and `.predict` has shape (T, L, N) where L is the lookback window and N is the number of features at each time step. `ForecastExperiment` automatically reshapes Z_t into this format for `SequenceEstimator` subclasses using the `lookback` field of `FeatureSpec`.

---

## KRRModel

Wraps `sklearn.kernel_approximation` + `sklearn.linear_model.Ridge` under the hood. Hyperparameters `alpha` (regularization strength) and `gamma` (RBF bandwidth) are selected by the active `CVScheme`. Log-spaced grids are used for both.

```python
from macrocast.pipeline import KRRModel

model = KRRModel()
model.fit(Z_train, y_train)
y_hat = model.predict(Z_test)
```

---

## LSTMModel

A two-layer LSTM with a linear output head. Input shape (T, L, N). Trained with Adam and early stopping against a held-out validation slice drawn from the end of the training window.

```python
from macrocast.pipeline import LSTMModel

model = LSTMModel(hidden_size=64, n_layers=2, lr=1e-3, n_epochs=100)
# X_seq.shape == (T, lookback, n_features)
model.fit(X_seq, y_train)
y_hat = model.predict(X_test_seq)
```

Note that `LSTMModel` requires `FeatureSpec.lookback` to be set in the experiment configuration. The default is 12 months.

---

## NNModel

A feedforward network with 1 or 2 ReLU hidden layers. Trained with Adam. Hyperparameters `hidden_size`, `n_layers`, `lr`, and `n_epochs` are tuned by the active CVScheme. Input is the standard Z_t matrix (T, n_features); no reshaping is required.
