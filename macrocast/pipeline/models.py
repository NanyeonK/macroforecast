"""Python-side forecasting models for the macrocast pipeline.

All models implement either MacrocastEstimator (cross-sectional) or
SequenceEstimator (LSTM).  Hyperparameter grids are defined as class attributes
so the experiment runner can build CV search spaces without inspecting the model
internals.

Model list (Python side, nonlinear):
  - KRRModel     -- Kernel Ridge Regression, RBF kernel (sklearn)
  - SVRRBFModel  -- Support Vector Regression, RBF kernel (sklearn)
  - SVRLinearModel -- Support Vector Regression, linear kernel (sklearn)
  - RFModel      -- Random Forest (sklearn)
  - XGBoostModel -- Gradient Boosted Trees (xgboost)
  - GBModel      -- Gradient Boosted Trees (sklearn GradientBoostingRegressor)
  - NNModel      -- Feedforward neural network, ReLU, 1-2 hidden layers (pytorch)
  - LSTMModel    -- LSTM sequence model (pytorch)

KRR vs SVR-RBF: same kernel, different loss (L2 vs ε-insensitive).
This pair isolates the LossFunction treatment effect in CLSS 2022.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

    class _DummyModule:
        """Placeholder so nn.Module subclasses parse without torch installed."""
        class Module:
            pass

    class _DummyTorch:
        class Tensor:
            pass

    nn = _DummyModule()  # type: ignore[assignment]
    torch = _DummyTorch()  # type: ignore[assignment]
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor

    _RF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RF_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor

from macrocast.pipeline.components import Nonlinearity
from macrocast.pipeline.estimator import MacrocastEstimator, SequenceEstimator

# ---------------------------------------------------------------------------
# Utility: fit a sklearn estimator with optional cross-validation
# ---------------------------------------------------------------------------


def _fit_with_cv(
    estimator,
    param_grid: dict[str, list[Any]],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
) -> Any:
    """Wrap estimator in GridSearchCV if param_grid is non-empty.

    Shortcut: if every param list is a singleton, skip GridSearchCV entirely
    and just set the params directly.  This avoids the 5-fold CV overhead
    when a fixed hyperparameter value is desired (e.g. min_samples_leaf=5
    as in CLSS 2021).
    """
    if not param_grid:
        return estimator.fit(X, y)
    # Singleton shortcut: all lists have exactly one candidate → no CV needed
    if all(len(v) == 1 for v in param_grid.values()):
        single_params = {k: v[0] for k, v in param_grid.items()}
        estimator.set_params(**single_params)
        return estimator.fit(X, y)
    gs = GridSearchCV(
        estimator,
        param_grid,
        cv=KFold(n_splits=cv, shuffle=False),
        scoring=scoring,
        refit=True,
        n_jobs=1,
    )
    gs.fit(X, y)
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# KRR — Kernel Ridge Regression (RBF)
# ---------------------------------------------------------------------------


class KRRModel(MacrocastEstimator):
    """Kernel Ridge Regression with RBF kernel.

    Loss function: L2 (squared loss).  Comparing this model against SVRRBFModel
    (same kernel, ε-insensitive loss) isolates the LossFunction treatment effect.

    Parameters
    ----------
    alpha_grid : list of float
        Regularization strengths to search over in CV.
    gamma_grid : list of float
        RBF bandwidth parameters to search over in CV.
    cv_folds : int
        K-fold splits for HP selection.
    """

    nonlinearity_type = Nonlinearity.KRR

    _DEFAULT_ALPHA_GRID: list[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    _DEFAULT_GAMMA_GRID: list[float] = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    def __init__(
        self,
        alpha_grid: list[float] | None = None,
        gamma_grid: list[float] | None = None,
        cv_folds: int = 5,
    ) -> None:
        self.alpha_grid = alpha_grid or self._DEFAULT_ALPHA_GRID
        self.gamma_grid = gamma_grid or self._DEFAULT_GAMMA_GRID
        self.cv_folds = cv_folds
        self._estimator: KernelRidge | None = None
        self._scaler = StandardScaler()

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> KRRModel:
        X_scaled = self._scaler.fit_transform(X)
        param_grid = {"alpha": self.alpha_grid, "gamma": self.gamma_grid}
        base = KernelRidge(kernel="rbf")
        self._estimator = _fit_with_cv(base, param_grid, X_scaled, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_scaled = self._scaler.transform(X)
        return self._estimator.predict(X_scaled)

    @property
    def best_params(self) -> dict[str, Any]:
        if self._estimator is None:
            raise RuntimeError("Model not fitted.")
        return {"alpha": self._estimator.alpha, "gamma": self._estimator.gamma}


# ---------------------------------------------------------------------------
# SVR-RBF — Support Vector Regression, RBF kernel
# ---------------------------------------------------------------------------


class SVRRBFModel(MacrocastEstimator):
    """Support Vector Regression with RBF kernel (ε-insensitive loss).

    Pairing with KRRModel isolates the LossFunction treatment effect:
    both use RBF kernel; KRR minimises L2, SVR-RBF minimises ε-insensitive.

    Parameters
    ----------
    C_grid : list of float
        Regularization parameter C (inverse of penalty strength).
    gamma_grid : list of float
        RBF bandwidth parameters.
    epsilon_grid : list of float
        Width of the ε-insensitive tube.
    cv_folds : int
        K-fold splits for HP selection.
    """

    nonlinearity_type = Nonlinearity.SVR_RBF

    _DEFAULT_C_GRID: list[float] = [0.1, 1.0, 10.0, 100.0]
    _DEFAULT_GAMMA_GRID: list[float] = [1e-3, 1e-2, 1e-1, 1.0]
    _DEFAULT_EPSILON_GRID: list[float] = [0.01, 0.05, 0.1, 0.5]

    def __init__(
        self,
        C_grid: list[float] | None = None,
        gamma_grid: list[float] | None = None,
        epsilon_grid: list[float] | None = None,
        cv_folds: int = 5,
    ) -> None:
        self.C_grid = C_grid or self._DEFAULT_C_GRID
        self.gamma_grid = gamma_grid or self._DEFAULT_GAMMA_GRID
        self.epsilon_grid = epsilon_grid or self._DEFAULT_EPSILON_GRID
        self.cv_folds = cv_folds
        self._estimator: SVR | None = None
        self._scaler = StandardScaler()

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> SVRRBFModel:
        X_scaled = self._scaler.fit_transform(X)
        param_grid = {
            "C": self.C_grid,
            "gamma": self.gamma_grid,
            "epsilon": self.epsilon_grid,
        }
        base = SVR(kernel="rbf")
        self._estimator = _fit_with_cv(base, param_grid, X_scaled, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_scaled = self._scaler.transform(X)
        return self._estimator.predict(X_scaled)


# ---------------------------------------------------------------------------
# SVR-Linear — Support Vector Regression, linear kernel
# ---------------------------------------------------------------------------


class SVRLinearModel(MacrocastEstimator):
    """Support Vector Regression with linear kernel.

    Provides the LINEAR nonlinearity type with ε-insensitive loss.
    Compared against Ridge (LINEAR + L2) to isolate loss function effect
    in the linear regime.

    Parameters
    ----------
    C_grid : list of float
        Regularization parameter.
    epsilon_grid : list of float
        ε-tube width.
    cv_folds : int
        K-fold splits for HP selection.
    """

    nonlinearity_type = Nonlinearity.SVR_LINEAR

    _DEFAULT_C_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
    _DEFAULT_EPSILON_GRID: list[float] = [0.01, 0.05, 0.1, 0.5]

    def __init__(
        self,
        C_grid: list[float] | None = None,
        epsilon_grid: list[float] | None = None,
        cv_folds: int = 5,
    ) -> None:
        self.C_grid = C_grid or self._DEFAULT_C_GRID
        self.epsilon_grid = epsilon_grid or self._DEFAULT_EPSILON_GRID
        self.cv_folds = cv_folds
        self._estimator: SVR | None = None
        self._scaler = StandardScaler()

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> SVRLinearModel:
        X_scaled = self._scaler.fit_transform(X)
        param_grid = {"C": self.C_grid, "epsilon": self.epsilon_grid}
        base = SVR(kernel="linear")
        self._estimator = _fit_with_cv(base, param_grid, X_scaled, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_scaled = self._scaler.transform(X)
        return self._estimator.predict(X_scaled)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RFModel(MacrocastEstimator):
    """Random Forest regressor (sklearn), calibrated to CLSS 2021 defaults.

    Tree-based model; nonlinear in inputs.  No kernel scaling required.
    Defaults match the ranger-based RF in Coulombe et al. (2021):
    fully grown trees, mtry=p/3, 75% subsampling without replacement.
    Only min_samples_leaf is tuned via K-fold CV.

    Parameters
    ----------
    n_estimators : int
        Number of trees.  500 matches CLSS 2021.
    min_samples_leaf_grid : list of int
        Minimum leaf size candidates for K-fold tuning.
    max_features : float or str
        Feature fraction per split.  1/3 matches ranger mtry=floor(p/3).
    max_samples : float
        Fraction of training samples per tree (bootstrap with replacement).
        0.75 approximates ranger's default 75% subsample rate.  sklearn does
        not support subsampling without replacement, so bootstrap=True is used
        with max_samples to control the effective subsample size.
    cv_folds : int
        K-fold splits for HP selection.
    """

    nonlinearity_type = Nonlinearity.RANDOM_FOREST

    _DEFAULT_MIN_SAMPLES_LEAF_GRID: list[int] = [5, 10, 20]

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf_grid: list[int] | None = None,
        max_features: float | str = 1 / 3,
        max_samples: float = 0.75,
        cv_folds: int = 5,
        rf_n_jobs: int = -1,
    ) -> None:
        if not _RF_AVAILABLE:  # pragma: no cover
            raise ImportError("scikit-learn RandomForestRegressor not available.")
        self.n_estimators = n_estimators
        self.min_samples_leaf_grid = (
            min_samples_leaf_grid or self._DEFAULT_MIN_SAMPLES_LEAF_GRID
        )
        self.max_features = max_features
        self.max_samples = max_samples
        self.cv_folds = cv_folds
        self._rf_n_jobs = rf_n_jobs
        self._estimator = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> RFModel:
        param_grid = {"min_samples_leaf": self.min_samples_leaf_grid}
        base = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=None,
            max_features=self.max_features,
            bootstrap=True,
            max_samples=self.max_samples,
            n_jobs=self._rf_n_jobs,
            random_state=42,
        )
        self._estimator = _fit_with_cv(base, param_grid, X, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._estimator.predict(X)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostModel(MacrocastEstimator):
    """Gradient boosted trees via XGBoost.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds.
    max_depth_grid : list of int
        Max depth candidates.
    learning_rate_grid : list of float
        Step-size shrinkage candidates.
    subsample_grid : list of float
        Row subsampling ratio candidates.
    cv_folds : int
        K-fold splits for HP selection.
    """

    nonlinearity_type = Nonlinearity.XGBOOST

    _DEFAULT_MAX_DEPTH_GRID: list[int] = [3, 5, 7]
    _DEFAULT_LR_GRID: list[float] = [0.01, 0.05, 0.1]
    _DEFAULT_SUBSAMPLE_GRID: list[float] = [0.7, 0.9, 1.0]

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth_grid: list[int] | None = None,
        learning_rate_grid: list[float] | None = None,
        subsample_grid: list[float] | None = None,
        cv_folds: int = 5,
    ) -> None:
        if not _XGBOOST_AVAILABLE:  # pragma: no cover
            raise ImportError("xgboost package not installed.")
        self.n_estimators = n_estimators
        self.max_depth_grid = max_depth_grid or self._DEFAULT_MAX_DEPTH_GRID
        self.learning_rate_grid = learning_rate_grid or self._DEFAULT_LR_GRID
        self.subsample_grid = subsample_grid or self._DEFAULT_SUBSAMPLE_GRID
        self.cv_folds = cv_folds
        self._estimator = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> XGBoostModel:
        param_grid = {
            "max_depth": self.max_depth_grid,
            "learning_rate": self.learning_rate_grid,
            "subsample": self.subsample_grid,
        }
        base = XGBRegressor(
            n_estimators=self.n_estimators,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        self._estimator = _fit_with_cv(base, param_grid, X, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._estimator.predict(X)


# ---------------------------------------------------------------------------
# Gradient Boosting (sklearn)
# ---------------------------------------------------------------------------


class GBModel(MacrocastEstimator):
    """Gradient boosted trees via scikit-learn GradientBoostingRegressor.

    Uses sklearn's native implementation rather than xgboost.  This matches
    the "gradient boosting" model in Coulombe et al. (2021) which is a
    standard sklearn tree ensemble, not XGBoost.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting stages.  Default 500.
    max_depth_grid : list of int
        Max tree depth candidates.  Default [3, 5, 7].
    learning_rate_grid : list of float
        Step-size shrinkage candidates.  Default [0.01, 0.05, 0.1].
    min_samples_leaf_grid : list of int
        Min leaf samples candidates.  Default [1, 5, 10].
    subsample_grid : list of float
        Row subsampling fraction candidates.  Default [0.7, 0.9, 1.0].
    cv_folds : int
        K-fold splits for HP selection.  Default 5.
    """

    nonlinearity_type = Nonlinearity.GRADIENT_BOOSTING

    _DEFAULT_MAX_DEPTH_GRID: list[int] = [3, 5, 7]
    _DEFAULT_LR_GRID: list[float] = [0.01, 0.05, 0.1]
    _DEFAULT_MIN_SAMPLES_LEAF_GRID: list[int] = [1, 5, 10]
    _DEFAULT_SUBSAMPLE_GRID: list[float] = [0.7, 0.9, 1.0]

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth_grid: list[int] | None = None,
        learning_rate_grid: list[float] | None = None,
        min_samples_leaf_grid: list[int] | None = None,
        subsample_grid: list[float] | None = None,
        cv_folds: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth_grid = max_depth_grid or self._DEFAULT_MAX_DEPTH_GRID
        self.learning_rate_grid = learning_rate_grid or self._DEFAULT_LR_GRID
        self.min_samples_leaf_grid = (
            min_samples_leaf_grid or self._DEFAULT_MIN_SAMPLES_LEAF_GRID
        )
        self.subsample_grid = subsample_grid or self._DEFAULT_SUBSAMPLE_GRID
        self.cv_folds = cv_folds
        self._estimator = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> GBModel:
        param_grid = {
            "max_depth": self.max_depth_grid,
            "learning_rate": self.learning_rate_grid,
            "min_samples_leaf": self.min_samples_leaf_grid,
            "subsample": self.subsample_grid,
        }
        base = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            random_state=42,
        )
        self._estimator = _fit_with_cv(base, param_grid, X, y, cv=self.cv_folds)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._estimator.predict(X)


# ---------------------------------------------------------------------------
# Feedforward Neural Network (pytorch)
# ---------------------------------------------------------------------------


class _FFNN(nn.Module):
    """Feedforward network with 1 or 2 hidden layers and ReLU activations."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend(
                [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NNModel(MacrocastEstimator):
    """Feedforward neural network with 1-2 ReLU hidden layers (pytorch).

    Hyperparameters (hidden_dim, n_layers, lr, dropout) are tuned by simple
    hold-out validation (last 20% of training window) rather than K-fold to
    keep training cost tractable.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden layer width candidates.
    n_layers_options : list of int
        Number of hidden layers (1 or 2).
    lr_options : list of float
        Adam learning rate candidates.
    dropout_options : list of float
        Dropout rate candidates.
    max_epochs : int
        Maximum training epochs per configuration.
    patience : int
        Early stopping patience (epochs without validation improvement).
    batch_size : int
        Mini-batch size.
    device : str or None
        Pytorch device string.  Auto-detected if None.
    """

    nonlinearity_type = Nonlinearity.NEURAL_NET

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        n_layers_options: list[int] | None = None,
        lr_options: list[float] | None = None,
        dropout_options: list[float] | None = None,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("NNModel requires torch. Install with: pip install torch")
        self.hidden_dims = hidden_dims or [64, 128, 256]
        self.n_layers_options = n_layers_options or [1, 2]
        self.lr_options = lr_options or [1e-3, 5e-4, 1e-4]
        self.dropout_options = dropout_options or [0.0, 0.1, 0.3]
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: _FFNN | None = None
        self._scaler = StandardScaler()
        self.best_params_: dict[str, Any] = {}

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> NNModel:
        X_scaled = self._scaler.fit_transform(X)
        # Hold-out last 20% for HP search
        val_size = max(1, int(0.2 * len(X_scaled)))
        X_tr, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_tr, y_val = y[:-val_size], y[-val_size:]

        best_val_loss = float("inf")
        best_cfg: tuple = (
            self.hidden_dims[0],
            self.n_layers_options[0],
            self.lr_options[0],
            self.dropout_options[0],
        )

        # Grid search over HP combinations
        for h_dim in self.hidden_dims:
            for n_lay in self.n_layers_options:
                for lr in self.lr_options:
                    for drop in self.dropout_options:
                        val_loss = self._train_and_evaluate(
                            X_tr,
                            y_tr,
                            X_val,
                            y_val,
                            input_dim=X_scaled.shape[1],
                            hidden_dim=h_dim,
                            n_layers=n_lay,
                            lr=lr,
                            dropout=drop,
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_cfg = (h_dim, n_lay, lr, drop)

        # Refit on full training data with best HP
        h_dim, n_lay, lr, drop = best_cfg
        self.best_params_ = {
            "hidden_dim": h_dim,
            "n_layers": n_lay,
            "lr": lr,
            "dropout": drop,
        }
        self._model = self._train(
            X_scaled,
            y,
            X_scaled,
            y,
            input_dim=X_scaled.shape[1],
            hidden_dim=h_dim,
            n_layers=n_lay,
            lr=lr,
            dropout=drop,
            full_train=True,
        )
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            y_hat = self._model(X_t).cpu().numpy()
        return y_hat

    def _train_and_evaluate(
        self, X_tr, y_tr, X_val, y_val, input_dim, hidden_dim, n_layers, lr, dropout
    ) -> float:
        model = self._train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            lr=lr,
            dropout=dropout,
            full_train=False,
        )
        model.eval()
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            loss = nn.MSELoss()(model(X_v), y_v).item()
        return loss

    def _train(
        self,
        X_tr,
        y_tr,
        X_val,
        y_val,
        input_dim,
        hidden_dim,
        n_layers,
        lr,
        dropout,
        full_train: bool = False,
    ) -> _FFNN:
        model = _FFNN(input_dim, hidden_dim, n_layers, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_tr, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=self.device)
        if not full_train:
            X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_val = float("inf")
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _epoch in range(self.max_epochs):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            if not full_train:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_v), y_v).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model


# ---------------------------------------------------------------------------
# LSTM (pytorch)
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Single-layer LSTM followed by a linear readout."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Use the last time step's hidden state for prediction
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(SequenceEstimator):
    """LSTM sequence model for direct multi-step forecasting.

    Input tensor X has shape (T, L, N) where L is the sequence look-back
    length (a tuning parameter) and N is the number of features per step.
    The target y_{t+h} is aligned with the last step of each window.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden state size candidates.
    n_layers_options : list of int
        Number of stacked LSTM layers.
    lr_options : list of float
        Adam learning rate candidates.
    dropout_options : list of float
        Dropout rate between LSTM layers.
    max_epochs : int
        Maximum training epochs per configuration.
    patience : int
        Early stopping patience.
    batch_size : int
        Mini-batch size.
    device : str or None
        Pytorch device.  Auto-detected if None.
    """

    nonlinearity_type = Nonlinearity.LSTM

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        n_layers_options: list[int] | None = None,
        lr_options: list[float] | None = None,
        dropout_options: list[float] | None = None,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("LSTMModel requires torch. Install with: pip install torch")
        self.hidden_dims = hidden_dims or [32, 64, 128]
        self.n_layers_options = n_layers_options or [1, 2]
        self.lr_options = lr_options or [1e-3, 5e-4]
        self.dropout_options = dropout_options or [0.0, 0.1]
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: _LSTMNet | None = None
        self._scaler = StandardScaler()  # applied per feature across T
        self.best_params_: dict[str, Any] = {}

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> LSTMModel:
        """Fit LSTM on training sequences.

        Parameters
        ----------
        X : array of shape (T, L, N)
        y : array of shape (T,)
        """
        T, L, N = X.shape
        # Scale per feature: reshape to (T*L, N), fit, reshape back
        X_flat = X.reshape(-1, N)
        X_flat_scaled = self._scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(T, L, N)

        # Hold-out last 20% for HP search
        val_size = max(1, int(0.2 * T))
        X_tr, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_tr, y_val = y[:-val_size], y[-val_size:]

        best_val_loss = float("inf")
        best_cfg: tuple = (
            self.hidden_dims[0],
            self.n_layers_options[0],
            self.lr_options[0],
            self.dropout_options[0],
        )

        for h_dim in self.hidden_dims:
            for n_lay in self.n_layers_options:
                for lr in self.lr_options:
                    for drop in self.dropout_options:
                        val_loss = self._train_and_evaluate(
                            X_tr,
                            y_tr,
                            X_val,
                            y_val,
                            input_dim=N,
                            hidden_dim=h_dim,
                            n_layers=n_lay,
                            lr=lr,
                            dropout=drop,
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_cfg = (h_dim, n_lay, lr, drop)

        h_dim, n_lay, lr, drop = best_cfg
        self.best_params_ = {
            "hidden_dim": h_dim,
            "n_layers": n_lay,
            "lr": lr,
            "dropout": drop,
        }
        self._model = self._train(
            X_scaled,
            y,
            X_scaled,
            y,
            input_dim=N,
            hidden_dim=h_dim,
            n_layers=n_lay,
            lr=lr,
            dropout=drop,
            full_train=True,
        )
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        T, L, N = X.shape
        X_flat = X.reshape(-1, N)
        X_flat_scaled = self._scaler.transform(X_flat)
        X_scaled = X_flat_scaled.reshape(T, L, N)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            y_hat = self._model(X_t).cpu().numpy()
        return y_hat

    def _train_and_evaluate(
        self, X_tr, y_tr, X_val, y_val, input_dim, hidden_dim, n_layers, lr, dropout
    ) -> float:
        model = self._train(
            X_tr,
            y_tr,
            X_val,
            y_val,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            lr=lr,
            dropout=dropout,
            full_train=False,
        )
        model.eval()
        X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            loss = nn.MSELoss()(model(X_v), y_v).item()
        return loss

    def _train(
        self,
        X_tr,
        y_tr,
        X_val,
        y_val,
        input_dim,
        hidden_dim,
        n_layers,
        lr,
        dropout,
        full_train: bool = False,
    ) -> _LSTMNet:
        model = _LSTMNet(input_dim, hidden_dim, n_layers, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_tr, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=self.device)
        if not full_train:
            X_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_val = float("inf")
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _epoch in range(self.max_epochs):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            if not full_train:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_v), y_v).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model
