from __future__ import annotations

from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, optional_import

RecurrentKind = Literal["lstm", "gru", "transformer"]
TorchDevice = Literal["auto", "cpu", "cuda"]
TorchOptimizer = Literal["adam", "sgd", "rmsprop"]
TorchLoss = Literal["mse", "huber"]


def nn(
    X: Any,
    y: Any | None = None,
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    activation: str = "relu",
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    optimizer: TorchOptimizer = "adam",
    loss: TorchLoss = "mse",
    random_state: int = 0,
    device: TorchDevice = "auto",
) -> ModelFit:
    """Fit a torch-backed feed-forward neural-network regressor."""

    params = {
        "hidden_layer_sizes": tuple(int(value) for value in hidden_layer_sizes),
        "activation": str(activation),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "optimizer": optimizer,
        "loss": loss,
        "random_state": int(random_state),
        "device": device,
    }
    return fit_estimator(
        _TorchNNRegressor(
            hidden_layer_sizes=tuple(int(value) for value in hidden_layer_sizes),
            activation=str(activation),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            weight_decay=float(weight_decay),
            optimizer=optimizer,
            loss=loss,
            random_state=int(random_state),
            device=device,
        ),
        X,
        y,
        model="nn",
        metadata=params,
    )


class _TorchNNRegressor:
    """Torch-backed feed-forward regressor for tabular forecast matrices."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        optimizer: TorchOptimizer = "adam",
        loss: TorchLoss = "mse",
        random_state: int = 0,
        device: TorchDevice = "auto",
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if optimizer not in {"adam", "sgd", "rmsprop"}:
            raise ValueError("optimizer must be 'adam', 'sgd', or 'rmsprop'")
        if loss not in {"mse", "huber"}:
            raise ValueError("loss must be 'mse' or 'huber'")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.hidden_layer_sizes = tuple(
            max(1, int(value)) for value in hidden_layer_sizes
        )
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one layer")
        self.activation = str(activation)
        self.dropout = float(np.clip(dropout, 0.0, 0.95))
        self.learning_rate = float(learning_rate)
        self.max_epochs = max(1, int(max_epochs))
        self.batch_size = max(1, int(batch_size))
        self.weight_decay = max(0.0, float(weight_decay))
        self.optimizer = optimizer
        self.loss = loss
        self.random_state = int(random_state)
        self.device = device
        self.device_: str | None = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.model_: Any = None
        self.training_history_: dict[str, list[float]] = {"loss": []}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchNNRegressor":
        torch = optional_import("torch", extra="deep")
        device = _resolve_torch_device(torch, self.device)
        # Torch-native implementation: no R parity target. The model contract is
        # tabular X/y input, fit-window scaling, deterministic seeds, and
        # predictions mapped back to target units.
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        self.x_mean_, self.x_scale_, x_scaled = _standardize_matrix(values)
        y_values = target.to_numpy(dtype=float)
        self.y_mean_, self.y_scale_, y_scaled = _standardize_vector(y_values)

        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        model = _torch_nn_net(
            torch=torch,
            n_features=x_scaled.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            dropout=self.dropout,
        ).to(device)
        optimizer = _torch_optimizer(
            torch,
            self.optimizer,
            model.parameters(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = _torch_loss(torch, self.loss)
        tensor_x = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        tensor_y = torch.tensor(
            y_scaled.reshape(-1, 1), dtype=torch.float32, device=device
        )
        n_obs = tensor_x.shape[0]
        indices = np.arange(n_obs)
        rng = np.random.default_rng(self.random_state)
        model.train()
        for _ in range(self.max_epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            epoch_n = 0
            for start in range(0, n_obs, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = model(tensor_x[batch_idx])
                loss = loss_fn(pred, tensor_y[batch_idx])
                loss.backward()
                optimizer.step()
                batch_n = int(len(batch_idx))
                epoch_loss += float(loss.detach().cpu().item()) * batch_n
                epoch_n += batch_n
            self.training_history_["loss"].append(epoch_loss / max(1, epoch_n))
        self.model_ = model
        self.device_ = str(device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch = optional_import("torch", extra="deep")
        if self.model_ is None or self.x_mean_ is None or self.x_scale_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(
            float
        )
        values = _standardize_prediction_matrix(
            frame.to_numpy(dtype=float),
            self.x_mean_,
            self.x_scale_,
        )
        if len(values) == 0:
            return np.array([], dtype=float)
        device = torch.device(self.device_ or "cpu")
        self.model_.eval()
        with torch.no_grad():
            pred = (
                self.model_(torch.tensor(values, dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        return pred * self.y_scale_ + self.y_mean_


class _TorchRecurrentRegressor:
    """Small torch-backed LSTM/GRU regressor for tabular forecast matrices."""

    def __init__(
        self,
        *,
        kind: RecurrentKind,
        sequence_length: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 0,
        device: TorchDevice = "auto",
    ) -> None:
        if kind not in {"lstm", "gru", "transformer"}:
            raise ValueError("kind must be 'lstm', 'gru', or 'transformer'")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.kind = kind
        self.sequence_length = max(1, int(sequence_length))
        self.hidden_size = max(1, int(hidden_size))
        self.num_layers = max(1, int(num_layers))
        self.dropout = float(np.clip(dropout, 0.0, 0.95))
        self.learning_rate = float(learning_rate)
        self.max_epochs = max(1, int(max_epochs))
        self.batch_size = max(1, int(batch_size))
        self.random_state = int(random_state)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.device = device
        self.device_: str | None = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.train_tail_: np.ndarray | None = None
        self.sequence_context_: dict[str, Any] = {}
        self.model_: Any = None
        self.training_history_: dict[str, list[float]] = {"loss": []}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchRecurrentRegressor":
        torch = optional_import("torch", extra="deep")
        device = _resolve_torch_device(torch, self.device)
        # Recurrent models are torch-native. Sequence construction is deliberately
        # local: each test row receives only the last fitted rows as prefix, so
        # forecasting runners can pass contiguous test blocks without leakage.

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if len(frame) < self.sequence_length:
            raise ValueError("sequence_length cannot exceed the fitted sample size")
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        _require_finite(values, "X")
        self.x_mean_ = np.nanmean(values, axis=0)
        x_std = np.nanstd(values, axis=0, ddof=1)
        self.x_scale_ = np.where(np.isfinite(x_std) & (x_std > 1e-12), x_std, 1.0)
        x_scaled = (values - self.x_mean_) / self.x_scale_
        y_values = target.to_numpy(dtype=float)
        _require_finite(y_values, "y")
        self.y_mean_ = float(np.nanmean(y_values))
        y_std = float(np.nanstd(y_values, ddof=1))
        self.y_scale_ = y_std if np.isfinite(y_std) and y_std > 1e-12 else 1.0
        y_scaled = (y_values - self.y_mean_) / self.y_scale_

        seq_x, seq_y = _make_sequences(x_scaled, y_scaled, self.sequence_length)
        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        model = _torch_recurrent_net(
            torch=torch,
            kind=self.kind,
            n_features=x_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        tensor_x = torch.tensor(seq_x, dtype=torch.float32, device=device)
        tensor_y = torch.tensor(
            seq_y.reshape(-1, 1), dtype=torch.float32, device=device
        )
        n_obs = tensor_x.shape[0]
        indices = np.arange(n_obs)
        rng = np.random.default_rng(self.random_state)
        model.train()
        for _ in range(self.max_epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            epoch_n = 0
            for start in range(0, n_obs, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = model(tensor_x[batch_idx])
                loss = loss_fn(pred, tensor_y[batch_idx])
                loss.backward()
                optimizer.step()
                batch_n = int(len(batch_idx))
                epoch_loss += float(loss.detach().cpu().item()) * batch_n
                epoch_n += batch_n
            self.training_history_["loss"].append(epoch_loss / max(1, epoch_n))
        self.model_ = model
        self.device_ = str(device)
        self.train_tail_ = x_scaled[-max(0, self.sequence_length - 1) :].copy()
        self.sequence_context_ = {
            "sequence_length": self.sequence_length,
            "train_tail_rows": int(len(self.train_tail_)),
            "test_sequence_prefix": "last fitted rows only",
            "fit_sample_size": int(len(frame)),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch = optional_import("torch", extra="deep")
        if self.model_ is None or self.x_mean_ is None or self.x_scale_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(
            float
        )
        values = _standardize_prediction_matrix(
            frame.to_numpy(dtype=float),
            self.x_mean_,
            self.x_scale_,
        )
        if len(values) == 0:
            return np.array([], dtype=float)
        prefix = (
            self.train_tail_
            if self.train_tail_ is not None
            else np.empty((0, values.shape[1]))
        )
        combined = np.vstack([prefix, values])
        seq_x = np.stack(
            [combined[i : i + self.sequence_length] for i in range(len(values))]
        )
        device = torch.device(self.device_ or "cpu")
        self.model_.eval()
        with torch.no_grad():
            pred = (
                self.model_(torch.tensor(seq_x, dtype=torch.float32, device=device))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        return pred * self.y_scale_ + self.y_mean_


def _resolve_torch_device(torch: Any, requested: TorchDevice) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested, but torch.cuda is not available")
    return torch.device(requested)


def _standardize_matrix(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_finite(values, "X")
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean, scale, (values - mean) / scale


def _standardize_vector(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    _require_finite(values, "y")
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))
    scale = std if np.isfinite(std) and std > 1e-12 else 1.0
    return mean, scale, (values - mean) / scale


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _standardize_prediction_matrix(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    scaled = (values - mean) / scale
    # Fit-time rows are required finite. At prediction time, nonfinite cells are
    # mapped to the fit-window mean in standardized space instead of letting
    # torch propagate invalid values through the network.
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_activation(torch: Any, activation: str) -> Any:
    key = activation.lower()
    if key == "relu":
        return torch.nn.ReLU()
    if key == "tanh":
        return torch.nn.Tanh()
    if key in {"sigmoid", "logistic"}:
        return torch.nn.Sigmoid()
    if key == "gelu":
        return torch.nn.GELU()
    if key == "identity":
        return torch.nn.Identity()
    raise ValueError(
        "activation must be relu, tanh, sigmoid, logistic, gelu, or identity"
    )


def _torch_nn_net(
    *,
    torch: Any,
    n_features: int,
    hidden_layer_sizes: tuple[int, ...],
    activation: str,
    dropout: float,
) -> Any:
    layers: list[Any] = []
    in_features = int(n_features)
    for width in hidden_layer_sizes:
        layers.append(torch.nn.Linear(in_features, int(width)))
        layers.append(_torch_activation(torch, activation))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        in_features = int(width)
    layers.append(torch.nn.Linear(in_features, 1))
    return torch.nn.Sequential(*layers)


def _torch_optimizer(
    torch: Any,
    optimizer: TorchOptimizer,
    parameters: Any,
    *,
    learning_rate: float,
    weight_decay: float,
) -> Any:
    if optimizer == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer == "rmsprop":
        return torch.optim.RMSprop(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError("optimizer must be 'adam', 'sgd', or 'rmsprop'")


def _torch_loss(torch: Any, loss: TorchLoss) -> Any:
    if loss == "mse":
        return torch.nn.MSELoss()
    if loss == "huber":
        return torch.nn.HuberLoss()
    raise ValueError("loss must be 'mse' or 'huber'")


def _torch_recurrent_net(
    *,
    torch: Any,
    kind: RecurrentKind,
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> Any:
    if kind == "transformer":
        return _TorchTransformerNet(
            torch=torch,
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    return _TorchRecurrentNet(
        torch=torch,
        kind=kind,
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


class _TorchRecurrentNet:
    """Pickleable wrapper around torch recurrent modules."""

    def __init__(
        self,
        *,
        torch: Any,
        kind: RecurrentKind,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        module = torch.nn.LSTM if kind == "lstm" else torch.nn.GRU
        self.rnn = module(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = torch.nn.Linear(hidden_size, 1)

    def to(self, device: Any) -> "_TorchRecurrentNet":
        self.rnn.to(device)
        self.head.to(device)
        return self

    def train(self) -> "_TorchRecurrentNet":
        self.rnn.train()
        self.head.train()
        return self

    def eval(self) -> "_TorchRecurrentNet":
        self.rnn.eval()
        self.head.eval()
        return self

    def parameters(self) -> Any:
        yield from self.rnn.parameters()
        yield from self.head.parameters()

    def __call__(self, x: Any) -> Any:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class _TorchTransformerNet:
    """Pickleable wrapper around a compact torch Transformer encoder."""

    def __init__(
        self,
        *,
        torch: Any,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=n_features,
            nhead=1,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = torch.nn.Linear(n_features, 1)

    def to(self, device: Any) -> "_TorchTransformerNet":
        self.encoder.to(device)
        self.head.to(device)
        return self

    def train(self) -> "_TorchTransformerNet":
        self.encoder.train()
        self.head.train()
        return self

    def eval(self) -> "_TorchTransformerNet":
        self.encoder.eval()
        self.head.eval()
        return self

    def parameters(self) -> Any:
        yield from self.encoder.parameters()
        yield from self.head.parameters()

    def __call__(self, x: Any) -> Any:
        out = self.encoder(x)
        return self.head(out[:, -1, :])


def _make_sequences(
    values: np.ndarray, target: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    seq_x = []
    seq_y = []
    for end in range(length - 1, len(values)):
        start = end - length + 1
        seq_x.append(values[start : end + 1])
        seq_y.append(target[end])
    return np.asarray(seq_x, dtype=float), np.asarray(seq_y, dtype=float)


class _TorchHemisphereNNRegressor:
    """Bagged dual-head neural net for mean and variance forecasts."""

    def __init__(
        self,
        *,
        lc: int = 2,
        lm: int = 2,
        lv: int = 2,
        neurons: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        n_estimators: int = 100,
        subsample: float = 0.8,
        nu: float | None = None,
        variance_penalty: float = 1.0,
        patience: int = 15,
        validation_fraction: float = 0.2,
        random_state: int = 0,
        device: TorchDevice = "auto",
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> None:
        self.lc = max(1, int(lc))
        self.lm = max(1, int(lm))
        self.lv = max(1, int(lv))
        self.neurons = max(2, int(neurons))
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self.max_epochs = max(1, int(max_epochs))
        self.n_estimators = max(1, int(n_estimators))
        self.subsample = float(np.clip(subsample, 0.1, 1.0))
        self.nu = None if nu is None else float(np.clip(nu, 1e-3, 0.99))
        self.variance_penalty = max(0.0, float(variance_penalty))
        self.patience = max(1, int(patience))
        self.validation_fraction = float(np.clip(validation_fraction, 0.05, 0.5))
        self.random_state = int(random_state)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.device = device
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        if not self.quantile_levels or any(
            level <= 0.0 or level >= 1.0 for level in self.quantile_levels
        ):
            raise ValueError("quantile_levels must contain values in (0, 1)")
        self.device_: str | None = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.fallback_: float = 0.0
        self.target_variance_: float = 1.0
        self.nu_: float = 0.5
        self.models_: list[Any] = []
        self.training_history_: dict[str, list[float]] = {"loss": []}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchHemisphereNNRegressor":
        torch = optional_import("torch", extra="deep")
        device = _resolve_torch_device(torch, self.device)
        self.device_ = str(device)
        # Hemisphere is implemented as a compact torch-native dual-head density
        # model. The logical contract is mean/variance prediction with blocked
        # bagging and a chronological validation block, not R source parity.
        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.default_rng(self.random_state)

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        self.x_mean_, self.x_scale_, x_scaled = _standardize_matrix(values)
        y_values = target.to_numpy(dtype=float)
        self.fallback_ = float(np.nanmean(y_values)) if y_values.size else 0.0
        self.target_variance_ = float(np.nanvar(y_values, ddof=1))
        if not np.isfinite(self.target_variance_) or self.target_variance_ <= 1e-12:
            self.target_variance_ = 1.0
        self.nu_ = self.nu if self.nu is not None else 0.5
        target_variance_mean = float(self.nu_ * self.target_variance_)

        n_obs, n_features = x_scaled.shape
        if n_obs < 4 or n_features == 0:
            return self
        val_size = max(1, int(round(self.validation_fraction * n_obs)))
        val_size = min(val_size, n_obs - 2)
        train_idx = np.arange(0, n_obs - val_size)
        val_idx = np.arange(n_obs - val_size, n_obs)
        x_train_all = torch.tensor(x_scaled[train_idx], dtype=torch.float32, device=device)
        y_train_all = torch.tensor(y_values[train_idx], dtype=torch.float32, device=device)
        x_val = torch.tensor(x_scaled[val_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(y_values[val_idx], dtype=torch.float32, device=device)
        target_variance_tensor = torch.tensor(
            target_variance_mean,
            dtype=torch.float32,
            device=device,
        )
        self.models_ = []
        losses = []
        for bag in range(self.n_estimators):
            in_idx = _blocked_subsample_indices(len(train_idx), self.subsample, rng)
            if len(in_idx) < 2:
                continue
            model = _TorchHemisphereNet(
                torch=torch,
                n_features=n_features,
                lc=self.lc,
                lm=self.lm,
                lv=self.lv,
                neurons=self.neurons,
                dropout=self.dropout,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            x_bag = x_train_all[in_idx]
            y_bag = y_train_all[in_idx]
            best_loss = float("inf")
            stale = 0
            best_state = _torch_clone_state(model.state_dict())
            for _ in range(self.max_epochs):
                model.train()
                optimizer.zero_grad()
                mean, variance = model(x_bag)
                nll = ((y_bag - mean) ** 2 / variance + torch.log(variance)).mean()
                # Anchor the variance penalty on the training subsample only.
                # Using x_all (the whole sample, including the chronological
                # validation block used for early stopping) let validation rows
                # feed the training gradient and weakened the early-stopping
                # independence (mild validation leakage).
                _, full_variance = model(x_bag)
                scale = max(self.target_variance_**2, 1e-12)
                penalty = (full_variance.mean() - target_variance_tensor) ** 2 / scale
                loss = nll + self.variance_penalty * penalty
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_mean, val_variance = model(x_val)
                    val_loss = float(
                        (
                            (y_val - val_mean) ** 2 / val_variance
                            + torch.log(val_variance)
                        )
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                if val_loss < best_loss:
                    best_loss = val_loss
                    stale = 0
                    best_state = _torch_clone_state(model.state_dict())
                else:
                    stale += 1
                if stale >= self.patience:
                    break
            model.load_state_dict(best_state)
            self.models_.append(model)
            losses.append(best_loss)
        if not self.models_:
            model = _TorchHemisphereNet(
                torch=torch,
                n_features=n_features,
                lc=self.lc,
                lm=self.lm,
                lv=self.lv,
                neurons=self.neurons,
                dropout=self.dropout,
            ).to(device)
            self.models_.append(model)
        self.training_history_["loss"] = [float(value) for value in losses]
        self.device_ = str(device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean, _ = self.predict_distribution(X)
        return mean

    def predict_variance(self, X: pd.DataFrame) -> np.ndarray:
        _, variance = self.predict_distribution(X)
        return variance

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] | None = None,
    ) -> dict[float, np.ndarray]:
        mean, variance = self.predict_distribution(X)
        sigma = np.sqrt(np.maximum(variance, 1e-12))
        normal = NormalDist()
        quantile_levels = (
            self.quantile_levels if levels is None else tuple(float(level) for level in levels)
        )
        if not quantile_levels or any(
            level <= 0.0 or level >= 1.0 for level in quantile_levels
        ):
            raise ValueError("quantile levels must be in (0, 1)")
        return {
            float(level): mean + sigma * normal.inv_cdf(float(level))
            for level in quantile_levels
        }

    def predict_distribution(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        torch = optional_import("torch", extra="deep")
        if not self.models_ or self.x_mean_ is None or self.x_scale_ is None:
            n = len(X)
            return (
                np.full(n, self.fallback_, dtype=float),
                np.full(n, self.target_variance_, dtype=float),
            )
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        values = _standardize_prediction_matrix(
            frame.to_numpy(dtype=float),
            self.x_mean_,
            self.x_scale_,
        )
        if len(values) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        device = torch.device(self.device_ or "cpu")
        tensor_x = torch.tensor(values, dtype=torch.float32, device=device)
        means = []
        variances = []
        with torch.no_grad():
            for model in self.models_:
                model.eval()
                mean, variance = model(tensor_x)
                means.append(mean.detach().cpu().numpy())
                variances.append(variance.detach().cpu().numpy())
        mean_matrix = np.column_stack(means)
        variance_matrix = np.column_stack(variances)
        mean_pred = np.mean(mean_matrix, axis=1)
        variance_pred = np.mean(variance_matrix, axis=1) + np.var(mean_matrix, axis=1)
        variance_pred = np.where(
            np.isfinite(variance_pred) & (variance_pred > 1e-12),
            variance_pred,
            self.target_variance_,
        )
        return mean_pred, variance_pred


class _TorchHemisphereNet:
    """Small torch module wrapper with shared core, mean head, and variance head."""

    def __init__(
        self,
        *,
        torch: Any,
        n_features: int,
        lc: int,
        lm: int,
        lv: int,
        neurons: int,
        dropout: float,
    ) -> None:
        self.softplus = torch.nn.Softplus()
        self.core = _torch_dense_stack(
            torch, n_features, neurons, depth=lc, dropout=dropout
        )
        self.mean_head = torch.nn.Sequential(
            _torch_dense_stack(torch, neurons, neurons, depth=lm, dropout=dropout),
            torch.nn.Linear(neurons, 1),
        )
        self.variance_head = torch.nn.Sequential(
            _torch_dense_stack(torch, neurons, neurons, depth=lv, dropout=dropout),
            torch.nn.Linear(neurons, 1),
        )

    def to(self, device: Any) -> "_TorchHemisphereNet":
        self.core.to(device)
        self.mean_head.to(device)
        self.variance_head.to(device)
        return self

    def train(self) -> "_TorchHemisphereNet":
        self.core.train()
        self.mean_head.train()
        self.variance_head.train()
        return self

    def eval(self) -> "_TorchHemisphereNet":
        self.core.eval()
        self.mean_head.eval()
        self.variance_head.eval()
        return self

    def parameters(self) -> Any:
        yield from self.core.parameters()
        yield from self.mean_head.parameters()
        yield from self.variance_head.parameters()

    def state_dict(self) -> dict[str, Any]:
        return {
            "core": self.core.state_dict(),
            "mean_head": self.mean_head.state_dict(),
            "variance_head": self.variance_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.core.load_state_dict(state["core"])
        self.mean_head.load_state_dict(state["mean_head"])
        self.variance_head.load_state_dict(state["variance_head"])

    def __call__(self, x: Any) -> tuple[Any, Any]:
        latent = self.core(x)
        mean = self.mean_head(latent).squeeze(-1)
        variance = self.softplus(self.variance_head(latent)).squeeze(-1) + 1e-6
        return mean, variance


class _TorchDensityHNNRegressor:
    """Aionx DensityHNN-style dual-head density forecaster.

    This estimator follows the public Python source linked from Goulet
    Coulombe, Frenette, and Klieber (2025). The source-of-truth code is
    Aionx ``aionx/models.py::DensityHNN``:

    - ``prior_dnn_architecture``: fit a plain DNN ensemble first.
    - ``TimeSeriesBlockBootstrap`` and ``OutOfBagPredictor``: compute blocked
      OOB forecasts.
    - ``base_architecture``: shared core, conditional-mean head, and positive
      volatility head, with mean volatility scaled to the emphasis parameter.
    - ``volatility_rescaling_algorithm``: regress log OOB squared residuals on
      log predicted volatility squared, then rescale all volatility forecasts.

    The implementation is torch-native to avoid adding TensorFlow to the public
    dependency surface. It keeps Aionx's statistical logic but consumes the
    macroforecast callable ``X, y`` feature-matrix contract.
    """

    def __init__(
        self,
        *,
        common_layers: int = 2,
        mean_layers: int = 2,
        volatility_layers: int = 2,
        prior_layers: int = 3,
        neurons: int = 400,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        n_estimators: int = 100,
        prior_estimators: int = 50,
        subsample: float = 0.8,
        block_size: int = 8,
        volatility_emphasis: float | None = None,
        rescale_volatility: bool = True,
        patience: int = 15,
        random_state: int = 0,
        device: TorchDevice = "auto",
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
        volatility_clip: float = 0.05,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.common_layers = max(1, int(common_layers))
        self.mean_layers = max(1, int(mean_layers))
        self.volatility_layers = max(1, int(volatility_layers))
        self.prior_layers = max(1, int(prior_layers))
        self.neurons = max(2, int(neurons))
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        self.learning_rate = float(learning_rate)
        self.max_epochs = max(1, int(max_epochs))
        self.n_estimators = max(1, int(n_estimators))
        self.prior_estimators = max(0, int(prior_estimators))
        self.subsample = float(np.clip(subsample, 0.05, 0.99))
        self.block_size = max(1, int(block_size))
        self.volatility_emphasis = (
            None if volatility_emphasis is None else float(volatility_emphasis)
        )
        self.rescale_volatility = bool(rescale_volatility)
        self.patience = max(1, int(patience))
        self.random_state = int(random_state)
        self.device = device
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        if not self.quantile_levels or any(
            level <= 0.0 or level >= 1.0 for level in self.quantile_levels
        ):
            raise ValueError("quantile_levels must contain values in (0, 1)")
        self.volatility_clip = max(1e-6, float(volatility_clip))
        self.device_: str | None = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.fallback_: float = 0.0
        self.target_variance_: float = 1.0
        self.models_: list[Any] = []
        self.oob_indices_: list[np.ndarray] = []
        self.training_history_: dict[str, list[float]] = {
            "prior_loss": [],
            "density_loss": [],
        }
        self.volatility_emphasis_: float = 1.0
        self.prior_oob_mse_: float | None = None
        self.oob_rescaling_: dict[str, float | int | bool] = {
            "enabled": False,
            "intercept": 0.0,
            "slope": 1.0,
            "scaler": 1.0,
            "n_obs": 0,
        }
        self.oob_prediction_: pd.DataFrame | None = None
        self.density_diagnostics_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TorchDensityHNNRegressor":
        torch = optional_import("torch", extra="deep")
        device = _resolve_torch_device(torch, self.device)
        self.device_ = str(device)
        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.default_rng(self.random_state)

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        self.x_mean_, self.x_scale_, x_scaled = _standardize_matrix(values)
        y_values = target.to_numpy(dtype=float)
        self.y_mean_, self.y_scale_, y_scaled = _standardize_vector(y_values)
        self.fallback_ = self.y_mean_
        self.target_variance_ = float(self.y_scale_**2)

        n_obs, n_features = x_scaled.shape
        if n_obs < 4 or n_features == 0:
            self.volatility_emphasis_ = 1.0
            self.density_diagnostics_ = self._density_diagnostics()
            return self

        x_all = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        y_all = torch.tensor(y_scaled, dtype=torch.float32, device=device)

        if self.volatility_emphasis is None and self.prior_estimators > 0:
            prior = self._fit_prior_ensemble(torch, x_all, y_all, rng)
            self.prior_oob_mse_ = prior["oob_mse"]
            raw_emphasis = self.prior_oob_mse_
        else:
            raw_emphasis = self.volatility_emphasis
        self.volatility_emphasis_ = _aionx_volatility_emphasis(raw_emphasis)

        self.models_ = []
        self.oob_indices_ = []
        density_losses: list[float] = []
        for _ in range(self.n_estimators):
            train_idx, oob_idx = _time_series_block_bootstrap_indices(
                n_obs,
                self.subsample,
                self.block_size,
                rng,
            )
            if len(train_idx) < 2:
                continue
            model = _TorchDensityHNNNet(
                torch=torch,
                n_features=n_features,
                common_layers=self.common_layers,
                mean_layers=self.mean_layers,
                volatility_layers=self.volatility_layers,
                neurons=self.neurons,
                dropout=self.dropout,
                volatility_emphasis=self.volatility_emphasis_,
            ).to(device)
            best_loss = self._fit_density_member(
                torch,
                model,
                x_all,
                y_all,
                train_idx,
                oob_idx,
            )
            self.models_.append(model)
            self.oob_indices_.append(oob_idx)
            density_losses.append(best_loss)
        if not self.models_:
            model = _TorchDensityHNNNet(
                torch=torch,
                n_features=n_features,
                common_layers=self.common_layers,
                mean_layers=self.mean_layers,
                volatility_layers=self.volatility_layers,
                neurons=self.neurons,
                dropout=self.dropout,
                volatility_emphasis=self.volatility_emphasis_,
            ).to(device)
            self.models_.append(model)
            self.oob_indices_.append(np.arange(n_obs, dtype=int))
        self.training_history_["density_loss"] = [float(value) for value in density_losses]

        mean_matrix, sigma_matrix = self._predict_scaled_matrices(torch, x_all)
        mean_oob = _aionx_oob_average(mean_matrix, self.oob_indices_, self.subsample)
        sigma_oob = _aionx_oob_average(sigma_matrix, self.oob_indices_, self.subsample)
        if self.rescale_volatility:
            self.oob_rescaling_ = _fit_aionx_volatility_rescaling(
                y_scaled,
                mean_oob,
                sigma_oob,
                volatility_clip=self.volatility_clip,
            )
        else:
            self.oob_rescaling_ = {
                "enabled": False,
                "intercept": 0.0,
                "slope": 1.0,
                "scaler": 1.0,
                "n_obs": 0,
            }
        scaled_sigma_oob = _apply_aionx_volatility_rescaling(
            sigma_oob,
            self.oob_rescaling_,
            volatility_clip=self.volatility_clip,
        )
        self.oob_prediction_ = pd.DataFrame(
            {
                "conditional_mean": mean_oob * self.y_scale_ + self.y_mean_,
                "conditional_volatility": scaled_sigma_oob * self.y_scale_,
                "conditional_variance": (scaled_sigma_oob * self.y_scale_) ** 2,
            },
            index=frame.index,
        )
        self.density_diagnostics_ = self._density_diagnostics()
        return self

    def _fit_prior_ensemble(
        self,
        torch: Any,
        x_all: Any,
        y_all: Any,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        predictions = []
        oob_indices = []
        losses = []
        for _ in range(self.prior_estimators):
            train_idx, oob_idx = _time_series_block_bootstrap_indices(
                len(y_all),
                self.subsample,
                self.block_size,
                rng,
            )
            model = _torch_nn_net(
                torch=torch,
                n_features=x_all.shape[1],
                hidden_layer_sizes=tuple([self.neurons] * self.prior_layers),
                activation="relu",
                dropout=self.dropout,
            ).to(x_all.device)
            loss = self._fit_prior_member(torch, model, x_all, y_all, train_idx, oob_idx)
            model.eval()
            with torch.no_grad():
                pred = (
                    model(x_all)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            predictions.append(pred)
            oob_indices.append(oob_idx)
            losses.append(loss)
        self.training_history_["prior_loss"] = [float(value) for value in losses]
        matrix = np.column_stack(predictions) if predictions else np.empty((len(y_all), 0))
        oob = _aionx_oob_average(matrix, oob_indices, self.subsample)
        y_np = y_all.detach().cpu().numpy().reshape(-1)
        valid = np.isfinite(oob) & np.isfinite(y_np)
        mse = float(np.mean((oob[valid] - y_np[valid]) ** 2)) if valid.any() else 1.0
        if not np.isfinite(mse) or mse <= 1e-12:
            mse = 1.0
        return {"prediction_matrix": matrix, "oob_prediction": oob, "oob_mse": mse}

    def _fit_prior_member(
        self,
        torch: Any,
        model: Any,
        x_all: Any,
        y_all: Any,
        train_idx: np.ndarray,
        oob_idx: np.ndarray,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_val = x_all[oob_idx] if len(oob_idx) else x_train
        y_val = y_all[oob_idx] if len(oob_idx) else y_train
        best_loss = float("inf")
        stale = 0
        best_state = _torch_clone_state(model.state_dict())
        for _ in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(x_train).squeeze(-1)
            loss = ((y_train - pred) ** 2).mean()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val).squeeze(-1)
                val_loss = float(((y_val - val_pred) ** 2).mean().detach().cpu().item())
            if val_loss < best_loss:
                best_loss = val_loss
                stale = 0
                best_state = _torch_clone_state(model.state_dict())
            else:
                stale += 1
            if stale >= self.patience:
                break
        model.load_state_dict(best_state)
        return best_loss

    def _fit_density_member(
        self,
        torch: Any,
        model: "_TorchDensityHNNNet",
        x_all: Any,
        y_all: Any,
        train_idx: np.ndarray,
        oob_idx: np.ndarray,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_val = x_all[oob_idx] if len(oob_idx) else x_train
        y_val = y_all[oob_idx] if len(oob_idx) else y_train
        best_loss = float("inf")
        stale = 0
        best_state = _torch_clone_state(model.state_dict())
        for _ in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            mean, sigma = model(x_train)
            loss = _torch_gaussian_sigma_nll(torch, y_train, mean, sigma, self.volatility_clip)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_mean, val_sigma = model(x_val)
                val_loss = float(
                    _torch_gaussian_sigma_nll(
                        torch,
                        y_val,
                        val_mean,
                        val_sigma,
                        self.volatility_clip,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            if val_loss < best_loss:
                best_loss = val_loss
                stale = 0
                best_state = _torch_clone_state(model.state_dict())
            else:
                stale += 1
            if stale >= self.patience:
                break
        model.load_state_dict(best_state)
        return best_loss

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean, _ = self.predict_distribution(X)
        return mean

    def predict_variance(self, X: pd.DataFrame) -> np.ndarray:
        _, variance = self.predict_distribution(X)
        return variance

    def predict_volatility(self, X: pd.DataFrame) -> np.ndarray:
        variance = self.predict_variance(X)
        return np.sqrt(np.maximum(variance, 1e-12))

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] | None = None,
    ) -> dict[float, np.ndarray]:
        mean, variance = self.predict_distribution(X)
        sigma = np.sqrt(np.maximum(variance, 1e-12))
        normal = NormalDist()
        quantile_levels = (
            self.quantile_levels if levels is None else tuple(float(level) for level in levels)
        )
        if not quantile_levels or any(
            level <= 0.0 or level >= 1.0 for level in quantile_levels
        ):
            raise ValueError("quantile levels must be in (0, 1)")
        return {
            float(level): mean + sigma * normal.inv_cdf(float(level))
            for level in quantile_levels
        }

    def predict_distribution(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        torch = optional_import("torch", extra="deep")
        if not self.models_ or self.x_mean_ is None or self.x_scale_ is None:
            n = len(X)
            return (
                np.full(n, self.fallback_, dtype=float),
                np.full(n, self.target_variance_, dtype=float),
            )
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        values = _standardize_prediction_matrix(
            frame.to_numpy(dtype=float),
            self.x_mean_,
            self.x_scale_,
        )
        if len(values) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        device = torch.device(self.device_ or "cpu")
        tensor_x = torch.tensor(values, dtype=torch.float32, device=device)
        mean_matrix, sigma_matrix = self._predict_scaled_matrices(torch, tensor_x)
        mean_scaled = np.mean(mean_matrix, axis=1)
        sigma_scaled = np.mean(sigma_matrix, axis=1)
        sigma_scaled = _apply_aionx_volatility_rescaling(
            sigma_scaled,
            self.oob_rescaling_,
            volatility_clip=self.volatility_clip,
        )
        mean = mean_scaled * self.y_scale_ + self.y_mean_
        volatility = np.maximum(sigma_scaled * self.y_scale_, 1e-12)
        return mean, volatility**2

    def _predict_scaled_matrices(self, torch: Any, tensor_x: Any) -> tuple[np.ndarray, np.ndarray]:
        means = []
        sigmas = []
        with torch.no_grad():
            for model in self.models_:
                model.eval()
                mean, sigma = model(tensor_x)
                means.append(mean.detach().cpu().numpy().reshape(-1))
                sigmas.append(sigma.detach().cpu().numpy().reshape(-1))
        return np.column_stack(means), np.column_stack(sigmas)

    def _density_diagnostics(self) -> dict[str, Any]:
        return {
            "source_reference": (
                "Goulet Coulombe, Frenette, and Klieber (2025) and "
                "Aionx aionx.models.DensityHNN"
            ),
            "backend_alignment": {
                "prior_dnn_architecture": "torch feed-forward prior ensemble",
                "base_architecture": "torch shared core plus mean/volatility heads",
                "TimeSeriesBlockBootstrap": "time-series block bootstrap indices",
                "OutOfBagPredictor": "Aionx denominator OOB averaging",
                "volatility_rescaling_algorithm": "log residual-square calibration",
            },
            "volatility_emphasis": float(self.volatility_emphasis_),
            "prior_oob_mse": None
            if self.prior_oob_mse_ is None
            else float(self.prior_oob_mse_),
            "oob_rescaling": dict(self.oob_rescaling_),
            "n_estimators": int(len(self.models_)),
            "prior_estimators": int(self.prior_estimators),
            "subsample": float(self.subsample),
            "block_size": int(self.block_size),
            "output_scale": "mean in target units; variance in target units squared",
        }


class _TorchDensityHNNNet:
    """Shared-core HNN with an Aionx-style positive volatility hemisphere."""

    def __init__(
        self,
        *,
        torch: Any,
        n_features: int,
        common_layers: int,
        mean_layers: int,
        volatility_layers: int,
        neurons: int,
        dropout: float,
        volatility_emphasis: float,
    ) -> None:
        self.torch = torch
        self.softplus = torch.nn.Softplus()
        self.volatility_emphasis = float(volatility_emphasis)
        self.core = _torch_dense_stack(
            torch, n_features, neurons, depth=common_layers, dropout=dropout
        )
        self.mean_head = torch.nn.Sequential(
            _torch_dense_stack(torch, neurons, neurons, depth=mean_layers, dropout=dropout),
            torch.nn.Linear(neurons, 1),
        )
        self.volatility_head = torch.nn.Sequential(
            _torch_dense_stack(
                torch,
                neurons,
                neurons,
                depth=volatility_layers,
                dropout=dropout,
            ),
            torch.nn.Linear(neurons, 1),
        )

    def to(self, device: Any) -> "_TorchDensityHNNNet":
        self.core.to(device)
        self.mean_head.to(device)
        self.volatility_head.to(device)
        return self

    def train(self) -> "_TorchDensityHNNNet":
        self.core.train()
        self.mean_head.train()
        self.volatility_head.train()
        return self

    def eval(self) -> "_TorchDensityHNNNet":
        self.core.eval()
        self.mean_head.eval()
        self.volatility_head.eval()
        return self

    def parameters(self) -> Any:
        yield from self.core.parameters()
        yield from self.mean_head.parameters()
        yield from self.volatility_head.parameters()

    def state_dict(self) -> dict[str, Any]:
        return {
            "core": self.core.state_dict(),
            "mean_head": self.mean_head.state_dict(),
            "volatility_head": self.volatility_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.core.load_state_dict(state["core"])
        self.mean_head.load_state_dict(state["mean_head"])
        self.volatility_head.load_state_dict(state["volatility_head"])

    def __call__(self, x: Any) -> tuple[Any, Any]:
        latent = self.core(x)
        mean = self.mean_head(latent).squeeze(-1)
        raw_sigma = self.softplus(self.volatility_head(latent)).squeeze(-1) + 1e-6
        # Aionx DensityHNN.base_architecture uses
        #   vol_emphasis * vol_output / reduce_mean(vol_output).
        # This fixes the average predicted volatility to the OOB-derived
        # emphasis parameter and is central to the paper's variance allocation.
        sigma = self.volatility_emphasis * raw_sigma / raw_sigma.mean().clamp_min(1e-6)
        return mean, sigma


def _torch_dense_stack(
    torch: Any,
    in_features: int,
    width: int,
    *,
    depth: int,
    dropout: float,
) -> Any:
    layers = []
    current = int(in_features)
    for _ in range(max(1, int(depth))):
        layers.append(torch.nn.Linear(current, int(width)))
        layers.append(torch.nn.ReLU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        current = int(width)
    return torch.nn.Sequential(*layers)


def _torch_clone_state(state: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            cloned[key] = {
                inner_key: tensor.detach().clone()
                for inner_key, tensor in value.items()
            }
        else:
            cloned[key] = value.detach().clone()
    return cloned


def _torch_gaussian_sigma_nll(
    torch: Any,
    y_true: Any,
    mean: Any,
    sigma: Any,
    volatility_clip: float,
) -> Any:
    # Aionx kerasnn.losses.GaussianLogLikelihood computes
    # mean((y - mu)^2 / sigma^2 + log(sigma^2)) after clipping sigma.
    clipped = torch.clamp(sigma, min=float(volatility_clip))
    return (((y_true - mean) ** 2) / (clipped**2) + torch.log(clipped**2)).mean()


def _aionx_volatility_emphasis(value: float | None) -> float:
    # Aionx DensityHNN.base_architecture warns outside [0.01, 1.0] and sets
    # the emphasis to 0.99. Keep that behavior for paper/source alignment.
    if value is None or not np.isfinite(float(value)):
        return 0.99
    resolved = float(value)
    if not 0.01 <= resolved <= 1.0:
        return 0.99
    return resolved


def _time_series_block_bootstrap_indices(
    n_obs: int,
    sampling_rate: float,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return blocked bootstrap and OOB indices for time-series samples."""

    if n_obs <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    block_size = max(1, int(block_size))
    blocks = [
        np.arange(start, min(start + block_size, n_obs), dtype=int)
        for start in range(0, n_obs, block_size)
    ]
    n_train = max(1, int(round(float(sampling_rate) * n_obs)))
    n_blocks = max(1, int(np.ceil(n_train / block_size)))
    chosen = rng.choice(len(blocks), size=n_blocks, replace=True)
    train_idx = np.concatenate([blocks[int(index)] for index in chosen])
    if len(train_idx) > n_train:
        train_idx = train_idx[:n_train]
    train_idx = np.asarray(train_idx, dtype=int)
    in_bag_unique = np.unique(train_idx)
    oob_idx = np.setdiff1d(np.arange(n_obs, dtype=int), in_bag_unique, assume_unique=True)
    if len(oob_idx) == 0:
        fallback = max(1, min(n_obs - 1, block_size))
        oob_idx = np.arange(n_obs - fallback, n_obs, dtype=int)
    return np.sort(train_idx), np.asarray(oob_idx, dtype=int)


def _aionx_oob_average(
    forecasts: np.ndarray,
    oob_indices: list[np.ndarray],
    sampling_rate: float,
) -> np.ndarray:
    """Aionx OutOfBagPredictor average for rows exposed as OOB by estimators."""

    if forecasts.size == 0 or forecasts.shape[1] == 0:
        return np.full(forecasts.shape[0], np.nan, dtype=float)
    work = np.asarray(forecasts, dtype=float).copy()
    all_rows = np.arange(work.shape[0])
    for col, oob_idx in enumerate(oob_indices[: work.shape[1]]):
        rows_to_nan = np.where(~np.isin(all_rows, np.asarray(oob_idx, dtype=int)))[0]
        work[rows_to_nan, col] = np.nan
    row_sums = np.nansum(work, axis=1)
    denominator = max((1.0 - float(sampling_rate)) * work.shape[1], 1e-12)
    averaged = row_sums / denominator
    averaged[averaged == 0.0] = np.nan
    return averaged


def _fit_aionx_volatility_rescaling(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_pred: np.ndarray,
    *,
    volatility_clip: float,
) -> dict[str, float | int | bool]:
    """Fit Aionx's log squared-residual volatility recalibration."""

    residual_sq = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    sigma_sq = np.asarray(sigma_pred, dtype=float) ** 2
    valid = (
        np.isfinite(residual_sq)
        & np.isfinite(sigma_sq)
        & (residual_sq > 1e-12)
        & (sigma_sq > volatility_clip**2)
    )
    if int(valid.sum()) < 3:
        return {
            "enabled": False,
            "intercept": 0.0,
            "slope": 1.0,
            "scaler": 1.0,
            "n_obs": int(valid.sum()),
        }
    x = np.log(sigma_sq[valid])
    y = np.log(residual_sq[valid])
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    projection = intercept + slope * x
    scaler = float(np.sqrt(np.mean(np.exp(y - projection))))
    if not np.isfinite(scaler) or scaler <= 0:
        scaler = 1.0
    return {
        "enabled": True,
        "intercept": float(intercept),
        "slope": float(slope),
        "scaler": scaler,
        "n_obs": int(valid.sum()),
    }


def _apply_aionx_volatility_rescaling(
    sigma_pred: np.ndarray,
    rescaling: dict[str, float | int | bool],
    *,
    volatility_clip: float,
) -> np.ndarray:
    sigma = np.asarray(sigma_pred, dtype=float)
    sigma = np.where(
        np.isfinite(sigma) & (sigma > volatility_clip),
        sigma,
        float(volatility_clip),
    )
    if not rescaling.get("enabled", False):
        return sigma
    intercept = float(rescaling.get("intercept", 0.0))
    slope = float(rescaling.get("slope", 1.0))
    scaler = float(rescaling.get("scaler", 1.0))
    projection = intercept + slope * np.log(np.maximum(sigma**2, volatility_clip**2))
    adjusted = np.sqrt(np.exp(projection)) * scaler
    return np.where(np.isfinite(adjusted) & (adjusted > 0), adjusted, sigma)


def _blocked_subsample_indices(
    n_obs: int,
    subsample: float,
    rng: np.random.Generator,
) -> np.ndarray:
    block_size = max(2, n_obs // 10)
    blocks = [
        np.arange(start, min(start + block_size, n_obs))
        for start in range(0, n_obs, block_size)
    ]
    n_blocks = max(1, int(round(float(subsample) * len(blocks))))
    chosen = rng.choice(len(blocks), size=n_blocks, replace=False)
    return np.sort(np.concatenate([blocks[int(index)] for index in chosen]))


def lstm(
    X: Any,
    y: Any | None = None,
    *,
    sequence_length: int = 4,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 0,
    device: TorchDevice = "auto",
) -> ModelFit:
    """Fit a torch-backed LSTM regressor. Requires ``macroforecast[deep]``."""

    params = {
        "sequence_length": int(sequence_length),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "device": device,
    }
    return fit_estimator(
        _TorchRecurrentRegressor(
            kind="lstm",
            sequence_length=int(sequence_length),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            random_state=int(random_state),
            device=device,
        ),
        X,
        y,
        model="lstm",
        metadata=params,
    )


def gru(
    X: Any,
    y: Any | None = None,
    *,
    sequence_length: int = 4,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 0,
    device: TorchDevice = "auto",
) -> ModelFit:
    """Fit a torch-backed GRU regressor. Requires ``macroforecast[deep]``."""

    params = {
        "sequence_length": int(sequence_length),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "device": device,
    }
    return fit_estimator(
        _TorchRecurrentRegressor(
            kind="gru",
            sequence_length=int(sequence_length),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            random_state=int(random_state),
            device=device,
        ),
        X,
        y,
        model="gru",
        metadata=params,
    )


def transformer(
    X: Any,
    y: Any | None = None,
    *,
    sequence_length: int = 4,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 0,
    device: TorchDevice = "auto",
) -> ModelFit:
    """Fit a torch-backed Transformer encoder regressor. Requires ``macroforecast[deep]``."""

    params = {
        "sequence_length": int(sequence_length),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "device": device,
    }
    return fit_estimator(
        _TorchRecurrentRegressor(
            kind="transformer",
            sequence_length=int(sequence_length),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            random_state=int(random_state),
            device=device,
        ),
        X,
        y,
        model="transformer",
        metadata=params,
    )


def hemisphere_nn(
    X: Any,
    y: Any | None = None,
    *,
    lc: int = 2,
    lm: int = 2,
    lv: int = 2,
    neurons: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    n_estimators: int = 100,
    subsample: float = 0.8,
    nu: float | None = None,
    variance_penalty: float = 1.0,
    patience: int = 15,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    device: TorchDevice = "auto",
    lr: float | None = None,
    n_epochs: int | None = None,
    B: int | None = None,
    sub_rate: float | None = None,
    lambda_emphasis: float | None = None,
    val_frac: float | None = None,
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> ModelFit:
    """Fit a torch-backed Hemisphere neural network mean/variance forecaster."""

    resolved_learning_rate = float(learning_rate if lr is None else lr)
    resolved_max_epochs = int(max_epochs if n_epochs is None else n_epochs)
    resolved_n_estimators = int(n_estimators if B is None else B)
    resolved_subsample = float(subsample if sub_rate is None else sub_rate)
    resolved_variance_penalty = float(
        variance_penalty if lambda_emphasis is None else lambda_emphasis
    )
    resolved_validation_fraction = float(
        validation_fraction if val_frac is None else val_frac
    )
    params = {
        "lc": int(lc),
        "lm": int(lm),
        "lv": int(lv),
        "neurons": int(neurons),
        "dropout": float(dropout),
        "learning_rate": resolved_learning_rate,
        "max_epochs": resolved_max_epochs,
        "n_estimators": resolved_n_estimators,
        "subsample": resolved_subsample,
        "nu": nu,
        "variance_penalty": resolved_variance_penalty,
        "patience": int(patience),
        "validation_fraction": resolved_validation_fraction,
        "random_state": int(random_state),
        "device": device,
        "quantile_levels": tuple(float(level) for level in quantile_levels),
        "legacy_aliases": {
            "lr": lr,
            "n_epochs": n_epochs,
            "B": B,
            "sub_rate": sub_rate,
            "lambda_emphasis": lambda_emphasis,
            "val_frac": val_frac,
        },
    }
    return fit_estimator(
        _TorchHemisphereNNRegressor(
            lc=int(lc),
            lm=int(lm),
            lv=int(lv),
            neurons=int(neurons),
            dropout=float(dropout),
            learning_rate=resolved_learning_rate,
            max_epochs=resolved_max_epochs,
            n_estimators=resolved_n_estimators,
            subsample=resolved_subsample,
            nu=nu,
            variance_penalty=resolved_variance_penalty,
            patience=int(patience),
            validation_fraction=resolved_validation_fraction,
            random_state=int(random_state),
            device=device,
            quantile_levels=tuple(float(level) for level in quantile_levels),
        ),
        X,
        y,
        model="hemisphere_nn",
        metadata=params,
    )


def density_hnn(
    X: Any,
    y: Any | None = None,
    *,
    common_layers: int = 2,
    mean_layers: int = 2,
    volatility_layers: int = 2,
    prior_layers: int = 3,
    neurons: int = 400,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    max_epochs: int = 100,
    n_estimators: int = 100,
    prior_estimators: int = 50,
    subsample: float = 0.8,
    block_size: int = 8,
    volatility_emphasis: float | None = None,
    rescale_volatility: bool = True,
    patience: int = 15,
    random_state: int = 0,
    device: TorchDevice = "auto",
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    volatility_clip: float = 0.05,
) -> ModelFit:
    """Fit the paper-faithful Density Hemisphere neural-network forecaster."""

    params = {
        "common_layers": int(common_layers),
        "mean_layers": int(mean_layers),
        "volatility_layers": int(volatility_layers),
        "prior_layers": int(prior_layers),
        "neurons": int(neurons),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "n_estimators": int(n_estimators),
        "prior_estimators": int(prior_estimators),
        "subsample": float(subsample),
        "block_size": int(block_size),
        "volatility_emphasis": None
        if volatility_emphasis is None
        else float(volatility_emphasis),
        "rescale_volatility": bool(rescale_volatility),
        "patience": int(patience),
        "random_state": int(random_state),
        "device": device,
        "quantile_levels": tuple(float(level) for level in quantile_levels),
        "volatility_clip": float(volatility_clip),
        "source_reference": (
            "Goulet Coulombe, Frenette, and Klieber (2025), "
            "From Reactive to Proactive Volatility Modeling with Hemisphere "
            "Neural Networks; Aionx DensityHNN"
        ),
    }
    return fit_estimator(
        _TorchDensityHNNRegressor(
            common_layers=int(common_layers),
            mean_layers=int(mean_layers),
            volatility_layers=int(volatility_layers),
            prior_layers=int(prior_layers),
            neurons=int(neurons),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            max_epochs=int(max_epochs),
            n_estimators=int(n_estimators),
            prior_estimators=int(prior_estimators),
            subsample=float(subsample),
            block_size=int(block_size),
            volatility_emphasis=volatility_emphasis,
            rescale_volatility=bool(rescale_volatility),
            patience=int(patience),
            random_state=int(random_state),
            device=device,
            quantile_levels=tuple(float(level) for level in quantile_levels),
            volatility_clip=float(volatility_clip),
        ),
        X,
        y,
        model="density_hnn",
        metadata=params,
    )


__all__ = ["density_hnn", "gru", "hemisphere_nn", "lstm", "nn", "transformer"]
