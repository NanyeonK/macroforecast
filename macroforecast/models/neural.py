from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, optional_import

RecurrentKind = Literal["lstm", "gru"]
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
        values = (frame.to_numpy(dtype=float) - self.x_mean_) / self.x_scale_
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

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if len(frame) < self.sequence_length:
            raise ValueError("sequence_length cannot exceed the fitted sample size")
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        values = frame.to_numpy(dtype=float)
        self.x_mean_ = np.nanmean(values, axis=0)
        x_std = np.nanstd(values, axis=0, ddof=1)
        self.x_scale_ = np.where(np.isfinite(x_std) & (x_std > 1e-12), x_std, 1.0)
        x_scaled = (values - self.x_mean_) / self.x_scale_
        y_values = target.to_numpy(dtype=float)
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
        values = (frame.to_numpy(dtype=float) - self.x_mean_) / self.x_scale_
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
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean, scale, (values - mean) / scale


def _standardize_vector(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))
    scale = std if np.isfinite(std) and std > 1e-12 else 1.0
    return mean, scale, (values - mean) / scale


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


__all__ = ["gru", "lstm", "nn"]
