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
        if not self.quantile_levels or any(level <= 0.0 or level >= 1.0 for level in self.quantile_levels):
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
        x_all = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        target_variance_tensor = torch.tensor(target_variance_mean, dtype=torch.float32, device=device)
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
                _, full_variance = model(x_all)
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
        quantile_levels = self.quantile_levels if levels is None else tuple(float(level) for level in levels)
        if not quantile_levels or any(level <= 0.0 or level >= 1.0 for level in quantile_levels):
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
        values = (frame.to_numpy(dtype=float) - self.x_mean_) / self.x_scale_
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
    return {
        key: {inner_key: tensor.detach().clone() for inner_key, tensor in value.items()}
        for key, value in state.items()
    }


def _blocked_subsample_indices(
    n_obs: int,
    subsample: float,
    rng: np.random.Generator,
) -> np.ndarray:
    block_size = max(2, n_obs // 10)
    blocks = [np.arange(start, min(start + block_size, n_obs)) for start in range(0, n_obs, block_size)]
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


__all__ = ["gru", "hemisphere_nn", "lstm", "nn", "transformer"]
