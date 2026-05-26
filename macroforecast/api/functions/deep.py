"""Standalone deep learning regression wrappers for the L4 deep sub-family.

Exposes four fit callables -- ``mlp_fit``, ``lstm_fit``, ``gru_fit``,
``transformer_fit`` -- each returning a frozen dataclass that conforms
structurally to :class:`~macroforecast.functions.FitResultBase`.

All callables follow the C28 lazy-import paradigm: ``_build_l4_model``
from ``macroforecast.core.runtime`` is imported inside the function body
to avoid circular imports.  For the sklearn MLP family no extra is needed;
torch families require ``pip install macroforecast[deep]``.

Cycle 36 -- L4 deep family standalone-ization (4 ops).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLPFitResult:
    """Result of :func:`mlp_fit`.

    Attributes
    ----------
    n_params :
        Total number of trainable parameters (weights + biases across all
        layers).
    n_features_in_ :
        Number of input features seen during fit.
    hidden_layer_sizes :
        Tuple of hidden layer widths, e.g. ``(32, 16)``.
    epochs_used :
        Number of optimiser iterations completed (= ``n_iter_`` from sklearn).
    final_loss :
        Training MSE at the end of fitting (``loss_`` from sklearn).
    _model :
        Internal fitted ``sklearn.neural_network.MLPRegressor``.
        Not part of the public contract.
    """

    n_params: int
    n_features_in_: int
    hidden_layer_sizes: tuple
    epochs_used: int
    final_loss: float
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predictions for new data.

        Parameters
        ----------
        X :
            Feature matrix.  Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(self._model.predict(X), dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable architecture metadata table.

        Returns
        -------
        str
            Statsmodels-style table showing model_type, hidden_size
            (architecture), n_features, n_params, epochs_used, final_loss.
        """
        sep = "=" * 78
        dash = "-" * 78
        lines = [
            sep,
            f"{'MLP Results':^78}",
            sep,
            f"{'model_type:':35s} {'mlp':>20s}",
            f"{'hidden_layer_sizes:':35s} {str(self.hidden_layer_sizes):>20s}",
            f"{'n_features:':35s} {self.n_features_in_:>20d}",
            f"{'n_params:':35s} {self.n_params:>20d}",
            f"{'epochs_used:':35s} {self.epochs_used:>20d}",
            f"{'final_loss:':35s} {self.final_loss:>20.6f}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class LSTMFitResult:
    """Result of :func:`lstm_fit`.

    Attributes
    ----------
    n_params :
        Total number of trainable parameters in the LSTM + head.
    n_features_in_ :
        Number of input features seen during fit.
    hidden_size :
        Width of the LSTM hidden state.
    epochs_used :
        Number of training epochs completed (= ``n_epochs`` parameter).
    final_loss :
        Training MSE computed via a no-grad forward pass after fitting.
    _model :
        Internal fitted torch model (``torch.nn.Module``).
        Not part of the public contract.
    """

    n_params: int
    n_features_in_: int
    hidden_size: int
    epochs_used: int
    final_loss: float
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predictions for new data.

        Parameters
        ----------
        X :
            Feature matrix.  Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.fillna(0.0).values
        x_arr = np.asarray(X, dtype="float32")
        seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.from_numpy(seq)).numpy()
        return np.asarray(preds, dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable architecture metadata table.

        Returns
        -------
        str
            Statsmodels-style table showing model_type, hidden_size,
            n_features, n_params, epochs_used, final_loss.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'LSTM Results':^78}",
            sep,
            f"{'model_type:':35s} {'lstm':>20s}",
            f"{'hidden_size:':35s} {self.hidden_size:>20d}",
            f"{'n_features:':35s} {self.n_features_in_:>20d}",
            f"{'n_params:':35s} {self.n_params:>20d}",
            f"{'epochs_used:':35s} {self.epochs_used:>20d}",
            f"{'final_loss:':35s} {self.final_loss:>20.6f}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class GRUFitResult:
    """Result of :func:`gru_fit`.

    Attributes
    ----------
    n_params :
        Total number of trainable parameters in the GRU + head.
    n_features_in_ :
        Number of input features seen during fit.
    hidden_size :
        Width of the GRU hidden state.
    epochs_used :
        Number of training epochs completed (= ``n_epochs`` parameter).
    final_loss :
        Training MSE computed via a no-grad forward pass after fitting.
    _model :
        Internal fitted torch model (``torch.nn.Module``).
        Not part of the public contract.
    """

    n_params: int
    n_features_in_: int
    hidden_size: int
    epochs_used: int
    final_loss: float
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predictions for new data.

        Parameters
        ----------
        X :
            Feature matrix.  Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.fillna(0.0).values
        x_arr = np.asarray(X, dtype="float32")
        seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.from_numpy(seq)).numpy()
        return np.asarray(preds, dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable architecture metadata table.

        Returns
        -------
        str
            Statsmodels-style table showing model_type, hidden_size,
            n_features, n_params, epochs_used, final_loss.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'GRU Results':^78}",
            sep,
            f"{'model_type:':35s} {'gru':>20s}",
            f"{'hidden_size:':35s} {self.hidden_size:>20d}",
            f"{'n_features:':35s} {self.n_features_in_:>20d}",
            f"{'n_params:':35s} {self.n_params:>20d}",
            f"{'epochs_used:':35s} {self.epochs_used:>20d}",
            f"{'final_loss:':35s} {self.final_loss:>20.6f}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class TransformerFitResult:
    """Result of :func:`transformer_fit`.

    Attributes
    ----------
    n_params :
        Total number of trainable parameters in the Transformer encoder + head.
    n_features_in_ :
        Number of input features seen during fit (= d_model).
    hidden_size :
        dim_feedforward of the single TransformerEncoderLayer.
    epochs_used :
        Number of training epochs completed (= ``n_epochs`` parameter).
    final_loss :
        Training MSE computed via a no-grad forward pass after fitting.
    _model :
        Internal fitted torch model (``torch.nn.Module``).
        Not part of the public contract.
    """

    n_params: int
    n_features_in_: int
    hidden_size: int
    epochs_used: int
    final_loss: float
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predictions for new data.

        Parameters
        ----------
        X :
            Feature matrix.  Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.fillna(0.0).values
        x_arr = np.asarray(X, dtype="float32")
        seq = x_arr.reshape(x_arr.shape[0], 1, x_arr.shape[1])
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.from_numpy(seq)).numpy()
        return np.asarray(preds, dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable architecture metadata table.

        Returns
        -------
        str
            Statsmodels-style table showing model_type, hidden_size,
            n_features, n_params, epochs_used, final_loss.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'Transformer Results':^78}",
            sep,
            f"{'model_type:':35s} {'transformer':>20s}",
            f"{'hidden_size:':35s} {self.hidden_size:>20d}",
            f"{'n_features:':35s} {self.n_features_in_:>20d}",
            f"{'n_params:':35s} {self.n_params:>20d}",
            f"{'epochs_used:':35s} {self.epochs_used:>20d}",
            f"{'final_loss:':35s} {self.final_loss:>20.6f}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helper: build + train a torch sequence model
# ---------------------------------------------------------------------------

def _fit_torch_sequence(
    kind: str,
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    hidden_size: int,
    n_epochs: int,
    random_state: int,
) -> tuple[Any, int, int, float]:
    """Fit a torch sequence model (lstm/gru/transformer).

    Returns
    -------
    tuple of (model, n_params, n_features, final_loss)
    """
    try:
        import torch
    except ImportError as e:
        raise NotImplementedError(
            "Torch-based sequence models (LSTM/GRU/Transformer) require PyTorch. "
            "Install with `pip install macroforecast[torch]` or `pip install torch`."
        ) from e
    from torch import nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    x_arr = X_df.fillna(0.0).to_numpy(dtype="float32")
    y_arr = np.asarray(y_ser, dtype="float32")
    n_features = x_arr.shape[1]

    seq = x_arr.reshape(x_arr.shape[0], 1, n_features)
    tensor_x = torch.from_numpy(seq)
    tensor_y = torch.from_numpy(y_arr)

    # Build underlying cell
    if kind == "lstm":
        cell = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
    elif kind == "gru":
        cell = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
    else:
        # transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_features,
            nhead=1,
            dim_feedforward=hidden_size,
            batch_first=True,
            dropout=0.1,
        )
        cell = nn.TransformerEncoder(enc_layer, num_layers=1)

    # Capture architecture params in closure (avoids capturing mutable self)
    _kind = kind
    _hidden = hidden_size

    class _Sequence(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = cell
            # LSTM/GRU: head maps hidden_size -> 1
            # Transformer: head maps d_model (= n_features) -> 1
            head_in = _hidden if _kind != "transformer" else n_features
            self.head = nn.Linear(head_in, 1)

        def forward(self, x):
            out = self.cell(x)
            if _kind in {"lstm", "gru"}:
                return self.head(out[0][:, -1, :]).squeeze(-1)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = _Sequence()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for _ in range(n_epochs):
        optim.zero_grad()
        preds = model(tensor_x)
        loss = loss_fn(preds, tensor_y)
        loss.backward()
        optim.step()

    # Compute final loss via no-grad forward pass (spec §3 planner decision)
    with torch.no_grad():
        final_preds = model(tensor_x)
        final_loss = float(loss_fn(final_preds, tensor_y).item())

    n_params = int(sum(p.numel() for p in model.parameters()))
    return model, n_params, n_features, final_loss


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def mlp_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    hidden_layer_sizes: tuple = (32, 16),
    max_iter: int = 500,
    random_state: int = 0,
) -> MLPFitResult:
    """Standalone multi-layer perceptron (sklearn) regression.

    Calls the L4 mlp family adapter
    (``_build_l4_model("mlp", params)``) directly; bypasses the recipe pipeline.
    Produces bit-exact numeric output identical to recipe-based MLP with the
    same parameter values.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    hidden_layer_sizes :
        Tuple of hidden layer widths.  Must be non-empty and all widths
        >= 1.  Default ``(32, 16)``.
    max_iter :
        Maximum number of optimiser iterations (epochs).  Must be >= 1.
        Default 500.
    random_state :
        Random seed for reproducibility.  Default 0.

    Returns
    -------
    MLPFitResult
        Fitted result exposing ``n_params``, ``n_features_in_``,
        ``hidden_layer_sizes``, ``epochs_used``, ``final_loss``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``max_iter < 1``, ``hidden_layer_sizes`` is empty, or any
        layer width < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mlp_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = mlp_fit(X, y, max_iter=50)
    >>> result.n_features_in_
    3

    References
    ----------
    Goodfellow, Bengio & Courville (2016) 'Deep Learning', MIT Press.
    """
    from ...core.runtime import _build_l4_model  # lazy import -- C28 paradigm

    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    if not hidden_layer_sizes:
        raise ValueError("hidden_layer_sizes must be non-empty")
    for i, w in enumerate(hidden_layer_sizes):
        if w < 1:
            raise ValueError(
                f"hidden_layer_sizes[{i}] must be >= 1, got {w!r}"
            )

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "hidden_layer_sizes": tuple(int(w) for w in hidden_layer_sizes),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    model = _build_l4_model("mlp", params)
    model.fit(X, y)

    n_params = int(
        sum(c.size for c in model.coefs_)
        + sum(b.size for b in model.intercepts_)
    )
    epochs_used = int(model.n_iter_)
    final_loss = float(model.loss_)

    return MLPFitResult(
        n_params=n_params,
        n_features_in_=int(model.n_features_in_),
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        epochs_used=epochs_used,
        final_loss=final_loss,
        _model=model,
    )


def lstm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    hidden_size: int = 32,
    n_epochs: int = 50,
    random_state: int = 0,
) -> LSTMFitResult:
    """Standalone LSTM regression (torch).

    Calls the L4 lstm family adapter
    (``_build_l4_model("lstm", params)``) architecture directly; bypasses the
    recipe pipeline.  Requires ``pip install macroforecast[deep]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    hidden_size :
        Width of the LSTM hidden state.  Must be >= 2.  Default 32.
    n_epochs :
        Number of training epochs.  Must be >= 1.  Default 50.
    random_state :
        Random seed for reproducibility.  Sets both ``torch.manual_seed``
        and ``np.random.seed``.  Default 0.

    Returns
    -------
    LSTMFitResult
        Fitted result exposing ``n_params``, ``n_features_in_``,
        ``hidden_size``, ``epochs_used``, ``final_loss``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``hidden_size < 2`` or ``n_epochs < 1``.
    NotImplementedError
        If the ``torch`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import lstm_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = lstm_fit(X, y, n_epochs=5)
    >>> result.n_features_in_
    3

    References
    ----------
    Hochreiter & Schmidhuber (1997) 'Long short-term memory',
    Neural Computation 9(8).
    """
    if hidden_size < 2:
        raise ValueError(f"hidden_size must be >= 2, got {hidden_size!r}")
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    # Fit using the shared helper (mirrors _TorchSequenceModel.fit internals
    # bit-exactly, with identical torch/np seeds and identical architecture)
    torch_model, n_params, n_features, final_loss = _fit_torch_sequence(
        kind="lstm",
        X_df=X,
        y_ser=y,
        hidden_size=int(hidden_size),
        n_epochs=int(n_epochs),
        random_state=int(random_state),
    )

    return LSTMFitResult(
        n_params=n_params,
        n_features_in_=n_features,
        hidden_size=int(hidden_size),
        epochs_used=int(n_epochs),
        final_loss=final_loss,
        _model=torch_model,
    )


def gru_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    hidden_size: int = 32,
    n_epochs: int = 50,
    random_state: int = 0,
) -> GRUFitResult:
    """Standalone GRU regression (torch).

    Calls the L4 gru family adapter
    (``_build_l4_model("gru", params)``) architecture directly; bypasses the
    recipe pipeline.  Requires ``pip install macroforecast[deep]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    hidden_size :
        Width of the GRU hidden state.  Must be >= 2.  Default 32.
    n_epochs :
        Number of training epochs.  Must be >= 1.  Default 50.
    random_state :
        Random seed for reproducibility.  Sets both ``torch.manual_seed``
        and ``np.random.seed``.  Default 0.

    Returns
    -------
    GRUFitResult
        Fitted result exposing ``n_params``, ``n_features_in_``,
        ``hidden_size``, ``epochs_used``, ``final_loss``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``hidden_size < 2`` or ``n_epochs < 1``.
    NotImplementedError
        If the ``torch`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import gru_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = gru_fit(X, y, n_epochs=5)
    >>> result.n_features_in_
    3

    References
    ----------
    Cho et al. (2014) 'Learning Phrase Representations using RNN
    Encoder-Decoder for Statistical Machine Translation', EMNLP.
    """
    if hidden_size < 2:
        raise ValueError(f"hidden_size must be >= 2, got {hidden_size!r}")
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    torch_model, n_params, n_features, final_loss = _fit_torch_sequence(
        kind="gru",
        X_df=X,
        y_ser=y,
        hidden_size=int(hidden_size),
        n_epochs=int(n_epochs),
        random_state=int(random_state),
    )

    return GRUFitResult(
        n_params=n_params,
        n_features_in_=n_features,
        hidden_size=int(hidden_size),
        epochs_used=int(n_epochs),
        final_loss=final_loss,
        _model=torch_model,
    )


def transformer_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    hidden_size: int = 32,
    n_epochs: int = 50,
    random_state: int = 0,
) -> TransformerFitResult:
    """Standalone Transformer encoder regression (torch).

    Calls the L4 transformer family adapter
    (``_build_l4_model("transformer", params)``) architecture directly;
    bypasses the recipe pipeline.  Single ``TransformerEncoderLayer`` with
    ``nhead=1`` and ``dim_feedforward=hidden_size``.  Requires
    ``pip install macroforecast[deep]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    hidden_size :
        ``dim_feedforward`` of the TransformerEncoderLayer.  Must be >= 2.
        Default 32.
    n_epochs :
        Number of training epochs.  Must be >= 1.  Default 50.
    random_state :
        Random seed for reproducibility.  Sets both ``torch.manual_seed``
        and ``np.random.seed``.  Default 0.

    Returns
    -------
    TransformerFitResult
        Fitted result exposing ``n_params``, ``n_features_in_``,
        ``hidden_size``, ``epochs_used``, ``final_loss``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``hidden_size < 2`` or ``n_epochs < 1``.
    NotImplementedError
        If the ``torch`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import transformer_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = transformer_fit(X, y, n_epochs=5)
    >>> result.n_features_in_
    3

    References
    ----------
    Vaswani et al. (2017) 'Attention is all you need', NeurIPS.
    """
    if hidden_size < 2:
        raise ValueError(f"hidden_size must be >= 2, got {hidden_size!r}")
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1, got {n_epochs!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    torch_model, n_params, n_features, final_loss = _fit_torch_sequence(
        kind="transformer",
        X_df=X,
        y_ser=y,
        hidden_size=int(hidden_size),
        n_epochs=int(n_epochs),
        random_state=int(random_state),
    )

    return TransformerFitResult(
        n_params=n_params,
        n_features_in_=n_features,
        hidden_size=int(hidden_size),
        epochs_used=int(n_epochs),
        final_loss=final_loss,
        _model=torch_model,
    )


__all__ = [
    "MLPFitResult",
    "mlp_fit",
    "LSTMFitResult",
    "lstm_fit",
    "GRUFitResult",
    "gru_fit",
    "TransformerFitResult",
    "transformer_fit",
    # C64: HemisphereNN gap callable
    "HemisphereNNFitResult",
    "hemisphere_nn_fit",
]


# ---------------------------------------------------------------------------
# C64: HemisphereNN gap callable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HemisphereNNFitResult:
    """Result of :func:`hemisphere_nn_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_HemisphereNN`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point predictions (mean hemisphere output) for new data.

        Parameters
        ----------
        X :
            Feature matrix. Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        return np.asarray(self._model.predict(X), dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable summary of the HemisphereNN fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and key parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        neurons = params.get("neurons", getattr(self._model, "neurons", "?"))
        B = params.get("B", getattr(self._model, "B", "?"))
        lines = [
            sep,
            f"{'HemisphereNN Results':^78}",
            sep,
            f"{'neurons:':35s} {str(neurons):>20s}",
            f"{'B (bags):':35s} {str(B):>20s}",
            sep,
        ]
        return "\n".join(lines)


def hemisphere_nn_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> HemisphereNNFitResult:
    """Fit a Hemisphere Neural Networks density forecaster (Goulet Coulombe et al. 2025).

    Standalone callable that constructs a ``_HemisphereNN`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Raises NotImplementedError if macroforecast[deep] is not installed.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_HemisphereNN`` constructor
        (e.g., ``neurons=64``, ``B=100``, ``nu=None``).

    Returns
    -------
    HemisphereNNFitResult
        Fitted result exposing ``.predict(X)`` and ``.summary()`` methods.

    Raises
    ------
    NotImplementedError
        If macroforecast[deep] (PyTorch) is not installed.

    References
    ----------
    Goulet Coulombe, Frenette, Klieber (2025),
    "Hemisphere Neural Networks for Density Forecasting",
    Journal of Applied Econometrics.
    """
    from ...core.runtime import _HemisphereNN

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _HemisphereNN(**kwargs)
    model.fit(X, y)
    return HemisphereNNFitResult(_model=model)
