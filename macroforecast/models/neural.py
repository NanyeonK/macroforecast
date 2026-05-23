"""macroforecast.models.neural -- public neural network model classes.

Cycle 64 -- 2 public classes (thin subclasses of private L4 implementations).

Design note on __init__ overrides:
Both public classes override __init__ and restore self.<param> = <raw_value>
AFTER calling super().__init__(). The private class __init__ applies type
coercions (max(), int(), float(), np.clip()) which produce new objects that
break sklearn clone() identity checks. By restoring raw values after
super().__init__(), get_params() reads the original objects, satisfying
sklearn's identity check in _clone_parametrized.

HemisphereNN additionally fixes the nu/nu_target mismatch: _HemisphereNN
stores nu as self.nu_target, which would break get_params() introspection
(BaseEstimator reads self.<param_name> to match __init__ param names).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from macroforecast.core.runtime import (
    _TorchSequenceModel,
    _HemisphereNN,
)

__all__ = [
    "SequenceModel",
    "HemisphereNN",
]


# ---------------------------------------------------------------------------
# Internal helper: set sklearn fit-tracking attributes
# ---------------------------------------------------------------------------

def _store_sklearn_fit_attrs(estimator: Any, X: pd.DataFrame) -> None:
    """Set feature_names_in_ and n_features_in_ per sklearn convention."""
    if isinstance(X, pd.DataFrame):
        estimator.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
        estimator.n_features_in_ = len(X.columns)
    else:
        # numpy array path: track n_features_in_ but not column names
        estimator.n_features_in_ = X.shape[1]


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------

class SequenceModel(_TorchSequenceModel, BaseEstimator, RegressorMixin):
    """LSTM / GRU / Transformer sequence model on lagged feature windows.

    Uses PyTorch when available (macroforecast[deep] extra). Raises
    NotImplementedError if torch is not installed. The sequence kind is
    selected via the ``kind`` parameter.

    Parameters
    ----------
    kind : str, default "lstm"
        Architecture: "lstm", "gru", or "transformer".
    hidden_size : int, default 32
        Hidden layer dimension.
    n_epochs : int, default 50
        Training epochs.
    random_state : int, default 0
        Random seed for torch and numpy.

    Note: Raises NotImplementedError if macroforecast[deep] is not installed.

    Standalone usage::

        from macroforecast.models import SequenceModel
        m = SequenceModel(kind="lstm", hidden_size=64)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    Reference: Hochreiter & Schmidhuber (1997) LSTM; Cho et al. (2014) GRU;
    Vaswani et al. (2017) Transformer.
    """

    def __init__(
        self,
        kind: str = "lstm",
        hidden_size: int = 32,
        n_epochs: int = 50,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            kind=kind,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            random_state=random_state,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        # _TorchSequenceModel coerces: max(2, int(hidden_size)), max(1, int(n_epochs)), etc.
        self.kind = kind
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SequenceModel":
        """Fit the sequence model and record sklearn feature tracking attributes.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        self
        """
        _store_sklearn_fit_attrs(self, X)
        return super().fit(X, y)  # type: ignore[return-value]


class HemisphereNN(_HemisphereNN, BaseEstimator, RegressorMixin):
    """Hemisphere Neural Networks density forecaster (Goulet Coulombe et al. 2025 JAE).

    Dual-head architecture: shared common core, then a mean hemisphere (h_m)
    and a variance hemisphere (h_v). Optimizes Gaussian negative log-likelihood.
    The variance head uses a softplus activation; the mean head is linear.
    After fit(), density attributes are accessible via self._hv_path and
    self._ensemble_preds.

    Parameters
    ----------
    lc : int, default 2
        Common-core depth (number of shared ReLU layers).
    lm : int, default 2
        Mean-hemisphere depth beyond the common core.
    lv : int, default 2
        Variance-hemisphere depth beyond the common core.
    neurons : int, default 64
        Width of each hidden layer (paper uses 400).
    dropout : float, default 0.2
        Dropout rate applied in both hemispheres.
    lr : float, default 1e-3
        Adam learning rate.
    n_epochs : int, default 100
        Maximum training epochs (with early stopping via patience).
    B : int, default 100
        Number of blocked-subsample bags (paper uses 1000; reduced for cost).
    sub_rate : float, default 0.80
        Per-bag row subsample fraction.
    nu : float or None, default None
        Variance-emphasis target (ratio mean(h_v)/var(y)); None triggers
        data-driven calibration.
    lambda_emphasis : float, default 1.0
        Lagrangian multiplier on the variance-emphasis penalty.
    patience : int, default 15
        Early stopping patience (epochs without validation improvement).
    val_frac : float, default 0.20
        Fraction of training data held out for early stopping.
    random_state : int, default 0
        Random seed.

    Note: Raises NotImplementedError if macroforecast[deep] is not installed.

    Note: explicit __init__ required for two reasons:
    (1) _HemisphereNN stores ``nu`` as ``self.nu_target``, which breaks
        BaseEstimator.get_params() introspection. This class restores
        ``self.nu = nu`` after super().__init__() so get_params() finds it.
    (2) _HemisphereNN applies type coercions (max(), int(), float(), np.clip())
        that produce new objects and break sklearn clone() identity check.
        All params are restored as raw values after super().__init__().

    Reference: Goulet Coulombe, Frenette, Klieber (2025),
    "Hemisphere Neural Networks for Density Forecasting",
    Journal of Applied Econometrics.
    """

    def __init__(
        self,
        lc: int = 2,
        lm: int = 2,
        lv: int = 2,
        neurons: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 100,
        B: int = 100,
        sub_rate: float = 0.80,
        nu: Any = None,
        lambda_emphasis: float = 1.0,
        patience: int = 15,
        val_frac: float = 0.20,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            lc=lc, lm=lm, lv=lv, neurons=neurons, dropout=dropout, lr=lr,
            n_epochs=n_epochs, B=B, sub_rate=sub_rate, nu=nu,
            lambda_emphasis=lambda_emphasis, patience=patience,
            val_frac=val_frac, random_state=random_state,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        # Also fixes nu/nu_target mismatch: _HemisphereNN sets self.nu_target = nu
        # but BaseEstimator.get_params() reads self.nu to match the __init__ param.
        self.lc = lc
        self.lm = lm
        self.lv = lv
        self.neurons = neurons
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.B = B
        self.sub_rate = sub_rate
        self.nu = nu  # restores nu (nu_target is also present from super().__init__)
        self.lambda_emphasis = lambda_emphasis
        self.patience = patience
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HemisphereNN":
        """Fit the HemisphereNN and record sklearn feature tracking attributes.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        self
        """
        _store_sklearn_fit_attrs(self, X)
        return super().fit(X, y)  # type: ignore[return-value]
