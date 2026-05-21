"""Standalone miscellaneous regression wrappers for the L4 misc sub-family.

Exposes six fit callables:

    svr_linear_fit, svr_rbf_fit, svr_poly_fit, knn_fit,
    kernel_ridge_fit, mars_fit

each returning a frozen dataclass that conforms structurally to
:class:`~macroforecast.functions.FitResultBase`.

All callables call ``_build_l4_model`` from ``macroforecast.core.runtime``
directly, producing bit-exact numeric output identical to the recipe DAG
with the same parameter values.

Cycle 37 -- L4 misc family standalone-ization (6 ops).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: convert ndarray inputs to DataFrame/Series
# ---------------------------------------------------------------------------

def _to_frame(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _to_series(y: np.ndarray | pd.Series) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y).ravel(), name="y")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SVRLinearFitResultV0:
    """Pre-consolidation result type for :func:`svr_linear_fit`.

    Superseded by :class:`SVRFitResult`. The public name
    ``SVRLinearFitResult`` is now a TypeAlias for ``SVRFitResult``.

    Attributes
    ----------
    C :
        Regularisation parameter used.
    n_support_vectors :
        Number of support vectors from the fitted SVR.
    _model :
        Internal fitted ``sklearn.svm.SVR(kernel='linear')`` instance.
        Not part of the public contract.
    """

    C: float
    n_support_vectors: int
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
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the SVR linear fit result.

        Returns
        -------
        str
            Statsmodels-style table showing C and support vector count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'SVR Linear Results':^78}",
            sep,
            f"{'C:':35s} {self.C:>20.4f}",
            f"{'n_support_vectors:':35s} {self.n_support_vectors:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class _SVRRBFFitResultV0:
    """Pre-consolidation result type for :func:`svr_rbf_fit`.

    Superseded by :class:`SVRFitResult`. The public name
    ``SVRRBFFitResult`` is now a TypeAlias for ``SVRFitResult``.

    Attributes
    ----------
    C :
        Regularisation parameter used.
    gamma :
        RBF kernel bandwidth parameter (string or float).
    n_support_vectors :
        Number of support vectors from the fitted SVR.
    _model :
        Internal fitted ``sklearn.svm.SVR(kernel='rbf')`` instance.
        Not part of the public contract.
    """

    C: float
    gamma: Any
    n_support_vectors: int
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
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the SVR RBF fit result.

        Returns
        -------
        str
            Statsmodels-style table showing C, gamma, and support vector count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'SVR RBF Results':^78}",
            sep,
            f"{'C:':35s} {self.C:>20.4f}",
            f"{'gamma:':35s} {str(self.gamma):>20s}",
            f"{'n_support_vectors:':35s} {self.n_support_vectors:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class _SVRPolyFitResultV0:
    """Pre-consolidation result type for :func:`svr_poly_fit`.

    Superseded by :class:`SVRFitResult`. The public name
    ``SVRPolyFitResult`` is now a TypeAlias for ``SVRFitResult``.

    Attributes
    ----------
    C :
        Regularisation parameter used.
    degree :
        Polynomial degree.
    n_support_vectors :
        Number of support vectors from the fitted SVR.
    _model :
        Internal fitted ``sklearn.svm.SVR(kernel='poly')`` instance.
        Not part of the public contract.
    """

    C: float
    degree: int
    n_support_vectors: int
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
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the SVR poly fit result.

        Returns
        -------
        str
            Statsmodels-style table showing C, degree, and support vector
            count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'SVR Poly Results':^78}",
            sep,
            f"{'C:':35s} {self.C:>20.4f}",
            f"{'degree:':35s} {self.degree:>20d}",
            f"{'n_support_vectors:':35s} {self.n_support_vectors:>20d}",
            sep,
        ]
        return "\n".join(lines)



@dataclass(frozen=True)
class SVRFitResult:
    """Consolidated result for SVR family callables.

    All three SVR kernel variants (:func:`svr_linear_fit`, :func:`svr_rbf_fit`,
    :func:`svr_poly_fit`) return this single type, distinguished by the
    ``kernel`` attribute.

    Attributes
    ----------
    kernel :
        SVR kernel type: ``"linear"`` | ``"rbf"`` | ``"poly"``.
    C :
        Regularisation parameter used.
    n_support_vectors :
        Number of support vectors from the fitted SVR.
    gamma :
        RBF/poly bandwidth parameter.  ``None`` for linear kernel.
    degree :
        Polynomial degree.  ``None`` for non-poly kernels.
    _model :
        Internal fitted ``sklearn.svm.SVR`` instance.
        Not part of the public contract.
    """

    kernel: str
    C: float
    n_support_vectors: int
    gamma: Any
    degree: Any
    _model: Any

    def predict(self, X: "np.ndarray | pd.DataFrame") -> "np.ndarray":
        """Return predictions for new data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the SVR fit result."""
        sep = "=" * 78
        label = f"SVR [{self.kernel.upper()}] Results"
        lines = [
            sep,
            f"{label:^78}",
            sep,
            f"{'kernel:':35s} {self.kernel:>20s}",
            f"{'C:':35s} {self.C:>20.4f}",
            f"{'n_support_vectors:':35s} {self.n_support_vectors:>20d}",
        ]
        if self.gamma is not None:
            lines.append(f"{'gamma:':35s} {str(self.gamma):>20s}")
        if self.degree is not None:
            lines.append(f"{'degree:':35s} {self.degree:>20d}")
        lines.append(sep)
        return "\n".join(lines)


# Backward-compat aliases so existing code referencing the old specific types
# still works at runtime.
SVRLinearFitResult: TypeAlias = SVRFitResult
SVRRBFFitResult: TypeAlias = SVRFitResult
SVRPolyFitResult: TypeAlias = SVRFitResult

@dataclass(frozen=True)
class KNNFitResult:
    """Result of :func:`knn_fit`.

    Attributes
    ----------
    n_neighbors :
        Neighbour count k requested (may be clipped by training set size).
    n_neighbors_used :
        Actual k used (= min(n_neighbors, n_train_samples)).
    n_features_in_ :
        Number of features seen at fit time.
    weights :
        Weight function used in prediction: ``"uniform"`` or ``"distance"``.
    _model :
        Internal fitted ``_AutoClipKNN`` instance.
        Not part of the public contract.
    """

    n_neighbors: int
    n_neighbors_used: int
    n_features_in_: int
    weights: str
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
        return np.asarray(
            self._model.predict(pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])),
            dtype=float,
        )

    def summary(self) -> str:
        """Return a human-readable text summary of the KNN fit result.

        Returns
        -------
        str
            Statsmodels-style table showing neighbour counts, weights, and features.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'KNN Results':^78}",
            sep,
            f"{'n_neighbors (requested):':35s} {self.n_neighbors:>20d}",
            f"{'n_neighbors (used):':35s} {self.n_neighbors_used:>20d}",
            f"{'weights:':35s} {self.weights:>20s}",
            f"{'n_features_in_:':35s} {self.n_features_in_:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class KernelRidgeFitResult:
    """Result of :func:`kernel_ridge_fit`.

    Attributes
    ----------
    alpha :
        Ridge penalty (regularisation strength).
    kernel :
        Kernel name, e.g. ``'rbf'``, ``'linear'``.
    n_features_in_ :
        Number of features seen at fit time.
    _model :
        Internal fitted ``sklearn.kernel_ridge.KernelRidge`` instance.
        Not part of the public contract.
    """

    alpha: float
    kernel: str
    n_features_in_: int
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
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the Kernel Ridge fit result.

        Returns
        -------
        str
            Statsmodels-style table showing alpha, kernel, and feature count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'Kernel Ridge Results':^78}",
            sep,
            f"{'alpha:':35s} {self.alpha:>20.4f}",
            f"{'kernel:':35s} {self.kernel:>20s}",
            f"{'n_features_in_:':35s} {self.n_features_in_:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MARSFitResult:
    """Result of :func:`mars_fit`.

    Attributes
    ----------
    n_terms :
        Number of basis terms in the fitted MARS model.
    n_features_in_ :
        Number of features seen at fit time.
    _model :
        Internal fitted ``pyearth.Earth`` instance.
        Not part of the public contract.
    """

    n_terms: int
    n_features_in_: int
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
        """Return a human-readable text summary of the MARS fit result.

        Returns
        -------
        str
            Statsmodels-style table showing number of basis terms and features.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'MARS Results':^78}",
            sep,
            f"{'n_terms:':35s} {self.n_terms:>20d}",
            f"{'n_features_in_:':35s} {self.n_features_in_:>20d}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def svr_linear_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    C: float = 1.0,
) -> SVRLinearFitResult:
    """Standalone SVR with linear kernel.

    Calls ``_build_l4_model("svr_linear", params)`` directly; bypasses the
    recipe DAG.  Epsilon-insensitive loss + L2 regularisation, sparse in
    support vectors.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    C :
        Regularisation parameter.  Must be > 0.  Default 1.0.

    Returns
    -------
    SVRLinearFitResult
        Fitted result exposing ``C``, ``n_support_vectors``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``C <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import svr_linear_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = svr_linear_fit(X, y)
    >>> result.C
    1.0

    References
    ----------
    Drucker, Burges, Kaufman, Smola & Vapnik (1997) 'Support Vector
    Regression Machines', NeurIPS.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if C <= 0:
        raise ValueError(f"C must be > 0, got {C!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"C": float(C)}
    model = _build_l4_model("svr_linear", params)
    model.fit(X_df.values, y_s.values)

    return SVRFitResult(
        kernel="linear",
        C=float(C),
        n_support_vectors=int(model.n_support_[0]),
        gamma=None,
        degree=None,
        _model=model,
    )


def svr_rbf_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    C: float = 1.0,
    gamma: str | float = "scale",
) -> SVRRBFFitResult:
    """Standalone SVR with RBF kernel.

    Calls ``_build_l4_model("svr_rbf", params)`` directly; bypasses the
    recipe DAG.  Non-linear regression via kernel trick.  Slow on large
    panels (O(n^3)).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    C :
        Regularisation parameter.  Must be > 0.  Default 1.0.
    gamma :
        RBF kernel bandwidth.  ``'scale'`` (default) = 1 / (n_features *
        X.var()); ``'auto'`` = 1 / n_features; or a positive float.

    Returns
    -------
    SVRRBFFitResult
        Fitted result exposing ``C``, ``gamma``, ``n_support_vectors``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``C <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import svr_rbf_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = svr_rbf_fit(X, y)
    >>> result.C
    1.0
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if C <= 0:
        raise ValueError(f"C must be > 0, got {C!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"C": float(C), "gamma": gamma}
    model = _build_l4_model("svr_rbf", params)
    model.fit(X_df.values, y_s.values)

    return SVRFitResult(
        kernel="rbf",
        C=float(C),
        n_support_vectors=int(model.n_support_[0]),
        gamma=gamma,
        degree=None,
        _model=model,
    )


def svr_poly_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    C: float = 1.0,
    degree: int = 3,
) -> SVRPolyFitResult:
    """Standalone SVR with polynomial kernel.

    Calls ``_build_l4_model("svr_poly", params)`` directly; bypasses the
    recipe DAG.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    C :
        Regularisation parameter.  Must be > 0.  Default 1.0.
    degree :
        Polynomial degree.  Must be >= 1.  Default 3.

    Returns
    -------
    SVRPolyFitResult
        Fitted result exposing ``C``, ``degree``, ``n_support_vectors``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``C <= 0`` or ``degree < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import svr_poly_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = svr_poly_fit(X, y)
    >>> result.degree
    3
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if C <= 0:
        raise ValueError(f"C must be > 0, got {C!r}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"C": float(C)}
    model = _build_l4_model("svr_poly", params)
    model.set_params(degree=int(degree))
    model.fit(X_df.values, y_s.values)

    return SVRFitResult(
        kernel="poly",
        C=float(C),
        n_support_vectors=int(model.n_support_[0]),
        gamma=None,
        degree=int(degree),
        _model=model,
    )


def knn_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_neighbors: int = 5,
) -> KNNFitResult:
    """Standalone k-nearest-neighbours regression.

    Calls ``_build_l4_model("knn", params)`` directly; uses ``_AutoClipKNN``
    which automatically clips ``n_neighbors`` to the training-set size.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_neighbors :
        Number of neighbours k.  Must be >= 1.  Default 5.  Automatically
        clipped to training-set size if smaller.

    Returns
    -------
    KNNFitResult
        Fitted result exposing ``n_neighbors``, ``n_neighbors_used``,
        ``n_features_in_``, ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_neighbors < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import knn_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = knn_fit(X, y)
    >>> result.n_neighbors
    5

    References
    ----------
    Cover & Hart (1967) 'Nearest neighbor pattern classification', IEEE
    Trans. on Information Theory 13(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"n_neighbors": int(n_neighbors)}
    model = _build_l4_model("knn", params)
    model.fit(X_df, y_s)

    # _AutoClipKNN clips n_neighbors to the actual training set size.
    n_used = model._knn.n_neighbors if model._knn is not None else min(n_neighbors, len(y_s))

    # Extract the weights string from the AutoClipKNN wrapper.
    weights_str = getattr(model, "weights", "uniform")

    return KNNFitResult(
        n_neighbors=int(n_neighbors),
        n_neighbors_used=int(n_used),
        n_features_in_=int(X_df.shape[1]),
        weights=str(weights_str),
        _model=model,
    )


def kernel_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> KernelRidgeFitResult:
    """Standalone Kernel Ridge Regression.

    Calls ``_build_l4_model("kernel_ridge", params)`` directly; bypasses the
    recipe DAG.  Produces bit-exact output identical to recipe-based KRR
    with the same parameters.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    alpha :
        Ridge regularisation strength.  Must be > 0.  Default 1.0.
    kernel :
        Kernel type.  Any sklearn-supported kernel string (``'rbf'`` default,
        ``'linear'``, ``'poly'``, etc.).
    gamma :
        Kernel bandwidth for RBF / poly / sigmoid.  ``None`` (default) uses
        sklearn's ``1 / n_features``.

    Returns
    -------
    KernelRidgeFitResult
        Fitted result exposing ``alpha``, ``kernel``, ``n_features_in_``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``alpha <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import kernel_ridge_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = kernel_ridge_fit(X, y)
    >>> result.kernel
    'rbf'

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2022) 'How is Machine
    Learning Useful for Macroeconomic Forecasting?', Journal of Applied
    Econometrics 37(5): 920-964.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {
        "alpha": float(alpha),
        "kernel": str(kernel),
        "gamma": gamma,
    }
    model = _build_l4_model("kernel_ridge", params)
    model.fit(X_df.values, y_s.values)

    return KernelRidgeFitResult(
        alpha=float(alpha),
        kernel=str(kernel),
        n_features_in_=int(X_df.shape[1]),
        _model=model,
    )


def mars_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> MARSFitResult:
    """Standalone Multivariate Adaptive Regression Splines (MARS).

    Calls ``_build_l4_model("mars", params)`` directly; requires the optional
    ``pyearth`` package (``pip install macroforecast[mars]``).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).

    Returns
    -------
    MARSFitResult
        Fitted result exposing ``n_terms``, ``n_features_in_``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    NotImplementedError
        If ``pyearth`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mars_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = mars_fit(X, y)  # doctest: +SKIP
    >>> result.n_features_in_  # doctest: +SKIP
    3

    References
    ----------
    Friedman (1991) 'Multivariate Adaptive Regression Splines', Annals of
    Statistics 19(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("mars", params)
    model.fit(X_df.values, y_s.values)

    n_terms = int(model.summary().count("x"))
    # pyearth Earth exposes the coef_ attribute after fit; use len
    # of basis functions as n_terms.
    try:
        n_terms = int(len(model.coef_))
    except AttributeError:
        n_terms = 0

    return MARSFitResult(
        n_terms=n_terms,
        n_features_in_=int(X_df.shape[1]),
        _model=model,
    )


__all__ = [
    "SVRFitResult",
    "SVRLinearFitResult",
    "svr_linear_fit",
    "SVRRBFFitResult",
    "svr_rbf_fit",
    "SVRPolyFitResult",
    "svr_poly_fit",
    "KNNFitResult",
    "knn_fit",
    "KernelRidgeFitResult",
    "kernel_ridge_fit",
    "MARSFitResult",
    "mars_fit",
]
