"""Standalone linear regression wrappers for the L4 linear sub-family.

Exposes seven fit callables -- ``ols_fit``, ``lasso_fit``,
``elastic_net_fit``, ``lasso_path_fit``, ``bayesian_ridge_fit``,
``huber_fit``, ``glmboost_fit`` -- each returning a frozen dataclass
that conforms structurally to :class:`~macroforecast.functions.FitResultBase`.

All callables call ``_build_l4_model`` from ``macroforecast.core.runtime``
directly, producing bit-exact numeric output identical to the recipe pipeline
with the same parameter values.

Ridge (``ridge_fit``) was implemented as the Cycle 22 POC in ``ridge.py``
and is not duplicated here.

Cycle 28 -- L4 linear family standalone-ization.
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
class OLSFitResult:
    """Result of :func:`ols_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar.
    _model :
        Internal fitted estimator (``sklearn.linear_model.LinearRegression``).
        Not part of the public contract.
    """

    coef_: np.ndarray
    intercept_: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the OLS fit result.

        Returns
        -------
        str
            Statsmodels-style table showing predictor count, intercept,
            and coefficient vector.

        Notes
        -----
        Inferential statistics (std err, t-stat, p-value) are deferred
        to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'OLS Results':^78}",
            sep,
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class LassoFitResult:
    """Result of :func:`lasso_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar.
    alpha :
        Regularisation strength passed by the caller (= ``_model.alpha``).
    _model :
        Internal fitted estimator (``sklearn.linear_model.Lasso``).
    """

    coef_: np.ndarray
    intercept_: float
    alpha: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the Lasso fit result.

        Returns
        -------
        str
            Statsmodels-style table showing regularisation strength,
            predictor count, intercept, and coefficient vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'Lasso Results':^78}",
            sep,
            f"{'alpha (regularisation):':35s} {self.alpha:>20.4f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class ElasticNetFitResult:
    """Result of :func:`elastic_net_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar.
    alpha :
        Regularisation strength passed by the caller.
    l1_ratio :
        L1 mixing parameter passed by the caller (0 = ridge, 1 = lasso).
    _model :
        Internal fitted estimator (``sklearn.linear_model.ElasticNet``).
    """

    coef_: np.ndarray
    intercept_: float
    alpha: float
    l1_ratio: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the ElasticNet fit result.

        Returns
        -------
        str
            Statsmodels-style table showing regularisation strength, l1_ratio,
            predictor count, intercept, and coefficient vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'ElasticNet Results':^78}",
            sep,
            f"{'alpha (regularisation):':35s} {self.alpha:>20.4f}",
            f"{'l1_ratio:':35s} {self.l1_ratio:>20.4f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class LassoPathFitResult:
    """Result of :func:`lasso_path_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar.
    alpha_selected :
        CV-selected regularisation strength (= ``_model.alpha_``).
    _model :
        Internal fitted estimator (``sklearn.linear_model.LassoCV``).
    """

    coef_: np.ndarray
    intercept_: float
    alpha_selected: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the LassoPath (LassoCV) fit result.

        Returns
        -------
        str
            Statsmodels-style table showing CV-selected alpha, predictor count,
            intercept, and coefficient vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'LassoPath (LassoCV) Results':^78}",
            sep,
            f"{'alpha (CV-selected):':35s} {self.alpha_selected:>20.6f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class BayesianRidgeFitResult:
    """Result of :func:`bayesian_ridge_fit`.

    Attributes
    ----------
    coef_ :
        Posterior mean coefficient vector, shape (n_features,).
    intercept_ :
        Posterior mean intercept scalar.
    alpha_ :
        Posterior noise precision (empirical Bayes estimate).
        *Not* the user-supplied regularisation ``alpha`` -- follows sklearn
        convention where ``BayesianRidge.alpha_`` is the noise precision.
    lambda_ :
        Posterior weight precision (empirical Bayes estimate).
    _model :
        Internal fitted estimator (``sklearn.linear_model.BayesianRidge``).
    """

    coef_: np.ndarray
    intercept_: float
    alpha_: float
    lambda_: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the BayesianRidge fit result.

        Returns
        -------
        str
            Statsmodels-style table showing empirical-Bayes noise and weight
            precision estimates, predictor count, intercept, and coefficient
            vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'BayesianRidge Results':^78}",
            sep,
            f"{'alpha_ (noise precision):':35s} {self.alpha_:>20.6f}",
            f"{'lambda_ (weight precision):':35s} {self.lambda_:>20.6f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class HuberFitResult:
    """Result of :func:`huber_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar.
    epsilon :
        Huber loss transition point supplied by the caller.
    scale_ :
        Robust scale estimate from the fitted model (``_model.scale_``).
    _model :
        Internal fitted estimator (``sklearn.linear_model.HuberRegressor``).
    """

    coef_: np.ndarray
    intercept_: float
    epsilon: float
    scale_: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the Huber fit result.

        Returns
        -------
        str
            Statsmodels-style table showing epsilon, robust scale estimate,
            predictor count, intercept, and coefficient vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'Huber Results':^78}",
            sep,
            f"{'epsilon (loss transition):':35s} {self.epsilon:>20.4f}",
            f"{'scale_ (robust scale):':35s} {self.scale_:>20.6f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass(frozen=True)
class GLMBoostFitResult:
    """Result of :func:`glmboost_fit`.

    Attributes
    ----------
    coef_ :
        Fitted coefficient vector, shape (n_features,).
    intercept_ :
        Fitted intercept scalar (initialised to ``mean(y)``).
    n_iter :
        Number of boosting iterations supplied by the caller.
    learning_rate :
        Shrinkage factor per iteration supplied by the caller.
    _model :
        Internal fitted estimator (``macroforecast.core.runtime._GLMBoost``).
    """

    coef_: np.ndarray
    intercept_: float
    n_iter: int
    learning_rate: float
    _model: Any

    def summary(self) -> str:
        """Return a human-readable text summary of the GLMBoost fit result.

        Returns
        -------
        str
            Statsmodels-style table showing boosting iterations, learning rate,
            predictor count, intercept, and coefficient vector.

        Notes
        -----
        Inferential statistics are deferred to a future cycle.
        """
        k = len(self.coef_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        lines: list[str] = [
            sep,
            f"{'GLMBoost Results':^78}",
            sep,
            f"{'n_iter:':35s} {self.n_iter:>20d}",
            f"{'learning_rate:':35s} {self.learning_rate:>20.4f}",
            f"{'No. Predictors:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'coef':>12s}",
            dash,
            f"{'const':30s} {self.intercept_:>12.6f}",
        ]
        for name, coef in zip(feat_names, self.coef_):
            lines.append(f"{name:30s} {coef:>12.6f}")
        lines.append(sep)
        lines.append("Note: inferential statistics are deferred to a future cycle.")
        return "\n".join(lines)

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
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def ols_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> OLSFitResult:
    """Standalone ordinary least squares regression.

    Calls the L4 OLS family adapter (``_build_l4_model("ols", {})``)
    directly; bypasses the recipe pipeline.  Produces bit-exact numeric
    output identical to recipe-based OLS.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.

    Returns
    -------
    OLSFitResult
        Fitted result exposing ``coef_``, ``intercept_``, and a
        ``.predict(X)`` method.

    Notes
    -----
    OLS has no tuning parameters.  When the design matrix is rank-deficient,
    sklearn raises an error.  For high-dimensional panels (p close to n),
    use ``ridge_fit`` or ``lasso_fit`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import ols_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1.0, 2.0, 3.0] + 0.1 * rng.randn(50)
    >>> result = ols_fit(X, y)
    >>> result.coef_.shape
    (3,)

    References
    ----------
    Greene (2018) *Econometric Analysis*, 8th ed., Pearson.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {}
    model = _build_l4_model("ols", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))

    return OLSFitResult(coef_=coef, intercept_=intercept, _model=model)


def lasso_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    max_iter: int = 20000,
) -> LassoFitResult:
    """Standalone Lasso regression (L1-regularised OLS).

    Calls the L4 Lasso family adapter (``_build_l4_model("lasso", params)``)
    directly; bypasses the recipe pipeline.  Produces bit-exact numeric output
    identical to recipe-based Lasso with the same parameter values.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    alpha :
        L1 regularisation strength.  Must be >= 0.  Default 1.0.
        Larger values force more coefficients to exactly zero.
    max_iter :
        Maximum number of coordinate descent iterations.  Default 20000
        (matches recipe default for convergence on large panels).

    Returns
    -------
    LassoFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``alpha``, and
        a ``.predict(X)`` method.

    Raises
    ------
    ValueError
        If ``alpha < 0`` or ``max_iter < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import lasso_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 10)
    >>> y = X[:, 0] + X[:, 1] + 0.1 * rng.randn(50)
    >>> result = lasso_fit(X, y, alpha=0.1)
    >>> result.alpha
    0.1

    References
    ----------
    Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso',
    JRSS-B 58(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {"alpha": float(alpha), "max_iter": int(max_iter)}
    model = _build_l4_model("lasso", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))

    return LassoFitResult(
        coef_=coef,
        intercept_=intercept,
        alpha=float(alpha),
        _model=model,
    )


def elastic_net_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
) -> ElasticNetFitResult:
    """Standalone elastic net regression (L1 + L2 hybrid).

    Calls the L4 elastic_net family adapter
    (``_build_l4_model("elastic_net", params)``) directly; bypasses
    the recipe pipeline.  Produces bit-exact numeric output identical to
    recipe-based elastic net with the same parameter values.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    alpha :
        Overall regularisation strength.  Must be >= 0.  Default 1.0.
    l1_ratio :
        Mixing parameter controlling the L1/L2 balance.  Must be in
        ``[0.0, 1.0]``.  ``0`` = pure ridge, ``1`` = pure lasso.
        Default 0.5.
    max_iter :
        Maximum coordinate descent iterations.  Default 20000.

    Returns
    -------
    ElasticNetFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``alpha``,
        ``l1_ratio``, and a ``.predict(X)`` method.

    Raises
    ------
    ValueError
        If ``alpha < 0``, ``l1_ratio`` not in ``[0, 1]``, or
        ``max_iter < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import elastic_net_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 5)
    >>> y = X[:, 0] + 0.1 * rng.randn(50)
    >>> result = elastic_net_fit(X, y, alpha=0.5, l1_ratio=0.3)
    >>> result.l1_ratio
    0.3

    References
    ----------
    Zou & Hastie (2005) 'Regularization and variable selection via the
    elastic net', JRSS-B 67(2).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")
    if not (0.0 <= l1_ratio <= 1.0):
        raise ValueError(f"l1_ratio must be in [0.0, 1.0], got {l1_ratio!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "max_iter": int(max_iter),
    }
    model = _build_l4_model("elastic_net", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))

    return ElasticNetFitResult(
        coef_=coef,
        intercept_=intercept,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        _model=model,
    )


def lasso_path_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    cv: int = 5,
    max_iter: int = 20000,
    random_state: int | None = None,
) -> LassoPathFitResult:
    """Standalone Lasso with CV-selected alpha (LassoCV).

    Calls the L4 lasso_path family adapter
    (``_build_l4_model("lasso_path", params)``) directly; bypasses the
    recipe pipeline.  Wraps ``sklearn.linear_model.LassoCV`` and selects
    alpha automatically from a regularisation path via k-fold CV.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    cv :
        Number of cross-validation folds.  Must be >= 2.  Default 5.
    max_iter :
        Maximum coordinate descent iterations per alpha.  Default 20000.
    random_state :
        Integer seed for cross-validation fold generation.  Pass an
        integer for reproducible results.  Default ``None`` (uses
        system entropy; note: ``_build_l4_model`` uses ``0`` as default
        seed when ``random_state=None``).

    Returns
    -------
    LassoPathFitResult
        Fitted result exposing ``coef_``, ``intercept_``,
        ``alpha_selected`` (CV-selected regularisation strength), and
        a ``.predict(X)`` method.

    Raises
    ------
    ValueError
        If ``cv < 2`` or ``max_iter < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import lasso_path_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(80, 10)
    >>> y = X[:, 0] + X[:, 1] + 0.1 * rng.randn(80)
    >>> result = lasso_path_fit(X, y, cv=5, random_state=0)
    >>> result.alpha_selected > 0
    True

    References
    ----------
    Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso',
    JRSS-B 58(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {"cv": int(cv), "max_iter": int(max_iter)}
    if random_state is not None:
        params["random_state"] = int(random_state)

    model = _build_l4_model("lasso_path", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))
    alpha_selected = float(model.alpha_)

    return LassoPathFitResult(
        coef_=coef,
        intercept_=intercept,
        alpha_selected=alpha_selected,
        _model=model,
    )


def bayesian_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> BayesianRidgeFitResult:
    """Standalone Bayesian ridge regression with empirical-Bayes prior.

    Calls the L4 bayesian_ridge family adapter
    (``_build_l4_model("bayesian_ridge", {})``) directly; bypasses the
    recipe pipeline.  Uses ``sklearn.linear_model.BayesianRidge`` with all
    defaults.  Estimates noise precision (``alpha_``) and weight
    precision (``lambda_``) via type-II ML (empirical Bayes).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.

    Returns
    -------
    BayesianRidgeFitResult
        Fitted result exposing ``coef_``, ``intercept_``,
        ``alpha_`` (posterior noise precision), ``lambda_`` (posterior
        weight precision), and a ``.predict(X)`` method.

    Notes
    -----
    ``alpha_`` and ``lambda_`` are empirical-Bayes estimates of the
    gamma prior hyperparameters -- they are *not* a user-supplied
    regularisation parameter.  This naming follows sklearn convention and
    differs from the ``alpha`` regularisation parameter in other L4 families.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import bayesian_ridge_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1.0, 2.0, 3.0] + 0.1 * rng.randn(50)
    >>> result = bayesian_ridge_fit(X, y)
    >>> result.alpha_ > 0
    True

    References
    ----------
    MacKay (1992) 'Bayesian Interpolation', Neural Computation 4(3).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {}
    model = _build_l4_model("bayesian_ridge", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))
    alpha_ = float(model.alpha_)
    lambda_ = float(model.lambda_)

    return BayesianRidgeFitResult(
        coef_=coef,
        intercept_=intercept,
        alpha_=alpha_,
        lambda_=lambda_,
        _model=model,
    )


def huber_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    epsilon: float = 1.35,
    max_iter: int = 1000,
) -> HuberFitResult:
    """Standalone Huber regression (robust to outliers).

    Calls the L4 Huber family adapter
    (``_build_l4_model("huber", params)``) directly; bypasses the recipe
    pipeline.  Uses ``sklearn.linear_model.HuberRegressor`` which replaces the
    squared loss with the Huber loss, down-weighting large residuals.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    epsilon :
        Huber loss transition point.  Must be > 1.0 (sklearn requirement).
        Residuals with ``|r| <= epsilon * scale_`` are treated as inliers
        (quadratic); larger residuals are treated as outliers (linear).
        Default 1.35 (95% efficiency under Gaussian noise).
    max_iter :
        Maximum number of LBFGS iterations.  Default 1000.

    Returns
    -------
    HuberFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``epsilon``,
        ``scale_`` (robust scale estimate), and a ``.predict(X)`` method.

    Raises
    ------
    ValueError
        If ``epsilon <= 1.0`` or ``max_iter < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import huber_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1.0, 2.0, 3.0] + 0.1 * rng.randn(50)
    >>> result = huber_fit(X, y, epsilon=1.5)
    >>> result.epsilon
    1.5

    References
    ----------
    Huber (1964) 'Robust Estimation of a Location Parameter',
    Annals of Mathematical Statistics 35(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if epsilon <= 1.0:
        raise ValueError(
            f"epsilon must be > 1.0 (HuberRegressor requirement), got {epsilon!r}"
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {"epsilon": float(epsilon), "max_iter": int(max_iter)}
    model = _build_l4_model("huber", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))
    scale_ = float(model.scale_)

    return HuberFitResult(
        coef_=coef,
        intercept_=intercept,
        epsilon=float(epsilon),
        scale_=scale_,
        _model=model,
    )


def glmboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_iter: int = 100,
    learning_rate: float = 0.1,
) -> GLMBoostFitResult:
    """Standalone componentwise L2-boosting with linear base learners.

    Calls the L4 glmboost family adapter
    (``_build_l4_model("glmboost", params)``) directly; bypasses the
    recipe pipeline.  Implements Buhlmann-Hothorn (2007) componentwise boosting:
    at each iteration, picks the predictor most correlated with the residual
    and updates only its coefficient.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_iter :
        Number of boosting iterations.  Must be >= 1.  Default 100.
        Internally passed to ``_GLMBoost`` as ``n_iter`` (mapped from
        ``n_estimators`` in the recipe params dict).
    learning_rate :
        Shrinkage factor applied to each coefficient update.  Must be > 0.
        Default 0.1.

    Returns
    -------
    GLMBoostFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``n_iter``,
        ``learning_rate``, and a ``.predict(X)`` method.

    Raises
    ------
    ValueError
        If ``n_iter < 1`` or ``learning_rate <= 0``.

    Notes
    -----
    The ``n_iter`` parameter is mapped to ``n_estimators`` in the
    ``_build_l4_model`` params dict (the recipe key), since
    ``_build_l4_model("glmboost", params)`` reads
    ``params.get("n_estimators", 100)``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import glmboost_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 5)
    >>> y = X[:, 0] + X[:, 1] + 0.1 * rng.randn(50)
    >>> result = glmboost_fit(X, y, n_iter=200, learning_rate=0.05)
    >>> result.n_iter
    200

    References
    ----------
    Buhlmann & Hothorn (2007) 'Boosting algorithms: Regularization,
    prediction and model fitting', Statistical Science 22(4).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    # n_iter (user-facing) maps to n_estimators (recipe params key)
    params: dict[str, Any] = {
        "n_estimators": int(n_iter),
        "learning_rate": float(learning_rate),
    }
    model = _build_l4_model("glmboost", params)
    model.fit(X, y)

    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))

    return GLMBoostFitResult(
        coef_=coef,
        intercept_=intercept,
        n_iter=int(n_iter),
        learning_rate=float(learning_rate),
        _model=model,
    )


__all__ = [
    "OLSFitResult",
    "ols_fit",
    "LassoFitResult",
    "lasso_fit",
    "ElasticNetFitResult",
    "elastic_net_fit",
    "LassoPathFitResult",
    "lasso_path_fit",
    "BayesianRidgeFitResult",
    "bayesian_ridge_fit",
    "HuberFitResult",
    "huber_fit",
    "GLMBoostFitResult",
    "glmboost_fit",
]
