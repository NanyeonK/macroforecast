"""Standalone tree / ensemble regression wrappers for the L4 tree sub-family.

Exposes six fit callables -- ``random_forest_fit``, ``extra_trees_fit``,
``gradient_boosting_fit``, ``xgboost_fit``, ``lightgbm_fit``,
``catboost_fit`` -- each returning a frozen dataclass that conforms
structurally to :class:`~macroforecast.functions.FitResultBase`.

All callables call ``_build_l4_model`` from ``macroforecast.core.runtime``
directly, producing bit-exact numeric output identical to the recipe pipeline
with the same parameter values.

Cycle 35 -- L4 tree/ensemble family standalone-ization.
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
class RandomForestFitResult:
    """Result of :func:`random_forest_fit`.

    Attributes
    ----------
    feature_importances_ :
        Mean decrease in impurity per feature, shape (n_features,).
        Raw sklearn ``feature_importances_`` (sums to 1.0).
    n_estimators_used :
        Actual number of trees grown (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator (``sklearn.ensemble.RandomForestRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
        """Return a human-readable text summary of the Random Forest fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances.
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        lines: list[str] = [
            sep,
            f"{'RandomForest Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            lines.append(
                f"{feat_names[fi]:30s} {self.feature_importances_[fi]:>12.6f}"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class ExtraTreesFitResult:
    """Result of :func:`extra_trees_fit`.

    Attributes
    ----------
    feature_importances_ :
        Mean decrease in impurity per feature, shape (n_features,).
        Raw sklearn ``feature_importances_`` (sums to 1.0).
    n_estimators_used :
        Actual number of trees grown (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator (``sklearn.ensemble.ExtraTreesRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
        """Return a human-readable text summary of the Extra Trees fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances.
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        lines: list[str] = [
            sep,
            f"{'ExtraTrees Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            lines.append(
                f"{feat_names[fi]:30s} {self.feature_importances_[fi]:>12.6f}"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class GradientBoostingFitResult:
    """Result of :func:`gradient_boosting_fit`.

    Attributes
    ----------
    feature_importances_ :
        Feature importances from the fitted GBM, shape (n_features,).
        Raw sklearn ``feature_importances_`` (sums to 1.0).
    n_estimators_used :
        Number of boosting iterations (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator
        (``sklearn.ensemble.GradientBoostingRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
        """Return a human-readable text summary of the Gradient Boosting fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances.
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        lines: list[str] = [
            sep,
            f"{'GradientBoosting Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            lines.append(
                f"{feat_names[fi]:30s} {self.feature_importances_[fi]:>12.6f}"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class XGBoostFitResult:
    """Result of :func:`xgboost_fit`.

    Attributes
    ----------
    feature_importances_ :
        Feature importances from the fitted XGBoost model, shape (n_features,).
        XGBoost ``feature_importances_`` (gain-based, normalised to sum to 1.0).
    n_estimators_used :
        Number of boosting rounds (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator (``xgboost.XGBRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
        """Return a human-readable text summary of the XGBoost fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances.
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        lines: list[str] = [
            sep,
            f"{'XGBoost Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            lines.append(
                f"{feat_names[fi]:30s} {self.feature_importances_[fi]:>12.6f}"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class LightGBMFitResult:
    """Result of :func:`lightgbm_fit`.

    Attributes
    ----------
    feature_importances_ :
        Feature importances from the fitted LightGBM model, shape (n_features,).
        LightGBM ``feature_importances_`` (split count, raw integer counts
        normalised to float array).
    n_estimators_used :
        Number of boosting rounds (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator (``lightgbm.LGBMRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
        """Return a human-readable text summary of the LightGBM fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances as a percentage of total split counts.
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_in_"):
            feat_names = list(self._model.feature_names_in_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        total = self.feature_importances_.sum()
        lines: list[str] = [
            sep,
            f"{'LightGBM Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            fi_pct = self.feature_importances_[fi] / total * 100 if total > 0 else 0.0
            lines.append(
                f"{feat_names[fi]:30s} {fi_pct:>11.2f}%"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class CatBoostFitResult:
    """Result of :func:`catboost_fit`.

    Attributes
    ----------
    feature_importances_ :
        Feature importances from the fitted CatBoost model, shape (n_features,).
        CatBoost ``get_feature_importance()`` normalised to a float array.
    n_estimators_used :
        Number of boosting iterations (= ``n_estimators`` parameter).
    _model :
        Internal fitted estimator (``catboost.CatBoostRegressor``).
        Not part of the public contract.
    """

    feature_importances_: np.ndarray
    n_estimators_used: int
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
            1-D float array of predictions, guaranteed 1-D via ``.ravel()``.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(self._model.predict(X), dtype=float).ravel()

    def summary(self) -> str:
        """Return a human-readable text summary of the CatBoost fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ensemble size and top feature
            importances as a percentage (CatBoost PredictionValuesChange
            natively sums to 100; displayed with ``%`` suffix).
        """
        k = len(self.feature_importances_)
        feat_names: list[str] = []
        if hasattr(self._model, "feature_names_"):
            feat_names = list(self._model.feature_names_)
        if not feat_names:
            feat_names = [f"x{i}" for i in range(k)]
        sep = "=" * 78
        dash = "-" * 78
        top_n = min(3, k)
        idx_sorted = np.argsort(self.feature_importances_)[::-1]
        total = self.feature_importances_.sum()
        lines: list[str] = [
            sep,
            f"{'CatBoost Results':^78}",
            sep,
            f"{'n_estimators:':35s} {self.n_estimators_used:>20d}",
            f"{'No. Features:':35s} {k:>20d}",
            sep,
            f"{'Feature':30s} {'importance':>12s}",
            dash,
        ]
        for i in range(top_n):
            fi = idx_sorted[i]
            fi_pct = self.feature_importances_[fi] / total * 100 if total > 0 else 0.0
            lines.append(
                f"{feat_names[fi]:30s} {fi_pct:>11.2f}%"
            )
        if k > top_n:
            lines.append(f"{'... (top 3 shown)':30s}")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def random_forest_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int = 1,
) -> RandomForestFitResult:
    """Standalone random forest regression.

    Calls the L4 random_forest family adapter
    (``_build_l4_model("random_forest", params)``) directly; bypasses the
    recipe pipeline.  Produces bit-exact numeric output identical to recipe-based
    random forest with the same parameter values.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of trees in the forest.  Must be >= 1.  Default 200.
    max_depth :
        Maximum depth of each tree.  ``None`` (default) grows trees until
        leaves are pure or contain fewer than ``min_samples_leaf`` samples.
    min_samples_leaf :
        Minimum number of samples required at a leaf node.  Must be >= 1.
        Default 1.
    random_state :
        Random seed for reproducibility.  Default 0.
    n_jobs :
        Number of parallel jobs for fitting and prediction.  ``1`` (default)
        uses a single core.  ``-1`` uses all available cores.

    Returns
    -------
    RandomForestFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``n_estimators < 1`` or ``min_samples_leaf < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import random_forest_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = random_forest_fit(X, y)
    >>> result.feature_importances_.shape
    (5,)

    References
    ----------
    Breiman (2001) 'Random Forests', Machine Learning 45(1).
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "random_state": int(random_state),
        "min_samples_leaf": int(min_samples_leaf),
    }
    if max_depth is not None:
        params["max_depth"] = int(max_depth)

    model = _build_l4_model("random_forest", params)
    # n_jobs override: _build_l4_model hardcodes n_jobs=1 for safety;
    # override here to allow parallelism when caller requests it.
    model.n_jobs = int(n_jobs)
    # min_samples_leaf: _build_l4_model does not forward this param; set post-construction.
    model.min_samples_leaf = int(min_samples_leaf)
    model.fit(X, y)

    fi = np.asarray(model.feature_importances_, dtype=float)

    return RandomForestFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


def extra_trees_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int = 1,
) -> ExtraTreesFitResult:
    """Standalone extremely randomized trees regression.

    Calls the L4 extra_trees family adapter
    (``_build_l4_model("extra_trees", params)``) directly; bypasses the
    recipe pipeline.  Unlike random forests, extra trees use random split
    thresholds at each node, which reduces variance further at the cost
    of slight bias.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of trees in the forest.  Must be >= 1.  Default 200.
    max_depth :
        Maximum depth of each tree.  ``None`` (default) grows trees until
        leaves are pure or contain fewer than ``min_samples_leaf`` samples.
    min_samples_leaf :
        Minimum number of samples required at a leaf node.  Must be >= 1.
        Default 1.
    random_state :
        Random seed for reproducibility.  Default 0.
    n_jobs :
        Number of parallel jobs for fitting and prediction.  ``1`` (default)
        uses a single core.  ``-1`` uses all available cores.

    Returns
    -------
    ExtraTreesFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``n_estimators < 1`` or ``min_samples_leaf < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import extra_trees_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = extra_trees_fit(X, y)
    >>> result.feature_importances_.shape
    (5,)

    References
    ----------
    Geurts, Ernst & Wehenkel (2006) 'Extremely randomized trees',
    Machine Learning 63(1).
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "random_state": int(random_state),
        "min_samples_leaf": int(min_samples_leaf),
    }
    if max_depth is not None:
        params["max_depth"] = int(max_depth)

    model = _build_l4_model("extra_trees", params)
    # n_jobs override: _build_l4_model hardcodes n_jobs=1 for safety;
    # override here to allow parallelism when caller requests it.
    model.n_jobs = int(n_jobs)
    # min_samples_leaf: _build_l4_model does not forward this param; set post-construction.
    model.min_samples_leaf = int(min_samples_leaf)
    model.fit(X, y)

    fi = np.asarray(model.feature_importances_, dtype=float)

    return ExtraTreesFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


def gradient_boosting_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 0,
) -> GradientBoostingFitResult:
    """Standalone gradient-boosted regression trees (sklearn).

    Calls the L4 gradient_boosting family adapter
    (``_build_l4_model("gradient_boosting", params)``) directly; bypasses
    the recipe pipeline.  Sequential boosting with shallow trees minimising the
    squared error.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of boosting iterations.  Must be >= 1.  Default 200.
    learning_rate :
        Shrinkage applied to each tree's contribution.  Must be > 0.
        Default 0.1.
    max_depth :
        Maximum depth of each regression tree.  Must be >= 1.  Default 3.
    random_state :
        Random seed for reproducibility.  Default 0.

    Returns
    -------
    GradientBoostingFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``n_estimators < 1``, ``learning_rate <= 0``, or ``max_depth < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import gradient_boosting_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = gradient_boosting_fit(X, y)
    >>> result.feature_importances_.shape
    (5,)

    References
    ----------
    Friedman (2001) 'Greedy function approximation: A gradient boosting
    machine', Annals of Statistics 29(5).
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
    }
    model = _build_l4_model("gradient_boosting", params)
    model.fit(X, y)

    fi = np.asarray(model.feature_importances_, dtype=float)

    return GradientBoostingFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


def xgboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    random_state: int = 0,
) -> XGBoostFitResult:
    """Standalone XGBoost gradient-boosted trees regression.

    Calls the L4 xgboost family adapter
    (``_build_l4_model("xgboost", params)``) directly; bypasses the recipe
    recipe pipeline.  Requires ``pip install macroforecast[xgboost]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of boosting rounds.  Must be >= 1.  Default 300.
    learning_rate :
        Step size shrinkage per boosting round.  Must be > 0.  Default 0.1.
    max_depth :
        Maximum tree depth.  Must be >= 1.  Default 6.
    subsample :
        Row sub-sampling ratio per boosting round.  Must be in (0, 1].
        Default 1.0 (no sub-sampling).
    random_state :
        Random seed for reproducibility.  Default 0.

    Returns
    -------
    XGBoostFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``n_estimators < 1``, ``learning_rate <= 0``, ``max_depth < 1``,
        or ``subsample`` not in (0, 1].
    NotImplementedError
        If the ``xgboost`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import xgboost_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = xgboost_fit(X, y)  # doctest: +SKIP
    >>> result.feature_importances_.shape  # doctest: +SKIP
    (5,)

    References
    ----------
    Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System', KDD.
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")
    if not (0 < subsample <= 1.0):
        raise ValueError(f"subsample must be in (0, 1], got {subsample!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
    }
    model = _build_l4_model("xgboost", params)
    # subsample: _build_l4_model does not expose this param; set via set_params
    model.set_params(subsample=float(subsample))
    model.fit(X, y)

    fi = np.asarray(model.feature_importances_, dtype=float)

    return XGBoostFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


def lightgbm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    num_leaves: int = 31,
    random_state: int = 0,
) -> LightGBMFitResult:
    """Standalone LightGBM gradient-boosted trees regression.

    Calls the L4 lightgbm family adapter
    (``_build_l4_model("lightgbm", params)``) directly; bypasses the recipe
    recipe pipeline.  Requires ``pip install macroforecast[lightgbm]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of boosting rounds.  Must be >= 1.  Default 300.
    learning_rate :
        Step size shrinkage per boosting round.  Must be > 0.  Default 0.1.
    max_depth :
        Maximum tree depth.  ``-1`` (default) means unlimited depth.
        Must be >= -1; ``0`` is invalid per LightGBM convention.
    num_leaves :
        Maximum number of leaves per tree.  Must be >= 2.  Default 31.
    random_state :
        Random seed for reproducibility.  Default 0.

    Returns
    -------
    LightGBMFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.

    Raises
    ------
    ValueError
        If ``n_estimators < 1``, ``learning_rate <= 0``,
        ``max_depth`` not in ``{-1} ∪ [1, ∞)``, or ``num_leaves < 2``.
    NotImplementedError
        If the ``lightgbm`` package is not installed.

    Notes
    -----
    ``max_depth=-1`` means unlimited depth (LightGBM convention).
    ``max_depth=0`` is invalid and raises ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import lightgbm_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = lightgbm_fit(X, y)  # doctest: +SKIP
    >>> result.feature_importances_.shape  # doctest: +SKIP
    (5,)

    References
    ----------
    Ke et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree', NeurIPS.
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    if max_depth == 0 or max_depth < -1:
        raise ValueError(
            f"max_depth must be -1 (unlimited) or >= 1, got {max_depth!r}. "
            f"LightGBM does not accept max_depth=0."
        )
    if num_leaves < 2:
        raise ValueError(f"num_leaves must be >= 2, got {num_leaves!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
    }
    model = _build_l4_model("lightgbm", params)
    # num_leaves: _build_l4_model does not expose this param; set via set_params
    model.set_params(num_leaves=int(num_leaves))
    model.fit(X, y)

    fi = np.asarray(model.feature_importances_, dtype=float)

    return LightGBMFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


def catboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 0,
) -> CatBoostFitResult:
    """Standalone CatBoost gradient-boosted trees regression.

    Calls the L4 catboost family adapter
    (``_build_l4_model("catboost", params)``) directly; bypasses the recipe
    recipe pipeline.  Requires ``pip install macroforecast[catboost]``.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y :
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    n_estimators :
        Number of boosting iterations.  Must be >= 1.  Default 300.
    learning_rate :
        Step size shrinkage per boosting iteration.  Must be > 0.  Default 0.1.
    max_depth :
        Maximum tree depth.  Must be >= 1.  Default 6.
    random_state :
        Random seed (``random_seed`` in CatBoost API).  Default 0.

    Returns
    -------
    CatBoostFitResult
        Fitted result exposing ``feature_importances_``, ``n_estimators_used``,
        and ``.predict(X)`` / ``.summary()`` methods.  Predictions are
        guaranteed 1-D via ``.ravel()``.

    Raises
    ------
    ValueError
        If ``n_estimators < 1``, ``learning_rate <= 0``, or ``max_depth < 1``.
    NotImplementedError
        If the ``catboost`` package is not installed.

    Notes
    -----
    The CatBoost API uses ``iterations`` instead of ``n_estimators``,
    ``depth`` instead of ``max_depth``, and ``random_seed`` instead of
    ``random_state``.  These mappings are handled internally via
    ``_build_l4_model``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import catboost_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> y = X @ [1, 2, 3, 4, 5] + 0.5 * rng.randn(100)
    >>> result = catboost_fit(X, y)  # doctest: +SKIP
    >>> result.feature_importances_.shape  # doctest: +SKIP
    (5,)

    References
    ----------
    Prokhorenkova et al. (2018) 'CatBoost: unbiased boosting with
    categorical features', NeurIPS.
    """
    from ...core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    params: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
    }
    model = _build_l4_model("catboost", params)
    model.fit(X, y)

    fi = np.asarray(model.get_feature_importance(), dtype=float)

    return CatBoostFitResult(
        feature_importances_=fi,
        n_estimators_used=int(n_estimators),
        _model=model,
    )


__all__ = [
    "RandomForestFitResult",
    "random_forest_fit",
    "ExtraTreesFitResult",
    "extra_trees_fit",
    "GradientBoostingFitResult",
    "gradient_boosting_fit",
    "XGBoostFitResult",
    "xgboost_fit",
    "LightGBMFitResult",
    "lightgbm_fit",
    "CatBoostFitResult",
    "catboost_fit",
    # C64: gap callables for tree-family private classes
    "SlowGrowingTreeFitResult",
    "slow_growing_tree_fit",
    "QuantileRegressionForestFitResult",
    "quantile_regression_forest_fit",
    "BaggingFitResult",
    "bagging_fit",
    "BoogingFitResult",
    "booging_fit",
    "MacroRandomForestFitResult",
    "macro_random_forest_fit",
]


# ---------------------------------------------------------------------------
# C64: Gap callables for tree-family private classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlowGrowingTreeFitResult:
    """Result of :func:`slow_growing_tree_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_SlowGrowingTree`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point predictions for new data.

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
        """Return a human-readable summary of the SGT fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        eta = params.get("eta", getattr(self._model, "eta", "?"))
        lines = [
            sep,
            f"{'SlowGrowingTree Results':^78}",
            sep,
            f"{'eta:':35s} {str(eta):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class QuantileRegressionForestFitResult:
    """Result of :func:`quantile_regression_forest_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_QuantileRegressionForest`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point predictions (forest mean) for new data.

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

    def predict_quantiles(
        self, X: np.ndarray | pd.DataFrame
    ) -> "dict[float, np.ndarray]":
        """Return per-quantile predictions.

        Parameters
        ----------
        X :
            Feature matrix.

        Returns
        -------
        dict[float, np.ndarray]
            Mapping from quantile level to 1-D prediction array.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        return self._model.predict_quantiles(X)

    def summary(self) -> str:
        """Return a human-readable summary of the QRF fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        n_est = params.get("n_estimators", getattr(self._model, "n_estimators", "?"))
        lines = [
            sep,
            f"{'QuantileRegressionForest Results':^78}",
            sep,
            f"{'n_estimators:':35s} {str(n_est):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BaggingFitResult:
    """Result of :func:`bagging_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_BaggingWrapper`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return bag-mean predictions for new data.

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
        """Return a human-readable summary of the Bagging fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        n_est = params.get("n_estimators", getattr(self._model, "n_estimators", "?"))
        base = params.get("base_family", getattr(self._model, "base_family", "?"))
        lines = [
            sep,
            f"{'Bagging Results':^78}",
            sep,
            f"{'base_family:':35s} {str(base):>20s}",
            f"{'n_estimators:':35s} {str(n_est):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BoogingFitResult:
    """Result of :func:`booging_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_BoogingWrapper`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return bag-mean predictions for new data.

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
        """Return a human-readable summary of the Booging fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        B = params.get("B", getattr(self._model, "B", "?"))
        lines = [
            sep,
            f"{'Booging Results':^78}",
            sep,
            f"{'B (outer bags):':35s} {str(B):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MacroRandomForestFitResult:
    """Result of :func:`macro_random_forest_fit`.

    Attributes
    ----------
    _model :
        Internal fitted ``_MRFExternalWrapper`` instance.
        Not part of the public contract.
    """

    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return MRF predictions for new data.

        Note: This is computationally expensive -- each call re-runs the full
        ensemble loop with the OOS rows appended to the training panel
        (the MRF algorithm is a fit-and-forecast pipeline, not a stored model).

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
        """Return a human-readable summary of the MRF fit result.

        Returns
        -------
        str
            Minimal statsmodels-style table showing model type and parameters.
        """
        sep = "=" * 78
        params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        B = params.get("B", getattr(self._model, "B", "?"))
        lines = [
            sep,
            f"{'MacroRandomForest Results':^78}",
            sep,
            f"{'B (bags):':35s} {str(B):>20s}",
            sep,
        ]
        return "\n".join(lines)


def slow_growing_tree_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> SlowGrowingTreeFitResult:
    """Fit a Slow-Growing Tree (Goulet Coulombe 2024).

    Standalone callable that constructs a ``_SlowGrowingTree`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_SlowGrowingTree`` constructor
        (e.g., ``eta=0.2``, ``herfindahl_threshold=0.3``).

    Returns
    -------
    SlowGrowingTreeFitResult
        Fitted result exposing ``.predict(X)`` and ``.summary()`` methods.

    References
    ----------
    Goulet Coulombe (2024), "The Slow-Growing Tree."
    """
    from ...core.runtime import _SlowGrowingTree

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _SlowGrowingTree(**kwargs)
    model.fit(X, y)
    return SlowGrowingTreeFitResult(_model=model)


def quantile_regression_forest_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> QuantileRegressionForestFitResult:
    """Fit a Quantile Regression Forest (Meinshausen 2006).

    Standalone callable that constructs a ``_QuantileRegressionForest`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_QuantileRegressionForest`` constructor
        (e.g., ``n_estimators=200``, ``quantile_levels=(0.1, 0.5, 0.9)``).

    Returns
    -------
    QuantileRegressionForestFitResult
        Fitted result exposing ``.predict(X)``, ``.predict_quantiles(X)``,
        and ``.summary()`` methods.

    References
    ----------
    Meinshausen (2006), "Quantile Regression Forests",
    Journal of Machine Learning Research 7, pp. 983-999.
    """
    from ...core.runtime import _QuantileRegressionForest

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _QuantileRegressionForest(**kwargs)
    model.fit(X, y)
    return QuantileRegressionForestFitResult(_model=model)


def bagging_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> BaggingFitResult:
    """Fit a Bagging meta-estimator (Breiman 1996).

    Standalone callable that constructs a ``_BaggingWrapper`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_BaggingWrapper`` constructor
        (e.g., ``base_family="ridge"``, ``n_estimators=50``,
        ``strategy="block"``).

    Returns
    -------
    BaggingFitResult
        Fitted result exposing ``.predict(X)`` and ``.summary()`` methods.

    References
    ----------
    Breiman (1996), "Bagging Predictors",
    Machine Learning 24(2), pp. 123-140.
    """
    from ...core.runtime import _BaggingWrapper

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _BaggingWrapper(**kwargs)
    model.fit(X, y)
    return BaggingFitResult(_model=model)


def booging_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> BoogingFitResult:
    """Fit a Booging estimator (Goulet Coulombe 2024).

    Standalone callable that constructs a ``_BoogingWrapper`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_BoogingWrapper`` constructor
        (e.g., ``B=100``, ``sample_frac=0.75``,
        ``inner_n_estimators=1500``).

    Returns
    -------
    BoogingFitResult
        Fitted result exposing ``.predict(X)`` and ``.summary()`` methods.

    References
    ----------
    Goulet Coulombe (2024), "To Bag is to Prune",
    Journal of Applied Econometrics (forthcoming).
    """
    from ...core.runtime import _BoogingWrapper

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _BoogingWrapper(**kwargs)
    model.fit(X, y)
    return BoogingFitResult(_model=model)


def macro_random_forest_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> MacroRandomForestFitResult:
    """Fit a Macroeconomic Random Forest (Coulombe 2024 JAE).

    Standalone callable that constructs a ``_MRFExternalWrapper`` directly,
    bypassing the recipe pipeline. kwargs are forwarded to the constructor.

    Note: Raises RuntimeError if the vendored macro_random_forest package is
    not available in the macroforecast installation.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,). Accepts numpy arrays or Series.
    **kwargs :
        Keyword arguments forwarded to ``_MRFExternalWrapper`` constructor
        (e.g., ``B=50``, ``ridge_lambda=0.1``, ``random_state=0``).

    Returns
    -------
    MacroRandomForestFitResult
        Fitted result exposing ``.predict(X)`` and ``.summary()`` methods.
        Note: predict() is computationally expensive.

    References
    ----------
    Coulombe (2024), "The Macroeconomy as a Random Forest",
    Journal of Applied Econometrics 39(4).
    """
    from ...core.runtime import _MRFExternalWrapper

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    model = _MRFExternalWrapper(**kwargs)
    model.fit(X, y)
    return MacroRandomForestFitResult(_model=model)
