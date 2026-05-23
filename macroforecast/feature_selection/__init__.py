"""Sklearn-style feature selection class wrappers.

Exposes five supervised feature selection classes with a uniform
``fit(X, y) -> self`` / ``transform(X) -> X_selected`` interface. Each
class delegates numeric computation to the corresponding private runtime
function in ``macroforecast.core.runtime``.

Classes
-------
- :class:`Boruta`            -- Kursa & Rudnicki (2010) Boruta algorithm.
- :class:`RFE`               -- Guyon et al. (2002) Recursive Feature Elimination.
- :class:`LassoPathSelector` -- Efron et al. (2004) LARS path entry-order selection.
- :class:`StabilitySelection`-- Meinshausen & Buhlmann (2010) stability selection.
- :class:`GeneticSelection`  -- Goldberg (1989) genetic algorithm feature selection.

Usage::

    import numpy as np
    import pandas as pd
    from macroforecast.feature_selection import Boruta

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 20), columns=[f"x{i}" for i in range(20)])
    y = pd.Series(X["x0"] + X["x1"] + 0.1 * rng.randn(200), name="y")

    sel = Boruta(n_estimators_rf=50, max_iter=20, random_state=0)
    sel.fit(X, y)
    X_sel = sel.transform(X)
    print(sel.selected_features_)   # list of column names

Cycle 63 -- L3 sklearn-style class wrappers (5 classes).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------

def _to_frame(
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert X to a DataFrame, preserving column names when available."""
    if isinstance(X, pd.DataFrame):
        return X
    cols = feature_names if feature_names else [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(np.asarray(X, dtype=float), columns=cols)


def _to_series(y: np.ndarray | pd.Series) -> pd.Series:
    """Convert y to a named Series."""
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y, dtype=float).ravel(), name="y")


# ---------------------------------------------------------------------------
# Boruta
# ---------------------------------------------------------------------------

class Boruta(BaseEstimator, TransformerMixin):
    """Boruta feature selection (Kursa & Rudnicki 2010, Algorithm 1).

    Implements Algorithm 1 with multi-shadow augmentation to control the
    false-positive rate under the Bonferroni correction. Internally calls
    :func:`~macroforecast.core.runtime._boruta_selection`.

    Parameters
    ----------
    n_estimators_rf : int
        Number of trees in the Random Forest used for importance scoring.
        Default 100.
    max_iter : int
        Maximum number of Boruta iterations. Default 100.
    alpha : float
        Bonferroni significance level. Default 0.05.
    include_tentative : bool
        If True, tentative features are included in the selected set after
        ``max_iter`` iterations. Default False (tentatives are rejected).
    random_state : int
        Random seed for reproducibility. Default 0.
    n_shadow_copies : int
        Number of independent shadow permutation matrices used per iteration
        to calibrate the MISA threshold (Bonferroni FP control). Default 6.

    Attributes
    ----------
    selected_features_ : list[str]
        Feature names selected after calling :meth:`fit`.

    References
    ----------
    Kursa, Rudnicki (2010) "Feature Selection with the Boruta Package."
    Journal of Statistical Software 36(11).
    """

    def __init__(
        self,
        *,
        n_estimators_rf: int = 100,
        max_iter: int = 100,
        alpha: float = 0.05,
        include_tentative: bool = False,
        random_state: int = 0,
        n_shadow_copies: int = 6,
    ) -> None:
        self.n_estimators_rf = n_estimators_rf
        self.max_iter = max_iter
        self.alpha = alpha
        self.include_tentative = include_tentative
        self.random_state = random_state
        self.n_shadow_copies = n_shadow_copies
        # selected_features_ is set only after a successful fit() call.

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "Boruta":
        """Fit the Boruta selector.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_samples, n_features).
        y :
            Target vector. Shape (n_samples,).

        Returns
        -------
        self
        """
        from macroforecast.core.runtime import _boruta_selection

        X_df = _to_frame(X)
        y_s = _to_series(y)
        # --- sklearn feature tracking (C64 BaseEstimator refactor) ---
        self.feature_names_in_ = np.array(X_df.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # --- existing selection logic ---
        params: dict[str, Any] = {
            "n_estimators_rf": self.n_estimators_rf,
            "max_iter": self.max_iter,
            "alpha": self.alpha,
            "include_tentative": self.include_tentative,
            "random_state": self.random_state,
            "n_shadow_copies": self.n_shadow_copies,
        }
        selected_frame = _boruta_selection(X_df, target=y_s, params=params)
        self.selected_features_ = list(selected_frame.columns)
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X :
            Feature matrix. Must have the same columns as the training data.

        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected feature columns.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        if not hasattr(self, "selected_features_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        X_df = _to_frame(X)
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df[cols]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the selector and return the selected feature columns.

        Convenience method equivalent to calling fit(X, y, **kwargs) followed
        by transform(X). Follows the sklearn transformer convention.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.
        **kwargs
            Forwarded to fit().

        Returns
        -------
        pd.DataFrame
            Subset of X containing only the selected columns.
        """
        return self.fit(X, y, **kwargs).transform(X)


# ---------------------------------------------------------------------------
# RFE
# ---------------------------------------------------------------------------

class RFE(BaseEstimator, TransformerMixin):
    """Recursive Feature Elimination (Guyon, Weston, Barnhill & Vapnik 2002).

    Iteratively prunes the feature with the lowest squared-coefficient ranking
    from a linear estimator until ``n_features_to_select`` features remain.
    Optionally uses RFECV for cross-validation-based automatic selection.
    Internally calls :func:`~macroforecast.core.runtime._recursive_feature_elimination`.

    Parameters
    ----------
    n_features_to_select : int or float
        Target number of features. If a float in (0, 1], interpreted as a
        fraction of the total features. Default 0.5.
    step : int
        Number of features to remove per iteration. Default 1.
    estimator : str
        Base estimator for RFE: "ridge" (default), "lasso", or "svr".
    use_cv : bool
        Use RFECV (cross-validation-based automatic selection). Default False.
    cv_folds : int
        Number of CV folds when ``use_cv=True``. Default 5.
    random_state : int
        Random seed. Default 0.

    Attributes
    ----------
    selected_features_ : list[str]
        Feature names selected after calling :meth:`fit`.

    References
    ----------
    Guyon, Weston, Barnhill, Vapnik (2002) "Gene Selection for Cancer
    Classification using Support Vector Machines." Machine Learning 46(1-3).
    """

    def __init__(
        self,
        *,
        n_features_to_select: int | float = 0.5,
        step: int = 1,
        estimator: str = "ridge",
        use_cv: bool = False,
        cv_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator = estimator
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.random_state = random_state
        # selected_features_ is set only after a successful fit() call.

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "RFE":
        """Fit the RFE selector.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_samples, n_features).
        y :
            Target vector. Shape (n_samples,).

        Returns
        -------
        self
        """
        from macroforecast.core.runtime import _recursive_feature_elimination

        X_df = _to_frame(X)
        y_s = _to_series(y)
        # --- sklearn feature tracking (C64 BaseEstimator refactor) ---
        self.feature_names_in_ = np.array(X_df.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # --- existing selection logic ---
        params: dict[str, Any] = {
            "n_features_to_select": self.n_features_to_select,
            "step": self.step,
            "estimator": self.estimator,
            "use_cv": self.use_cv,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }
        selected_frame = _recursive_feature_elimination(X_df, target=y_s, params=params)
        self.selected_features_ = list(selected_frame.columns)
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X :
            Feature matrix.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        if not hasattr(self, "selected_features_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        X_df = _to_frame(X)
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df[cols]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the selector and return the selected feature columns.

        Convenience method equivalent to calling fit(X, y, **kwargs) followed
        by transform(X). Follows the sklearn transformer convention.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.
        **kwargs
            Forwarded to fit().

        Returns
        -------
        pd.DataFrame
            Subset of X containing only the selected columns.
        """
        return self.fit(X, y, **kwargs).transform(X)


# ---------------------------------------------------------------------------
# LassoPathSelector
# ---------------------------------------------------------------------------

class LassoPathSelector(BaseEstimator, TransformerMixin):
    """LARS path entry-order feature selection (Efron et al. 2004).

    Selects features in the order they enter the LARS active set as
    regularization decreases from infinity toward zero. Internally calls
    :func:`~macroforecast.core.runtime._lasso_path_selection`.

    This op is distinct from ``feature_selection(method='lasso')`` which
    ranks features by LassoCV coefficient magnitude. Here the criterion is
    the LARS entry order (Efron et al. 2004, LARS Algorithm 1).

    Parameters
    ----------
    n_features_to_select : int or float
        Target number of features. If a float in (0, 1], fraction of total.
        Default 0.5.
    normalize_features : bool
        Standardize features before LARS path computation. Default True.
    random_state : int
        Accepted for interface consistency; LARS is deterministic. Default 0.

    Attributes
    ----------
    selected_features_ : list[str]
        Feature names selected after calling :meth:`fit`.

    References
    ----------
    Efron, Hastie, Johnstone, Tibshirani (2004) "Least Angle Regression."
    Annals of Statistics 32(2).
    """

    def __init__(
        self,
        *,
        n_features_to_select: int | float = 0.5,
        normalize_features: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.normalize_features = normalize_features
        self.random_state = random_state
        # selected_features_ is set only after a successful fit() call.

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "LassoPathSelector":
        """Fit the LARS path selector.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_samples, n_features).
        y :
            Target vector. Shape (n_samples,).

        Returns
        -------
        self
        """
        from macroforecast.core.runtime import _lasso_path_selection

        X_df = _to_frame(X)
        y_s = _to_series(y)
        # --- sklearn feature tracking (C64 BaseEstimator refactor) ---
        self.feature_names_in_ = np.array(X_df.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # --- existing selection logic ---
        params: dict[str, Any] = {
            "n_features_to_select": self.n_features_to_select,
            "normalize_features": self.normalize_features,
            "random_state": self.random_state,
        }
        selected_frame = _lasso_path_selection(X_df, target=y_s, params=params)
        self.selected_features_ = list(selected_frame.columns)
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X :
            Feature matrix.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        if not hasattr(self, "selected_features_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        X_df = _to_frame(X)
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df[cols]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the selector and return the selected feature columns.

        Convenience method equivalent to calling fit(X, y, **kwargs) followed
        by transform(X). Follows the sklearn transformer convention.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.
        **kwargs
            Forwarded to fit().

        Returns
        -------
        pd.DataFrame
            Subset of X containing only the selected columns.
        """
        return self.fit(X, y, **kwargs).transform(X)


# ---------------------------------------------------------------------------
# StabilitySelection
# ---------------------------------------------------------------------------

class StabilitySelection(BaseEstimator, TransformerMixin):
    """Stability selection (Meinshausen & Buhlmann 2010, Section 2).

    Repeatedly fits a Lasso or ElasticNet on random subsamples and selects
    features that appear in at least ``pi_thr`` fraction of subsamples.
    Internally calls :func:`~macroforecast.core.runtime._stability_selection`.

    Parameters
    ----------
    n_subsamples : int
        Number of random subsamples drawn. Default 100.
    subsample_fraction : float
        Fraction of training rows in each subsample. Default 0.5.
    pi_thr : float
        Selection probability threshold in (0, 1]. Default 0.6.
    base_estimator : str
        Base estimator for selection: "lasso" (default) or "elastic_net".
    alpha : float
        Regularisation parameter for the base estimator. Default 0.01.
    random_state : int
        Random seed for reproducibility. Default 0.

    Attributes
    ----------
    selected_features_ : list[str]
        Feature names selected after calling :meth:`fit`.

    References
    ----------
    Meinshausen, Buhlmann (2010) "Stability Selection." Journal of the Royal
    Statistical Society Series B 72(4).
    """

    def __init__(
        self,
        *,
        n_subsamples: int = 100,
        subsample_fraction: float = 0.5,
        pi_thr: float = 0.6,
        base_estimator: str = "lasso",
        alpha: float = 0.01,
        random_state: int = 0,
    ) -> None:
        self.n_subsamples = n_subsamples
        self.subsample_fraction = subsample_fraction
        self.pi_thr = pi_thr
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.random_state = random_state
        # selected_features_ is set only after a successful fit() call.

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "StabilitySelection":
        """Fit the stability selector.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_samples, n_features).
        y :
            Target vector. Shape (n_samples,).

        Returns
        -------
        self
        """
        from macroforecast.core.runtime import _stability_selection

        X_df = _to_frame(X)
        y_s = _to_series(y)
        # --- sklearn feature tracking (C64 BaseEstimator refactor) ---
        self.feature_names_in_ = np.array(X_df.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # --- existing selection logic ---
        params: dict[str, Any] = {
            "n_subsamples": self.n_subsamples,
            "subsample_fraction": self.subsample_fraction,
            "pi_thr": self.pi_thr,
            "base_estimator": self.base_estimator,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
        selected_frame = _stability_selection(X_df, target=y_s, params=params)
        self.selected_features_ = list(selected_frame.columns)
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X :
            Feature matrix.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        if not hasattr(self, "selected_features_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        X_df = _to_frame(X)
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df[cols]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the selector and return the selected feature columns.

        Convenience method equivalent to calling fit(X, y, **kwargs) followed
        by transform(X). Follows the sklearn transformer convention.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.
        **kwargs
            Forwarded to fit().

        Returns
        -------
        pd.DataFrame
            Subset of X containing only the selected columns.
        """
        return self.fit(X, y, **kwargs).transform(X)


# ---------------------------------------------------------------------------
# GeneticSelection
# ---------------------------------------------------------------------------

class GeneticSelection(BaseEstimator, TransformerMixin):
    """Genetic algorithm feature selection (Goldberg 1989, Chapters 1-3).

    Implements a binary-chromosome GA: tournament selection, single-point
    crossover, bit-flip mutation, and elitist replacement. Fitness is
    evaluated by cross-validated neg-MSE. Internally calls
    :func:`~macroforecast.core.runtime._genetic_algorithm_selection`.

    Parameters
    ----------
    population_size : int
        GA population size. Default 30.
    n_generations : int
        Number of GA generations. Default 50.
    crossover_prob : float
        Single-point crossover probability. Default 0.8.
    fitness_estimator : str
        Estimator for CV fitness evaluation: "ridge" (default), "lasso",
        or "ols".
    cv_folds : int
        Number of cross-validation folds. Default 3.
    random_state : int
        Random seed. Default 0.

    Attributes
    ----------
    selected_features_ : list[str]
        Feature names selected after calling :meth:`fit`.

    References
    ----------
    Goldberg (1989) "Genetic Algorithms in Search, Optimization, and Machine
    Learning." Addison-Wesley.
    """

    def __init__(
        self,
        *,
        population_size: int = 30,
        n_generations: int = 50,
        crossover_prob: float = 0.8,
        fitness_estimator: str = "ridge",
        cv_folds: int = 3,
        random_state: int = 0,
    ) -> None:
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.fitness_estimator = fitness_estimator
        self.cv_folds = cv_folds
        self.random_state = random_state
        # selected_features_ is set only after a successful fit() call.

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "GeneticSelection":
        """Fit the genetic selector.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_samples, n_features).
        y :
            Target vector. Shape (n_samples,).

        Returns
        -------
        self
        """
        from macroforecast.core.runtime import _genetic_algorithm_selection

        X_df = _to_frame(X)
        y_s = _to_series(y)
        # --- sklearn feature tracking (C64 BaseEstimator refactor) ---
        self.feature_names_in_ = np.array(X_df.columns.tolist(), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # --- existing selection logic ---
        params: dict[str, Any] = {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "crossover_prob": self.crossover_prob,
            "fitness_estimator": self.fitness_estimator,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
        }
        selected_frame = _genetic_algorithm_selection(X_df, target=y_s, params=params)
        self.selected_features_ = list(selected_frame.columns)
        return self

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Reduce X to the selected features.

        Parameters
        ----------
        X :
            Feature matrix.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        if not hasattr(self, "selected_features_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call fit() first."
            )
        X_df = _to_frame(X)
        cols = [c for c in self.selected_features_ if c in X_df.columns]
        return X_df[cols]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the selector and return the selected feature columns.

        Convenience method equivalent to calling fit(X, y, **kwargs) followed
        by transform(X). Follows the sklearn transformer convention.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target series.
        **kwargs
            Forwarded to fit().

        Returns
        -------
        pd.DataFrame
            Subset of X containing only the selected columns.
        """
        return self.fit(X, y, **kwargs).transform(X)


__all__ = [
    "Boruta",
    "RFE",
    "LassoPathSelector",
    "StabilitySelection",
    "GeneticSelection",
]
