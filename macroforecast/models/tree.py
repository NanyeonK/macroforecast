"""macroforecast.models.tree -- public tree, ensemble, and KNN model classes.

Cycle 64 -- 6 public classes (thin subclasses of private L4 implementations).

Design note on __init__ overrides:
Each public class overrides __init__ and sets self.<param> = <raw_value> AFTER
calling super().__init__(). The private class __init__ applies type coercions
(int(), float(), np.clip(), max(), str()) which produce new float/int objects
and break sklearn clone(). By restoring the raw parameter values after
super().__init__(), get_params() reads the original objects, satisfying
sklearn's identity check in _clone_parametrized.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from macroforecast.core.runtime import (
    _SlowGrowingTree,
    _QuantileRegressionForest,
    _BaggingWrapper,
    _BoogingWrapper,
    _MRFExternalWrapper,
    _AutoClipKNN,
)

__all__ = [
    "SlowGrowingTree",
    "QuantileRegressionForest",
    "Bagging",
    "Booging",
    "MacroRandomForest",
    "KNN",
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

class SlowGrowingTree(_SlowGrowingTree, BaseEstimator, RegressorMixin):
    """Goulet Coulombe (2024) Slow-Growing Tree (SGT).

    Soft-weighted CART with Herfindahl stopping and depth-based split-shrinkage.
    At each split, rows on the "losing" side retain fractional weight (1-eta)
    rather than being hard-excluded.

    Parameters
    ----------
    eta : float, default 0.1
        Split-shrinkage factor (0=no shrinkage, 1=hard CART).
    herfindahl_threshold : float, default 0.25
        Herfindahl concentration threshold H-bar for leaf-purity stopping.
    eta_depth_step : float, default 0.01
        Per-depth increment to eta (paper p.87: +0.01 per level).
    eta_max_plateau : float, default 0.5
        Plateau cap for eta after depth steps (paper p.87).
    mtry_frac : float, default 1.0
        Fraction of features considered at each split (1.0 = all).
    max_depth : int or None, default 10
        Hard maximum tree depth. None = unlimited (not recommended for SGT
        with soft weights -- the Herfindahl index stays low with uniform
        weights so max_depth is the primary depth-bounding mechanism).
    random_state : int, default 0
        Random seed for mtry sampling.
    min_leaf_size : int, default 5
        Minimum training samples per leaf.

    Standalone usage::

        from macroforecast.models import SlowGrowingTree
        m = SlowGrowingTree(eta=0.1)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

    Reference: Goulet Coulombe (2024), "The Slow-Growing Tree."
    """

    def __init__(
        self,
        eta: float = 0.1,
        herfindahl_threshold: float = 0.25,
        eta_depth_step: float = 0.01,
        eta_max_plateau: float = 0.5,
        mtry_frac: float = 1.0,
        max_depth: int | None = 10,
        random_state: int = 0,
        min_leaf_size: int = 5,
    ) -> None:
        # Call private __init__ first (applies type coercions like float(np.clip(...))).
        super().__init__(
            eta=eta,
            herfindahl_threshold=herfindahl_threshold,
            eta_depth_step=eta_depth_step,
            eta_max_plateau=eta_max_plateau,
            mtry_frac=mtry_frac,
            max_depth=max_depth,
            random_state=random_state,
            min_leaf_size=min_leaf_size,
        )
        # Restore raw values AFTER super().__init__() so that
        # BaseEstimator.get_params() reads the original objects (not coerced
        # copies), satisfying sklearn clone() identity check.
        self.eta = eta
        self.herfindahl_threshold = herfindahl_threshold
        self.eta_depth_step = eta_depth_step
        self.eta_max_plateau = eta_max_plateau
        self.mtry_frac = mtry_frac
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_leaf_size = min_leaf_size

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SlowGrowingTree":
        """Fit the SGT and record sklearn feature tracking attributes.

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


class QuantileRegressionForest(_QuantileRegressionForest, BaseEstimator, RegressorMixin):
    """Meinshausen (2006) Quantile Regression Forest (QRF).

    Trains a RandomForest and records the empirical distribution of training
    y values per leaf. At predict time, aggregates per-leaf CDFs for
    probabilistic forecasts. predict() returns point forecasts (mean);
    predict_quantiles() returns per-quantile arrays.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of trees in the forest.
    max_depth : int or None, default None
        Maximum tree depth.
    random_state : int, default 0
        Random seed.
    quantile_levels : tuple of float, default (0.05, 0.5, 0.95)
        Quantile levels to compute in predict_quantiles().

    Extended interface::

        preds = m.predict(X)             # point forecasts (forest mean)
        q_dict = m.predict_quantiles(X)  # dict[float, np.ndarray]

    Reference: Meinshausen (2006), "Quantile Regression Forests",
    Journal of Machine Learning Research 7, pp. 983-999.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 0,
        quantile_levels: tuple = (0.05, 0.5, 0.95),
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            quantile_levels=quantile_levels,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.quantile_levels = quantile_levels

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileRegressionForest":
        """Fit the QRF and record sklearn feature tracking attributes.

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


class Bagging(_BaggingWrapper, BaseEstimator, RegressorMixin):
    """Bootstrap-aggregating (Bagging) meta-estimator (Breiman 1996).

    Wraps any base L4 family string (e.g., "ridge", "lasso"). Supports
    i.i.d. bootstrap (strategy="standard") and moving-block bootstrap
    (strategy="block") for serially-correlated macro data. Point
    predictions are the bag mean; predict_quantiles() returns empirical
    quantile bands across the bag.

    Parameters
    ----------
    base_family : str, default "ridge"
        Name of the base L4 family (as used in recipe YAML).
    n_estimators : int, default 50
        Number of bootstrap replicates.
    max_samples : float, default 0.8
        Fraction of training rows per bootstrap draw.
    random_state : int, default 0
        Random seed.
    base_params : dict or None, default None
        Hyperparameters passed to the base family constructor.
    strategy : str, default "standard"
        Bootstrap strategy: "standard" (i.i.d.) or "block" (moving-block).
    block_length : int, default 4
        Block length for moving-block bootstrap (ignored when strategy="standard").

    Reference: Breiman (1996), "Bagging Predictors",
    Machine Learning 24(2), pp. 123-140.
    """

    def __init__(
        self,
        base_family: str = "ridge",
        n_estimators: int = 50,
        max_samples: float = 0.8,
        random_state: int = 0,
        base_params: dict | None = None,
        strategy: str = "standard",
        block_length: int = 4,
    ) -> None:
        super().__init__(
            base_family=base_family,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            base_params=base_params,
            strategy=strategy,
            block_length=block_length,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        self.base_family = base_family
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.base_params: dict | None = base_params  # type: ignore[assignment]
        self.strategy = strategy
        self.block_length = block_length

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Bagging":
        """Fit the Bagging ensemble and record sklearn feature tracking attributes.

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


class Booging(_BoogingWrapper, BaseEstimator, RegressorMixin):
    """Goulet Coulombe (2024) Booging (Bagging + Pruning via overfitting).

    Outer bagging of intentionally over-fitted inner Stochastic Gradient
    Boosted Trees with data augmentation. Noisy copies of predictor columns
    are appended and a random fraction of augmented columns is dropped per
    bag. The bag average prunes the overfitting without explicit depth tuning.

    Parameters
    ----------
    B : int, default 100
        Number of outer bootstrap bags.
    sample_frac : float, default 0.75
        Row subsample fraction per bag.
    inner_n_estimators : int, default 1500
        Boosting steps (S) inside each bag. High values intentionally
        overfit; the outer bag average prunes.
    inner_learning_rate : float, default 0.1
        GBM learning rate.
    inner_max_depth : int, default 3
        GBM tree depth per boosting step (paper section 4.1).
    inner_subsample : float, default 0.5
        Row sub-sampling inside each boosting step.
    da_noise_frac : float, default 1/3
        Noise fraction for data augmentation (sigma_k times da_noise_frac).
    da_drop_rate : float, default 0.2
        Fraction of augmented columns dropped per bag.
    random_state : int, default 0
        Random seed.

    Reference: Goulet Coulombe (2024), "To Bag is to Prune",
    Journal of Applied Econometrics (forthcoming).
    """

    def __init__(
        self,
        B: int = 100,
        sample_frac: float = 0.75,
        inner_n_estimators: int = 1500,
        inner_learning_rate: float = 0.1,
        inner_max_depth: int = 3,
        inner_subsample: float = 0.5,
        da_noise_frac: float = 1 / 3,
        da_drop_rate: float = 0.2,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            B=B,
            sample_frac=sample_frac,
            inner_n_estimators=inner_n_estimators,
            inner_learning_rate=inner_learning_rate,
            inner_max_depth=inner_max_depth,
            inner_subsample=inner_subsample,
            da_noise_frac=da_noise_frac,
            da_drop_rate=da_drop_rate,
            random_state=random_state,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        self.B = B
        self.sample_frac = sample_frac
        self.inner_n_estimators = inner_n_estimators
        self.inner_learning_rate = inner_learning_rate
        self.inner_max_depth = inner_max_depth
        self.inner_subsample = inner_subsample
        self.da_noise_frac = da_noise_frac
        self.da_drop_rate = da_drop_rate
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Booging":
        """Fit the Booging estimator and record sklearn feature tracking attributes.

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


class MacroRandomForest(_MRFExternalWrapper, BaseEstimator, RegressorMixin):
    """Macroeconomic Random Forest (Coulombe 2024 JAE).

    Generalized Time-Varying Parameter model via per-leaf local linear
    regressions and Block Bayesian Bootstrap ensembles. Backed by Ryan Lucas's
    reference implementation (vendored under macroforecast/_vendor/
    macro_random_forest/ with numpy/pandas compatibility patches).

    Each predict() call re-runs the full ensemble loop with OOS rows appended
    to the training panel (the paper algorithm is a fit-and-forecast pipeline,
    not a stored model). Note: this makes predict() computationally expensive.

    Parameters
    ----------
    B : int, default 50
        Number of Block Bayesian Bootstrap bags.
    ridge_lambda : float, default 0.1
        Ridge regularization for per-leaf local linear regression.
    rw_regul : float, default 0.75
        Random walk regularization on time-varying coefficients.
    mtry_frac : float, default 1/3
        Fraction of features considered at each split.
    trend_push : float, default 1
        Weight on recent observations (Olympic-podium smoothing).
    quantile_rate : float, default 0.3
        Quantile rate for leaf size stopping.
    subsampling_rate : float, default 0.75
        Per-bag subsample fraction (Bayesian Bootstrap).
    fast_rw : bool, default True
        Use fast random-walk regularization path.
    resampling_opt : int, default 2
        Resampling strategy option (1 or 2; paper default 2).
    parallelise : bool, default False
        Enable multicore parallelization via joblib.
    n_cores : int, default 1
        Number of cores for parallelization.
    block_size : int, default 24
        Block length for block bootstrap (paper default for monthly data).
    random_state : int, default 0
        Random seed.

    Note: Raises RuntimeError if the vendored macro_random_forest package is
    not available.

    Reference: Coulombe (2024), "The Macroeconomy as a Random Forest",
    Journal of Applied Econometrics 39(4).
    Citation for vendored implementation: Ryan Lucas (2024),
    github.com/RyanLucas3/MacroRandomForest.
    """

    def __init__(
        self,
        B: int = 50,
        ridge_lambda: float = 0.1,
        rw_regul: float = 0.75,
        mtry_frac: float = 1 / 3,
        trend_push: float = 1,
        quantile_rate: float = 0.3,
        subsampling_rate: float = 0.75,
        fast_rw: bool = True,
        resampling_opt: int = 2,
        parallelise: bool = False,
        n_cores: int = 1,
        block_size: int = 24,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            B=B,
            ridge_lambda=ridge_lambda,
            rw_regul=rw_regul,
            mtry_frac=mtry_frac,
            trend_push=trend_push,
            quantile_rate=quantile_rate,
            subsampling_rate=subsampling_rate,
            fast_rw=fast_rw,
            resampling_opt=resampling_opt,
            parallelise=parallelise,
            n_cores=n_cores,
            block_size=block_size,
            random_state=random_state,
        )
        # Restore raw values after super().__init__() for clone() compatibility.
        self.B = B
        self.ridge_lambda = ridge_lambda
        self.rw_regul = rw_regul
        self.mtry_frac = mtry_frac
        self.trend_push = trend_push
        self.quantile_rate = quantile_rate
        self.subsampling_rate = subsampling_rate
        self.fast_rw = fast_rw
        self.resampling_opt = resampling_opt
        self.parallelise = parallelise
        self.n_cores = n_cores
        self.block_size = block_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MacroRandomForest":
        """Fit the MRF and record sklearn feature tracking attributes.

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


class KNN(_AutoClipKNN, BaseEstimator, RegressorMixin):
    """K-Nearest Neighbor regressor with walk-forward safety clipping.

    Wraps sklearn KNeighborsRegressor and clamps n_neighbors to the
    training set size at each fit call. This ensures the estimator does
    not fail during walk-forward evaluation when the training window is
    small early in the OOS period.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbors. Clamped to min(n_neighbors, n_train) at fit.
    weights : str, default "uniform"
        Weight function: "uniform" or "distance".
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        # Restore raw values after super().__init__() for clone() compatibility.
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KNN":
        """Fit the KNN regressor and record sklearn feature tracking attributes.

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
