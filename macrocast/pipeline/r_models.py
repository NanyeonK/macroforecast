"""R model bridge for the macrocast pipeline.

Provides Python wrappers for the linear R models (Ridge, LASSO, etc.) that
live in macrocastR.  Each wrapper implements the MacrocastEstimator interface
so that R models slot into ForecastExperiment identically to Python models.

Internals
---------
fit() stores the training data.  predict() writes Z_train / y_train / Z_test
to a temporary directory as feather files, calls ``Rscript bridge.R`` via
subprocess, and reads the resulting ``output.json``.  The temp directory is
deleted automatically after each predict() call.

Data exchange format
--------------------
  tmpdir/Z_train.feather   (T_train, N_feat)  numpy → feather via pyarrow
  tmpdir/y_train.feather   (T_train, 1)
  tmpdir/Z_test.feather    (1, N_feat)
  tmpdir/config.json       model kwargs (cv_folds, nlambda, etc.)

  AR model additionally requires:
    tmpdir/y_train_full.feather   (T_full, 1)  un-shifted target series
    tmpdir/y_test_lags.feather    (p, 1)        last p lags for AR test row

  Output:
    tmpdir/output.json  {"y_hat": float, "hp": {str: scalar}}
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from macrocast.pipeline.components import Nonlinearity
from macrocast.pipeline.estimator import MacrocastEstimator

# ---------------------------------------------------------------------------
# Feather I/O helper
# ---------------------------------------------------------------------------


def _write_feather(arr: NDArray[np.floating], path: Path) -> None:
    """Write a 2-D numpy array to a feather file via pyarrow.

    Parameters
    ----------
    arr : NDArray of shape (T, K)
        Array to write.  1-D inputs are reshaped to (T, 1).
    path : Path
        Destination file path.
    """
    import pyarrow as pa
    import pyarrow.feather as pf

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    table = pa.table({f"c{i}": arr[:, i].astype(float) for i in range(arr.shape[1])})
    pf.write_feather(table, str(path))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RModelEstimator(MacrocastEstimator):
    """Bridge to R linear models via subprocess + feather exchange.

    Subclasses set ``model_name`` and pass model-specific kwargs through
    the constructor.  The bridge script (``macrocastR/inst/bridge.R``) is
    called once per predict() invocation.

    Parameters
    ----------
    model_name : str
        Name dispatched to the R bridge (e.g. ``"ridge"``, ``"lasso"``).
    **model_kwargs
        Keyword arguments forwarded to the R model function as JSON config.
    """

    nonlinearity_type: Nonlinearity = Nonlinearity.LINEAR

    # Path to bridge.R resolved relative to this file's package root
    _BRIDGE_SCRIPT: ClassVar[Path] = (
        Path(__file__).parent.parent.parent / "macrocastR" / "inst" / "bridge.R"
    )

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.best_params_: dict[str, Any] = {}
        self._X_train: NDArray[np.floating] | None = None
        self._y_train: NDArray[np.floating] | None = None

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> RModelEstimator:
        """Store training data for the combined fit+predict R call.

        R models perform fitting and prediction in a single bridge invocation
        (inside predict()), so fit() just caches the training arrays.

        Parameters
        ----------
        X : NDArray of shape (T, N)
            Feature matrix.
        y : NDArray of shape (T,)
            Target vector.

        Returns
        -------
        self
        """
        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y, dtype=float)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Run the R bridge and return the point forecast.

        Parameters
        ----------
        X : NDArray of shape (T_test, N)
            Test feature matrix.

        Returns
        -------
        y_hat : NDArray of shape (T_test,)
            Point forecast(s).

        Raises
        ------
        RuntimeError
            If Rscript is not found or the bridge script exits non-zero.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit() before predict().")

        if shutil.which("Rscript") is None:
            raise RuntimeError(
                "Rscript not found on PATH. Install R to use R model wrappers."
            )

        Z_test = np.asarray(X, dtype=float)
        y_hat, hp = self._call_r(self._X_train, self._y_train, Z_test)
        self.best_params_ = hp
        return np.array([y_hat])

    def _call_r(
        self,
        Z_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        Z_test: NDArray[np.floating],
    ) -> tuple[float, dict[str, Any]]:
        """Write inputs, invoke bridge.R, read output.

        Parameters
        ----------
        Z_train : NDArray of shape (T_train, N)
        y_train : NDArray of shape (T_train,)
        Z_test  : NDArray of shape (T_test, N)

        Returns
        -------
        (y_hat, hp)
            Point forecast scalar and hyperparameter dict.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_feather(Z_train, tmppath / "Z_train.feather")
            _write_feather(y_train.reshape(-1, 1), tmppath / "y_train.feather")
            _write_feather(Z_test, tmppath / "Z_test.feather")

            config = dict(self.model_kwargs)
            (tmppath / "config.json").write_text(json.dumps(config))

            result = subprocess.run(
                ["Rscript", str(self._BRIDGE_SCRIPT), self.model_name, tmpdir],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"R bridge failed for model '{self.model_name}':\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            output = json.loads((tmppath / "output.json").read_text())

        return float(output["y_hat"]), output.get("hp", {})


# ---------------------------------------------------------------------------
# Concrete model classes
# ---------------------------------------------------------------------------


class ARDIModel(RModelEstimator):
    """ARDI: OLS on PCA factors + AR lags (feature matrix from FeatureBuilder).

    Parameters
    ----------
    intercept : bool
        Include intercept in OLS.  Default True.
    """

    def __init__(self, intercept: bool = True) -> None:
        super().__init__("ardi", intercept=intercept)


class RidgeModel(RModelEstimator):
    """Ridge regression via glmnet (alpha=0) with CV penalty selection.

    Parameters
    ----------
    cv_folds : int
        Number of cross-validation folds.
    nlambda : int
        Number of lambda values in the regularisation path.
    """

    def __init__(self, cv_folds: int = 5, nlambda: int = 50) -> None:
        super().__init__("ridge", cv_folds=cv_folds, nlambda=nlambda)


class LassoModel(RModelEstimator):
    """LASSO via glmnet (alpha=1) with CV penalty selection.

    Parameters
    ----------
    cv_folds : int
    nlambda : int
    """

    def __init__(self, cv_folds: int = 5, nlambda: int = 50) -> None:
        super().__init__("lasso", cv_folds=cv_folds, nlambda=nlambda)


class AdaptiveLassoModel(RModelEstimator):
    """Adaptive LASSO with Ridge-initialised penalty weights.

    Two-step: fit Ridge to get initial beta_hat, compute penalty weights
    w_j = 1 / |beta_hat_j|^gamma, then fit LASSO with those weights.

    Parameters
    ----------
    cv_folds : int
    nlambda : int
    gamma : float
        Exponent for penalty weight computation.  Default 1.
    """

    def __init__(
        self, cv_folds: int = 5, nlambda: int = 50, gamma: float = 1.0
    ) -> None:
        super().__init__("adaptive_lasso", cv_folds=cv_folds, nlambda=nlambda, gamma=gamma)


class GroupLassoModel(RModelEstimator):
    """Group LASSO via grpreg with CV penalty selection.

    Group structure maps features to FRED variable categories.  Pass
    ``groups`` as a list of integers (one per feature column) or leave
    as None to infer from column-name prefixes (requires named Z matrix,
    which Python feather does not preserve — prefer explicit groups).

    Parameters
    ----------
    groups : list[int] or None
        Group membership vector of length N_feat.  If None, R infers groups
        from column name prefixes (only works when Z has named columns).
    cv_folds : int
    """

    def __init__(
        self, groups: list[int] | None = None, cv_folds: int = 5
    ) -> None:
        super().__init__("group_lasso", groups=groups, cv_folds=cv_folds)


class ElasticNetModel(RModelEstimator):
    """Elastic Net via glmnet with CV penalty selection.

    Parameters
    ----------
    cv_folds : int
    nlambda : int
    alpha : float
        Mixing parameter: 0 = Ridge, 1 = LASSO.  Default 0.5.
    """

    def __init__(
        self, cv_folds: int = 5, nlambda: int = 50, alpha: float = 0.5
    ) -> None:
        super().__init__("elastic_net", cv_folds=cv_folds, nlambda=nlambda, alpha=alpha)


class TVPRidgeModel(RModelEstimator):
    """Time-Varying Parameter Ridge via Legendre polynomial expansion.

    Expands Z by interacting each predictor with a polynomial time-trend
    basis, then applies Ridge.  Coefficients drift smoothly over time.

    Parameters
    ----------
    n_poly : int
        Degree of the Legendre polynomial basis.  Default 3.
    cv_folds : int
    """

    def __init__(self, n_poly: int = 3, cv_folds: int = 5) -> None:
        super().__init__("tvp_ridge", n_poly=n_poly, cv_folds=cv_folds)


class BoogingModel(RModelEstimator):
    """Booging: bootstrap aggregating of OLS with pruning.

    Draws n_boot bootstrap samples, fits OLS on a random predictor subset
    each time, then keeps the top-performing fraction by in-sample MSFE.

    Parameters
    ----------
    n_boot : int
        Number of bootstrap draws.  Default 200.
    prune_quantile : float
        Fraction of bootstrap models to keep (lowest in-sample MSFE).
        Default 0.5 (top half).
    """

    def __init__(self, n_boot: int = 200, prune_quantile: float = 0.5) -> None:
        super().__init__("booging", n_boot=n_boot, prune_quantile=prune_quantile)


class BVARModel(RModelEstimator):
    """Bayesian VAR with Minnesota prior (Litterman 1986).

    Implements a Bayesian linear regression with a diagonal Normal prior
    (Minnesota-style) on the regression coefficients.  Applied to the
    Z_train feature matrix from FeatureBuilder, this serves as a standard
    Bayesian shrinkage benchmark in macro forecasting horse races.

    The prior precision is diagonal: D_jj = lambda / Var(z_j), which
    shrinks high-variance predictors less aggressively.  The intercept
    receives a near-flat prior (D_11 ≈ 0).

    The MAP estimate is analytical: beta = (Z'Z + lambda*D)^{-1} Z'y.
    When lambda is None, it is tuned by leave-one-out cross-validation
    via the fast hat-matrix diagonal formula.

    Parameters
    ----------
    lambda_ : float or None
        Global shrinkage hyperparameter.  None (default) tunes by LOO-CV
        over a log-spaced grid from 1e-3 to 1e3.
    intercept : bool
        Include intercept column.  Default True.
    n_grid : int
        Number of lambda candidates in the LOO-CV grid.  Default 20.
    """

    def __init__(
        self,
        lambda_: float | None = None,
        intercept: bool = True,
        n_grid: int = 20,
    ) -> None:
        # Pass lambda as None (JSON null) when unset; R bridge treats NULL as LOO-CV
        super().__init__(
            "bvar",
            lambda_=lambda_,
            intercept=intercept,
            n_grid=n_grid,
        )

    def _call_r(
        self,
        Z_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        Z_test: NDArray[np.floating],
    ) -> tuple[float, dict[str, Any]]:
        """Override to map Python's ``lambda_`` kwarg to R's ``lambda``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _write_feather(Z_train, tmppath / "Z_train.feather")
            _write_feather(y_train.reshape(-1, 1), tmppath / "y_train.feather")
            _write_feather(Z_test, tmppath / "Z_test.feather")

            # Remap: Python uses lambda_ to avoid shadowing built-in; R expects lambda
            config: dict[str, Any] = {
                "lambda": self.model_kwargs.get("lambda_"),  # None → JSON null → R NULL
                "intercept": self.model_kwargs.get("intercept", True),
                "n_grid": self.model_kwargs.get("n_grid", 20),
            }
            (tmppath / "config.json").write_text(json.dumps(config))

            result = subprocess.run(
                ["Rscript", str(self._BRIDGE_SCRIPT), "bvar", tmpdir],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"R bridge failed for model 'bvar':\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            output = json.loads((tmppath / "output.json").read_text())

        return float(output["y_hat"]), output.get("hp", {})


# ---------------------------------------------------------------------------
# AR model (special interface)
# ---------------------------------------------------------------------------


class ARModel(RModelEstimator):
    """AR(p) benchmark with BIC lag selection, implemented in R.

    Unlike other R models, AR does not use the FeatureBuilder's Z matrix.
    It receives the raw (un-shifted) target series and selects the lag order
    internally via BIC.

    ForecastExperiment is responsible for setting ``_y_train_full`` and
    ``_y_test_lags`` on this estimator before calling predict().

    Parameters
    ----------
    h : int
        Forecast horizon.  Passed to the R bridge for internal alignment.
    max_lag : int
        Maximum AR lag order considered by BIC.
    """

    def __init__(self, h: int = 1, max_lag: int = 12) -> None:
        super().__init__("ar", h=h, max_lag=max_lag)
        self._h = h
        self._y_train_full: NDArray[np.floating] | None = None
        self._y_test_lags: NDArray[np.floating] | None = None

    def _call_r(
        self,
        Z_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        Z_test: NDArray[np.floating],
    ) -> tuple[float, dict[str, Any]]:
        """Override to include AR-specific feather files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Standard files (Z_train/y_train/Z_test still written for
            # consistency; bridge.R ignores them for AR)
            _write_feather(Z_train, tmppath / "Z_train.feather")
            _write_feather(y_train.reshape(-1, 1), tmppath / "y_train.feather")
            _write_feather(Z_test, tmppath / "Z_test.feather")

            # AR-specific: un-shifted target and test lags
            if self._y_train_full is not None:
                _write_feather(
                    self._y_train_full.reshape(-1, 1),
                    tmppath / "y_train_full.feather",
                )
            else:
                # Fall back: use the (shifted) y_train; bridge will still work
                # but lag selection may be slightly off
                _write_feather(
                    y_train.reshape(-1, 1),
                    tmppath / "y_train_full.feather",
                )

            if self._y_test_lags is not None:
                _write_feather(
                    self._y_test_lags.reshape(-1, 1),
                    tmppath / "y_test_lags.feather",
                )

            config = dict(self.model_kwargs)
            (tmppath / "config.json").write_text(json.dumps(config))

            result = subprocess.run(
                ["Rscript", str(self._BRIDGE_SCRIPT), "ar", tmpdir],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"R bridge failed for model 'ar':\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            output = json.loads((tmppath / "output.json").read_text())

        return float(output["y_hat"]), output.get("hp", {})
