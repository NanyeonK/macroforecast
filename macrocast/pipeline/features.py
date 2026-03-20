"""Feature construction for macrocast pipeline.

FeatureBuilder assembles the predictor matrix Z_t that is passed to each
MacrocastEstimator.  Two modes are supported:

* **Factors mode** (FACTORS / ARDI regularization): Z_t contains PCA diffusion
  index factors f_1..f_{p_f} plus AR lags y_{t-1}..y_{t-p_y}.  Number of
  factors p_f and number of lags p_y are both tuning parameters exposed to the
  CV loop.

* **AR-only mode** (all other regularizations): Z_t contains AR lags only
  y_{t-1}..y_{t-p_y}.  Used for data-poor linear baselines (OLS, Ridge, etc.)
  that do not use a large predictor set directly.

The builder is stateful: ``fit`` computes PCA loadings on the training window;
``transform`` applies those loadings to any window (train or test) without
refitting.  This ensures strict pseudo-OOS discipline — no look-ahead.

Notes
-----
PCA is applied to the *full* predictor panel X (T, N), NOT to the target
series y.  The target is included only through the AR lag columns.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureBuilder:
    """Construct the predictor matrix Z_t for a single training window.

    Parameters
    ----------
    n_factors : int
        Number of PCA factors to extract from the large predictor panel.
        Ignored when ``use_factors=False``.  Tuned by the CV loop.
    n_lags : int
        Number of AR lags of the target series to append.  Applies in both
        modes.  Tuned by the CV loop.
    use_factors : bool
        If True, prepend PCA factors to the feature matrix (FACTORS / ARDI
        mode).  If False, use AR lags only.
    standardize_X : bool
        Standardize the predictor panel before PCA.  Recommended (default True).
    standardize_Z : bool
        Standardize the final feature matrix Z before returning.  Useful for
        kernel-based models (KRR, SVR) and NNs that are sensitive to scale.
        Default False (sklearn pipelines can handle scaling internally).
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_lags: int = 4,
        use_factors: bool = True,
        standardize_X: bool = True,
        standardize_Z: bool = False,
    ) -> None:
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.use_factors = use_factors
        self.standardize_X = standardize_X
        self.standardize_Z = standardize_Z

        # Fitted objects — set by fit()
        self._scaler_X: StandardScaler | None = None
        self._pca: PCA | None = None
        self._scaler_Z: StandardScaler | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> FeatureBuilder:
        """Fit PCA (and optional scalers) on the training window.

        Parameters
        ----------
        X_panel : array of shape (T_train, N)
            Large predictor panel for the training window.  Columns should
            already be stationary-transformed (tcode applied).
        y : array of shape (T_train,)
            Target series for the training window.  Used to construct AR lags.
            PCA is NOT fitted on y.

        Returns
        -------
        self
        """
        if self.use_factors:
            if self.standardize_X:
                self._scaler_X = StandardScaler()
                X_scaled = self._scaler_X.fit_transform(X_panel)
            else:
                X_scaled = X_panel

            # Clamp n_factors to available dimensions
            n_factors_actual = min(
                self.n_factors, X_scaled.shape[1], X_scaled.shape[0] - 1
            )
            self._pca = PCA(n_components=n_factors_actual)
            self._pca.fit(X_scaled)

        if self.standardize_Z:
            # Fit scaler on the training features; apply later in transform
            Z_train = self._build_Z(X_panel, y, is_train=True)
            self._scaler_Z = StandardScaler()
            self._scaler_Z.fit(Z_train)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        is_train: bool = False,
    ) -> NDArray[np.floating]:
        """Construct Z_t for a given window.

        Parameters
        ----------
        X_panel : array of shape (T, N)
            Predictor panel (stationary-transformed).
        y : array of shape (T,)
            Target series used to construct AR lag columns.
        is_train : bool
            Must be True when called on the training window so that row counts
            are handled consistently.  False for the test / OOS window.

        Returns
        -------
        Z : array of shape (T - n_lags, n_features)
            Feature matrix with the first ``n_lags`` rows dropped because those
            observations lack a full lag history.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        Z = self._build_Z(X_panel, y, is_train=is_train)
        if self.standardize_Z and self._scaler_Z is not None:
            Z = self._scaler_Z.transform(Z)
        return Z

    def fit_transform(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Fit on training data and return the training feature matrix."""
        self.fit(X_panel, y)
        return self.transform(X_panel, y, is_train=True)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_Z(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        is_train: bool,
    ) -> NDArray[np.floating]:
        """Assemble Z without the final scaler step.

        Two modes:
        * Training (T > p): sliding-window AR lags, rows 0..T-p-1.
        * Single test row (T <= p): y is interpreted as the lag history
          [y_{t-1}, y_{t-2}, ..., y_{t-p}] (or the last p values of training
          history). Returns exactly 1 row.
        """
        T = X_panel.shape[0]
        p = self.n_lags

        if p >= T:
            # Single test-row mode: y holds the p most-recent training values
            # [y_{T-p}, ..., y_{T-1}].  Reverse to get [y_{T-1}, ..., y_{T-p}].
            y_tail = y[-p:] if len(y) >= p else np.pad(y, (p - len(y), 0))
            ar_lags = y_tail[::-1].reshape(1, p)

            if self.use_factors and self._pca is not None:
                if self.standardize_X and self._scaler_X is not None:
                    X_scaled = self._scaler_X.transform(X_panel)
                else:
                    X_scaled = X_panel
                factors = self._pca.transform(X_scaled)  # (1, n_factors)
                Z = np.concatenate([factors, ar_lags], axis=1)
            else:
                Z = ar_lags
            return Z

        # Normal training-window mode (T > p)
        # AR lags: row i corresponds to time t = p + i.
        # Column j (lag j+1): y[p - j - 1 + i] = y_{t - j - 1}
        ar_lags = np.column_stack(
            [y[p - lag - 1 : T - lag - 1] for lag in range(p)]
        )  # shape (T - p, p)

        if self.use_factors and self._pca is not None:
            if self.standardize_X and self._scaler_X is not None:
                X_scaled = self._scaler_X.transform(X_panel)
            else:
                X_scaled = X_panel
            factors = self._pca.transform(X_scaled)  # (T, n_factors)
            factors = factors[p:, :]  # align: drop first p rows

            Z = np.concatenate([factors, ar_lags], axis=1)
        else:
            Z = ar_lags

        return Z

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Total number of columns in the output Z matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        factor_cols = self._pca.n_components_ if (self.use_factors and self._pca) else 0
        return factor_cols + self.n_lags

    @property
    def is_fitted(self) -> bool:
        return self._fitted
