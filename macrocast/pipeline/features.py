"""Feature construction for macrocast pipeline.

FeatureBuilder assembles the predictor matrix Z_t that is passed to each
MacrocastEstimator.  Four modes are supported, following Coulombe et al.
(2021, CLSS):

* **Factors mode** (FACTORS / ARDI regularization): Z_t contains PCA diffusion
  index factors f_1..f_{p_f} plus AR lags y_{t-1}..y_{t-p_y}.  Number of
  factors p_f and number of lags p_y are both tuning parameters exposed to the
  CV loop.

* **AR-only mode** (all other regularizations): Z_t contains AR lags only
  y_{t-1}..y_{t-p_y}.  Used for data-poor linear baselines (OLS, Ridge, etc.)
  that do not use a large predictor set directly.

* **MARX mode** (Moving Average Rotation of X): Each variable k at order p is
  replaced by the cumulative moving average (1/p) sum_{p'=1}^{p} X_{t-p'+1,k}.
  The MARX panel (K * p_marx columns) either replaces raw X before PCA, or is
  used directly as columns without PCA reduction.  Implements Eq. 7 of
  Coulombe et al. (2021).

* **MAF mode** (Moving Average Factors): PCA applied to the MARX-transformed
  panel instead of raw X.  Activated by ``use_maf=True``, which forces both
  ``use_marx=True`` and ``use_factors=True`` internally.

* **Raw-X mode** (``include_raw_x=True``): Append the standardized raw X
  columns to Z alongside factors, MARX columns, and AR lags.  Enables the
  F-X, X, X-MAF, and related information sets from Coulombe et al. (2021).

* **PCA-on-raw-X with MARX columns** (``use_factors=True``,
  ``use_marx=True``, ``marx_for_pca=False``): PCA is applied to the raw
  scaled X panel; MARX columns are then appended as separate Z columns
  alongside the factors.  Contrast with MAF mode where PCA is applied to
  the MARX panel itself.

Additionally, level-form predictors and the target can be appended to Z_t via
``include_levels=True`` (Coulombe et al. 2021, p. 1342).

The builder is stateful: ``fit`` computes PCA loadings on the training window;
``transform`` applies those loadings to any window (train or test) without
refitting.  This ensures strict pseudo-OOS discipline with no look-ahead.

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
        Number of AR lags of the target series to append.  Applies in all
        modes.  Tuned by the CV loop.
    use_factors : bool
        If True, prepend PCA factors to the feature matrix (FACTORS / ARDI
        mode).  If False, use AR lags only (or MARX columns when
        ``use_marx=True``).
    standardize_X : bool
        Standardize the predictor panel before PCA.  Recommended (default True).
    standardize_Z : bool
        Standardize the final feature matrix Z before returning.  Useful for
        kernel-based models (KRR, SVR) and NNs that are sensitive to scale.
        Default False (sklearn pipelines can handle scaling internally).
    use_marx : bool
        If True, apply the MARX transformation to the predictor panel before
        PCA (or directly as columns when ``use_factors=False``).  Implements
        the Moving Average Rotation of X from Coulombe et al. (2021).
    p_marx : int
        Maximum moving average order for MARX.  Produces p_marx moving averages
        per variable.  Default 12 (suited to monthly data).
    use_maf : bool
        If True, apply PCA to the MARX-transformed panel (Moving Average
        Factors).  Forces ``use_factors=True`` and ``use_marx=True``
        internally.  Coulombe et al. (2021), p. 1341.
    include_levels : bool
        If True, append level-form predictor columns and the current target
        value to Z.  Requires ``X_levels`` to be passed to ``fit()`` and
        ``transform()``.  Coulombe et al. (2021), p. 1342.
    include_raw_x : bool
        If True, append the standardized raw X columns to Z alongside factors
        or MARX columns.  Enables the F-X, X, X-MAF, etc. information sets
        from Coulombe et al. (2021).  Default False.
    marx_for_pca : bool
        Controls what panel PCA is applied to when both ``use_factors=True``
        and ``use_marx=True``.  When True (default), PCA is applied to the
        MARX panel (MAF mode).  When False, PCA is applied to the raw scaled
        X panel and the MARX columns are appended as additional Z columns.
        Ignored when ``use_factors`` and ``use_marx`` are not both True.
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_lags: int = 4,
        use_factors: bool = True,
        standardize_X: bool = True,
        standardize_Z: bool = False,
        use_marx: bool = False,
        p_marx: int = 12,
        use_maf: bool = False,
        include_levels: bool = False,
        include_raw_x: bool = False,
        marx_for_pca: bool = True,
    ) -> None:
        # MAF is syntactic sugar: forces both use_factors and use_marx on
        if use_maf:
            use_factors = True
            use_marx = True

        self.n_factors = n_factors
        self.n_lags = n_lags
        self.use_factors = use_factors
        self.standardize_X = standardize_X
        self.standardize_Z = standardize_Z
        self.use_marx = use_marx
        self.p_marx = p_marx
        self.use_maf = use_maf
        self.include_levels = include_levels
        self.include_raw_x = include_raw_x
        self.marx_for_pca = marx_for_pca

        # Fitted objects — populated by fit()
        self._scaler_X: StandardScaler | None = None
        self._pca: PCA | None = None
        self._scaler_Z: StandardScaler | None = None
        self._scaler_levels: StandardScaler | None = None
        # Last p_marx rows of unscaled training X, needed for test-time MARX
        self._last_X_rows: NDArray[np.floating] | None = None
        self._fitted: bool = False

        # Feature name tracking — populated by fit()
        self._feature_names_out_: list[str] = []
        self._feature_group_map_: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _marx_transform(
        X: NDArray[np.floating], p_marx: int
    ) -> NDArray[np.floating]:
        """Moving Average Rotation of X (Coulombe et al. 2021).

        For variable k and order p, the MARX feature at time t is:
            Z_{t,k,p} = (1/p) sum_{p'=1}^{p} X_{t-p'+1, k}

        Uses the cumsum trick for O(T * K * p_marx) efficiency.

        Parameters
        ----------
        X : NDArray of shape (T, K)
            Predictor panel (already standardized if required).
        p_marx : int
            Maximum moving average order.  All orders 1..p_marx are computed.

        Returns
        -------
        NDArray of shape (T - p_marx + 1, K * p_marx)
            Columns ordered as [MA_1_col0, ..., MA_1_colK-1,
            MA_2_col0, ..., MA_p_colK-1].  The first p_marx - 1 rows of X
            are dropped because the longest MA requires a full history of
            length p_marx.
        """
        T, K = X.shape
        # cs[i] = sum of X[0 .. i-1], with cs[0] = 0.  Shape (T+1, K).
        cs = np.zeros((T + 1, K), dtype=float)
        cs[1:] = np.cumsum(X, axis=0)

        parts: list[NDArray[np.floating]] = []
        for p in range(1, p_marx + 1):
            # MA_p at time t (0-indexed): mean of X[t-p+1 .. t]
            #   = (cs[t+1] - cs[t+1-p]) / p
            # Valid for t >= p - 1.  We evaluate for t = p_marx-1 .. T-1,
            # which corresponds to cs row indices p_marx .. T.
            ma_p = (cs[p_marx : T + 1] - cs[p_marx - p : T + 1 - p]) / p
            parts.append(ma_p)

        return np.concatenate(parts, axis=1)  # (T - p_marx + 1, K * p_marx)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        X_levels: NDArray[np.floating] | None = None,
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
        X_levels : array of shape (T_train, N_levels) or None
            Predictor panel in levels (pre-tcode).  Required when
            ``include_levels=True``.

        Returns
        -------
        self
        """
        # Store trailing unscaled X rows for test-time MARX computation.
        # Must be stored before any fitting so the raw values are preserved.
        if self.use_marx:
            self._last_X_rows = X_panel[-self.p_marx :].copy()

        if self.use_factors:
            if self.standardize_X:
                self._scaler_X = StandardScaler()
                X_scaled = self._scaler_X.fit_transform(X_panel)
            else:
                X_scaled = X_panel.astype(float)

            # When MARX is active, fit PCA on the MARX panel (MAF mode) when
            # marx_for_pca=True (default).  When marx_for_pca=False, PCA is
            # fitted on the raw scaled panel; MARX columns are appended later.
            if self.use_marx and self.marx_for_pca:
                X_for_pca = self._marx_transform(X_scaled, self.p_marx)
            else:
                X_for_pca = X_scaled

            # Clamp n_factors to available dimensions to avoid PCA errors
            n_factors_actual = min(
                self.n_factors, X_for_pca.shape[1], X_for_pca.shape[0] - 1
            )
            self._pca = PCA(n_components=n_factors_actual)
            self._pca.fit(X_for_pca)

        elif self.use_marx:
            # MARX without PCA: still fit the X scaler if requested so that
            # test-time scaling is consistent with training.
            if self.standardize_X:
                self._scaler_X = StandardScaler()
                self._scaler_X.fit(X_panel)

        elif self.include_raw_x:
            # Raw-X mode with no factors or MARX: fit the X scaler so that
            # test-time transform is consistent with training.
            if self.standardize_X:
                self._scaler_X = StandardScaler()
                self._scaler_X.fit(X_panel)

        if self.include_levels and X_levels is not None:
            self._scaler_levels = StandardScaler()
            self._scaler_levels.fit(X_levels)

        # Build training Z (needed for both Z-scaler fitting and name tracking).
        Z_train = self._build_Z(X_panel, y, is_train=True, X_levels=X_levels)

        if self.standardize_Z:
            self._scaler_Z = StandardScaler()
            self._scaler_Z.fit(Z_train)

        # ------------------------------------------------------------------
        # Populate feature name tracking
        # ------------------------------------------------------------------
        names: list[str] = []

        # Determine actual number of PCA factors (may have been clamped).
        n_factors_actual: int = (
            self._pca.n_components_ if (self.use_factors and self._pca is not None) else 0
        )

        # Determine whether the factors come from a MARX panel (MAF) or raw X.
        is_maf = self.use_factors and self.use_marx and self.marx_for_pca

        if self.use_factors:
            if is_maf:
                # MAF mode: factors derived from the MARX-transformed panel.
                names.extend([f"MAF_factor_{i+1}" for i in range(n_factors_actual)])
            else:
                names.extend([f"factor_{i+1}" for i in range(n_factors_actual)])

            if self.use_marx and not self.marx_for_pca:
                # PCA on raw X with MARX columns appended separately.
                n_marx_cols = X_panel.shape[1] * self.p_marx
                names.extend([f"MARX_{i}" for i in range(n_marx_cols)])

        elif self.use_marx:
            # MARX columns without PCA reduction.
            n_marx_cols = X_panel.shape[1] * self.p_marx
            names.extend([f"MARX_{i}" for i in range(n_marx_cols)])

        # AR lags of the target — always present in all modes.
        names.extend([f"y_lag_{i+1}" for i in range(self.n_lags)])

        # Raw X columns appended when include_raw_x=True.
        if self.include_raw_x:
            names.extend([f"X_{i}" for i in range(X_panel.shape[1])])

        # Level columns (X_levels columns + y level scalar) appended last.
        if self.include_levels and X_levels is not None:
            n_levels = X_levels.shape[1]
            names.extend([f"level_{i}" for i in range(n_levels)])
            names.append("level_y")

        if len(names) != Z_train.shape[1]:
            raise RuntimeError(
                f"Feature name count ({len(names)}) does not match Z columns "
                f"({Z_train.shape[1]}). This is a bug in FeatureBuilder."
            )

        # Build group map from the assembled names.
        group_map: dict[str, str] = {}
        for name in names:
            if name.startswith("MAF_factor_") or name.startswith("factor_"):
                group_map[name] = "factors"
            elif name.startswith("MARX_"):
                group_map[name] = "marx"
            elif name.startswith("y_lag_"):
                group_map[name] = "ar"
            elif name.startswith("X_"):
                group_map[name] = "x"
            elif name.startswith("level_"):
                group_map[name] = "levels"

        self._feature_names_out_ = names
        self._feature_group_map_ = group_map

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
        X_levels: NDArray[np.floating] | None = None,
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
        X_levels : array of shape (T, N_levels) or None
            Predictor panel in levels.  Required when ``include_levels=True``.

        Returns
        -------
        Z : NDArray of shape (effective_rows, n_features)
            Feature matrix.  For the training window the effective row count
            is T - max(n_lags, p_marx - 1) when MARX is active, or T - n_lags
            otherwise.  For the test window it is always 1.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        Z = self._build_Z(X_panel, y, is_train=is_train, X_levels=X_levels)
        if self.standardize_Z and self._scaler_Z is not None:
            Z = self._scaler_Z.transform(Z)
        return Z

    def fit_transform(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        X_levels: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Fit on training data and return the training feature matrix."""
        self.fit(X_panel, y, X_levels=X_levels)
        return self.transform(X_panel, y, is_train=True, X_levels=X_levels)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_Z(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        is_train: bool,
        X_levels: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Assemble Z without the final scaler step.

        Two modes:
        * Training (T > p): sliding-window AR lags, rows 0..T-p-1.
        * Single test row (T <= p): y is interpreted as the lag history
          [y_{t-1}, y_{t-2}, ..., y_{t-p}] (or the last p values of training
          history).  Returns exactly 1 row.
        """
        T = X_panel.shape[0]
        p = self.n_lags

        # ------------------------------------------------------------------
        # Test-row (single-step) path: T <= p
        # ------------------------------------------------------------------
        if p >= T:
            # y holds the p most-recent training values [y_{T-p}, ..., y_{T-1}].
            # Reverse to get [y_{T-1}, ..., y_{T-p}] (lag-1 first).
            y_tail = y[-p:] if len(y) >= p else np.pad(y, (p - len(y), 0))
            ar_lags = y_tail[::-1].reshape(1, p)

            if self.use_factors and self._pca is not None:
                if self.use_marx and self.marx_for_pca and self._last_X_rows is not None:
                    # MAF test path: build the MARX window (last p_marx training
                    # rows + test row), apply PCA to the MARX representation.
                    if self.standardize_X and self._scaler_X is not None:
                        last_scaled = self._scaler_X.transform(self._last_X_rows)
                        cur_scaled = self._scaler_X.transform(X_panel)
                    else:
                        last_scaled = self._last_X_rows.astype(float)
                        cur_scaled = X_panel.astype(float)
                    # X_window has p_marx + 1 rows; _marx_transform returns 2 rows.
                    # Take the last row which corresponds to the test observation.
                    X_window = np.vstack([last_scaled, cur_scaled])
                    marx_out = self._marx_transform(X_window, self.p_marx)
                    factors = self._pca.transform(marx_out[-1:, :])  # (1, n_factors)
                    Z = np.concatenate([factors, ar_lags], axis=1)
                elif self.use_marx and not self.marx_for_pca:
                    # PCA on raw X + MARX columns appended separately (test path).
                    if self.standardize_X and self._scaler_X is not None:
                        X_scaled = self._scaler_X.transform(X_panel)
                    else:
                        X_scaled = X_panel.astype(float)
                    factors = self._pca.transform(X_scaled)  # (1, n_factors)
                    # Build MARX columns using last p_marx training rows + test row.
                    if self._last_X_rows is not None:
                        if self.standardize_X and self._scaler_X is not None:
                            last_scaled = self._scaler_X.transform(self._last_X_rows)
                        else:
                            last_scaled = self._last_X_rows.astype(float)
                        X_window = np.vstack([last_scaled, X_scaled])
                        marx_out = self._marx_transform(X_window, self.p_marx)
                        Z = np.concatenate([factors, marx_out[-1:, :], ar_lags], axis=1)
                    else:
                        Z = np.concatenate([factors, ar_lags], axis=1)
                else:
                    if self.standardize_X and self._scaler_X is not None:
                        X_scaled = self._scaler_X.transform(X_panel)
                    else:
                        X_scaled = X_panel.astype(float)
                    factors = self._pca.transform(X_scaled)  # (1, n_factors)
                    Z = np.concatenate([factors, ar_lags], axis=1)

            elif self.use_marx:
                # MARX columns without PCA reduction, test row
                if self._last_X_rows is not None:
                    if self.standardize_X and self._scaler_X is not None:
                        last_scaled = self._scaler_X.transform(self._last_X_rows)
                        cur_scaled = self._scaler_X.transform(X_panel)
                    else:
                        last_scaled = self._last_X_rows.astype(float)
                        cur_scaled = X_panel.astype(float)
                    X_window = np.vstack([last_scaled, cur_scaled])
                    marx_out = self._marx_transform(X_window, self.p_marx)
                    Z = np.concatenate([marx_out[-1:, :], ar_lags], axis=1)
                else:
                    Z = ar_lags

            else:
                Z = ar_lags

            # Append raw standardized X columns if requested (test-row path).
            if self.include_raw_x:
                if self.standardize_X and self._scaler_X is not None:
                    X_raw = self._scaler_X.transform(X_panel)
                else:
                    X_raw = X_panel.astype(float)
                Z = np.concatenate([Z, X_raw], axis=1)

            # Append level columns + current y level if requested
            if (
                self.include_levels
                and X_levels is not None
                and self._scaler_levels is not None
            ):
                levels_scaled = self._scaler_levels.transform(X_levels)
                y_level = np.array([[y[-1]]])  # (1, 1) — last training y value
                Z = np.concatenate([Z, levels_scaled, y_level], axis=1)

            return Z

        # ------------------------------------------------------------------
        # Training-window path: T > p
        # ------------------------------------------------------------------
        # AR lags: row i corresponds to time t = p + i.
        # Column j (lag j+1): y[p - j - 1 + i] = y_{t - j - 1}
        ar_lags = np.column_stack(
            [y[p - lag - 1 : T - lag - 1] for lag in range(p)]
        )  # shape (T - p, p)

        if self.use_factors and self._pca is not None:
            if self.standardize_X and self._scaler_X is not None:
                X_scaled = self._scaler_X.transform(X_panel)
            else:
                X_scaled = X_panel.astype(float)

            if self.use_marx and self.marx_for_pca:
                # MAF path: apply MARX before PCA.
                # _marx_transform returns rows for time indices p_marx-1 .. T-1.
                X_marx = self._marx_transform(X_scaled, self.p_marx)
                # X_marx.shape = (T - p_marx + 1, K * p_marx)
                factors_full = self._pca.transform(X_marx)
                # Align factors_full and ar_lags to the same time index.
                #   factors_full[i] -> time index p_marx - 1 + i
                #   ar_lags[i]      -> time index p + i
                # Common start time: max(p_marx - 1, p)
                marx_offset = self.p_marx - 1
                ar_offset = p
                common_start = max(marx_offset, ar_offset)
                factors = factors_full[common_start - marx_offset :]
                ar_lags = ar_lags[common_start - ar_offset :]
                Z = np.concatenate([factors, ar_lags], axis=1)
            elif self.use_marx and not self.marx_for_pca:
                # PCA on raw X; MARX columns appended as separate Z columns.
                factors_full = self._pca.transform(X_scaled)  # (T, n_factors)
                # Compute MARX panel for the separate columns.
                X_marx = self._marx_transform(X_scaled, self.p_marx)
                # Align all three: factors_full, X_marx, ar_lags.
                #   factors_full[i]  -> time index i
                #   X_marx[i]        -> time index p_marx - 1 + i
                #   ar_lags[i]       -> time index p + i
                marx_offset = self.p_marx - 1
                ar_offset = p
                common_start = max(marx_offset, ar_offset)
                factors = factors_full[common_start:]
                X_marx_aligned = X_marx[common_start - marx_offset:]
                ar_lags_aligned = ar_lags[common_start - ar_offset:]
                Z = np.concatenate([factors, X_marx_aligned, ar_lags_aligned], axis=1)
            else:
                # Standard factors path: PCA on raw scaled X.
                factors = self._pca.transform(X_scaled)  # (T, n_factors)
                factors = factors[p:, :]  # align: drop first p rows
                Z = np.concatenate([factors, ar_lags], axis=1)

        elif self.use_marx:
            # MARX columns without PCA reduction
            if self.standardize_X and self._scaler_X is not None:
                X_scaled = self._scaler_X.transform(X_panel)
            else:
                X_scaled = X_panel.astype(float)
            X_marx = self._marx_transform(X_scaled, self.p_marx)
            # Align X_marx and ar_lags
            marx_offset = self.p_marx - 1
            ar_offset = p
            common_start = max(marx_offset, ar_offset)
            X_marx_aligned = X_marx[common_start - marx_offset :]
            ar_lags_aligned = ar_lags[common_start - ar_offset :]
            Z = np.concatenate([X_marx_aligned, ar_lags_aligned], axis=1)

        else:
            Z = ar_lags

        # Append raw standardized X columns if requested.  This enables the
        # F-X, X, and related information sets from Coulombe et al. (2021).
        # Works regardless of which branch above ran; X_scaled may or may not
        # be in scope, so we recompute it lazily here.
        if self.include_raw_x:
            n_rows = Z.shape[0]
            if self.standardize_X and self._scaler_X is not None:
                X_raw = self._scaler_X.transform(X_panel)
            else:
                X_raw = X_panel.astype(float)
            # Drop the first rows to align with Z's time index.
            X_raw_aligned = X_raw[X_raw.shape[0] - n_rows:]
            Z = np.concatenate([Z, X_raw_aligned], axis=1)

        # Append level columns + current y level if requested.
        # Z has n_rows rows aligned to the common start time index.
        if (
            self.include_levels
            and X_levels is not None
            and self._scaler_levels is not None
        ):
            n_rows = Z.shape[0]
            # Take the last n_rows rows of X_levels and y to match Z's time index.
            levels_slice = X_levels[X_levels.shape[0] - n_rows :]
            levels_scaled = self._scaler_levels.transform(levels_slice)
            y_level_col = y[len(y) - n_rows :].reshape(-1, 1)
            Z = np.concatenate([Z, levels_scaled, y_level_col], axis=1)

        return Z

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_names_out_(self) -> list[str]:
        """Column names of the Z matrix, populated after fit().

        Names are positional/index-based (not FRED series names) because
        FeatureBuilder receives a numpy array rather than a named DataFrame.

        Returns
        -------
        list[str]
            Copy of the internal list; mutating the returned value has no
            effect on the builder's state.
        """
        return list(self._feature_names_out_)

    @property
    def feature_group_map_(self) -> dict[str, str]:
        """Maps each feature name to its semantic group tag.

        Group tags: ``"ar"``, ``"factors"``, ``"marx"``, ``"x"``,
        ``"levels"``.  Populated after fit().

        Note: both MAF factors (``MAF_factor_*``) and PCA factors
        (``factor_*``) map to group ``"factors"``.  Callers who need to
        distinguish them must inspect the name prefix directly.

        Returns
        -------
        dict[str, str]
            Copy of the internal dict; mutating the returned value has no
            effect on the builder's state.
        """
        return dict(self._feature_group_map_)

    @property
    def n_features(self) -> int:
        """Total number of features in Z after fit().

        Before fit(), returns an estimate based on factors and AR lags only.
        After fit(), returns the exact count matching feature_names_out_.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._feature_names_out_:
            return len(self._feature_names_out_)
        # Pre-fit estimate: factors + AR lags
        factor_cols = self._pca.n_components_ if (self.use_factors and self._pca) else 0
        return factor_cols + self.n_lags

    @property
    def is_fitted(self) -> bool:
        return self._fitted
