"""Feature construction for macrocast pipeline.

FeatureBuilder assembles the predictor matrix Z_t passed to each
MacrocastEstimator.  The information set is controlled by three orthogonal
choices following Coulombe et al. (2021, CLSS) Table 1:

1. **factor_type** — what panel PCA is applied to:
   - ``"none"``  : no dimensionality reduction; use appended columns only
   - ``"X"``     : standard diffusion factors (PCA on stationary X panel)
   - ``"MARX"``  : moving average factors / MAF (PCA on MARX-transformed panel)

2. **Append flags** — orthogonal columns added to Z regardless of factor_type:
   - ``append_x_factors`` : prepend standard X-PCA factors alongside MAF
                            factors (active only when factor_type="MARX");
                            enables F-MAF, F-X-MAF
   - ``append_marx``      : append raw MARX columns to Z; enables F-MARX,
                            X-MARX, etc.
   - ``append_raw_x``     : append standardized stationary X columns; enables
                            F-X, X, X-MAF, etc.
   - ``append_levels``    : append level-form X columns and y_t level; enables
                            F-Level, X-Level, etc.

3. **AR lags of y** — always included in Z regardless of the information set.
   ``n_lags`` controls the lag order P_y (= P_f in the paper; CLSS uses 12).

This design maps all 16 CLSS 2021 Table 1 information sets onto unique flag
combinations (see FeatureSpec docstring for the full mapping table).

The builder is stateful: ``fit`` computes PCA loadings and scalers on the
training window; ``transform`` applies them without refitting, enforcing
strict pseudo-OOS discipline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from macrocast.preprocessing.transforms import apply_marx as _apply_marx


class FeatureBuilder:
    """Construct the predictor matrix Z_t for a single training window.

    Parameters
    ----------
    factor_type : str
        Primary dimensionality-reduction mode: ``"none"``, ``"X"``, or
        ``"MARX"``.  See module docstring for details.
    n_factors : int
        Number of PCA factors to extract.  Ignored when
        ``factor_type="none"``.
    n_lags : int
        AR lag order for the target (P_y).  Also controls row alignment for
        factor and X-column blocks (P_f = P_y throughout).  Applied in all
        modes; AR lags are always included in Z.
    p_marx : int
        MARX lag order (P_MARX in CLSS 2021).  Used when
        ``factor_type="MARX"`` or ``append_marx=True``.
    append_x_factors : bool
        When ``factor_type="MARX"``, also prepend standard X-PCA factors to
        Z.  Enables the F-MAF and F-X-MAF information sets.
    append_marx : bool
        Append raw MARX columns to Z.  Enables F-MARX, X-MARX, etc.
    append_raw_x : bool
        Append standardized stationary X columns to Z.  Enables F-X, X, etc.
    append_levels : bool
        Append level-form predictor columns and the current target level to Z.
        Requires ``X_levels`` to be passed to ``fit()`` and ``transform()``.
    standardize_X : bool
        Standardize the predictor panel before PCA (recommended).
    standardize_Z : bool
        Standardize the assembled Z matrix.  Useful for kernel-based models
        (KRR, SVR) and neural networks.
    """

    def __init__(
        self,
        factor_type: str = "X",
        n_factors: int = 8,
        n_lags: int = 4,
        p_marx: int = 12,
        append_x_factors: bool = False,
        append_marx: bool = False,
        append_raw_x: bool = False,
        append_levels: bool = False,
        standardize_X: bool = True,
        standardize_Z: bool = False,
    ) -> None:
        if factor_type not in {"none", "X", "MARX"}:
            raise ValueError(
                f"factor_type must be 'none', 'X', or 'MARX'; got {factor_type!r}"
            )
        self.factor_type = factor_type
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.p_marx = p_marx
        self.append_x_factors = append_x_factors
        self.append_marx = append_marx
        self.append_raw_x = append_raw_x
        self.append_levels = append_levels
        self.standardize_X = standardize_X
        self.standardize_Z = standardize_Z

        # Fitted objects — populated by fit()
        self._imputer_X: SimpleImputer | None = None
        self._scaler_X: StandardScaler | None = None
        self._pca: PCA | None = None           # primary: on X or MARX panel
        self._pca_x: PCA | None = None         # secondary: on raw X (append_x_factors)
        self._scaler_Z: StandardScaler | None = None
        self._scaler_levels: StandardScaler | None = None
        # Last p_marx rows of scaled X, needed for test-time MARX computation
        self._last_X_rows: NDArray[np.floating] | None = None
        self._fitted: bool = False

        # Feature name / group tracking — populated by fit()
        self._feature_names_out_: list[str] = []
        self._feature_group_map_: dict[str, str] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _marx_transform(
        X: NDArray[np.floating], p_marx: int
    ) -> NDArray[np.floating]:
        """Delegate to :func:`macrocast.preprocessing.transforms.apply_marx`."""
        return _apply_marx(X, p=p_marx, scale=False)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_panel: NDArray[np.floating],
        y: NDArray[np.floating],
        X_levels: NDArray[np.floating] | None = None,
    ) -> FeatureBuilder:
        """Fit scalers and PCA on the training window.

        Parameters
        ----------
        X_panel : (T_train, N)
            Stationary-transformed predictor panel.
        y : (T_train,)
            Target series (used for AR lags; PCA is not applied to y).
        X_levels : (T_train, N) or None
            Level-form predictor panel.  Required when
            ``append_levels=True``.
        """
        needs_marx = self.factor_type == "MARX" or self.append_marx
        needs_x = self.factor_type in {"X", "MARX"} or self.append_raw_x or self.append_x_factors

        # Impute NaN in X_panel (column-wise median, fit on training data only)
        self._imputer_X = SimpleImputer(strategy="median", keep_empty_features=True)
        X_panel = self._imputer_X.fit_transform(X_panel)

        # Fit X scaler whenever X is touched
        if needs_x and self.standardize_X:
            self._scaler_X = StandardScaler()
            X_scaled = self._scaler_X.fit_transform(X_panel)
        else:
            X_scaled = X_panel.astype(float)

        # Store trailing scaled X rows for test-time MARX computation
        if needs_marx:
            self._last_X_rows = X_scaled[-self.p_marx :].copy()

        # Fit primary PCA
        if self.factor_type == "X":
            n_act = min(self.n_factors, X_scaled.shape[1], X_scaled.shape[0] - 1)
            self._pca = PCA(n_components=n_act)
            self._pca.fit(X_scaled)
        elif self.factor_type == "MARX":
            X_marx = self._marx_transform(X_scaled, self.p_marx)
            n_act = min(self.n_factors, X_marx.shape[1], X_marx.shape[0] - 1)
            self._pca = PCA(n_components=n_act)
            self._pca.fit(X_marx)

        # Fit secondary X-PCA (for append_x_factors alongside MAF)
        if self.append_x_factors and self.factor_type == "MARX":
            n_act2 = min(self.n_factors, X_scaled.shape[1], X_scaled.shape[0] - 1)
            self._pca_x = PCA(n_components=n_act2)
            self._pca_x.fit(X_scaled)

        # Fit level scaler
        if self.append_levels and X_levels is not None:
            self._scaler_levels = StandardScaler()
            self._scaler_levels.fit(X_levels)

        # Build training Z for Z-scaler and name tracking
        Z_train = self._build_Z(X_panel, y, is_train=True, X_levels=X_levels)

        if self.standardize_Z:
            self._scaler_Z = StandardScaler()
            self._scaler_Z.fit(Z_train)

        # Populate feature names and group map
        self._build_names(X_panel.shape[1], Z_train.shape[1])

        self._fitted = True
        return self

    def _build_names(self, n_cols: int, n_z: int) -> None:
        """Populate _feature_names_out_ and _feature_group_map_."""
        names: list[str] = []

        # Primary factors
        if self.factor_type == "X" and self._pca is not None:
            for i in range(self._pca.n_components_):
                names.append(f"factor_{i + 1}")
        elif self.factor_type == "MARX" and self._pca is not None:
            for i in range(self._pca.n_components_):
                names.append(f"MAF_factor_{i + 1}")

        # Secondary X factors (append_x_factors alongside MAF)
        if self.append_x_factors and self._pca_x is not None:
            for i in range(self._pca_x.n_components_):
                names.append(f"F_factor_{i + 1}")

        # AR lags of y (always present)
        for i in range(self.n_lags):
            names.append(f"y_lag_{i + 1}")

        # MARX columns
        if self.append_marx:
            for i in range(n_cols * self.p_marx):
                names.append(f"MARX_{i}")

        # Raw X columns
        if self.append_raw_x:
            for i in range(n_cols):
                names.append(f"X_{i}")

        # Level columns (N_levels may differ from n_cols when X_levels has fewer series)
        if self.append_levels:
            n_level_cols = (
                self._scaler_levels.n_features_in_
                if self._scaler_levels is not None
                else n_cols
            )
            for i in range(n_level_cols):
                names.append(f"level_{i}")
            names.append("level_y")

        if len(names) != n_z:
            raise RuntimeError(
                f"Feature name count ({len(names)}) != Z columns ({n_z}). "
                "Bug in FeatureBuilder._build_names."
            )

        group_map: dict[str, str] = {}
        for name in names:
            if name.startswith(("factor_", "MAF_factor_", "F_factor_")):
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
        X_panel : (T, N)
            Stationary-transformed predictor panel.
        y : (T,) for training; last ``n_lags`` values for test.
            Used to construct AR lag columns.
        is_train : bool
            True when called on the full training window.
        X_levels : (T, N) or None
            Level-form panel.  Required when ``append_levels=True``.
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
        """Assemble the feature matrix Z without the final scaler step.

        Training path (is_train=True):
            Returns (T - offset, n_features) where offset = max(n_lags, p_marx-1).
        Test path (is_train=False):
            X_panel is (1, N); y holds the last n_lags training values.
            Returns (1, n_features).
        """
        if self._imputer_X is not None:
            X_panel = self._imputer_X.transform(X_panel)

        # Forward-fill then zero-fill NaN in y
        if np.isnan(y).any():
            y = y.copy().astype(float)
            for i in range(1, len(y)):
                if np.isnan(y[i]):
                    y[i] = y[i - 1]
            y = np.where(np.isnan(y), 0.0, y)

        # Scale X
        if self._scaler_X is not None:
            X_scaled = self._scaler_X.transform(X_panel)
        else:
            X_scaled = X_panel.astype(float)

        p = self.n_lags

        if not is_train:
            return self._build_test_row(X_scaled, y, p, X_levels)
        return self._build_train_block(X_scaled, y, p, X_levels)

    def _build_test_row(
        self,
        X_scaled: NDArray[np.floating],
        y: NDArray[np.floating],
        p: int,
        X_levels: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Build a single test-row Z. X_scaled is (1, N)."""
        # AR lags: y holds last p training values [y_{T-p}, ..., y_{T-1}]
        y_tail = y[-p:] if len(y) >= p else np.pad(y, (p - len(y), 0))
        ar_lags = y_tail[::-1].reshape(1, p)  # lag-1 first

        parts: list[NDArray[np.floating]] = []

        # Primary factors
        if self.factor_type == "X" and self._pca is not None:
            parts.append(self._pca.transform(X_scaled))
        elif self.factor_type == "MARX" and self._pca is not None and self._last_X_rows is not None:
            X_window = np.vstack([self._last_X_rows, X_scaled])
            marx_out = self._marx_transform(X_window, self.p_marx)
            parts.append(self._pca.transform(marx_out[-1:]))

        # Secondary X factors alongside MAF
        if self.append_x_factors and self._pca_x is not None:
            parts.append(self._pca_x.transform(X_scaled))

        # AR lags (always)
        parts.append(ar_lags)

        # MARX columns
        if self.append_marx and self._last_X_rows is not None:
            X_window = np.vstack([self._last_X_rows, X_scaled])
            marx_out = self._marx_transform(X_window, self.p_marx)
            parts.append(marx_out[-1:])

        # Raw X
        if self.append_raw_x:
            parts.append(X_scaled)

        # Level columns
        if self.append_levels and X_levels is not None and self._scaler_levels is not None:
            parts.append(self._scaler_levels.transform(X_levels))
            parts.append(np.array([[y[-1]]]))

        return np.concatenate(parts, axis=1)

    def _build_train_block(
        self,
        X_scaled: NDArray[np.floating],
        y: NDArray[np.floating],
        p: int,
        X_levels: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Build the full training-window Z. X_scaled is (T, N)."""
        T = X_scaled.shape[0]

        # Determine common start row (max alignment offset)
        ar_offset = p          # ar_lags starts at row p
        marx_offset = self.p_marx - 1  # MARX valid from row p_marx-1

        needs_marx_alignment = (
            self.factor_type == "MARX" or self.append_marx
        )
        common_start = max(ar_offset, marx_offset) if needs_marx_alignment else ar_offset

        # AR lags: row i → time t = p + i; lag j → y[p - j - 1 + i]
        ar_full = np.column_stack(
            [y[p - lag - 1 : T - lag - 1] for lag in range(p)]
        )  # (T - p, p)

        parts: list[NDArray[np.floating]] = []

        # Primary factors
        if self.factor_type == "X" and self._pca is not None:
            # PCA on raw X → (T, k); align by dropping first common_start rows
            f_full = self._pca.transform(X_scaled)  # (T, k)
            parts.append(f_full[common_start:])
        elif self.factor_type == "MARX" and self._pca is not None:
            X_marx = self._marx_transform(X_scaled, self.p_marx)  # (T-pm+1, K*pm)
            f_full = self._pca.transform(X_marx)                   # (T-pm+1, k)
            # f_full[i] → time index p_marx - 1 + i
            parts.append(f_full[common_start - marx_offset:])

        # Secondary X factors (append_x_factors alongside MAF)
        if self.append_x_factors and self._pca_x is not None:
            fx_full = self._pca_x.transform(X_scaled)  # (T, k)
            parts.append(fx_full[common_start:])

        # AR lags (always): ar_full[i] → time index p + i
        parts.append(ar_full[common_start - ar_offset:])

        # MARX columns
        if self.append_marx:
            X_marx_cols = self._marx_transform(X_scaled, self.p_marx)
            parts.append(X_marx_cols[common_start - marx_offset:])

        # Raw X columns
        if self.append_raw_x:
            n_rows = parts[0].shape[0] if parts else T - common_start
            parts.append(X_scaled[X_scaled.shape[0] - n_rows:])

        # Level columns
        if self.append_levels and X_levels is not None and self._scaler_levels is not None:
            n_rows = parts[0].shape[0] if parts else T - common_start
            lev_slice = X_levels[X_levels.shape[0] - n_rows:]
            parts.append(self._scaler_levels.transform(lev_slice))
            parts.append(y[len(y) - n_rows:].reshape(-1, 1))

        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_names_out_(self) -> list[str]:
        """Column names of Z, populated after fit()."""
        return list(self._feature_names_out_)

    @property
    def feature_group_map_(self) -> dict[str, str]:
        """Maps each feature name to its semantic group.

        Groups: ``"ar"``, ``"factors"``, ``"marx"``, ``"x"``, ``"levels"``.
        """
        return dict(self._feature_group_map_)

    @property
    def n_features(self) -> int:
        """Total number of features in Z (requires fit())."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return len(self._feature_names_out_)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
