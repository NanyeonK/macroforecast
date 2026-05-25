"""Standalone L2 clean-panel function-op namespace.

Cycle 34: L2 clean panel ops standalone-ization (14 ops).

Each callable wraps the corresponding runtime primitive from
``macroforecast.core.runtime`` to preserve bit-exact results with
the recipe-path dispatch. The full-sample variant is used (no
per-origin cutoff_ts); recipe-path OOS safety is handled by the
runtime's _apply_*_per_origin helpers.

Import pattern follows C28/C29/C30-C33 (lazy runtime imports inside
each function body to avoid circular imports).

14 ops across 5 clusters:
    L2.A freq-align (2):  freq_align_quarterly_to_monthly_clean,
                          freq_align_monthly_to_quarterly_clean
    L2.B tcode (1):       apply_tcode_transform
    L2.C outlier (3):     iqr_outlier_clean, zscore_outlier_clean,
                          winsorize_clean
    L2.D imputation (5):  em_factor_impute_clean, em_multivariate_impute_clean,
                          mean_impute_clean, forward_fill_clean,
                          linear_interpolate_clean
    L2.E frame edge (3):  truncate_to_balanced_clean,
                          drop_unbalanced_series_clean,
                          zero_fill_leading_clean
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------

def _require_non_empty(panel: pd.DataFrame, *, name: str = "panel") -> None:
    """Raise ValueError when the DataFrame has zero rows or zero columns."""
    if panel.empty:
        raise ValueError(
            f"{name} must not be empty; got shape {panel.shape}"
        )


# ============================================================================
# L2.C Outlier ops
# ============================================================================


def iqr_outlier_clean(
    panel: pd.DataFrame,
    *,
    threshold: float = 10.0,
    action: str = "flag_as_nan",
) -> pd.DataFrame:
    """Flag or replace outliers using the McCracken-Ng IQR-multiple rule.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
    threshold : float, default 10.0
        IQR multiplier for outlier flagging. McCracken-Ng's published
        default is 10.0; set to a smaller value (e.g. 3.0) for a tighter
        filter. Must be strictly positive.
    action : str, default "flag_as_nan"
        What to do with flagged values. One of:

        * ``"flag_as_nan"`` -- replace flagged cells with NaN (default;
          pairs with L2.D imputation to recover values).
        * ``"replace_with_median"`` -- replace flagged cells with the
          per-column full-sample median.
        * ``"replace_with_cap_value"`` -- cap flagged cells at the
          per-column 1st / 99th percentile.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; flagged cells replaced according to *action*.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_outlier_policy`` for
    ``policy="mccracken_ng_iqr"``):

    1. Compute per-column median and IQR = q(0.75) - q(0.25).
    2. ``iqr.replace(0, np.nan)`` ensures zero-IQR columns never flag any
       value.
    3. Flag ``|x - median| > threshold * IQR``.

    Equivalent recipe configuration::

        l2:
          outlier_policy: mccracken_ng_iqr
          outlier_action: flag_as_nan
          leaf_config:
            outlier_iqr_threshold: 10.0

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 3), columns=list("abc"))
    >>> panel.iloc[5, 0] = 1000.0  # extreme outlier
    >>> out = iqr_outlier_clean(panel, threshold=10.0)
    >>> out.shape
    (50, 3)
    >>> out.iloc[5, 0]  # flagged -> NaN
    nan

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    _VALID_ACTIONS = {"flag_as_nan", "replace_with_median", "replace_with_cap_value"}
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold!r}")
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"action must be one of {sorted(_VALID_ACTIONS)}; got {action!r}"
        )

    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result

    median = numeric.median()
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    # Critical: replace zero-IQR columns with np.nan (not pd.NA) before masking
    # so that constant-valued columns never flag any observation.
    # Using np.nan avoids the TypeError raised by pandas 3.x when pd.NA participates
    # in a boolean comparison ("boolean value of NA is ambiguous").
    iqr_safe = iqr.replace(0, np.nan)
    mask = (numeric - median).abs() > threshold * iqr_safe
    mask = mask.fillna(False)  # NaN comparison result -> False (no outlier flag)

    if action == "flag_as_nan":
        result[numeric.columns] = numeric.mask(mask)
    elif action == "replace_with_median":
        result[numeric.columns] = numeric.mask(mask, median, axis=1)
    elif action == "replace_with_cap_value":
        upper = numeric.quantile(0.99)
        lower = numeric.quantile(0.01)
        capped = numeric.clip(lower=lower, upper=upper, axis=1)
        result[numeric.columns] = numeric.where(~mask.fillna(False), capped)
    return result


def zscore_outlier_clean(
    panel: pd.DataFrame,
    *,
    threshold: float = 3.0,
    action: str = "flag_as_nan",
) -> pd.DataFrame:
    """Flag or replace outliers beyond a z-score threshold.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
    threshold : float, default 3.0
        Z-score cut-off. Values with ``|z| > threshold`` are flagged.
        Must be strictly positive.
    action : str, default "flag_as_nan"
        What to do with flagged values. One of ``"flag_as_nan"``,
        ``"replace_with_median"``, or ``"replace_with_cap_value"``.
        Same semantics as :func:`iqr_outlier_clean`.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; flagged cells replaced according to *action*.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_outlier_policy`` for
    ``policy="zscore_threshold"``):

    1. Compute per-column mean and std (ddof=0).
    2. ``std.replace(0, pd.NA)`` ensures constant columns never flag any
       value.
    3. Flag ``|(x - mean) / std| > threshold``.

    Equivalent recipe configuration::

        l2:
          outlier_policy: zscore_threshold
          outlier_action: flag_as_nan
          leaf_config:
            zscore_threshold_value: 3.0

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 3), columns=list("abc"))
    >>> panel.iloc[5, 0] = 20.0  # extreme outlier
    >>> out = zscore_outlier_clean(panel, threshold=3.0)
    >>> out.shape
    (50, 3)
    >>> import numpy as np; np.isnan(out.iloc[5, 0])
    True

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    _VALID_ACTIONS = {"flag_as_nan", "replace_with_median", "replace_with_cap_value"}
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold!r}")
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"action must be one of {sorted(_VALID_ACTIONS)}; got {action!r}"
        )

    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result

    mask = (
        (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, pd.NA)
    ).abs() > threshold

    if action == "flag_as_nan":
        result[numeric.columns] = numeric.mask(mask)
    elif action == "replace_with_median":
        result[numeric.columns] = numeric.mask(mask, numeric.median(), axis=1)
    elif action == "replace_with_cap_value":
        upper = numeric.quantile(0.99)
        lower = numeric.quantile(0.01)
        capped = numeric.clip(lower=lower, upper=upper, axis=1)
        result[numeric.columns] = numeric.where(~mask.fillna(False), capped)
    return result


def winsorize_clean(
    panel: pd.DataFrame,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Cap observations at user-supplied quantile thresholds (winsorization).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
    lower_quantile : float, default 0.01
        Lower clip quantile probability. Must satisfy
        ``0 <= lower_quantile < upper_quantile <= 1``.
    upper_quantile : float, default 0.99
        Upper clip quantile probability.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; each numeric column clipped to
        [q(lower_quantile), q(upper_quantile)].

    Notes
    -----
    Algorithm (bit-exact with ``_apply_outlier_policy`` for
    ``policy="winsorize"``):

    1. Compute per-column q(lower_quantile) and q(upper_quantile).
    2. ``panel.clip(lo_val, hi_val, axis=1)`` applies the cap.

    Unlike :func:`iqr_outlier_clean` and :func:`zscore_outlier_clean`,
    winsorize does **not** introduce new NaN values; it caps at the
    threshold value.

    Equivalent recipe configuration::

        l2:
          outlier_policy: winsorize
          leaf_config:
            winsorize_quantiles: [0.01, 0.99]

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 3), columns=list("abc"))
    >>> panel.iloc[0, 0] = 100.0  # extreme value
    >>> out = winsorize_clean(panel, lower_quantile=0.01, upper_quantile=0.99)
    >>> out.shape
    (50, 3)
    >>> out.iloc[0, 0] < 10.0  # capped
    True

    References
    ----------
    Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.
    """
    _require_non_empty(panel)
    if not (0 <= lower_quantile < upper_quantile <= 1):
        raise ValueError(
            f"Need 0 <= lower_quantile < upper_quantile <= 1; "
            f"got lower_quantile={lower_quantile!r}, upper_quantile={upper_quantile!r}"
        )

    result = panel.copy()
    numeric = result.select_dtypes("number")
    if numeric.empty:
        return result

    lo_val = numeric.quantile(lower_quantile)
    hi_val = numeric.quantile(upper_quantile)
    clipped = numeric.clip(lo_val, hi_val, axis=1)
    result[numeric.columns] = clipped
    return result


# ============================================================================
# L2.D Imputation ops
# ============================================================================


def em_factor_impute_clean(
    panel: pd.DataFrame,
    *,
    n_factors: int = 8,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Impute missing values using the McCracken-Ng PCA-EM algorithm (fixed rank).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    n_factors : int, default 8
        Rank for SVD truncation. McCracken-Ng production default is 8.
        Must be >= 1.
    max_iter : int, default 20
        Maximum number of EM iterations. Must be >= 1.
    tol : float, default 1e-4
        Convergence tolerance (relative Frobenius norm change). Must be > 0.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; missing cells filled by the PCA-EM
        reconstruction.

    Notes
    -----
    Algorithm (bit-exact with ``_pca_em_imputation(panel, n_factors=n_factors,
    max_iter=max_iter, tol=tol)`` in runtime):

    Calls ``_pca_em_imputation`` from ``macroforecast.core.runtime`` directly.
    The rank is clamped internally to ``min(n_factors, min(T,K) - 1)``.
    If the effective rank < 1, falls back to per-column mean imputation.

    Equivalent recipe configuration::

        l2:
          imputation_policy: em_factor
          leaf_config:
            em_n_factors: 8

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> panel.iloc[10:15, 1] = np.nan
    >>> out = em_factor_impute_clean(panel, n_factors=3)
    >>> out.shape
    (50, 5)
    >>> out.isna().sum().sum()
    0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    _require_non_empty(panel)
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1; got {n_factors!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter!r}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol!r}")

    from macroforecast.core.runtime import _pca_em_imputation  # noqa: PLC0415
    return _pca_em_imputation(panel, n_factors=n_factors, max_iter=max_iter, tol=tol)


def em_multivariate_impute_clean(
    panel: pd.DataFrame,
    *,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Impute missing values using the PCA-EM algorithm (uncapped rank).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
    max_iter : int, default 20
        Maximum number of EM iterations. Must be >= 1.
    tol : float, default 1e-4
        Convergence tolerance (relative Frobenius norm change). Must be > 0.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; missing cells filled by the PCA-EM
        reconstruction.

    Notes
    -----
    Algorithm (bit-exact with ``_pca_em_imputation(panel, n_factors=None,
    max_iter=max_iter, tol=tol)`` in runtime):

    Passes ``n_factors=None`` to ``_pca_em_imputation``, which internally
    uses ``rank = min(T, K) // 2``. More flexible than
    :func:`em_factor_impute_clean` (no hard rank cap) but more expensive
    on large panels.

    Equivalent recipe configuration::

        l2:
          imputation_policy: em_multivariate

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(30, 4), columns=list("abcd"))
    >>> panel.iloc[5:10, 2] = np.nan
    >>> out = em_multivariate_impute_clean(panel)
    >>> out.shape
    (30, 4)
    >>> out.isna().sum().sum()
    0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter!r}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol!r}")

    from macroforecast.core.runtime import _pca_em_imputation  # noqa: PLC0415
    return _pca_em_imputation(panel, n_factors=None, max_iter=max_iter, tol=tol)


def mean_impute_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace missing cells with the per-column full-sample mean.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; NaN cells replaced by the per-column mean.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_imputation`` for
    ``policy="mean"``):

    ``panel.fillna(panel.mean(numeric_only=True))``

    Non-numeric columns are passed through unchanged. If a column is
    entirely NaN the mean is NaN and those cells remain NaN.

    Equivalent recipe configuration::

        l2:
          imputation_policy: mean

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
    >>> mean_impute_clean(panel)
         a    b
    0  1.0  3.0
    1  2.0  2.0
    2  3.0  4.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    return panel.fillna(panel.mean(numeric_only=True))


def forward_fill_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Carry the last observed value forward to fill missing cells.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; interior and trailing NaN cells replaced
        by the most-recent observed value. Leading NaN values (no prior
        observation) remain as NaN.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_imputation`` for
    ``policy="forward_fill"``):

    ``panel.ffill()``

    Equivalent recipe configuration::

        l2:
          imputation_policy: forward_fill

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
    >>> forward_fill_clean(panel)
         a
    0  1.0
    1  1.0
    2  1.0
    3  4.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    return panel.ffill()


def linear_interpolate_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Impute missing cells by linear interpolation between adjacent observations.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; interior NaN cells replaced by linear
        interpolation. Leading and trailing NaN values remain since
        ``interpolate(method="linear")`` does not extrapolate beyond
        observed endpoints.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_imputation`` for
    ``policy="linear_interpolation"``):

    ``panel.interpolate(method="linear")``

    Equivalent recipe configuration::

        l2:
          imputation_policy: linear_interpolation

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
    >>> linear_interpolate_clean(panel)
         a
    0  1.0
    1  2.0
    2  3.0
    3  4.0
    4  5.0

    References
    ----------
    Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution,
    and Extrapolation of Time Series by Related Series', RES 53(4).
    """
    _require_non_empty(panel)
    return panel.interpolate(method="linear")


# ============================================================================
# L2.E Frame-edge ops
# ============================================================================


def truncate_to_balanced_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing any NaN to produce a balanced panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Subset of rows (T' <= T) with no missing values; same columns
        and index subset as input.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_frame_edge`` for
    ``policy="truncate_to_balanced"``):

    ``panel.dropna(axis=0, how="any")``

    Equivalent recipe configuration::

        l2:
          frame_edge_policy: truncate_to_balanced

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [np.nan, 2.0, 3.0], "b": [1.0, np.nan, 3.0]})
    >>> truncate_to_balanced_clean(panel)
         a    b
    2  3.0  3.0

    References
    ----------
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    _require_non_empty(panel)
    return panel.dropna(axis=0, how="any")


def drop_unbalanced_series_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that contain any NaN (retain only fully-observed series).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Same rows; subset of columns (K' <= K) that have no missing
        values. All rows are preserved.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_frame_edge`` for
    ``policy="drop_unbalanced_series"``):

    ``panel.dropna(axis=1, how="any")``

    Equivalent recipe configuration::

        l2:
          frame_edge_policy: drop_unbalanced_series

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 2.0]})
    >>> drop_unbalanced_series_clean(panel)
         a
    0  1.0
    1  2.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    return panel.dropna(axis=1, how="any")


def zero_fill_leading_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill ALL NaN cells with zero (name refers to the leading-edge use case).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; all NaN cells replaced by 0. Despite the
        "leading" naming, the runtime implementation fills every NaN with
        zero regardless of position.

    Notes
    -----
    Algorithm (bit-exact with ``_apply_frame_edge`` for
    ``policy="zero_fill_leading"``):

    ``panel.fillna(0)``

    Despite the name "zero_fill_leading", the runtime implementation
    fills ALL NaN cells with zero (not only leading NaN). The standalone
    callable matches this exact runtime behaviour.

    Equivalent recipe configuration::

        l2:
          frame_edge_policy: zero_fill_leading

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0]})
    >>> zero_fill_leading_clean(panel)
         a
    0  0.0
    1  1.0
    2  0.0
    3  3.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    return panel.fillna(0)


# ============================================================================
# L2.B Tcode op
# ============================================================================


def apply_tcode_transform(
    panel: pd.DataFrame,
    tcode_map: dict[str, int],
) -> pd.DataFrame:
    """Apply McCracken-Ng t-code stationarity transforms per column.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
    tcode_map : dict[str, int]
        Mapping from column name to integer t-code (1..7). Columns not
        present in *tcode_map* are passed through unchanged. Must be a
        non-empty dict with string keys and integer values in {1..7}.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; transformed columns replaced in-place.

    Notes
    -----
    T-code mapping (McCracken-Ng):

    * 1 = level (identity)
    * 2 = first difference ``y_t - y_{t-1}``
    * 3 = second difference ``(y_t - y_{t-1}) - (y_{t-1} - y_{t-2})``
    * 4 = log (safe: NaN for <= 0 values)
    * 5 = first difference of log (≈ growth rate)
    * 6 = second difference of log
    * 7 = percentage change ``y_t / y_{t-1} - 1``

    Algorithm (bit-exact with ``_apply_transform`` / ``_apply_tcode``
    in runtime):

    Calls ``_apply_tcode`` from ``macroforecast.core.runtime`` for each
    column present in both *panel* and *tcode_map*.

    Equivalent recipe configuration::

        l2:
          transform_policy: apply_official_tcode
          # or:
          transform_policy: custom_tcode
          leaf_config:
            custom_tcode_map: {col_name: tcode_int}

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [10.0, 20.0, 30.0, 40.0]})
    >>> apply_tcode_transform(panel, {"a": 2, "b": 5})
         a         b
    0  NaN       NaN
    1  1.0  0.693147
    2  2.0  0.405465
    3  4.0  0.287682

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    if not isinstance(tcode_map, dict) or not tcode_map:
        raise ValueError("tcode_map must be a non-empty dict")
    _VALID_TCODES = {1, 2, 3, 4, 5, 6, 7}
    for key, val in tcode_map.items():
        if not isinstance(key, str):
            raise ValueError(
                f"tcode_map keys must be strings; got key {key!r} of type {type(key).__name__!r}"
            )
        if int(val) not in _VALID_TCODES:
            raise ValueError(
                f"tcode_map values must be integers in {{1..7}}; got {val!r} for key {key!r}"
            )

    from macroforecast.core.runtime import _apply_tcode  # noqa: PLC0415

    result = panel.copy()
    for col, tcode in tcode_map.items():
        if col not in result.columns:
            continue
        result[col] = _apply_tcode(result[col], int(tcode))
    return result


# ============================================================================
# L2.A Frequency-alignment ops
# ============================================================================


def freq_align_quarterly_to_monthly_clean(
    panel: pd.DataFrame,
    quarterly_columns: list[str],
    *,
    rule: str = "step_backward",
) -> pd.DataFrame:
    """Align quarterly series to a monthly grid using a chosen interpolation rule.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel with ``pd.DatetimeIndex``. Each column is a variable;
        rows are monthly time periods.
    quarterly_columns : list[str]
        Column names to treat as quarterly series. Names not present in
        *panel* are silently skipped.
    rule : str, default "step_backward"
        Alignment rule. One of:

        * ``"step_backward"`` (default) -- hold each quarterly observation
          constant backward: ``.bfill().ffill()``. Conservative: no
          smoothing, no extrapolation.
        * ``"step_forward"`` -- hold each quarterly observation constant
          forward: ``.ffill()``.
        * ``"linear_interpolation"`` -- linearly interpolate between
          quarterly observations in both directions.

    Returns
    -------
    pd.DataFrame
        Same shape as *panel*; specified quarterly columns re-sampled to
        the monthly grid.

    Notes
    -----
    Algorithm (bit-exact with the Q-to-M branch of
    ``_apply_fred_sd_frequency_alignment``):

    For ``rule="step_backward"``: ``series.bfill().ffill()`` (NOT
    ``.ffill().bfill()``; order matters for leading / trailing NaN
    behaviour).

    Column names not in *quarterly_columns* are passed through unchanged.

    Equivalent recipe configuration::

        l2:
          quarterly_to_monthly_policy: step_backward
          # (within _apply_fred_sd_frequency_alignment)

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    >>> panel = pd.DataFrame({"q": [1.0, np.nan, np.nan, 2.0, np.nan, np.nan,
    ...                             3.0, np.nan, np.nan, 4.0, np.nan, np.nan]},
    ...                      index=idx)
    >>> out = freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="step_backward")
    >>> out["q"].tolist()
    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution,
    and Extrapolation of Time Series by Related Series', RES 53(4).
    """
    _require_non_empty(panel)
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError(
            "freq_align_quarterly_to_monthly_clean requires a DatetimeIndex; "
            f"got {type(panel.index).__name__!r}"
        )
    _VALID_RULES = {"step_backward", "step_forward", "linear_interpolation"}
    if rule not in _VALID_RULES:
        raise ValueError(
            f"rule must be one of {sorted(_VALID_RULES)}; got {rule!r}"
        )

    result = panel.copy()
    for col in quarterly_columns:
        if col not in result.columns:
            continue
        series = result[col]
        if rule == "linear_interpolation":
            result[col] = series.interpolate(method="linear", limit_direction="both")
        elif rule == "step_forward":
            result[col] = series.ffill()
        else:  # step_backward (default)
            # Critical: bfill first, then ffill — NOT the reverse
            result[col] = series.bfill().ffill()
    return result


def freq_align_monthly_to_quarterly_clean(
    panel: pd.DataFrame,
    monthly_columns: list[str],
    *,
    rule: str = "quarterly_average",
) -> pd.DataFrame:
    """Aggregate monthly columns to quarterly frequency.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel with ``pd.DatetimeIndex`` at (approximately) quarterly
        frequency. Monthly columns are aggregated then re-joined on the
        input index.
    monthly_columns : list[str]
        Column names to aggregate from monthly to quarterly. Names not
        present in *panel* are silently skipped.
    rule : str, default "quarterly_average"
        Aggregation rule. One of:

        * ``"quarterly_average"`` (default) -- resample to ``"QE"`` using
          ``.mean()``. Standard for stock variables.
        * ``"quarterly_endpoint"`` -- resample to ``"QE"`` using
          ``.last()``. For end-of-period series (balance sheets, M2
          month-end).
        * ``"quarterly_sum"`` -- resample to ``"QE"`` using ``.sum()``.
          For flow variables (production, sales).

    Returns
    -------
    pd.DataFrame
        DataFrame at the quarterly cadence of the input index; monthly
        columns replaced by their quarterly aggregates. Non-monthly columns
        are preserved unchanged.

    Notes
    -----
    Algorithm (bit-exact with the M-to-Q branch of
    ``_apply_fred_sd_frequency_alignment``):

    Resamples *monthly_columns* to quarterly frequency via
    ``resample("QE").{mean,last,sum}()``, then joins non-monthly columns
    via ``other_agg.join(agg)`` returning a quarterly-indexed frame
    directly (no secondary reindex step).

    Equivalent recipe configuration::

        l2:
          monthly_to_quarterly_policy: quarterly_average
          # (within _apply_fred_sd_frequency_alignment)

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    >>> panel = pd.DataFrame({"m": [1.0, 2.0, 3.0, 4.0], "other": [10, 20, 30, 40]},
    ...                      index=idx)
    >>> out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
    >>> out.shape
    (4, 2)

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589.
    """
    _require_non_empty(panel)
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError(
            "freq_align_monthly_to_quarterly_clean requires a DatetimeIndex; "
            f"got {type(panel.index).__name__!r}"
        )
    _VALID_RULES = {"quarterly_average", "quarterly_endpoint", "quarterly_sum"}
    if rule not in _VALID_RULES:
        raise ValueError(
            f"rule must be one of {sorted(_VALID_RULES)}; got {rule!r}"
        )

    result = panel.copy()
    monthly_cols_present = [c for c in monthly_columns if c in result.columns]
    if not monthly_cols_present:
        return result

    if rule == "quarterly_average":
        agg = result[monthly_cols_present].resample("QE").mean()
    elif rule == "quarterly_endpoint":
        agg = result[monthly_cols_present].resample("QE").last()
    else:  # quarterly_sum
        agg = result[monthly_cols_present].resample("QE").sum()

    # Aggregate all remaining columns to quarterly cadence as well, then join.
    # Returning the quarterly-indexed frame (not reindexing back to the monthly
    # input index) gives len(result) == len(panel) // 3 for a standard monthly input.
    other_cols = [c for c in result.columns if c not in monthly_cols_present]
    if other_cols:
        if rule == "quarterly_average":
            other_agg = result[other_cols].resample("QE").mean()
        elif rule == "quarterly_endpoint":
            other_agg = result[other_cols].resample("QE").last()
        else:  # quarterly_sum
            other_agg = result[other_cols].resample("QE").sum()
        result_out = other_agg.join(agg)
    else:
        result_out = agg
    return result_out


__all__ = [
    "iqr_outlier_clean",
    "zscore_outlier_clean",
    "winsorize_clean",
    "em_factor_impute_clean",
    "em_multivariate_impute_clean",
    "mean_impute_clean",
    "forward_fill_clean",
    "linear_interpolate_clean",
    "truncate_to_balanced_clean",
    "drop_unbalanced_series_clean",
    "zero_fill_leading_clean",
    "apply_tcode_transform",
    "freq_align_quarterly_to_monthly_clean",
    "freq_align_monthly_to_quarterly_clean",
]
