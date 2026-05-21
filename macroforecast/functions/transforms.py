"""Standalone L3 panel-transform functions.

Cycle 30: L3 basic panel transforms standalone-ization (10 ops).

Each callable wraps the corresponding runtime primitive from
``macroforecast.core.runtime`` to preserve bit-exact results with
the recipe-path dispatch.  Import pattern follows C28/C29 (linear.py,
tests.py): runtime helpers are imported lazily inside each function
body to avoid circular imports and keep the module self-contained at
definition time.

Basic stationary / lag / aggregation / scale ops:
    diff_transform, log_transform, log_diff_transform,
    pct_change_transform, cumsum_transform, ma_window_transform,
    lag_matrix, seasonal_lag_matrix, ma_increasing_order_transform,
    scale_transform
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


# ---------------------------------------------------------------------------
# 1. diff_transform
# ---------------------------------------------------------------------------

def diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Compute a simple finite difference along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        DataFrame or Series; Series is promoted to a single-column
        DataFrame internally.
    periods : int, default 1
        Number of lag periods to difference.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Differenced panel of the same shape.  The first ``periods`` rows
        contain ``NaN``.

    Notes
    -----
    Calls ``_as_frame`` followed by ``_diff_like`` from
    ``macroforecast.core.runtime``.  Equivalent recipe configuration::

        op: diff
        params:
          n_diff: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(10, 2), columns=["a", "b"])
    >>> diff_transform(panel).shape
    (10, 2)
    >>> diff_transform(panel).iloc[0].isna().all()
    True

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589. Transformation code 2 (first
    difference).
    """
    from macroforecast.core.runtime import _as_frame, _diff_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("diff_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _diff_like(frame, periods=periods)


# ---------------------------------------------------------------------------
# 2. log_transform
# ---------------------------------------------------------------------------

def log_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Element-wise natural logarithm of a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  All values must be strictly positive.  NaN values
        are preserved.

    Returns
    -------
    pd.DataFrame
        Log-transformed panel with the same shape and index/columns.

    Notes
    -----
    Wraps ``_as_frame`` from ``macroforecast.core.runtime`` then applies
    ``np.log`` directly (via pandas ``applymap``-equivalent path).
    Values <= 0 are not silently coerced; callers must ensure positivity.
    The recipe-path uses a cell-by-cell guard (``pd.NA`` on <= 0 cells)
    whereas this standalone uses ``np.log`` directly to preserve NaN
    propagation. Equivalent recipe configuration::

        op: log

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 4.0, 8.0]})
    >>> log_transform(panel)
         a         b
    0  0.0  0.693147
    1  0.693147  1.386294
    2  1.386294  2.079442

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 4
    (log level).
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return np.log(frame)


# ---------------------------------------------------------------------------
# 3. log_diff_transform
# ---------------------------------------------------------------------------

def log_diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Log then first-difference: ``ln(y_t) - ln(y_{t-periods})``.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  All values must be strictly positive.
    periods : int, default 1
        Number of lag periods to difference.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Log-differenced panel. The first ``periods`` rows contain NaN.

    Notes
    -----
    Applies ``np.log`` after ``_as_frame``, then calls ``_diff_like``
    from ``macroforecast.core.runtime``.  Values <= 0 are not silently
    coerced; callers must ensure positivity.  The recipe-path uses a
    cell-by-cell guard (``pd.NA`` on <= 0 cells) whereas this standalone
    uses ``np.log`` directly to preserve NaN propagation.  Equivalent recipe
    configuration::

        op: log_diff
        params:
          n_diff: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, np.e, np.e**2]})
    >>> log_diff_transform(panel)
         a
    0  NaN
    1  1.0
    2  1.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 5
    (log first-difference -- monthly growth rate approximation).
    """
    from macroforecast.core.runtime import _as_frame, _diff_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("log_diff_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    logged = np.log(frame)
    return _diff_like(logged, periods=periods)


# ---------------------------------------------------------------------------
# 4. pct_change_transform
# ---------------------------------------------------------------------------

def pct_change_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Percentage change along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    periods : int, default 1
        Number of lag periods for the percentage change.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Percentage-change panel: ``(y_t - y_{t-periods}) / |y_{t-periods}|``.
        The first ``periods`` rows contain NaN.

    Notes
    -----
    Calls ``_pct_change_like`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: pct_change
        params:
          n_periods: 1

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [100.0, 110.0, 121.0]})
    >>> pct_change_transform(panel)
              a
    0       NaN
    1  0.100000
    2  0.100000

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 3
    (percent change).
    """
    from macroforecast.core.runtime import _as_frame, _pct_change_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("pct_change_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _pct_change_like(frame, periods=periods)


# ---------------------------------------------------------------------------
# 5. cumsum_transform
# ---------------------------------------------------------------------------

def cumsum_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative sum along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.

    Returns
    -------
    pd.DataFrame
        Cumulative-sum panel of the same shape.  NaNs in the input are
        treated as 0 by pandas ``cumsum`` (NaN propagation is disabled).

    Notes
    -----
    Calls ``_as_frame(panel).cumsum()`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: cumsum

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    >>> cumsum_transform(panel)
         a
    0  1.0
    1  3.0
    2  6.0

    References
    ----------
    macroforecast design Part 2, L3: feature engineering DAG step library.
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return frame.cumsum()


# ---------------------------------------------------------------------------
# 6. ma_window_transform
# ---------------------------------------------------------------------------

def ma_window_transform(panel: pd.DataFrame, *, window: int = 3) -> pd.DataFrame:
    """Centred rolling moving average with fixed window width.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    window : int, default 3
        Rolling window size in periods.  Must be >= 1.  The first
        ``window - 1`` rows will contain NaN (min_periods = window).

    Returns
    -------
    pd.DataFrame
        Rolling-mean panel of the same shape.

    Notes
    -----
    Equivalent to ``_as_frame(panel).rolling(window, min_periods=window).mean()``,
    matching the runtime dispatch for ``op: ma_window``.  Equivalent
    recipe configuration::

        op: ma_window
        params:
          window: 3

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> ma_window_transform(panel, window=3)
              a
    0       NaN
    1       NaN
    2  2.000000
    3  3.000000
    4  4.000000

    References
    ----------
    macroforecast design Part 2, L3: step library, ``ma_window`` op.
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    if window < 1:
        raise ValueError("ma_window_transform requires window >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return frame.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# 7. lag_matrix
# ---------------------------------------------------------------------------

def lag_matrix(
    panel: pd.DataFrame,
    *,
    n_lag: int = 4,
    include_contemporaneous: bool = False,
) -> pd.DataFrame:
    """Build a wide lag matrix from a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  Each column is lagged ``n_lag`` times.
    n_lag : int, default 4
        Number of lags.  Must be >= 1.
    include_contemporaneous : bool, default False
        If ``True``, also include lag 0 (the contemporaneous column),
        suffixed ``_lag0``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_lag1``, ``_lag2``, ...,
        ``_lag{n_lag}``.  If ``include_contemporaneous=True``, also
        includes ``_lag0``.  Shape: ``(T, K * n_lags)`` where K is the
        number of input columns.

    Notes
    -----
    Calls ``_lagged_predictors`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: lag
        params:
          n_lag: 4
          include_contemporaneous: false

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": range(6), "b": range(6, 12)})
    >>> lag_matrix(panel, n_lag=2).columns.tolist()
    ['a_lag1', 'a_lag2', 'b_lag1', 'b_lag2']

    References
    ----------
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    from macroforecast.core.runtime import _as_frame, _lagged_predictors  # noqa: PLC0415

    if n_lag < 1:
        raise ValueError("lag_matrix requires n_lag >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _lagged_predictors(frame, n_lag, include_contemporaneous=include_contemporaneous)


# ---------------------------------------------------------------------------
# 8. seasonal_lag_matrix
# ---------------------------------------------------------------------------

def seasonal_lag_matrix(
    panel: pd.DataFrame,
    *,
    seasonal_period: int = 12,
    n_seasonal_lags: int = 1,
) -> pd.DataFrame:
    """Build a seasonal lag matrix from a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  Each column is seasonally lagged.
    seasonal_period : int, default 12
        Seasonal cycle length (e.g. 12 for monthly data, 4 for quarterly).
        Must be >= 2.
    n_seasonal_lags : int, default 1
        Number of seasonal lags to include.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_s{seasonal_period}_lag{i}``
        for ``i`` in ``1, ..., n_seasonal_lags``.  Each lag shifts by
        ``seasonal_period * i`` periods.

    Notes
    -----
    Calls ``_seasonal_lagged_predictors`` from
    ``macroforecast.core.runtime``.  Equivalent recipe configuration::

        op: seasonal_lag
        params:
          seasonal_period: 12
          n_seasonal_lags: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame({"a": rng.randn(24)})
    >>> seasonal_lag_matrix(panel, seasonal_period=12, n_seasonal_lags=1).shape
    (24, 1)

    References
    ----------
    Hylleberg, Engle, Granger & Yoo (1990) 'Seasonal Integration and
    Cointegration', Journal of Econometrics 44(1-2): 215-238.
    """
    from macroforecast.core.runtime import _as_frame, _seasonal_lagged_predictors  # noqa: PLC0415

    if seasonal_period < 2:
        raise ValueError("seasonal_lag_matrix requires seasonal_period >= 2")
    if n_seasonal_lags < 1:
        raise ValueError("seasonal_lag_matrix requires n_seasonal_lags >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _seasonal_lagged_predictors(
        frame,
        seasonal_period=seasonal_period,
        n_seasonal_lags=n_seasonal_lags,
    )


# ---------------------------------------------------------------------------
# 9. ma_increasing_order_transform
# ---------------------------------------------------------------------------

def ma_increasing_order_transform(
    panel: pd.DataFrame,
    *,
    max_order: int = 12,
) -> pd.DataFrame:
    """Compute moving averages of all orders from 2 to ``max_order``.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    max_order : int, default 12
        Maximum window order.  Must be >= 2.  Generates windows
        2, 3, ..., max_order.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_ma{order}`` for each
        order from 2 to ``max_order``.  Shape:
        ``(T, K * (max_order - 1))`` where K is the input column count.

    Notes
    -----
    Calls ``_ma_increasing_order`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: ma_increasing_order
        params:
          max_order: 12

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": range(10)})
    >>> ma_increasing_order_transform(panel, max_order=3).columns.tolist()
    ['a_ma2', 'a_ma3']

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data
    Transformations Matter', International Journal of Forecasting 37(4):
    1338-1354.
    """
    from macroforecast.core.runtime import _as_frame, _ma_increasing_order  # noqa: PLC0415

    if max_order < 2:
        raise ValueError("ma_increasing_order_transform requires max_order >= 2")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _ma_increasing_order(frame, max_order=max_order)


# ---------------------------------------------------------------------------
# 10. scale_transform
# ---------------------------------------------------------------------------

def scale_transform(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
) -> pd.DataFrame:
    """Standardise a panel column-by-column using a named scale method.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    method : str, default "zscore"
        Scaling method.  One of:

        * ``"zscore"`` / ``"standard"`` / ``"standardize"`` --
          ``(x - mean) / std`` (population std, ddof=0).
        * ``"robust"`` -- ``(x - median) / IQR`` where IQR is the
          75th minus 25th percentile gap.
        * ``"minmax"`` -- ``(x - min) / (max - min)``.

    Returns
    -------
    pd.DataFrame
        Scaled panel of the same shape.  Columns with zero spread are
        divided by ``pd.NA`` (result: all-NaN column).

    Notes
    -----
    Calls ``_scale_frame`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: scale
        params:
          method: zscore

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
    >>> scale_transform(panel)["a"].mean()  # doctest: +ELLIPSIS
    0.0

    References
    ----------
    macroforecast design Part 2, L3: step library, ``scale`` op.
    Matches sklearn ``StandardScaler`` (zscore), ``RobustScaler``
    (robust), and ``MinMaxScaler`` (minmax) column-by-column behaviour.
    """
    from macroforecast.core.runtime import _as_frame, _scale_frame  # noqa: PLC0415

    _VALID_METHODS = {"zscore", "standard", "standardize", "robust", "minmax"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method: {method!r}. Expected zscore/robust/minmax/winsorize/quantile."
        )
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _scale_frame(frame, method=method)


# ---------------------------------------------------------------------------
# 11. hp_filter_transform
# ---------------------------------------------------------------------------

def hp_filter_transform(
    panel: pd.DataFrame,
    *,
    lambda_: float = 1600,
) -> pd.DataFrame:
    """Extract the cyclical component of a panel using the Hodrick-Prescott filter.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    lambda_ : float, default 1600
        HP smoothing parameter (``lambda``/``lam`` in statsmodels).
        Convention: 1600 for quarterly, 129600 for monthly (Ravn-Uhlig 2002).
        Must be > 0.

    Returns
    -------
    pd.DataFrame
        Cyclical component panel; columns suffixed ``_hp_cycle``.
        Trend component is discarded.  Rows with fewer than 4 non-NaN
        observations in a column become NaN in the output.

    Notes
    -----
    Calls ``_hp_filter`` from ``macroforecast.core.runtime`` with
    ``lam=lambda_``.  Runtime delegates to
    ``statsmodels.tsa.filters.hp_filter.hpfilter``.  Note that the
    standalone uses the parameter name ``lambda_`` (trailing underscore
    avoids shadowing the Python built-in) while the runtime expects
    ``lam``.  Equivalent recipe configuration::

        op: hp_filter
        params:
          lamb: 1600

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 2), columns=["a", "b"])
    >>> out = hp_filter_transform(panel)
    >>> "a_hp_cycle" in out.columns
    True

    References
    ----------
    Hodrick & Prescott (1997) 'Postwar U.S. Business Cycles: An Empirical
    Investigation', Journal of Money, Credit and Banking 29(1): 1-16.

    Ravn & Uhlig (2002) 'On Adjusting the Hodrick-Prescott Filter for the
    Frequency of Observations', Review of Economics and Statistics 84(2):
    371-376.
    """
    from macroforecast.core.runtime import _as_frame, _hp_filter  # noqa: PLC0415

    if lambda_ <= 0:
        raise ValueError("hp_filter_transform requires lambda_ > 0")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _hp_filter(frame, lam=lambda_)


# ---------------------------------------------------------------------------
# 12. hamilton_filter_transform
# ---------------------------------------------------------------------------

def hamilton_filter_transform(
    panel: pd.DataFrame,
    *,
    h: int = 8,
    p: int = 4,
) -> pd.DataFrame:
    """Extract the cyclical component of a panel using the Hamilton (2018) filter.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    h : int, default 8
        Forecast horizon (number of periods ahead).  Hamilton (2018) uses
        h=8 for quarterly data (2 years) and h=24 for monthly data.
        Must be >= 1.
    p : int, default 4
        Number of lags used in the regression.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Cyclical component panel; columns suffixed ``_hamilton_cycle``.
        The leading ``h + p`` rows will contain NaN (lag-panel boundary).

    Notes
    -----
    Calls ``_hamilton_filter`` from ``macroforecast.core.runtime`` with
    ``n_horizon=h``, ``n_lags=p``.  Algorithm: regress ``y_{t+h}`` on
    ``[y_t, y_{t-1}, ..., y_{t-p+1}]``; the residuals are the cycle.
    Equivalent recipe configuration::

        op: hamilton_filter
        params:
          h: 8
          p: 4

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(60, 2), columns=["a", "b"])
    >>> out = hamilton_filter_transform(panel)
    >>> "a_hamilton_cycle" in out.columns
    True

    References
    ----------
    Hamilton (2018) 'Why You Should Never Use the Hodrick-Prescott Filter',
    Review of Economics and Statistics 100(5): 831-843.
    """
    from macroforecast.core.runtime import _as_frame, _hamilton_filter  # noqa: PLC0415

    if h < 1:
        raise ValueError("hamilton_filter_transform requires h >= 1")
    if p < 1:
        raise ValueError("hamilton_filter_transform requires p >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _hamilton_filter(frame, n_horizon=h, n_lags=p)


# ---------------------------------------------------------------------------
# 13. savitzky_golay_transform
# ---------------------------------------------------------------------------

def savitzky_golay_transform(
    panel: pd.DataFrame,
    *,
    window: int = 7,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Smooth a panel column-by-column with the Savitzky-Golay polynomial filter.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    window : int, default 7
        Length of the smoothing window.  Must be >= 3.  If even, the
        runtime rounds up to the next odd integer (scipy requirement).
    polyorder : int, default 3
        Degree of the polynomial used to fit within each window.
        Must be < ``window``.

    Returns
    -------
    pd.DataFrame
        Smoothed panel; columns suffixed ``_savgol``.  Shape is the same
        as the input (no rows dropped).

    Notes
    -----
    Calls ``_savitzky_golay_filter`` from ``macroforecast.core.runtime``
    with ``window_length=window``.  Runtime delegates to
    ``scipy.signal.savgol_filter``; scipy is a required dependency.
    Note: the standalone uses ``window`` while the runtime parameter is
    ``window_length`` -- the wrapper maps accordingly.  Equivalent recipe
    configuration::

        op: savitzky_golay_filter
        params:
          window_length: 7
          polyorder: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(30, 2), columns=["a", "b"])
    >>> out = savitzky_golay_transform(panel)
    >>> "a_savgol" in out.columns
    True

    References
    ----------
    Savitzky & Golay (1964) 'Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures', Analytical Chemistry 36(8).
    """
    from macroforecast.core.runtime import _as_frame, _savitzky_golay_filter  # noqa: PLC0415

    if window < 3:
        raise ValueError("savitzky_golay_transform requires window >= 3")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _savitzky_golay_filter(frame, window_length=window, polyorder=polyorder)


# ---------------------------------------------------------------------------
# 14. polynomial_expansion_transform
# ---------------------------------------------------------------------------

def polynomial_expansion_transform(
    panel: pd.DataFrame,
    *,
    degree: int = 2,
) -> pd.DataFrame:
    """Expand a panel with polynomial powers up to the specified degree.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    degree : int, default 2
        Maximum polynomial degree.  Must be >= 1.  Degree 1 returns the
        panel unchanged; degree 2 appends ``col_pow2`` columns; degree 3
        also appends ``col_pow3``; and so on.

    Returns
    -------
    pd.DataFrame
        Expanded panel.  Shape: ``(T, K * degree)``.  Original columns are
        preserved; new columns are suffixed ``_pow{k}`` for k = 2..degree.

    Notes
    -----
    Calls ``_polynomial_expansion`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: polynomial_expansion
        params:
          degree: 2

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    >>> out = polynomial_expansion_transform(panel, degree=3)
    >>> out.columns.tolist()
    ['a', 'b', 'a_pow2', 'b_pow2', 'a_pow3', 'b_pow3']

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data
    Transformations Matter', International Journal of Forecasting 37(4):
    1338-1354.
    """
    from macroforecast.core.runtime import _as_frame, _polynomial_expansion  # noqa: PLC0415

    if degree < 1:
        raise ValueError("polynomial_expansion_transform requires degree >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _polynomial_expansion(frame, degree=degree)


# ---------------------------------------------------------------------------
# 15. interaction_terms_transform
# ---------------------------------------------------------------------------

def interaction_terms_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute all pairwise interaction terms for a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Requires at least 2 columns to produce any output.

    Returns
    -------
    pd.DataFrame
        Interaction-terms panel.  Contains one column ``{left}_x_{right}``
        for each unique unordered pair (left, right) of input columns.
        Shape: ``(T, K*(K-1)/2)`` where K is the input column count.
        Returns an empty DataFrame (zero columns) when K < 2.

    Notes
    -----
    Calls ``_interaction_terms`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: interaction

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    >>> out = interaction_terms_transform(panel)
    >>> out.columns.tolist()
    ['a_x_b', 'a_x_c', 'b_x_c']

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data
    Transformations Matter', International Journal of Forecasting 37(4):
    1338-1354.
    """
    from macroforecast.core.runtime import _as_frame, _interaction_terms  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _interaction_terms(frame)


# ---------------------------------------------------------------------------
# 16. pca_transform
# ---------------------------------------------------------------------------

def pca_transform(
    panel: pd.DataFrame,
    *,
    n_components: "int | str" = 3,
) -> pd.DataFrame:
    """Extract principal components from a panel (standard PCA, Stock-Watson 2002).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_components : int or 'all', default 3
        Number of principal components to extract.  Clamped internally to
        ``min(T, K) - 1`` where T is the number of clean rows and K is
        the column count.  Must be >= 1.  Sentinel ``'all'`` extracts the
        full effective rank ``min(T_clean, K)`` (no safety margin).

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``factor_1``, ``factor_2``,
        ..., ``factor_{n_components}``.  Rows dropped for NaN are filled
        back with NaN on output.

    Notes
    -----
    Calls ``_pca_factors`` from ``macroforecast.core.runtime`` with
    ``variant="pca"``.  Runtime uses sklearn ``PCA(random_state=0)``
    on the centred data matrix.  Equivalent recipe configuration::

        op: pca
        params:
          n_components: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = pca_transform(panel, n_components=2)
    >>> out.shape
    (50, 2)
    >>> out.columns.tolist()
    ['factor_1', 'factor_2']

    References
    ----------
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    from macroforecast.core.runtime import _as_frame, _pca_factors  # noqa: PLC0415

    if isinstance(n_components, str):
        if n_components != "all":
            raise ValueError(
                f"pca_transform: n_components must be a positive int or 'all'; got {n_components!r}"
            )
        # 'all' sentinel: pass through to runtime which resolves to min(T_clean, K)
    else:
        if n_components < 1:
            raise ValueError("pca_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _pca_factors(frame, n_components=n_components, variant="pca")


# ---------------------------------------------------------------------------
# 17. maf_per_variable_pca_transform
# ---------------------------------------------------------------------------

def maf_per_variable_pca_transform(
    panel: pd.DataFrame,
    *,
    n_lags: int = 12,
    n_components_per_var: int = 2,
) -> pd.DataFrame:
    """Per-variable PCA MAF -- Coulombe et al. (2021 IJF) Eq. (7).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    n_lags : int, default 12
        Number of lags to include in the lag-panel per variable.  Paper
        default is 12 (monthly data).  Must be >= 1.
    n_components_per_var : int, default 2
        Number of PCA components per variable.  Paper default is 2
        (footnote 11: 'We keep two MAFs for each series and they are
        obtained by PCA.').  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Factor panel; shape ``(T, K * n_components_per_var)``.  Columns
        named ``{col}_maf1``, ``{col}_maf2``, ..., for each input column.
        The leading ``n_lags`` rows (default 12) per variable are NaN.

    Notes
    -----
    Calls ``_maf_per_variable_pca`` from ``macroforecast.core.runtime``
    forwarding ``n_lags`` and ``n_components_per_var``.  For each
    variable k, a ``(T, n_lags+1)`` lag-panel is built, NaN rows
    dropped, then PCA is fit and projected back to the full T-length
    index.  Equivalent recipe configuration::

        op: maf_per_variable_pca
        params:
          n_components_per_var: 2

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
    >>> out = maf_per_variable_pca_transform(panel)
    >>> out.shape[1]
    6

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data
    Transformations Matter', International Journal of Forecasting 37(4):
    1338-1354.
    """
    from macroforecast.core.runtime import _as_frame, _maf_per_variable_pca  # noqa: PLC0415

    if n_lags < 1:
        raise ValueError("maf_per_variable_pca_transform requires n_lags >= 1")
    if n_components_per_var < 1:
        raise ValueError("maf_per_variable_pca_transform requires n_components_per_var >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _maf_per_variable_pca(frame, n_lags=n_lags, n_components_per_var=n_components_per_var)


# ---------------------------------------------------------------------------
# 18. adaptive_ma_rf_transform
# ---------------------------------------------------------------------------

def adaptive_ma_rf_transform(
    panel: pd.DataFrame,
    *,
    n_estimators: int = 100,
    min_samples_leaf: int = 40,
    sided: str = "two",
    random_state: "int | None" = 0,
) -> pd.DataFrame:
    """Adaptive Moving Average via Random Forest -- AlbaMA smoother.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    n_estimators : int, default 100
        Number of trees in the forest.  Paper recommends 500 for final
        results; 100 is the default here for speed.  Must be >= 1.
    min_samples_leaf : int, default 40
        Minimum samples per leaf (lower-bounds the effective window
        length).  Paper default: 40.  Must be >= 1.
    sided : str, default "two"
        ``"two"`` fits the forest once on the full sample (retrospective
        smoother).  ``"one"`` fits an expanding-window forest per time
        index t (real-time / nowcasting variant; O(T) RF fits per column).
        Must be ``"two"`` or ``"one"``.
    random_state : int or None, default 0
        RNG seed for sklearn ``RandomForestRegressor``.  Pass ``None``
        for non-reproducible results.

    Returns
    -------
    pd.DataFrame
        Smoothed panel; columns suffixed ``_albama``.  Shape matches input.

    Notes
    -----
    Calls ``_adaptive_ma_rf`` from ``macroforecast.core.runtime``
    forwarding all four parameters.  The sole regressor is the time
    index; CART splitting learns the adaptive bandwidth per observation.
    Performance note: ``sided='one'`` is O(T) RF fits per column and
    is slow for large T.  Equivalent recipe configuration::

        op: adaptive_ma_rf
        params:
          n_estimators: 100

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 2), columns=["a", "b"])
    >>> out = adaptive_ma_rf_transform(panel)
    >>> "a_albama" in out.columns
    True

    References
    ----------
    Goulet Coulombe & Klieber (2025) 'An Adaptive Moving Average for
    Macroeconomic Monitoring', arXiv:2501.13222.
    """
    from macroforecast.core.runtime import _as_frame, _adaptive_ma_rf  # noqa: PLC0415

    if n_estimators < 1:
        raise ValueError("adaptive_ma_rf_transform requires n_estimators >= 1")
    if min_samples_leaf < 1:
        raise ValueError("adaptive_ma_rf_transform requires min_samples_leaf >= 1")
    if sided not in {"two", "one"}:
        raise ValueError(
            f"adaptive_ma_rf_transform: sided must be 'two' or 'one'; got {sided!r}"
        )
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _adaptive_ma_rf(
        frame,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        sided=sided,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 19. wavelet_transform
# ---------------------------------------------------------------------------

def wavelet_transform(
    panel: pd.DataFrame,
    *,
    wavelet: str = "db4",
    n_levels: int = 3,
) -> pd.DataFrame:
    """Multi-resolution wavelet decomposition of a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    wavelet : str, default "db4"
        Wavelet family name (e.g., ``"db4"`` for Daubechies-4, ``"haar"``).
        This parameter is accepted for API consistency with the recipe
        interface; the runtime uses a rolling-mean approximation that is
        family-independent (see Notes).
    n_levels : int, default 3
        Number of decomposition levels.  Must be >= 1.  Each level
        produces an approximation (``_wA{level}``) and detail
        (``_wD{level}``) pair.

    Returns
    -------
    pd.DataFrame
        Multi-resolution features panel.  Columns: ``{col}_wA{level}``
        (approximation) and ``{col}_wD{level}`` (detail) for each input
        column and each level from 1 to ``n_levels``.
        Shape: ``(T, 2 * K * n_levels)``.

    Notes
    -----
    Calls ``_wavelet_decomposition`` from ``macroforecast.core.runtime``
    with ``n_levels=n_levels``.  The runtime uses a rolling-mean
    low-pass approximation (window = 2^level) rather than a true DWT;
    the ``wavelet`` parameter is stored for recipe compatibility but does
    not affect computation.  Equivalent recipe configuration::

        op: wavelet
        params:
          wavelet: db4
          n_levels: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(30, 2), columns=["a", "b"])
    >>> out = wavelet_transform(panel, n_levels=2)
    >>> out.shape
    (30, 8)

    References
    ----------
    Mallat (1989) 'A Theory for Multiresolution Signal Decomposition',
    IEEE Transactions on Pattern Analysis and Machine Intelligence 11(7):
    674-693.
    """
    from macroforecast.core.runtime import _as_frame, _wavelet_decomposition  # noqa: PLC0415

    if n_levels < 1:
        raise ValueError("wavelet_transform requires n_levels >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _wavelet_decomposition(frame, n_levels=n_levels)


# ---------------------------------------------------------------------------
# 20. fourier_transform
# ---------------------------------------------------------------------------

def fourier_transform(
    panel: pd.DataFrame,
    *,
    n_terms: int = 4,
    period: int = 12,
) -> pd.DataFrame:
    """Generate Fourier basis (sin/cos) features for a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        The index is used to compute phase positions (positional for
        non-DatetimeIndex; day-offset for DatetimeIndex).
    n_terms : int, default 4
        Number of harmonic pairs (sin + cos) to generate.  Must be >= 1.
        Total output columns: ``2 * n_terms``.
    period : int, default 12
        Fundamental period of the seasonal pattern (e.g., 12 for monthly
        data with annual cycle, 4 for quarterly).  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Fourier feature panel independent of input columns (same set of
        features regardless of the number of input variables); columns
        named ``fourier_sin_{k}`` and ``fourier_cos_{k}`` for
        k = 1..n_terms.  Shape: ``(T, 2 * n_terms)``.

    Notes
    -----
    Calls ``_fourier_features`` from ``macroforecast.core.runtime``.
    The output does not depend on the *values* of the input panel --
    only the index and shape are used.  Equivalent recipe configuration::

        op: fourier
        params:
          n_terms: 4
          period: 12

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": np.ones(12)})
    >>> out = fourier_transform(panel, n_terms=2, period=12)
    >>> out.shape
    (12, 4)
    >>> list(out.columns)
    ['fourier_sin_1', 'fourier_cos_1', 'fourier_sin_2', 'fourier_cos_2']

    References
    ----------
    Harvey & Shephard (1993) 'Structural Time Series Models', in
    Handbook of Statistics 11, Elsevier, 261-302.
    """
    from macroforecast.core.runtime import _as_frame, _fourier_features  # noqa: PLC0415

    if n_terms < 1:
        raise ValueError("fourier_transform requires n_terms >= 1")
    if period < 1:
        raise ValueError("fourier_transform requires period >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _fourier_features(frame, n_terms=n_terms, period=period)


# ---------------------------------------------------------------------------
# 21. asymmetric_trim_transform
# ---------------------------------------------------------------------------

def asymmetric_trim_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-period rank-space transformation (Albacore-family asymmetric trim).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel ``(T, K)`` of contemporaneous component series.
        Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Requires at least 2 columns for the rank-sort to be meaningful.

    Returns
    -------
    pd.DataFrame
        Rank-sorted panel of the same shape ``(T, K)``.  Column
        ``rank_{r+1}`` (for r = 0..K-1) contains the r-th ascending
        order statistic at each period.  Downstream nonneg-ridge learns
        rank-position weights that yield asymmetric trimming.

    Notes
    -----
    Calls ``_asymmetric_trim`` from ``macroforecast.core.runtime`` with
    default ``smooth_window=0`` (no centred MA post-processing).  To
    apply a smoothing window, chain with ``ma_window_transform``.
    Equivalent recipe configuration::

        op: asymmetric_trim

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [3.0, 1.0], "b": [1.0, 3.0], "c": [2.0, 2.0]})
    >>> out = asymmetric_trim_transform(panel)
    >>> out.iloc[0].tolist()
    [1.0, 2.0, 3.0]

    References
    ----------
    Goulet Coulombe, Klieber, Barrette & Goebel (2024) 'Maximally
    Forward-Looking Core Inflation', technical report (R package:
    assemblage).
    """
    from macroforecast.core.runtime import _as_frame, _asymmetric_trim  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _asymmetric_trim(frame)


# ---------------------------------------------------------------------------
# 22. season_dummy_transform
# ---------------------------------------------------------------------------

def season_dummy_transform(
    panel: pd.DataFrame,
    *,
    season: str = "quarter",
) -> pd.DataFrame:
    """Generate calendar dummy variables from the panel index.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        The index is used to extract the seasonal period; column values
        are not used.
    season : str, default "quarter"
        Seasonal granularity hint.  Accepted values: ``"quarter"`` and
        ``"month"``.  Currently validated but has no effect on output
        (deprecated -- kept for API compatibility).  The runtime output
        is driven solely by the index type: ``DatetimeIndex`` inputs
        produce ``month_*`` columns; all other index types produce
        ``season_*`` columns.

    Returns
    -------
    pd.DataFrame
        Dummy-variable panel.  For ``DatetimeIndex`` inputs, columns are
        ``month_1`` through ``month_12`` (one-hot).  For all other index
        types, columns are ``season_1`` through ``season_12``.

    Notes
    -----
    Delegates unconditionally to recipe ``_season_dummy``
    (``macroforecast.core.runtime``).  The ``season`` kwarg is validated
    but has no effect on output -- it is retained for API compatibility
    only.  Non-DatetimeIndex inputs produce ``season_*`` columns;
    DatetimeIndex inputs produce ``month_*`` columns.
    Equivalent recipe configuration::

        op: season_dummy

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.period_range("2000-01", periods=12, freq="M")
    >>> panel = pd.DataFrame({"a": np.ones(12)}, index=idx)
    >>> out = season_dummy_transform(panel, season="month")
    >>> out.shape
    (12, 12)

    References
    ----------
    macroforecast design Part 2, L3: step library, ``season_dummy`` op.
    """
    from macroforecast.core.runtime import _as_frame, _season_dummy  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)

    _VALID_SEASONS = {"quarter", "month"}
    if season not in _VALID_SEASONS:
        raise ValueError(
            f"Unknown season: {season!r}. Expected one of {sorted(_VALID_SEASONS)}."
        )

    # Delegate to _season_dummy for all paths.
    # DatetimeIndex -> month_1..month_12; non-DatetimeIndex -> season_1..season_12.
    # The season kwarg is validated above but does not alter runtime output;
    # _season_dummy is index-type-driven.  Spec §3.12.3 requires 'season_*'
    # or 'month_*' prefixes only -- no 'qtr_*' prefix is produced.
    return _season_dummy(frame)


# ---------------------------------------------------------------------------
# 23. scaled_pca_transform
# ---------------------------------------------------------------------------

def scaled_pca_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    n_components: int = 3,
) -> pd.DataFrame:
    """Huang/Jiang/Li/Tong/Zhou (2022) Scaled PCA -- target-supervised factor extraction.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    target : pd.Series
        Supervisory signal used to compute per-column OLS slopes beta_j.
        Must share at least one index value with ``panel``; raises
        ``ValueError`` if the intersection is empty.
    n_components : int, default 3
        Number of principal components to extract. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``factor_1``, ``factor_2``,
        ..., ``factor_{n_components}``. Rows dropped for NaN are filled
        back with NaN on output.

    Notes
    -----
    Calls ``_pca_factors`` from ``macroforecast.core.runtime`` with
    ``variant="scaled_pca"`` and ``target_signal=target``.  Implements
    the Huang-Jiang-Li-Tong-Zhou (2022) sPCAest algorithm: standardise X
    column-wise, compute univariate OLS slope beta_j per column against the
    target, scale each column by its signed beta_j, then run PCA on the
    scaled matrix.  Equivalent recipe configuration::

        op: scaled_pca
        params:
          n_components: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> target = pd.Series(rng.randn(50), name="y")
    >>> out = scaled_pca_transform(panel, target, n_components=2)
    >>> out.shape
    (50, 2)
    >>> list(out.columns[:2])
    ['factor_1', 'factor_2']

    References
    ----------
    Huang, Jiang, Li, Tong & Zhou (2022) 'Scaled PCA: A New Approach to
    Dimension Reduction', Management Science 68(3): 1678-1695.
    """
    from macroforecast.core.runtime import _as_frame, _pca_factors  # noqa: PLC0415

    if isinstance(n_components, str):
        if n_components == "all":
            pass  # "all" is a valid passthrough; runtime handles it
        else:
            raise ValueError(
                f"scaled_pca_transform: n_components string value must be \"all\"; "
                f"got {n_components!r}"
            )
    elif n_components < 1:
        raise ValueError("scaled_pca_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    if not isinstance(target, pd.Series):
        raise TypeError(
            "scaled_pca_transform requires target to be a pd.Series; "
            f"got {type(target).__name__}"
        )
    if frame.index.intersection(target.index).empty:
        raise ValueError(
            "scaled_pca_transform: target and panel share no common index values; "
            "cannot align supervisory signal with panel rows."
        )
    return _pca_factors(
        frame, n_components=n_components, variant="scaled_pca", target_signal=target
    )


# ---------------------------------------------------------------------------
# 24. supervised_pca_transform
# ---------------------------------------------------------------------------

def supervised_pca_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    n_components: int = 3,
    q: float = 0.5,
) -> pd.DataFrame:
    """Giglio-Xiu-Zhang (2025) Supervised PCA -- screen-then-PCA factor extraction.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    target : pd.Series
        Supervisory signal used to rank panel columns by univariate
        correlation. Must share at least one index value with ``panel``;
        raises ``ValueError`` if the intersection is empty.
    n_components : int, default 3
        Number of supervised principal components (P). Must be >= 1.
    q : float, default 0.5
        Fraction of panel columns to retain after univariate correlation
        screening. Must satisfy ``0 < q < 1``; raises ``ValueError``
        outside this range. Forwarded to ``_supervised_pca``.

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``spca_1``, ``spca_2``,
        ..., ``spca_{n_components}``. Rows dropped for NaN are filled
        back with NaN on output.

    Notes
    -----
    Calls ``_supervised_pca`` from ``macroforecast.core.runtime``.
    Two-stage procedure: (1) rank panel columns by univariate Pearson
    correlation with the target, keep the top q-fraction (default
    q=0.5); (2) run PCA on the screened sub-panel via SVD.  Distinct
    from ``partial_least_squares`` (NIPALS over all columns) and
    ``scaled_pca`` (column-weight scaling before PCA).  Equivalent
    recipe configuration::

        op: supervised_pca
        params:
          n_components: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> target = pd.Series(rng.randn(50), name="y")
    >>> out = supervised_pca_transform(panel, target, n_components=2)
    >>> out.shape[1]
    2
    >>> out.columns[0]
    'spca_1'

    References
    ----------
    Giglio, Xiu & Zhang (2025) 'Test Assets and Weak Factors', Journal of
    Finance, forthcoming.
    Rapach & Zhou (2025) 'Sparse Macro-Finance Factors', working paper.
    """
    from macroforecast.core.runtime import _as_frame, _supervised_pca  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("supervised_pca_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    if not isinstance(target, pd.Series):
        raise TypeError(
            "supervised_pca_transform requires target to be a pd.Series; "
            f"got {type(target).__name__}"
        )
    if frame.index.intersection(target.index).empty:
        raise ValueError(
            "supervised_pca_transform: target and panel share no common index values; "
            "cannot align supervisory signal with panel rows."
        )
    if not (0.0 < q < 1.0):
        raise ValueError(
            f"supervised_pca_transform: q must satisfy 0 < q < 1; got {q!r}"
        )
    return _supervised_pca(frame, target=target, n_components=n_components, q=q)


# ---------------------------------------------------------------------------
# 25. partial_least_squares_transform
# ---------------------------------------------------------------------------

def partial_least_squares_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    n_components: int = 3,
) -> pd.DataFrame:
    """Partial least squares regression -- supervised factor extraction.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    target : pd.Series
        Supervisory signal whose covariance with latent components is
        maximised by the NIPALS algorithm. Must share at least one index
        value with ``panel``; raises ``ValueError`` if the intersection
        is empty.
    n_components : int, default 3
        Number of PLS latent components. Must be >= 1. Clamped
        internally to ``min(T_clean - 1, K_clean)``.

    Returns
    -------
    pd.DataFrame
        Latent-component scores panel; columns named ``pls_1``,
        ``pls_2``, ..., ``pls_{n_components}``. Rows dropped for NaN
        are filled back with NaN on output.

    Notes
    -----
    Calls ``_partial_least_squares`` from ``macroforecast.core.runtime``.
    Uses sklearn ``PLSRegression`` (NIPALS) on the jointly-aligned
    (panel, target) matrix after listwise NaN deletion.  Distinct from
    ``scaled_pca`` (column beta-scaling then PCA) and ``supervised_pca``
    (hard-screen by correlation then PCA).  Equivalent recipe
    configuration::

        op: partial_least_squares
        params:
          n_components: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> target = pd.Series(rng.randn(50), name="y")
    >>> out = partial_least_squares_transform(panel, target, n_components=2)
    >>> out.columns.tolist()
    ['pls_1', 'pls_2']

    References
    ----------
    Wold, Sjostrom & Eriksson (2001) 'PLS-regression: a basic tool of
    chemometrics', Chemometrics and Intelligent Laboratory Systems
    58(2): 109-130.
    """
    from macroforecast.core.runtime import _as_frame, _partial_least_squares  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("partial_least_squares_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    if not isinstance(target, pd.Series):
        raise TypeError(
            "partial_least_squares_transform requires target to be a pd.Series; "
            f"got {type(target).__name__}"
        )
    if frame.index.intersection(target.index).empty:
        raise ValueError(
            "partial_least_squares_transform: target and panel share no common "
            "index values; cannot align supervisory signal with panel rows."
        )
    # BLK-6: guard against small-N (fewer than 2 clean rows)
    aligned_check = pd.concat([frame, target.rename("__target__")], axis=1).dropna()
    if len(aligned_check) < 2:
        return pd.DataFrame(
            np.full((len(frame), n_components), np.nan),
            index=frame.index,
            columns=[f"pls_{i + 1}" for i in range(n_components)],
        )
    # BLK-4: clamp n_components to min(T_clean-1, K_clean) [NOTE-A fix: K_clean not K_clean-1]
    T_clean = len(aligned_check)
    K_clean = aligned_check.shape[1] - 1  # exclude target column
    n_components = min(n_components, min(T_clean - 1, K_clean))
    n_components = max(1, n_components)
    return _partial_least_squares(frame, target=target, n_components=n_components)


# ---------------------------------------------------------------------------
# 26. sliced_inverse_regression_transform
# ---------------------------------------------------------------------------

def sliced_inverse_regression_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    *,
    n_components: int = 3,
    n_slices: int = 10,
    scaling_method: str = "scaled_pca",
) -> pd.DataFrame:
    """Fan-Xue-Yao (2017) Sliced Inverse Regression with Huang-Zhou predictive scaling.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    target : pd.Series
        Supervisory signal used to sort rows into slices. Must share at
        least one index value with ``panel``; raises ``ValueError`` if
        the intersection is empty.
    n_components : int, default 3
        Number of SIR directions (effective rank of the between-slice
        covariance matrix). Must be >= 1.
    n_slices : int, default 10
        Number of contiguous slices of the target distribution. Must
        be >= 2. Clamped internally to the number of aligned rows.
    scaling_method : str, default ``"scaled_pca"``
        Predictive scaling variant forwarded to ``_sliced_inverse_regression``.
        Allowed values: ``"scaled_pca"`` (Huang-Zhou 2022 sSUFF augmentation)
        and ``"none"`` (plain SIR without predictive scaling). Raises
        ``ValueError`` for any other value.

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``factor_1``, ``factor_2``,
        ..., ``factor_{n_components}``. Rows without a valid aligned
        target value are filled with zeros on output (runtime convention
        for SIR).

    Notes
    -----
    Calls ``_sliced_inverse_regression`` from
    ``macroforecast.core.runtime`` with ``scaling_method="scaled_pca"``
    (sSUFF variant; Huang-Jiang-Li-Tong-Zhou 2022 augmentation applies
    a univariate predictive slope weight per column before slicing).
    Equivalent recipe configuration::

        op: sliced_inverse_regression
        params:
          n_components: 3
          n_slices: 10

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> target = pd.Series(rng.randn(50), name="y")
    >>> out = sliced_inverse_regression_transform(panel, target, n_components=2)
    >>> out.shape
    (50, 2)
    >>> out.columns[0]
    'factor_1'

    References
    ----------
    Li (1991) 'Sliced Inverse Regression for Dimension Reduction',
    JASA 86(414): 316-327.
    Fan, Xue & Yao (2017) 'Sufficient forecasting using factor models',
    Journal of Econometrics 201(2): 292-306.
    Huang, Jiang, Li, Tong & Zhou (2022) 'Scaled PCA: A New Approach to
    Dimension Reduction', Management Science 68(3): 1678-1695.
    """
    from macroforecast.core.runtime import (  # noqa: PLC0415
        _as_frame,
        _sliced_inverse_regression,
    )

    if n_components < 1:
        raise ValueError("sliced_inverse_regression_transform requires n_components >= 1")
    if n_slices < 2:
        raise ValueError("sliced_inverse_regression_transform requires n_slices >= 2")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    if not isinstance(target, pd.Series):
        raise TypeError(
            "sliced_inverse_regression_transform requires target to be a pd.Series; "
            f"got {type(target).__name__}"
        )
    if frame.index.intersection(target.index).empty:
        raise ValueError(
            "sliced_inverse_regression_transform: target and panel share no common "
            "index values; cannot align supervisory signal with panel rows."
        )
    _ALLOWED_SCALING_METHODS = {"scaled_pca", "none"}
    if scaling_method not in _ALLOWED_SCALING_METHODS:
        raise ValueError(
            f"sliced_inverse_regression_transform: scaling_method must be one of "
            f"{sorted(_ALLOWED_SCALING_METHODS)!r}; got {scaling_method!r}"
        )
    return _sliced_inverse_regression(
        frame,
        target=target,
        n_components=n_components,
        n_slices=n_slices,
        scaling_method=scaling_method,
    )


# ---------------------------------------------------------------------------
# 27. dfm_transform
# ---------------------------------------------------------------------------

def dfm_transform(
    panel: pd.DataFrame,
    *,
    n_factors: int = 3,
) -> pd.DataFrame:
    """Static dynamic factor model approximation via PCA on standardised panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_factors : int, default 3
        Number of latent dynamic factors to extract. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``dfm_1``, ``dfm_2``,
        ..., ``dfm_{n_factors}``. Rows dropped for NaN are filled back
        with NaN on output.

    Notes
    -----
    Calls ``_dfm_factors`` from ``macroforecast.core.runtime``.  Static
    approximation: standardise the panel column-wise (zero mean, unit
    standard deviation with ddof=0), then apply PCA.  Renames factor
    columns from ``factor_{k}`` to ``dfm_{k}`` to distinguish from
    plain ``pca_transform`` output.  No target required -- fully
    unsupervised.  Equivalent recipe configuration::

        op: dfm
        params:
          n_factors: 3

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = dfm_transform(panel, n_factors=2)
    >>> out.shape
    (50, 2)
    >>> out.columns.tolist()
    ['dfm_1', 'dfm_2']

    References
    ----------
    Mariano & Murasawa (2003) 'A new coincident index of business cycles
    based on monthly and quarterly series', JAE 18(4): 427-443.
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    from macroforecast.core.runtime import _as_frame, _dfm_factors  # noqa: PLC0415

    if n_factors < 1:
        raise ValueError("dfm_transform requires n_factors >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    result = _dfm_factors(frame, n_factors=n_factors)
    return result.reindex(frame.index)


# ---------------------------------------------------------------------------
# 28. feature_selection_transform
# ---------------------------------------------------------------------------

def feature_selection_transform(
    panel: pd.DataFrame,
    target: "pd.Series | None" = None,
    *,
    n_features: "int | float" = 0.5,
    method: str = "variance",
) -> pd.DataFrame:
    """Filter panel columns by variance, target correlation, or lasso pre-screen.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
    target : pd.Series or None, default None
        Supervisory signal required by ``method="correlation"`` and
        ``method="lasso"``.  Ignored for ``method="variance"``.
        Raises ``ValueError`` when a supervised method is requested but
        ``target is None``.
    n_features : int or float, default 0.5
        Number of features to keep.  If a float in ``(0, 1]``, treated
        as a fraction of the total column count.  If an integer, used
        as a direct count (clamped to ``[1, K]``).
    method : str, default "variance"
        Selection criterion.  One of ``"variance"``, ``"correlation"``,
        or ``"lasso"``.  ``"correlation"`` and ``"lasso"`` require
        ``target is not None``.

    Returns
    -------
    pd.DataFrame
        Subset of input ``panel`` columns selected by the criterion.
        Row count and index are unchanged.

    Notes
    -----
    Calls ``_feature_selection`` from ``macroforecast.core.runtime``.
    The ``variance`` path is purely unsupervised; ``correlation`` and
    ``lasso`` require the target signal.  Equivalent recipe
    configuration::

        op: feature_selection
        params:
          n_features: 0.5
          method: variance

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 8), columns=[f"x{i}" for i in range(8)])
    >>> target = pd.Series(rng.randn(50), name="y")
    >>> out = feature_selection_transform(panel)
    >>> out.shape
    (50, 4)
    >>> out2 = feature_selection_transform(panel, target, method="correlation")
    >>> out2.shape
    (50, 4)

    References
    ----------
    macroforecast design Part 2, L3: step library, ``feature_selection`` op.
    """
    from macroforecast.core.runtime import _as_frame, _feature_selection  # noqa: PLC0415

    _VALID_METHODS = {"variance", "correlation", "lasso"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"feature_selection_transform: unknown method {method!r}. "
            f"Expected one of {sorted(_VALID_METHODS)}."
        )
    if method in {"correlation", "lasso"} and target is None:
        raise ValueError(
            f"feature_selection_transform: method={method!r} requires target; "
            "pass a pd.Series as the target argument."
        )
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _feature_selection(frame, target=target, n_features=n_features, method=method)


# ---------------------------------------------------------------------------
# 29. sparse_pca_transform
# ---------------------------------------------------------------------------

def sparse_pca_transform(
    panel: pd.DataFrame,
    *,
    n_components: int = 8,
) -> pd.DataFrame:
    """L1-penalised Sparse PCA factor extraction (Zou-Hastie-Tibshirani 2006).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_components : int, default 8
        Number of sparse principal components to extract. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Factor scores panel; columns named ``factor_1``, ``factor_2``,
        ..., ``factor_{n_components}``. Rows dropped for NaN are filled
        back with NaN on output.

    Notes
    -----
    Calls ``_pca_factors`` from ``macroforecast.core.runtime`` with
    ``variant="sparse_pca"``.  Applies sklearn's ``SparsePCA`` (dictionary
    learning with L1 penalty) to yield interpretable factors that load on
    a small subset of predictors.  For the Chen-Rohe (2023) non-diagonal-D
    SCA variant, use ``sparse_pca_chen_rohe_transform`` instead.
    Equivalent recipe configuration::

        op: sparse_pca
        params:
          n_components: 8

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = sparse_pca_transform(panel, n_components=3)
    >>> out.shape
    (50, 3)
    >>> list(out.columns)
    ['factor_1', 'factor_2', 'factor_3']

    References
    ----------
    Zou, Hastie & Tibshirani (2006) 'Sparse Principal Component Analysis',
    Journal of Computational and Graphical Statistics 15(2): 265-286.
    """
    from macroforecast.core.runtime import _as_frame, _pca_factors  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("sparse_pca_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _pca_factors(frame, n_components=n_components, variant="sparse_pca")


# ---------------------------------------------------------------------------
# 30. sparse_pca_chen_rohe_transform
# ---------------------------------------------------------------------------

def sparse_pca_chen_rohe_transform(
    panel: pd.DataFrame,
    *,
    n_components: int = 4,
    zeta: float = 0.0,
    max_iter: int = 200,
    var_innovations: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """Chen-Rohe (2023) Sparse Component Analysis (SCA) -- non-diagonal D variant.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_components : int, default 4
        Number of sparse components (= J in the SCA objective). Must be >= 1.
    zeta : float, default 0.0
        L1 budget for loadings Theta. A value of ``0.0`` routes to
        ``zeta = n_components`` (the most-binding boundary the paper finds
        optimal in cross-validation). Must be >= 0.
    max_iter : int, default 200
        Maximum number of alternating-maximisation iterations. Must be >= 1.
    var_innovations : bool, default False
        If ``True``, fit a VAR(1) on the SCA scores and return the residuals
        as sparse macro-finance factors (Rapach-Zhou 2025 Strategy step 2).
    random_state : int, default 0
        Seed for NumPy random number generator (used for Z/Theta init).

    Returns
    -------
    pd.DataFrame
        SCA factor scores; columns named ``sca_1``, ..., ``sca_{n_components}``.
        Rows dropped for NaN are filled back with NaN on output.

    Notes
    -----
    Calls ``_sparse_pca_chen_rohe`` from ``macroforecast.core.runtime``.
    Solves the bilinear convex-hull form
    ``max_{Z,Theta} ||Z' X Theta||_F`` s.t. ``Z in H(T,J)``,
    ``Theta in H(M,J)``, ``||Theta||_1 <= zeta`` (Rapach-Zhou 2025 eq. 4).
    Used as the macro-side stage in Rapach & Zhou (2025) Sparse Macro-Finance
    Factors.  Equivalent recipe configuration::

        op: sparse_pca_chen_rohe
        params:
          n_components: 4
          zeta: 0.0

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = sparse_pca_chen_rohe_transform(panel, n_components=2)
    >>> out.shape
    (50, 2)

    References
    ----------
    Chen & Rohe (2023) 'A New Basis for Sparse Principal Component Analysis',
    Journal of Computational and Graphical Statistics. arXiv:2007.00596.
    Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper, eqs. (3)-(4).
    """
    from macroforecast.core.runtime import _as_frame, _sparse_pca_chen_rohe  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("sparse_pca_chen_rohe_transform requires n_components >= 1")
    if zeta < 0:
        raise ValueError("sparse_pca_chen_rohe_transform requires zeta >= 0")
    if max_iter < 1:
        raise ValueError("sparse_pca_chen_rohe_transform requires max_iter >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _sparse_pca_chen_rohe(
        frame,
        n_components=n_components,
        zeta=zeta,
        max_iter=max_iter,
        var_innovations=var_innovations,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 31. varimax_transform
# ---------------------------------------------------------------------------

def varimax_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Varimax-rotated factor scores via orthogonal rotation.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel of factor scores (e.g., output of ``pca_transform``).
        Each column is a factor; rows are time periods. Series is promoted
        to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).

    Returns
    -------
    pd.DataFrame
        Varimax-rotated factor scores; columns named ``varimax_1``,
        ..., ``varimax_{K}``. Rows dropped for NaN are filled back with
        NaN on output.

    Notes
    -----
    Calls ``_varimax_rotation`` from ``macroforecast.core.runtime``.
    Applies an iterative orthogonal rotation that maximises the variance
    of squared loadings within each factor, producing sparser (more
    interpretable) loading patterns than plain PCA.  Equivalent recipe
    configuration::

        op: varimax
        params: {}

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 3), columns=list("abc"))
    >>> out = varimax_transform(panel)
    >>> out.shape
    (50, 3)

    References
    ----------
    Kaiser (1958) 'The varimax criterion for analytic rotation in factor
    analysis', Psychometrika 23(3): 187-200.
    """
    from macroforecast.core.runtime import _as_frame, _varimax_rotation  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _varimax_rotation(frame)


# ---------------------------------------------------------------------------
# 32. random_projection_transform
# ---------------------------------------------------------------------------

def random_projection_transform(
    panel: pd.DataFrame,
    *,
    n_components: int = 8,
) -> pd.DataFrame:
    """Johnson-Lindenstrauss random Gaussian projection.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_components : int, default 8
        Number of output dimensions. Clamped internally to
        ``min(n_components, K)`` where K is the number of panel columns.
        Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Projected panel; columns named ``rp_1``, ``rp_2``,
        ..., ``rp_{n_components}``. Rows dropped for NaN are filled back
        with NaN on output.

    Notes
    -----
    Calls ``_random_projection`` from ``macroforecast.core.runtime``.
    Multiplies the panel by a random Gaussian matrix (sklearn's
    ``GaussianRandomProjection``) scaled to approximately preserve pairwise
    distances (Johnson-Lindenstrauss lemma).  Useful as a cheap baseline
    against structured reductions (PCA, sparse PCA).  Equivalent recipe
    configuration::

        op: random_projection
        params:
          n_components: 8

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = random_projection_transform(panel, n_components=3)
    >>> out.shape
    (50, 3)
    >>> list(out.columns)
    ['rp_1', 'rp_2', 'rp_3']

    References
    ----------
    Johnson & Lindenstrauss (1984) 'Extensions of Lipschitz mappings into a
    Hilbert space', Contemporary Mathematics 26: 189-206.
    """
    from macroforecast.core.runtime import _as_frame, _random_projection  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("random_projection_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _random_projection(frame, n_components=n_components)


# ---------------------------------------------------------------------------
# 33. kernel_features_transform
# ---------------------------------------------------------------------------

def kernel_features_transform(
    panel: pd.DataFrame,
    *,
    kind: str = "rbf",
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Exact kernel Gram matrix -- RBF or polynomial pairwise similarities.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    kind : str, default "rbf"
        Kernel type.  One of ``"rbf"`` (radial basis function / Gaussian
        kernel) or ``"polynomial"`` (degree-2 polynomial kernel).
    gamma : float, default 1.0
        Kernel bandwidth / scale parameter.  For ``"rbf"``:
        ``K(x, z) = exp(-gamma * ||x - z||^2)``.  For ``"polynomial"``:
        ``K(x, z) = (gamma * <x, z> + 1)^2``.  Must be > 0.

    Returns
    -------
    pd.DataFrame
        Gram matrix of shape ``(T_clean, T_clean)`` where ``T_clean`` is
        the number of rows without any NaN.  Columns named ``kernel_1``,
        ..., ``kernel_{T_clean}``.  Rows dropped for NaN are **not**
        reindexed back (the output is square over the clean rows only).

    Notes
    -----
    Calls ``_kernel_features`` from ``macroforecast.core.runtime``.  The
    output is the *exact* T_clean x T_clean Gram matrix -- not an
    approximation.  For large panels use ``nystroem_transform`` (Nystroem
    low-rank approximation) or random Fourier features instead.
    Equivalent recipe configuration::

        op: kernel_features
        params:
          kind: rbf
          gamma: 1.0

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = kernel_features_transform(panel, kind="rbf", gamma=0.5)
    >>> out.shape
    (50, 50)

    References
    ----------
    Rahimi & Recht (2007) 'Random Features for Large-Scale Kernel Machines',
    NeurIPS.
    """
    from macroforecast.core.runtime import _as_frame, _kernel_features  # noqa: PLC0415

    _VALID_KINDS = {"rbf", "polynomial"}
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kernel_features_transform: unknown kind {kind!r}. "
            f"Expected one of {sorted(_VALID_KINDS)}."
        )
    if gamma <= 0:
        raise ValueError("kernel_features_transform requires gamma > 0")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _kernel_features(frame, kind=kind, gamma=gamma)


# ---------------------------------------------------------------------------
# 34. nystroem_transform
# ---------------------------------------------------------------------------

def nystroem_transform(
    panel: pd.DataFrame,
    *,
    n_components: int = 32,
) -> pd.DataFrame:
    """Nystroem low-rank kernel approximation feature map.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        Rows with any NaN are dropped before fitting (listwise deletion).
    n_components : int, default 32
        Number of landmark points for the Nystroem approximation.
        Clamped internally to ``min(n_components, T_clean)``.
        Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Nystroem feature map; columns named ``nystroem_1``,
        ..., ``nystroem_{n_components}``. Rows dropped for NaN are filled
        back with NaN on output.

    Notes
    -----
    Calls ``_nystroem_features`` from ``macroforecast.core.runtime``.
    Uses sklearn's ``Nystroem`` with ``random_state=0`` for reproducibility.
    Constructs a low-rank approximation of the RBF kernel matrix using a
    random subsample of training points (landmark points).  More accurate
    than Random Fourier Features for non-RBF kernels but with larger memory
    footprint.  Equivalent recipe configuration::

        op: nystroem_features
        params:
          n_components: 32

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 5), columns=list("abcde"))
    >>> out = nystroem_transform(panel, n_components=10)
    >>> out.shape
    (50, 10)
    >>> list(out.columns[:3])
    ['nystroem_1', 'nystroem_2', 'nystroem_3']

    References
    ----------
    Williams & Seeger (2001) 'Using the Nystroem method to speed up kernel
    machines', NeurIPS.
    """
    from macroforecast.core.runtime import _as_frame, _nystroem_features  # noqa: PLC0415

    if n_components < 1:
        raise ValueError("nystroem_transform requires n_components >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _nystroem_features(frame, n_components=n_components)


# ---------------------------------------------------------------------------
# 35. time_trend_transform
# ---------------------------------------------------------------------------

def time_trend_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Deterministic linear time trend (t = 1, 2, ..., T).

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        The panel index is used only to align the output; actual trend
        values are positional (1-indexed).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with column ``"time_trend"`` containing
        ``1, 2, ..., T`` as float64.  Shape is ``(T, 1)`` where T is
        ``len(panel)``.

    Notes
    -----
    Generated inline via ``np.arange(1, T + 1)`` -- no runtime helper
    is required for this trivial deterministic feature.  Equivalent
    recipe configuration::

        op: time_trend
        params: {}

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(5, 2), columns=list("ab"))
    >>> out = time_trend_transform(panel)
    >>> out.shape
    (5, 1)
    >>> list(out["time_trend"])
    [1.0, 2.0, 3.0, 4.0, 5.0]

    References
    ----------
    macroforecast design Part 2, L3: step library, ``time_trend`` op.
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    T = len(frame)
    trend: np.ndarray = np.arange(1, T + 1, dtype=float)
    return pd.DataFrame({"time_trend": trend}, index=frame.index)


# ---------------------------------------------------------------------------
# 36. holiday_transform
# ---------------------------------------------------------------------------

def holiday_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """US federal holiday indicator column (0/1) over the panel index.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        Series is promoted to a single-column DataFrame internally.
        When the panel index is a ``pd.DatetimeIndex``, dates matching
        US federal holidays are flagged ``1.0``; all other dates are
        ``0.0``.  Non-DatetimeIndex inputs always return all zeros.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with column ``"is_holiday"`` containing
        ``0.0`` or ``1.0``.  Shape is ``(T, 1)`` where T is
        ``len(panel)``.

    Notes
    -----
    Calls ``_holiday_indicator`` from ``macroforecast.core.runtime``.
    Uses ``pd.tseries.offsets.USFederalHolidayCalendar`` for holiday
    detection.  Equivalent recipe configuration::

        op: holiday
        params: {}

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(50, 2), columns=list("ab"))
    >>> out = holiday_transform(panel)
    >>> out.shape
    (50, 1)
    >>> out.columns.tolist()
    ['is_holiday']

    References
    ----------
    macroforecast design Part 2, L3: step library, ``holiday`` op.
    """
    from macroforecast.core.runtime import _as_frame, _holiday_indicator  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _holiday_indicator(frame)
