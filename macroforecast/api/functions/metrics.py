"""Standalone L5 scalar metric functions.

Promotes the L5 scalar metrics to first-class callables (v0.1.0).
Each function is pure-numeric and returns a float.  Results are bit-exact with
the recipe-based L5 evaluation path where a recipe-path implementation exists.

References: see individual function docstrings.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Private validation helper
# ---------------------------------------------------------------------------

def _check_inputs(
    *arrays: np.ndarray,
    names: tuple[str, ...] | None = None,
) -> None:
    """Validate 1-D, same-length, non-empty arrays.

    Parameters
    ----------
    *arrays:
        Arrays to validate (already converted to float ndarray).
    names:
        Optional tuple of names for error messages; defaults to
        ``("arr0", "arr1", ...)``.

    Raises
    ------
    ValueError
        When any array is not 1-D, arrays have different lengths, or all
        arrays are empty.
    """
    if names is None:
        names = tuple(f"arr{i}" for i in range(len(arrays)))

    for arr, name in zip(arrays, names):
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D array; got shape {arr.shape}."
            )

    if len(arrays) == 0:
        return

    n = len(arrays[0])
    if n == 0:
        raise ValueError(
            f"{names[0]} (and all other inputs) must be non-empty."
        )

    for arr, name in zip(arrays[1:], names[1:]):
        if len(arr) != n:
            raise ValueError(
                f"All arrays must have the same length; "
                f"got {n} ({names[0]}) vs {len(arr)} ({name})."
            )


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean squared error.

    ``MSE = (1/N) Σ (y_true - y_pred)²``

    Produces bit-exact the same value as recipe-based L5 ``mse``
    (extracted from the ``groupby().agg(mse=("squared_error", "mean"))``
    computation in ``macroforecast.core.runtime``).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Must be the same length as ``y_true``.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty,
        or when either is not 1-D.

    Notes
    -----
    MSE penalises large residuals super-linearly.  Under Gaussian-residual /
    squared-loss decision theory this is the natural evaluation criterion; it
    matches the L4 fit objective for OLS / ridge / elastic-net.  A single
    outlier in the OOS sample can dominate the score.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mse
    >>> mse(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    0.25

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root mean squared error.

    ``RMSE = sqrt(MSE) = sqrt((1/N) Σ (y_true - y_pred)²)``

    Produces bit-exact the same value as recipe-based L5 ``rmse``
    (extracted from ``metrics["rmse"] = metrics["mse"] ** 0.5`` in
    ``macroforecast.core.runtime``).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Must be the same length as ``y_true``.

    Returns
    -------
    float
        Root mean squared error, expressed in the same units as ``y_true``.

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty,
        or when either is not 1-D.

    Notes
    -----
    RMSE has the same outlier sensitivity as MSE but is expressed in target
    units rather than squared-target units.  Standard reporting metric in
    macro / finance papers; pairs naturally with confidence-band charts
    since RMSE has the same units as the prediction interval.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import rmse
    >>> rmse(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    0.5

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean absolute error.

    ``MAE = (1/N) Σ |y_true - y_pred|``

    Produces bit-exact the same value as recipe-based L5 ``mae``
    (extracted from the ``groupby().agg(mae=("absolute_error", "mean"))``
    computation in ``macroforecast.core.runtime``).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Must be the same length as ``y_true``.

    Returns
    -------
    float
        Mean absolute error.

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty,
        or when either is not 1-D.

    Notes
    -----
    L1 loss; robust alternative to MSE.  Equally weighs every absolute
    residual rather than penalising large errors super-linearly.  The
    implicit decision rule under MAE is the median of the predictive
    distribution (vs the mean for MSE).

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mae
    >>> mae(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    0.5

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    return float(np.mean(np.abs(y_true - y_pred)))


def medae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Median absolute error.

    ``MedAE = median(|y_true - y_pred|)``

    Produces bit-exact the same value as recipe-based L5 ``medae``
    (extracted from ``_add_l5_extended_metrics`` in
    ``macroforecast.core.runtime``, line 7681).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Must be the same length as ``y_true``.

    Returns
    -------
    float
        Median absolute error.

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty,
        or when either is not 1-D.

    Notes
    -----
    Maximally robust point-forecast metric: substitution by median
    completely insulates the score from a constant share of extreme residuals.
    Common in robust-statistics papers; rarer in mainstream forecasting.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import medae
    >>> medae(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 4.5]))
    0.5

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    return float(np.median(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    eps: float = 1e-10,
) -> float:
    """Mean absolute percentage error.

    ``MAPE = (100/N) Σ |y_true - y_pred| / max(|y_true|, eps)``

    No recipe-path computation exists in ``macroforecast.core.runtime``
    for this metric; this function is the canonical implementation.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Must be the same length as ``y_true``.
    eps : float, optional
        Small positive value used in the denominator guard to avoid
        division by zero when targets are near zero.  Default ``1e-10``.
        Must be positive.

    Returns
    -------
    float
        Mean absolute percentage error (scale: 0–100, not 0–1).

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty,
        or when either is not 1-D, or when ``eps <= 0``.

    Notes
    -----
    Scale-free percentage version of MAE.  Allows comparing forecasts for
    targets on different scales (e.g. US GDP vs Korean GDP).  Pathological
    when targets can be zero or near-zero -- the metric blows up.  Hyndman
    & Koehler (2006) recommend MASE / sMAPE in those cases.

    No recipe-path computation exists for this metric; the standalone
    function is the first and canonical implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mape
    >>> mape(np.array([100.0, 200.0, 300.0]), np.array([110.0, 195.0, 285.0]))
    4.166666666666667

    References
    ----------
    Hyndman & Koehler (2006) *Another look at measures of forecast accuracy*,
    International Journal of Forecasting 22(4): 679-688.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    safe_denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / safe_denom) * 100)


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------

def relative_mse(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_benchmark: np.ndarray,
) -> float:
    """Relative mean squared error (model MSE / benchmark MSE).

    ``relative_mse = MSE(y_true, y_model) / MSE(y_true, y_benchmark)``

    Produces bit-exact the same value as recipe-based L5 ``relative_mse``
    (extracted from ``_add_l5_relative_metrics`` in
    ``macroforecast.core.runtime``, line 7743).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_model : np.ndarray or pd.Series
        Candidate model forecast values.  Same length as ``y_true``.
    y_benchmark : np.ndarray or pd.Series
        Benchmark model forecast values.  Same length as ``y_true``.

    Returns
    -------
    float
        Ratio of model MSE to benchmark MSE.  Below 1.0 means the model
        beats the benchmark.  Returns ``nan`` when benchmark MSE is zero.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    The standard horse-race ratio.  Below 1 means the candidate beats the
    benchmark.  Requires two forecast arrays (model + benchmark), unlike
    the point metrics which take only ``y_pred``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import relative_mse
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_model = np.array([1.1, 2.1, 3.1])
    >>> y_bench = np.array([1.5, 2.5, 3.5])
    >>> relative_mse(y_true, y_model, y_bench)  # 0.01 / 0.25 = 0.04
    0.04

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_benchmark = np.asarray(y_benchmark, dtype=float)
    _check_inputs(y_true, y_model, y_benchmark,
                  names=("y_true", "y_model", "y_benchmark"))
    num = float(np.mean((y_true - y_model) ** 2))
    den = float(np.mean((y_true - y_benchmark) ** 2))
    return num / den if den > 0 else float("nan")


def relative_mae(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_benchmark: np.ndarray,
) -> float:
    """Relative mean absolute error (model MAE / benchmark MAE).

    ``relative_mae = MAE(y_true, y_model) / MAE(y_true, y_benchmark)``

    Produces bit-exact the same value as recipe-based L5 ``relative_mae``
    (extracted from ``_add_l5_relative_metrics`` in
    ``macroforecast.core.runtime``, line 7745).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_model : np.ndarray or pd.Series
        Candidate model forecast values.  Same length as ``y_true``.
    y_benchmark : np.ndarray or pd.Series
        Benchmark model forecast values.  Same length as ``y_true``.

    Returns
    -------
    float
        Ratio of model MAE to benchmark MAE.  Below 1.0 means the model
        beats the benchmark on absolute-loss criterion.  Returns ``nan``
        when benchmark MAE is zero.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    L1-loss analogue of ``relative_mse``.  Robust to heavy-tailed forecast
    errors.  Below 1 means the candidate beats the benchmark on the
    absolute-loss criterion.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import relative_mae
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_model = np.array([1.1, 2.1, 3.1])
    >>> y_bench = np.array([1.5, 2.5, 3.5])
    >>> relative_mae(y_true, y_model, y_bench)  # 0.1 / 0.5 = 0.2
    0.2

    References
    ----------
    Diebold (2017) *Forecasting in Economics, Business, Finance and Beyond*,
    University of Pennsylvania.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_benchmark = np.asarray(y_benchmark, dtype=float)
    _check_inputs(y_true, y_model, y_benchmark,
                  names=("y_true", "y_model", "y_benchmark"))
    num = float(np.mean(np.abs(y_true - y_model)))
    den = float(np.mean(np.abs(y_true - y_benchmark)))
    return num / den if den > 0 else float("nan")


def mse_reduction(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_benchmark: np.ndarray,
) -> float:
    """Absolute MSE reduction (benchmark MSE − model MSE).

    ``mse_reduction = MSE(y_true, y_benchmark) - MSE(y_true, y_model)``

    Produces bit-exact the same value as recipe-based L5 ``mse_reduction``
    (extracted from ``_add_l5_relative_metrics`` in
    ``macroforecast.core.runtime``, line 7746).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_model : np.ndarray or pd.Series
        Candidate model forecast values.  Same length as ``y_true``.
    y_benchmark : np.ndarray or pd.Series
        Benchmark model forecast values.  Same length as ``y_true``.

    Returns
    -------
    float
        Absolute difference ``benchmark_MSE − model_MSE``.  Positive means
        the model beats the benchmark.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    **Note**: the recipe-path computation (runtime.py line 7746) uses the
    absolute difference ``benchmark_MSE − model_MSE``, not the ratio-based
    ``1 − relative_mse`` as described in some documentation.  This function
    matches the recipe-path behavior.  The doc-string discrepancy (absolute
    vs. ratio) is flagged for a future documentation-fix cycle.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import mse_reduction
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_model = np.array([1.1, 2.1, 3.1])
    >>> y_bench = np.array([1.5, 2.5, 3.5])
    >>> mse_reduction(y_true, y_model, y_bench)  # 0.25 - 0.01 = 0.24
    0.24

    References
    ----------
    Campbell & Thompson (2008) *Predicting Excess Stock Returns Out of
    Sample*, Review of Financial Studies 21(4): 1509-1531.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_benchmark = np.asarray(y_benchmark, dtype=float)
    _check_inputs(y_true, y_model, y_benchmark,
                  names=("y_true", "y_model", "y_benchmark"))
    bench_mse = float(np.mean((y_true - y_benchmark) ** 2))
    model_mse = float(np.mean((y_true - y_model) ** 2))
    return bench_mse - model_mse


def r2_oos(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_benchmark: np.ndarray,
) -> float:
    """Out-of-sample R² (Campbell-Thompson 2008).

    ``R²_OOS = 1 - relative_mse(y_true, y_model, y_benchmark)``
             = ``1 - MSE(y_true, y_model) / MSE(y_true, y_benchmark)``

    Produces bit-exact the same value as recipe-based L5 ``r2_oos``
    (extracted from ``_add_l5_relative_metrics`` in
    ``macroforecast.core.runtime``, line 7744).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_model : np.ndarray or pd.Series
        Candidate model forecast values.  Same length as ``y_true``.
    y_benchmark : np.ndarray or pd.Series
        Benchmark model forecast values.  Same length as ``y_true``.

    Returns
    -------
    float
        Out-of-sample R².  Positive means the model beats the benchmark.
        Returns ``nan`` when benchmark MSE is zero.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    Standard return-predictability metric in finance (and increasingly in
    macro).  Identical formula to ``mse_reduction / benchmark_MSE`` when
    expressed as a fraction.  Campbell & Thompson (2008) popularised the
    metric for the empirical-asset-pricing literature.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import r2_oos
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_model = np.array([1.1, 2.1, 3.1])
    >>> y_bench = np.array([1.5, 2.5, 3.5])
    >>> r2_oos(y_true, y_model, y_bench)  # 1 - 0.04 = 0.96
    0.96

    References
    ----------
    Campbell & Thompson (2008) *Predicting Excess Stock Returns Out of
    Sample*, Review of Financial Studies 21(4): 1509-1531.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_benchmark = np.asarray(y_benchmark, dtype=float)
    _check_inputs(y_true, y_model, y_benchmark,
                  names=("y_true", "y_model", "y_benchmark"))
    rel = relative_mse(y_true, y_model, y_benchmark)
    if math.isnan(rel):
        return float("nan")
    return 1.0 - rel


# ---------------------------------------------------------------------------
# Interval / coverage metrics (density B1 subset)
# ---------------------------------------------------------------------------

def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    *,
    alpha: float = 0.05,
) -> float:
    """Winkler (1972) interval score.

    ``IS_α = mean((y_upper - y_lower)
               + (2/α) max(y_lower - y_true, 0)
               + (2/α) max(y_true - y_upper, 0))``

    No recipe-path computation exists in ``macroforecast.core.runtime``
    for this metric; this function is the canonical implementation.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_lower : np.ndarray or pd.Series
        Lower bound of the prediction interval.  Same length as ``y_true``.
    y_upper : np.ndarray or pd.Series
        Upper bound of the prediction interval.  Same length as ``y_true``.
    alpha : float, optional
        Miscoverage level (1 - confidence level).  E.g. ``alpha=0.05``
        for 95% prediction intervals.  Must be in ``(0, 1)``.  Default 0.05.

    Returns
    -------
    float
        Interval score averaged over all observations.  Lower = better.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D, or when ``alpha`` is not in ``(0, 1)``.

    Notes
    -----
    Strictly-proper scoring rule for the α-level prediction interval.
    Jointly penalises miscoverage and interval width.  For a nominal-α
    interval ``[L, U]``: reward tighter intervals, but incur a heavy penalty
    ``(2/α)`` for each unit by which the observation falls outside.

    No recipe-path computation exists for this metric; the standalone
    function is the first and canonical implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import interval_score
    >>> y_true = np.array([1.0, 2.0, 5.0])
    >>> y_lower = np.array([0.5, 1.5, 2.5])
    >>> y_upper = np.array([1.5, 2.5, 3.5])
    >>> interval_score(y_true, y_lower, y_upper, alpha=0.1)  # doctest: +ELLIPSIS
    14.0

    References
    ----------
    Winkler (1972) *A Decision-Theoretic Approach to Interval Estimation*,
    JASA 67(337): 187-191.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    _check_inputs(y_true, y_lower, y_upper,
                  names=("y_true", "y_lower", "y_upper"))
    width = y_upper - y_lower
    under = np.maximum(y_lower - y_true, 0.0)
    over = np.maximum(y_true - y_upper, 0.0)
    scores = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    return float(np.mean(scores))


def coverage_rate(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Empirical coverage rate.

    ``coverage_rate = (1/N) Σ 1{y_lower ≤ y_true ≤ y_upper}``

    No recipe-path computation exists in ``macroforecast.core.runtime``
    for this metric; this function is the canonical implementation.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_lower : np.ndarray or pd.Series
        Lower bound of the prediction interval.  Same length as ``y_true``.
    y_upper : np.ndarray or pd.Series
        Upper bound of the prediction interval.  Same length as ``y_true``.

    Returns
    -------
    float
        Share of observations falling within the interval.  Bounded in
        ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    Should equal ``1 - alpha`` if the model is well-calibrated.  Deviations
    indicate miscalibration: low coverage = intervals too narrow; high
    coverage = intervals too wide.  Pair with ``interval_score`` to capture
    both calibration and sharpness.

    No recipe-path computation exists for this metric; the standalone
    function is the first and canonical implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import coverage_rate
    >>> y_true = np.array([1.0, 2.0, 5.0])
    >>> y_lower = np.array([0.5, 1.5, 2.5])
    >>> y_upper = np.array([1.5, 2.5, 3.5])
    >>> coverage_rate(y_true, y_lower, y_upper)  # 2 of 3 covered
    0.6666666666666666

    References
    ----------
    Gneiting & Raftery (2007) *Strictly Proper Scoring Rules, Prediction,
    and Estimation*, JASA 102(477): 359-378.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    _check_inputs(y_true, y_lower, y_upper,
                  names=("y_true", "y_lower", "y_upper"))
    hits = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(hits.astype(float)))


# ---------------------------------------------------------------------------
# Direction metrics
# ---------------------------------------------------------------------------

def success_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray,
) -> float:
    """Directional success ratio (hit-rate of correct directional forecasts).

    ``success_ratio = mean(sign(y_pred - y_prev) == sign(y_true - y_prev))``
    evaluated on rows where ``y_prev`` is not ``nan``.

    Produces bit-exact the same value as recipe-based L5 ``success_ratio``
    (extracted from ``_add_l5_extended_metrics`` in
    ``macroforecast.core.runtime``, lines 7695-7706).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Same length as ``y_true``.
    y_prev : np.ndarray or pd.Series
        Lagged actual values (y at t-1).  Same length as ``y_true``.
        Pass ``np.nan`` for rows where the previous value is unavailable;
        those rows are excluded from the computation.

    Returns
    -------
    float
        Hit-rate of correct directional forecasts.  Bounded in ``[0.0, 1.0]``.
        Returns ``nan`` when fewer than 2 valid (non-NaN ``y_prev``) rows
        remain after masking.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    Direction is measured relative to the previous actual value ``y_prev``,
    not relative to zero.  Does not adjust for the unconditional direction
    frequency, so a constant 'always positive' forecast can score highly on
    a consistently rising target.  For statistical significance, pair with
    ``pesaran_timmermann_metric`` and the L6.F PT test.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import success_ratio
    >>> y_true = np.array([1.0, 2.0, 1.5, 3.0])
    >>> y_pred = np.array([1.5, 2.2, 1.2, 2.8])
    >>> y_prev = np.array([np.nan, 1.0, 2.0, 1.5])
    >>> success_ratio(y_true, y_pred, y_prev)  # 3 valid rows; 3/3 correct
    1.0

    References
    ----------
    Pesaran & Timmermann (1992) *A simple nonparametric test of predictive
    performance*, JBES 10(4): 461-465.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    _check_inputs(y_true, y_pred, y_prev,
                  names=("y_true", "y_pred", "y_prev"))
    valid = ~np.isnan(y_prev)
    if valid.sum() < 2:
        return float("nan")
    yt = y_true[valid]
    yp = y_pred[valid]
    yp_prev = y_prev[valid]
    sign_pred = np.sign(yp - yp_prev)
    sign_true = np.sign(yt - yp_prev)
    return float(np.mean(sign_pred == sign_true))


def pesaran_timmermann_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.0,
) -> float:
    """Pesaran-Timmermann (1992) directional-accuracy statistic.

    Computes the PT statistic for binary directional forecasts derived by
    comparing both ``y_true`` and ``y_pred`` against ``threshold``.  The
    statistic is asymptotically ``N(0, 1)`` under the null of no directional
    predictive skill.

    Produces bit-exact the same value as recipe-based L5
    ``pesaran_timmermann_metric`` (extracted from
    ``_pesaran_timmermann_test`` in ``macroforecast.core.runtime``,
    lines 11616-11660).

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual (realised) values.  1-D float array of length N.
    y_pred : np.ndarray or pd.Series
        Forecast values.  Same length as ``y_true``.
    threshold : float, optional
        Threshold for computing binary direction series.  A value above
        ``threshold`` = directional 'up'.  Default 0.0.

    Returns
    -------
    float
        PT test statistic.  Returns ``nan`` when ``N < 2`` or when the
        probability parameters fall outside ``(0, 1)``.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty, or when any
        array is not 1-D.

    Notes
    -----
    Adjusts the success ratio for the joint probability of agreement under
    independence (so a constant-sign forecast no longer scores high).
    Returns the statistic only (float), not the p-value; for the p-value
    and significance test, use the L6.F PT test.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import pesaran_timmermann_metric
    >>> rng = np.random.RandomState(0)
    >>> y_true = rng.choice([0, 1], size=100).astype(float)
    >>> y_pred = y_true + 0.1 * rng.randn(100)
    >>> stat = pesaran_timmermann_metric(y_true, y_pred)
    >>> isinstance(stat, float)
    True

    References
    ----------
    Pesaran & Timmermann (1992) *A simple nonparametric test of predictive
    performance*, JBES 10(4): 461-465.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_inputs(y_true, y_pred, names=("y_true", "y_pred"))
    n = len(y_true)
    if n < 2:
        return float("nan")
    # Convert to binary direction arrays using threshold.
    forecast = (y_pred > threshold).astype(int)
    actual = (y_true > threshold).astype(int)
    # success rate (proportion of matching directions)
    success = float((forecast == actual).mean())
    p_y = float(actual.mean())
    p_x = float(forecast.mean())
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    if p_star <= 0.0 or p_star >= 1.0:
        return float("nan")
    var_p = (p_star * (1.0 - p_star)) / n
    var_p_star = (
        ((2.0 * p_y - 1.0) ** 2 * p_x * (1.0 - p_x)) / n
        + ((2.0 * p_x - 1.0) ** 2 * p_y * (1.0 - p_y)) / n
        + (4.0 * p_y * p_x * (1.0 - p_y) * (1.0 - p_x)) / (n * n)
    )
    denom = max(var_p - var_p_star, 1e-12)
    return float((success - p_star) / math.sqrt(denom))
