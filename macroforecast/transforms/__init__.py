"""Public transform function wrappers.

Exposes the Chow-Lin (1971) temporal disaggregation function as a standalone
public callable. Internally delegates to the private runtime implementation in
``macroforecast.core.runtime``.

Functions
---------
- :func:`chow_lin_disaggregate` -- Chow-Lin regression-based disaggregation.

Usage::

    import pandas as pd
    from macroforecast.transforms import chow_lin_disaggregate

    # Quarterly GDP disaggregated to monthly using retail sales indicator.
    monthly = chow_lin_disaggregate(
        low_freq=quarterly_gdp,
        indicator_high_freq=monthly_retail_sales,
    )

Cycle 63 -- Chow-Lin transform function wrapper.
"""
from __future__ import annotations

import pandas as pd


def chow_lin_disaggregate(
    low_freq: pd.Series,
    indicator_high_freq: pd.Series,
) -> pd.Series:
    """Chow-Lin (1971) regression-based temporal disaggregation.

    Disaggregates a low-frequency series (e.g. quarterly) to a higher
    frequency (e.g. monthly) using a related high-frequency indicator via
    the Chow-Lin regression method (constant-only intercept + AR(0) error
    variant, the common chow_lin_litterman simplification).

    Algorithm
    ---------
    1. Aggregate the indicator to the low frequency via mean.
    2. Regress the observed low-frequency series on the aggregated indicator
       to estimate ``alpha`` + ``beta`` via OLS.
    3. Disaggregate to the high frequency:
       ``y^H_t = alpha / m + beta * X^H_t + smoothed_residual_t``
       where ``m`` is the aggregation ratio (e.g. 3 for quarterly-to-monthly)
       and the smoothed residual distributes the low-frequency residual evenly
       across the high-frequency periods within each low-frequency period.

    Parameters
    ----------
    low_freq : pd.Series
        Observed low-frequency series (e.g. quarterly). Must be a
        ``pd.Series`` with a ``DatetimeIndex`` for proper temporal
        alignment. Non-DatetimeIndex inputs receive a best-effort
        bfill/ffill fallback.
    indicator_high_freq : pd.Series
        High-frequency indicator series (e.g. monthly). The output is
        aligned to this series's index.

    Returns
    -------
    pd.Series
        Disaggregated high-frequency series aligned with
        ``indicator_high_freq.index``.

    Notes
    -----
    This is a direct wrapper for the private function
    :func:`~macroforecast.core.runtime._chow_lin_disaggregate`. The public
    parameter names ``low_freq`` / ``indicator_high_freq`` replace the
    internal ``quarterly`` / ``indicator_monthly`` names to be
    frequency-agnostic.

    References
    ----------
    Chow, Lin (1971) "Best Linear Unbiased Interpolation, Distribution,
    and Extrapolation of Time Series by Related Series." Review of
    Economics and Statistics 53(4).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx_m = pd.date_range("2010-01-31", periods=36, freq="ME")
    >>> idx_q = pd.date_range("2010-03-31", periods=12, freq="QE")
    >>> rng = np.random.RandomState(0)
    >>> indicator = pd.Series(rng.randn(36), index=idx_m, name="ind")
    >>> ind_q = indicator.resample("QE").mean()
    >>> # Build quarterly target correlated with indicator.
    >>> y_q = pd.Series(0.5 + 2.0 * ind_q.values + 0.1 * rng.randn(12),
    ...                 index=idx_q, name="y_q")
    >>> y_m = chow_lin_disaggregate(y_q, indicator)
    >>> len(y_m) == 36
    True
    """
    from macroforecast.core.runtime import _chow_lin_disaggregate

    return _chow_lin_disaggregate(low_freq, indicator_high_freq)


__all__ = ["chow_lin_disaggregate"]
