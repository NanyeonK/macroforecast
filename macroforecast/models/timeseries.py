"""Public thin-subclass promotions for the L4 time-series model family.

Each class is a zero-overhead subclass of the corresponding private runtime
class. No new behaviour is introduced.

Cycle 63 -- L4 time-series class promotion (3 classes: ETSWrapper,
ThetaWrapper, HoltWintersWrapper).
"""
from __future__ import annotations

from macroforecast.core.runtime import (
    _ETSWrapper,
    _ThetaWrapper,
    _HoltWintersWrapper,
)


class ETS(_ETSWrapper):
    """Exponential Smoothing State Space Model (ETS).

    Public alias for :class:`~macroforecast.core.runtime._ETSWrapper`.

    Fits an ETS(error, trend, seasonal) model via statsmodels
    ``ExponentialSmoothing``. Supports additive and multiplicative
    error/trend/seasonal components with automatic parameter estimation by
    maximum likelihood.

    Parameters
    ----------
    error : str
        Error component: "add" or "mul". Default "add".
    trend : str or None
        Trend component: "add", "mul", or None. Default None.
    seasonal : str or None
        Seasonal component: "add", "mul", or None. Default None.
    seasonal_periods : int
        Number of periods per seasonal cycle. Default 12.
    damped_trend : bool
        Whether to damp the trend component. Default False.
    """


class Theta(_ThetaWrapper):
    """Theta method for time-series forecasting (Assimakopoulos-Nikolopoulos 2000).

    Public alias for :class:`~macroforecast.core.runtime._ThetaWrapper`.

    Decomposes the time series into two "theta lines" (linear trend and
    a dampened version), then produces forecasts by combining them. The
    standard theta method (theta=2) is equivalent to SES + linear drift.

    Parameters
    ----------
    theta : float
        Theta decomposition coefficient. Default 2.0. Must be > 1.
    seasonal : bool
        Apply seasonal decomposition before fitting. Default False.
    seasonal_periods : int
        Number of periods per seasonal cycle. Default 12.

    References
    ----------
    Assimakopoulos, Nikolopoulos (2000) "The theta model: a decomposition
    approach to forecasting." International Journal of Forecasting 16(4).
    """


class HoltWinters(_HoltWintersWrapper):
    """Holt-Winters triple exponential smoothing.

    Public alias for :class:`~macroforecast.core.runtime._HoltWintersWrapper`.

    Extends simple exponential smoothing with trend and seasonal components
    via the Holt-Winters additive or multiplicative procedure. Equivalent
    to ETS(A,A,A) or ETS(A,A,M) depending on the seasonal type.

    Parameters
    ----------
    seasonal : str
        Seasonal component type: "add" or "mul". Default "add".
    seasonal_periods : int
        Number of periods per seasonal cycle. Default 12.
    trend : str
        Trend component type: "add" or "mul". Default "add".
    damped_trend : bool
        Whether to damp the trend component. Default False.
    """


__all__ = [
    "ETS",
    "Theta",
    "HoltWinters",
]
