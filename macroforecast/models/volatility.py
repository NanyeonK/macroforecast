"""Public thin-subclass promotions for the L4 volatility model family.

Each class is a zero-overhead subclass of the corresponding private runtime
class. No new behaviour is introduced.

Cycle 63 -- L4 volatility class promotion (2 classes: GARCHFamily,
RealizedGARCH).
"""
from __future__ import annotations

from macroforecast.core.runtime import (
    _GARCHFamily,
    _RealizedGARCHModel,
)


class GARCH(_GARCHFamily):
    """GARCH/EGARCH volatility model family wrapper.

    Public alias for :class:`~macroforecast.core.runtime._GARCHFamily`.

    Wraps the ``arch`` library's GARCH/EGARCH estimator for use in the L4
    forecasting pipeline. Supports GARCH(1,1) and EGARCH(1,1) specifications.
    When ``arch`` is not installed, raises ``NotImplementedError`` with an
    install hint.

    Parameters
    ----------
    variant : str
        Variance model variant: "garch11" or "egarch". Default "garch11".
    p : int
        ARCH lag order. Default 1.
    q : int
        GARCH lag order. Default 1.
    dist : str
        Error distribution: "normal", "t", "skewt". Default "normal".
    """

    def __init__(
        self,
        *,
        variant: str = "garch11",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        mean_model: str = "constant",
        dist: str = "normal",
        rescale: bool = False,
        random_state: int = 0,
        realized_variance: str | None = None,
    ) -> None:
        super().__init__(
            variant=variant,
            p=p,
            o=o,
            q=q,
            mean_model=mean_model,
            dist=dist,
            rescale=rescale,
            random_state=random_state,
            realized_variance=realized_variance,
        )


class RealizedGARCH(_RealizedGARCHModel):
    """Hansen-Huang-Shek (2012) Realized GARCH model.

    Public alias for :class:`~macroforecast.core.runtime._RealizedGARCHModel`.

    Models the joint dynamics of returns and a realized measure (e.g.
    realized variance) via the Realized GARCH measurement equation:

        log(h_t) = omega + beta * log(h_{t-1}) + gamma * log(x_{t-1})
        log(x_t) = xi + phi * log(h_t) + delta1 * z_t + delta2 * (z_t^2 - 1)
                   + tau_t

    where x_t is the realized measure and z_t = eps_t / sqrt(h_t).

    Estimation via numerical MLE. Falls back to GARCH(1,1) when no realized
    measure column is detected in X.

    References
    ----------
    Hansen, Huang, Shek (2012) "Realized GARCH: A Joint Model for Returns
    and Realized Measures of Volatility." Journal of Applied Econometrics 27.
    """


__all__ = [
    "GARCH",
    "RealizedGARCH",
]
