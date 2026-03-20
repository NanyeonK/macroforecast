"""Diebold-Mariano (1995) test for equal predictive accuracy.

Tests H0: E[d_t] = 0 where d_t = L(e_{1,t}) - L(e_{2,t}) is the loss
differential between two forecasts.

The DM test statistic is:
    DM = d_bar / sqrt(V_hat / T)

where V_hat is a HAC (Newey-West) estimate of the long-run variance of d_t.

Under H0, DM → N(0, 1).

Reference:
  Diebold, F.X. and Mariano, R.S. (1995).
  "Comparing Predictive Accuracy." JBES, 13(3), 253-263.

Note: The modified Harvey-Leybourne-Newbold (HLN 1997) correction for small
samples is applied by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class DMResult:
    """Result of the Diebold-Mariano test.

    Attributes
    ----------
    dm_stat : float
        DM test statistic (HLN-adjusted by default).
    p_value : float
        Two-tailed p-value.
    loss_diff_mean : float
        Mean of the loss differential d_bar.  Positive means model 1 is worse.
    hln_adjusted : bool
        Whether the HLN small-sample correction was applied.
    """

    dm_stat: float
    p_value: float
    loss_diff_mean: float
    hln_adjusted: bool


def dm_test(
    y_true: NDArray[np.floating],
    y_hat_1: NDArray[np.floating],
    y_hat_2: NDArray[np.floating],
    h: int = 1,
    loss: Literal["mse", "mae", "custom"] = "mse",
    loss_fn: callable | None = None,
    nw_bw: int | None = None,
    hln_adjust: bool = True,
) -> DMResult:
    """Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    y_true : array of shape (T,)
        Realised values.
    y_hat_1 : array of shape (T,)
        Forecasts from model 1.
    y_hat_2 : array of shape (T,)
        Forecasts from model 2.
    h : int
        Forecast horizon.  Used to set the Newey-West bandwidth to max(1, h-1)
        by default (as recommended by Diebold 2015).
    loss : str
        Loss function: "mse" (squared error), "mae" (absolute error), or
        "custom" (requires `loss_fn`).
    loss_fn : callable (y_true, y_hat) → array (T,) or None
        Custom element-wise loss function.  Required when `loss = "custom"`.
    nw_bw : int or None
        Newey-West bandwidth.  Defaults to max(1, h - 1).
    hln_adjust : bool
        Apply Harvey-Leybourne-Newbold (1997) finite-sample correction.

    Returns
    -------
    DMResult
    """
    T = len(y_true)
    e1 = y_true - y_hat_1
    e2 = y_true - y_hat_2

    if loss == "mse":
        L1 = e1**2
        L2 = e2**2
    elif loss == "mae":
        L1 = np.abs(e1)
        L2 = np.abs(e2)
    elif loss == "custom":
        if loss_fn is None:
            raise ValueError("Provide loss_fn when loss='custom'.")
        L1 = loss_fn(y_true, y_hat_1)
        L2 = loss_fn(y_true, y_hat_2)
    else:
        raise ValueError(f"Unknown loss: '{loss}'. Choose 'mse', 'mae', or 'custom'.")

    d = L1 - L2
    d_bar = d.mean()

    # Trivial case: all loss differentials are exactly zero
    if np.all(d == 0):
        return DMResult(
            dm_stat=0.0,
            p_value=1.0,
            loss_diff_mean=0.0,
            hln_adjusted=hln_adjust,
        )

    # Newey-West HAC variance
    bw = nw_bw if nw_bw is not None else max(0, h - 1)
    d_dm = d - d_bar
    gamma0 = np.dot(d_dm, d_dm) / T
    nw_var = gamma0
    for lag in range(1, bw + 1):
        gamma_l = np.dot(d_dm[lag:], d_dm[:-lag]) / T
        w = 1.0 - lag / (bw + 1)
        nw_var += 2 * w * gamma_l

    var_d_bar = nw_var / T

    if var_d_bar <= 0:
        return DMResult(
            dm_stat=float("nan"),
            p_value=float("nan"),
            loss_diff_mean=float(d_bar),
            hln_adjusted=hln_adjust,
        )

    dm_stat = d_bar / np.sqrt(var_d_bar)

    if hln_adjust:
        # HLN correction: multiply by sqrt((T + 1 - 2h + h(h-1)/T) / T)
        correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat *= correction
        # Use t distribution with T-1 degrees of freedom
        p_value = 2 * float(stats.t.sf(np.abs(dm_stat), df=T - 1))
    else:
        p_value = 2 * float(stats.norm.sf(np.abs(dm_stat)))

    return DMResult(
        dm_stat=float(dm_stat),
        p_value=float(p_value),
        loss_diff_mean=float(d_bar),
        hln_adjusted=hln_adjust,
    )
