"""Standalone L6 statistical test functions.

Cycle 29: L6 tests standalone-ization.
Each callable wraps the corresponding runtime primitive from
``macroforecast.core.runtime`` to preserve bit-exact results with
the recipe-path dispatch.

Import pattern follows C28 (linear.py): runtime helpers are imported
lazily inside each function body to avoid circular imports and keep
the module self-contained at definition time.

Private helpers (_diebold_mariano_test, _harvey_newbold_test,
_long_run_variance, _normal_two_sided_p)
are module-private in runtime.py (``_`` prefix) but accessible to
internal package modules.  This is intentional: the standalone
functions are part of the same package and share the runtime's
numerical primitives.

Pairwise equal-predictive tests (L6.A):
    dm_test, gw_test, dmp_test, hn_test

Nested model tests (L6.B):
    cw_test, enc_new_test, enc_t_test
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private validation helper
# ---------------------------------------------------------------------------

def _check_losses(
    *arrays: np.ndarray,
    names: tuple[str, ...] | None = None,
) -> None:
    """Validate 1-D, same-length, non-empty arrays.

    Raises
    ------
    ValueError
        When any array is not 1-D, arrays differ in length, or all arrays
        are empty.
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
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMTestResult:
    """Result of :func:`dm_test`.

    Attributes
    ----------
    stat :
        DM test statistic (None when n < 3 or variance <= 0).
    pvalue :
        Two-sided p-value (None when stat is None).
    decision :
        True when reject H0 at 5% significance.
    alternative :
        Always ``"two_sided"``.
    correction_method :
        ``"hln_nw"`` (with HLN correction) or ``"nw"`` (without).
    n_obs :
        Number of finite observations used.
    horizon :
        Forecast horizon supplied by caller.
    hln_correction :
        Whether the Harvey-Leybourne-Newbold small-sample correction
        was applied.
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int
    hln_correction: bool

    def summary(self) -> str:
        """Return a human-readable text summary of the DM test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "                    Diebold-Mariano Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"HLN correction:                                  {self.hln_correction}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     two_sided\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: H0: equal predictive ability. Reject implies model A and model B differ."
        )


@dataclass(frozen=True)
class GWTestResult:
    """Result of :func:`gw_test`.

    Identical fields to :class:`DMTestResult`. Class name and summary
    header distinguish the Giacomini-White framing.
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int
    hln_correction: bool

    def summary(self) -> str:
        """Return a human-readable text summary of the GW test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "                    Giacomini-White Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"HLN correction:                                  {self.hln_correction}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     two_sided\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: H0: equal predictive ability. Reject implies model A and model B differ."
        )


@dataclass(frozen=True)
class DMPTestResult:
    """Result of :func:`dmp_test`.

    Attributes
    ----------
    stat :
        DMP test statistic.
    pvalue :
        Two-sided p-value.
    decision :
        True when reject H0 at 5%.
    alternative :
        Always ``"two_sided"``.
    correction_method :
        Always ``"nw"`` (HAC Newey-West, no HLN).
    n_obs_stacked :
        Total stacked observations across all horizons.
    horizon :
        Always ``None`` (joint across horizons).
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs_stacked: int
    horizon: None

    def summary(self) -> str:
        """Return a human-readable text summary of the DMP test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "          Diebold-Mariano-Pesaran Multi-Horizon Test Results\n"
            "==============================================================================\n"
            f"N observations (stacked):                        {self.n_obs_stacked}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     two_sided\n"
            "Note: This is a joint test across all horizons.\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: H0: equal predictive ability across all horizons."
        )


@dataclass(frozen=True)
class HNTestResult:
    """Result of :func:`hn_test`.

    Attributes
    ----------
    stat :
        HN test statistic (None when n < 5 or se <= 0).
    pvalue :
        One-sided p-value (None when stat is None).
    decision :
        True when reject H0 at 5%.
    alternative :
        Always ``"one_sided"``.
    correction_method :
        Always ``"hln_nw"``.
    n_obs :
        Number of finite observations.
    horizon :
        Forecast horizon.
    encompassing :
        Always ``"a_over_b"`` (H0: A encompasses B).
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int
    encompassing: str

    def summary(self) -> str:
        """Return a human-readable text summary of the HN test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "              Harvey-Newbold Encompassing Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     one_sided\n"
            f"Encompassing direction:                          {self.encompassing}\n"
            "Note: H0: forecast A encompasses forecast B.\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: One-sided: reject implies combining forecasts improves accuracy."
        )


@dataclass(frozen=True)
class CWTestResult:
    """Result of :func:`cw_test`.

    Attributes
    ----------
    stat :
        CW test statistic.
    pvalue :
        One-sided p-value.
    decision :
        True when reject H0 at 5%.
    alternative :
        Always ``"one_sided"``.
    correction_method :
        Always ``"nw"`` (no HLN for nested tests).
    n_obs :
        Number of finite observations.
    horizon :
        Forecast horizon.
    cw_adjustment :
        Always ``True`` (CW squared-forecast-difference penalty always applied).
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int
    cw_adjustment: bool

    def summary(self) -> str:
        """Return a human-readable text summary of the CW test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "                      Clark-West Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"CW adjustment:                                   {self.cw_adjustment}\n"
            f"Alternative:                                     one_sided\n"
            "Note: H0: small model as accurate as large model.\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: One-sided: reject implies large model significantly improves on small."
        )


@dataclass(frozen=True)
class EncNewTestResult:
    """Result of :func:`enc_new_test`.

    Attributes
    ----------
    stat :
        Enc-New test statistic.
    pvalue :
        One-sided p-value.
    decision :
        True when reject H0 at 5%.
    alternative :
        Always ``"one_sided"``.
    correction_method :
        Always ``"nw"``.
    n_obs :
        Number of finite observations.
    horizon :
        Forecast horizon.
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int

    def summary(self) -> str:
        """Return a human-readable text summary of the Enc-New test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "             Enc-New Forecast Encompassing Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     one_sided\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: H0: small model forecast encompasses large model."
        )


@dataclass(frozen=True)
class EncTTestResult:
    """Result of :func:`enc_t_test`.

    Attributes
    ----------
    stat :
        Enc-T test statistic.
    pvalue :
        One-sided p-value.
    decision :
        True when reject H0 at 5%.
    alternative :
        Always ``"one_sided"``.
    correction_method :
        Always ``"nw"``.
    n_obs :
        Number of finite observations.
    horizon :
        Forecast horizon.
    """

    stat: float | None
    pvalue: float | None
    decision: bool
    alternative: str
    correction_method: str
    n_obs: int
    horizon: int

    def summary(self) -> str:
        """Return a human-readable text summary of the Enc-T test result."""
        stat_str = f"{self.stat:.6f}" if self.stat is not None else "None"
        pvalue_str = f"{self.pvalue:.6f}" if self.pvalue is not None else "None"
        decision_str = "Reject H0" if self.decision else "Fail to reject H0"
        return (
            "==============================================================================\n"
            "               Enc-T Forecast Encompassing Test Results\n"
            "==============================================================================\n"
            f"Horizon:                                         {self.horizon}\n"
            f"N observations:                                  {self.n_obs}\n"
            f"Correction method:                               {self.correction_method}\n"
            f"Alternative:                                     one_sided\n"
            "------------------------------------------------------------------------------\n"
            f"Statistic:                                       {stat_str}\n"
            f"P-value:                                         {pvalue_str}\n"
            f"Decision (5%):                                   {decision_str}\n"
            "==============================================================================\n"
            "Note: H0: small model forecast encompasses large model (Ericsson 1992 t-form)."
        )


# ---------------------------------------------------------------------------
# L6.A: Equal-predictive-ability tests
# ---------------------------------------------------------------------------

def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    correction: Literal["hln", "none"] = "hln",
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> DMTestResult:
    """Diebold-Mariano (1995) equal-predictive-ability test.

    Parameters
    ----------
    loss_a :
        Per-period losses for model A (e.g. squared errors), 1-D array.
    loss_b :
        Per-period losses for model B, same length as ``loss_a``.
    horizon :
        Forecast horizon h >= 1. Controls the Newey-West lag truncation
        and HLN small-sample correction factor.
    correction :
        ``"hln"`` (default): apply Harvey-Leybourne-Newbold (1997)
        small-sample correction.  ``"none"``: raw DM statistic.
    kernel :
        HAC kernel for long-run variance estimation.
        One of ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    DMTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    Calls ``_diebold_mariano_test`` from ``macroforecast.core.runtime``
    to ensure bit-exact results with the recipe-path dispatch.
    Reference: Diebold & Mariano (1995), JBES 13(3): 253-263.
    Harvey, Leybourne & Newbold (1997), IJF 13(2): 281-291.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import dm_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> loss_a = (y - rng.randn(100)) ** 2
    >>> loss_b = (y - rng.randn(100)) ** 2
    >>> result = dm_test(loss_a, loss_b)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _diebold_mariano_test  # lazy import -- avoids circular

    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    _check_losses(loss_a, loss_b, names=("loss_a", "loss_b"))

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    diff = pd.Series(loss_a - loss_b)
    hln = correction == "hln"
    stat, pvalue = _diebold_mariano_test(diff, horizon=horizon, hln=hln, kernel=kernel)

    # Count finite observations (mirrors runtime's clean = diff.dropna())
    n_obs = int(np.isfinite(diff.to_numpy()).sum())
    decision = pvalue is not None and pvalue < 0.05
    correction_method = "hln_nw" if hln else "nw"

    return DMTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="two_sided",
        correction_method=correction_method,
        n_obs=n_obs,
        horizon=int(horizon),
        hln_correction=hln,
    )


def gw_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    correction: Literal["hln", "none"] = "hln",
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> GWTestResult:
    """Giacomini-White (2006) conditional equal-predictive-ability test.

    Parameters
    ----------
    loss_a :
        Per-period losses for model A (e.g. squared errors), 1-D array.
    loss_b :
        Per-period losses for model B, same length as ``loss_a``.
    horizon :
        Forecast horizon h >= 1.
    correction :
        ``"hln"`` (default): apply HLN small-sample correction.
        ``"none"``: raw statistic.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    GWTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    The GW unconditional predictive ability framing is computationally
    identical to DM with HAC standard errors. Both route through the
    same ``_diebold_mariano_test`` primitive in ``macroforecast.core.runtime``.
    Reference: Giacomini & White (2006), Econometrica 74(6): 1545-1578.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import gw_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> loss_a = (y - rng.randn(100)) ** 2
    >>> loss_b = (y - rng.randn(100)) ** 2
    >>> result = gw_test(loss_a, loss_b)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _diebold_mariano_test  # lazy import -- avoids circular

    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    _check_losses(loss_a, loss_b, names=("loss_a", "loss_b"))

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    diff = pd.Series(loss_a - loss_b)
    hln = correction == "hln"
    stat, pvalue = _diebold_mariano_test(diff, horizon=horizon, hln=hln, kernel=kernel)

    n_obs = int(np.isfinite(diff.to_numpy()).sum())
    decision = pvalue is not None and pvalue < 0.05
    correction_method = "hln_nw" if hln else "nw"

    return GWTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="two_sided",
        correction_method=correction_method,
        n_obs=n_obs,
        horizon=int(horizon),
        hln_correction=hln,
    )


def dmp_test(
    loss_differentials: list[np.ndarray] | np.ndarray,
    *,
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> DMPTestResult:
    """Diebold-Mariano-Pesaran joint multi-horizon test.

    Parameters
    ----------
    loss_differentials :
        Either a single 1-D ``np.ndarray`` of pre-stacked loss
        differentials (loss_a - loss_b), or a list of 1-D arrays
        one per horizon.  Empty arrays within the list are skipped.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    DMPTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision``,
        ``n_obs_stacked`` fields and a ``.summary()`` method.

    Notes
    -----
    Calls ``_long_run_variance`` and ``_normal_two_sided_p`` from
    ``macroforecast.core.runtime`` to ensure bit-exact results.
    The DMP statistic is a HAC-adjusted z-test on the stacked mean
    loss differential across all horizons jointly.
    Reference: Pesaran & Timmermann (2007), JoE 137(1): 134-161.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import dmp_test
    >>> rng = np.random.RandomState(42)
    >>> diffs = [rng.randn(50) for _ in range(3)]
    >>> result = dmp_test(diffs)
    >>> result.stat  # doctest: +SKIP
    """
    from scipy import stats as _scipy_stats
    from ..core.runtime import _long_run_variance  # lazy import

    # Stack input
    if isinstance(loss_differentials, list):
        parts = [
            np.asarray(part, dtype=float)
            for part in loss_differentials
            if len(part) > 0
        ]
        stacked = np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        stacked = np.asarray(loss_differentials, dtype=float)

    # Filter to finite
    stacked = stacked[np.isfinite(stacked)]
    n = len(stacked)

    if n < 3:
        return DMPTestResult(
            stat=None,
            pvalue=None,
            decision=False,
            alternative="two_sided",
            correction_method="nw",
            n_obs_stacked=n,
            horizon=None,
        )

    mean_diff = float(stacked.mean())
    lr_var = _long_run_variance(stacked - mean_diff, kernel=kernel)
    se = float(math.sqrt(max(lr_var / n, 1e-12)))
    stat = mean_diff / se if se > 0 else 0.0
    pvalue = float(2 * (1 - _scipy_stats.norm.cdf(abs(stat))))
    decision = bool(pvalue < 0.05)

    return DMPTestResult(
        stat=float(stat),
        pvalue=float(pvalue),
        decision=decision,
        alternative="two_sided",
        correction_method="nw",
        n_obs_stacked=n,
        horizon=None,
    )


def hn_test(
    e_a: np.ndarray,
    e_b: np.ndarray,
    *,
    horizon: int = 1,
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
    small_sample: bool = True,
) -> HNTestResult:
    """Harvey-Leybourne-Newbold (1998) forecast-encompassing test.

    Parameters
    ----------
    e_a :
        Forecast errors for model A (actual - forecast_a), 1-D array.
        NOT squared losses.
    e_b :
        Forecast errors for model B (actual - forecast_b), same length.
    horizon :
        Forecast horizon h >= 1.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.
    small_sample :
        Apply the Harvey-Leybourne-Newbold (1998) small-sample correction
        (Eq. 5). Default ``True``.

    Returns
    -------
    HNTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    Calls ``_harvey_newbold_test`` from ``macroforecast.core.runtime``
    for bit-exact results.
    H0: forecast A encompasses forecast B.
    H1: combining forecasts improves accuracy (one-sided).
    Reference: Harvey, Leybourne & Newbold (1998), JBES 16(2): 254-259.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import hn_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> e_a = y - rng.randn(100)
    >>> e_b = y - rng.randn(100)
    >>> result = hn_test(e_a, e_b)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _harvey_newbold_test  # lazy import

    e_a = np.asarray(e_a, dtype=float)
    e_b = np.asarray(e_b, dtype=float)
    _check_losses(e_a, e_b, names=("e_a", "e_b"))

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    stat, pvalue = _harvey_newbold_test(
        e_a, e_b, horizon=horizon, kernel=kernel, small_sample=small_sample
    )

    # Count finite observations (d = e_a * (e_a - e_b), finite mask)
    d = e_a * (e_a - e_b)
    n_obs = int(np.isfinite(d).sum())
    decision = pvalue is not None and pvalue < 0.05

    return HNTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="one_sided",
        correction_method="hln_nw",
        n_obs=n_obs,
        horizon=int(horizon),
        encompassing="a_over_b",
    )


# ---------------------------------------------------------------------------
# L6.B: Nested model tests
# ---------------------------------------------------------------------------

def cw_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
    f_small: np.ndarray,
    f_large: np.ndarray,
    *,
    horizon: int = 1,
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> CWTestResult:
    """Clark-West (2006/2007) nested-model predictive ability test.

    Parameters
    ----------
    loss_small :
        Squared losses for the small (restricted) model, 1-D array.
    loss_large :
        Squared losses for the large (unrestricted) model, same length.
    f_small :
        Point forecasts for the small model, same length.
    f_large :
        Point forecasts for the large model, same length.
    horizon :
        Forecast horizon h >= 1.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    CWTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    Constructs the CW-adjusted loss differential:
    ``f_value = (loss_small - loss_large) + (f_small - f_large)^2``
    then calls ``_diebold_mariano_test`` from ``macroforecast.core.runtime``
    with ``hln=False`` (no HLN for nested tests).
    The one-sided p-value is ``p_two / 2`` when ``stat > 0``.
    Reference: Clark & West (2007), JoE 138(2): 291-311.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import cw_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> f_s = rng.randn(100)
    >>> f_l = rng.randn(100)
    >>> loss_s = (y - f_s) ** 2
    >>> loss_l = (y - f_l) ** 2
    >>> result = cw_test(loss_s, loss_l, f_s, f_l)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _diebold_mariano_test  # lazy import

    loss_small = np.asarray(loss_small, dtype=float)
    loss_large = np.asarray(loss_large, dtype=float)
    f_small = np.asarray(f_small, dtype=float)
    f_large = np.asarray(f_large, dtype=float)
    _check_losses(
        loss_small, loss_large, f_small, f_large,
        names=("loss_small", "loss_large", "f_small", "f_large"),
    )

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    improvement = loss_small - loss_large
    adjustment = (f_small - f_large) ** 2
    f_value = improvement + adjustment

    diff = pd.Series(f_value)
    n_obs = int(diff.notna().sum())

    if n_obs < 3:
        return CWTestResult(
            stat=None,
            pvalue=None,
            decision=False,
            alternative="one_sided",
            correction_method="nw",
            n_obs=n_obs,
            horizon=int(horizon),
            cw_adjustment=True,
        )

    stat, p_two = _diebold_mariano_test(diff, horizon=horizon, hln=False, kernel=kernel)

    # One-sided correction (H_a: large model improves on small)
    pvalue: float | None
    if p_two is not None and stat is not None and stat > 0:
        pvalue = p_two / 2.0
    else:
        pvalue = p_two

    decision = pvalue is not None and pvalue < 0.05

    return CWTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="one_sided",
        correction_method="nw",
        n_obs=n_obs,
        horizon=int(horizon),
        cw_adjustment=True,
    )


def enc_new_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
    *,
    horizon: int = 1,
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> EncNewTestResult:
    """Enc-New forecast encompassing test (Clark-McCracken 2001).

    Parameters
    ----------
    loss_small :
        Squared losses for the small model, 1-D array.
    loss_large :
        Squared losses for the large model, same length.
    horizon :
        Forecast horizon h >= 1.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    EncNewTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    Uses raw loss improvement ``f_value = loss_small - loss_large``
    (no CW adjustment term), then calls ``_diebold_mariano_test`` with
    ``hln=False``. One-sided p-value as in ``cw_test``.
    Reference: Clark & McCracken (2001), JoE 105(2): 1-28.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import enc_new_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> loss_s = (y - rng.randn(100)) ** 2
    >>> loss_l = (y - rng.randn(100)) ** 2
    >>> result = enc_new_test(loss_s, loss_l)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _diebold_mariano_test  # lazy import

    loss_small = np.asarray(loss_small, dtype=float)
    loss_large = np.asarray(loss_large, dtype=float)
    _check_losses(loss_small, loss_large, names=("loss_small", "loss_large"))

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    f_value = loss_small - loss_large
    diff = pd.Series(f_value)
    n_obs = int(diff.notna().sum())

    if n_obs < 3:
        return EncNewTestResult(
            stat=None,
            pvalue=None,
            decision=False,
            alternative="one_sided",
            correction_method="nw",
            n_obs=n_obs,
            horizon=int(horizon),
        )

    stat, p_two = _diebold_mariano_test(diff, horizon=horizon, hln=False, kernel=kernel)

    # One-sided correction
    pvalue: float | None
    if p_two is not None and stat is not None and stat > 0:
        pvalue = p_two / 2.0
    else:
        pvalue = p_two

    decision = pvalue is not None and pvalue < 0.05

    return EncNewTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="one_sided",
        correction_method="nw",
        n_obs=n_obs,
        horizon=int(horizon),
    )


def enc_t_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
    *,
    horizon: int = 1,
    kernel: Literal["newey_west", "andrews", "parzen"] = "newey_west",
) -> EncTTestResult:
    """Enc-T forecast encompassing test (Ericsson 1992 t-form).

    Parameters
    ----------
    loss_small :
        Squared losses for the small model, 1-D array.
    loss_large :
        Squared losses for the large model, same length.
    horizon :
        Forecast horizon h >= 1.
    kernel :
        HAC kernel: ``"newey_west"`` (default), ``"andrews"``, ``"parzen"``.

    Returns
    -------
    EncTTestResult
        Frozen dataclass with ``stat``, ``pvalue``, ``decision`` fields
        and a ``.summary()`` method.

    Raises
    ------
    ValueError
        When inputs are not 1-D, have different lengths, are empty, or
        ``horizon < 1``.

    Notes
    -----
    Identical computation to ``enc_new_test``. Both enc_new and enc_t
    use raw loss improvement with no CW adjustment and one-sided inference;
    the distinction is the test label (enc_t is the Ericsson 1992 t-form).
    This faithfully mirrors the runtime dispatch, where both fall into
    the same code branch.
    Reference: Ericsson (1992) 'Parameter Constancy, Mean Square Forecast
    Errors, and Measuring Forecast Performance', JoE 52(1-2): 113-153.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import enc_t_test
    >>> rng = np.random.RandomState(42)
    >>> y = rng.randn(100)
    >>> loss_s = (y - rng.randn(100)) ** 2
    >>> loss_l = (y - rng.randn(100)) ** 2
    >>> result = enc_t_test(loss_s, loss_l)
    >>> result.stat  # doctest: +SKIP
    """
    from ..core.runtime import _diebold_mariano_test  # lazy import

    loss_small = np.asarray(loss_small, dtype=float)
    loss_large = np.asarray(loss_large, dtype=float)
    _check_losses(loss_small, loss_large, names=("loss_small", "loss_large"))

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}.")

    f_value = loss_small - loss_large
    diff = pd.Series(f_value)
    n_obs = int(diff.notna().sum())

    if n_obs < 3:
        return EncTTestResult(
            stat=None,
            pvalue=None,
            decision=False,
            alternative="one_sided",
            correction_method="nw",
            n_obs=n_obs,
            horizon=int(horizon),
        )

    stat, p_two = _diebold_mariano_test(diff, horizon=horizon, hln=False, kernel=kernel)

    # One-sided correction
    pvalue: float | None
    if p_two is not None and stat is not None and stat > 0:
        pvalue = p_two / 2.0
    else:
        pvalue = p_two

    decision = pvalue is not None and pvalue < 0.05

    return EncTTestResult(
        stat=stat,
        pvalue=pvalue,
        decision=bool(decision),
        alternative="one_sided",
        correction_method="nw",
        n_obs=n_obs,
        horizon=int(horizon),
    )


__all__ = [
    "DMTestResult",
    "dm_test",
    "GWTestResult",
    "gw_test",
    "DMPTestResult",
    "dmp_test",
    "HNTestResult",
    "hn_test",
    "CWTestResult",
    "cw_test",
    "EncNewTestResult",
    "enc_new_test",
    "EncTTestResult",
    "enc_t_test",
]
