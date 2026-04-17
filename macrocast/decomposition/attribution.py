"""One-way ANOVA sum-of-squares — the numeric core of Phase 7 attribution.

Kept deliberately tiny; the engine calls this once per (axis, metric) pair
and assembles the per-component shares from the results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnovaResult:
    ss_between: float
    ss_total: float
    n_groups: int
    n_obs: int
    f_stat: float | None
    p_value: float | None


def one_way_anova(values: np.ndarray, groups: np.ndarray) -> AnovaResult:
    """Partition ``values`` by ``groups`` and return one-way ANOVA statistics.

    Parameters
    ----------
    values : 1-D ndarray, float
        The numeric observations (one per variant x horizon, typically).
    groups : 1-D ndarray
        Parallel categorical labels identifying which group each value
        belongs to. Any hashable dtype is fine; internally coerced via
        ``np.unique``.

    Returns
    -------
    AnovaResult
        ``ss_between``, ``ss_total`` (float64), group and observation
        counts, and optional F / p values. Degenerate inputs produce
        ``p_value=None``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    groups = np.asarray(groups)
    if groups.shape != values.shape:
        raise ValueError(
            f"groups shape {groups.shape} != values shape {values.shape}"
        )
    n_obs = int(len(values))
    if n_obs == 0:
        return AnovaResult(0.0, 0.0, 0, 0, None, None)

    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())

    unique_groups = np.unique(groups)
    n_groups = int(len(unique_groups))
    if n_groups < 2:
        return AnovaResult(0.0, ss_total, n_groups, n_obs, None, None)

    ss_between = 0.0
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        mean_g = float(values[mask].mean())
        ss_between += n_g * (mean_g - grand_mean) ** 2
        ss_within += float(((values[mask] - mean_g) ** 2).sum())

    df_between = n_groups - 1
    df_within = n_obs - n_groups
    if df_within <= 0 or ss_within <= 0 or ss_total <= 0:
        return AnovaResult(ss_between, ss_total, n_groups, n_obs, None, None)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within <= 0:
        return AnovaResult(ss_between, ss_total, n_groups, n_obs, None, None)

    f_stat = ms_between / ms_within

    try:
        from scipy.stats import f as _f_dist

        p_value = float(1.0 - _f_dist.cdf(f_stat, df_between, df_within))
    except Exception:  # pragma: no cover — scipy is a core dep but be safe
        p_value = None

    return AnovaResult(ss_between, ss_total, n_groups, n_obs, float(f_stat), p_value)


__all__ = ["AnovaResult", "one_way_anova"]
