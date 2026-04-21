"""Deterministic component augmentation for 1.4.5.

v1.0 semantics: the selected rule adds deterministic feature columns to the
raw-feature-panel X matrix. Values:

- ``none`` (default)    : no augmentation.
- ``constant_only``     : rely on the model's fit_intercept (sklearn default);
                          we insert a column of 1s so both raw_feature_panel and
                          autoreg paths see the intercept explicitly.
- ``linear_trend``      : add a `t` column (integer time index starting at 0).
- ``monthly_seasonal``  : add 11 monthly dummy columns (JanEffect .. NovEffect;
                          Dec is the reference month).
- ``quarterly_seasonal``: add 3 quarterly dummies (Q1 .. Q3; Q4 is the reference).
- ``break_dummies``     : 0/1 dummy per user-supplied break date (the dummy is 1
                          from the break date onward).

Applied AFTER preprocessing but BEFORE model fit. Both train and predict-time
frames are augmented identically.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


OPERATIONAL_COMPONENTS = frozenset({
    "none",
    "constant_only",
    "linear_trend",
    "monthly_seasonal",
    "quarterly_seasonal",
    "break_dummies",
})


def augment_frame(
    frame: pd.DataFrame,
    component: str,
    *,
    index: pd.DatetimeIndex | None = None,
    break_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append deterministic columns to `frame` per the `component` rule.

    `frame` is a pandas DataFrame of predictor columns with a DatetimeIndex.
    Returns a new DataFrame; the input is not mutated.
    """
    if component == "none" or component is None:
        return frame
    if component not in OPERATIONAL_COMPONENTS:
        raise ValueError(f"unsupported deterministic_components={component!r}; operational: {sorted(OPERATIONAL_COMPONENTS)}")

    out = frame.copy()
    idx = out.index if index is None else index

    if component == "constant_only":
        out["_dc_const"] = 1.0
        return out

    if component == "linear_trend":
        out["_dc_trend"] = np.arange(len(out), dtype=float)
        return out

    if component == "monthly_seasonal":
        if not hasattr(idx, "month"):
            raise ValueError("monthly_seasonal requires a DatetimeIndex")
        months = pd.Series(idx.month, index=out.index)
        for m in range(1, 12):  # skip December (reference)
            out[f"_dc_month_{m:02d}"] = (months == m).astype(float).values
        return out

    if component == "quarterly_seasonal":
        if not hasattr(idx, "quarter"):
            raise ValueError("quarterly_seasonal requires a DatetimeIndex")
        quarters = pd.Series(idx.quarter, index=out.index)
        for q in range(1, 4):  # skip Q4 (reference)
            out[f"_dc_q{q}"] = (quarters == q).astype(float).values
        return out

    if component == "break_dummies":
        if not break_dates:
            raise ValueError("break_dummies requires a non-empty break_dates list (via leaf_config.break_dates)")
        try:
            breaks = [pd.Timestamp(d) for d in break_dates]
        except Exception as exc:
            raise ValueError(f"break_dummies: invalid break_dates entry: {exc}") from exc
        for i, b in enumerate(breaks):
            col = f"_dc_break_{i+1}"
            out[col] = (pd.Series(idx, index=out.index) >= b).astype(float).values
        return out

    # Unreachable due to guard above; kept for safety.
    return out


def augment_array(
    X: np.ndarray,
    component: str,
    *,
    index: pd.DatetimeIndex,
    break_dates: Sequence[str] | None = None,
    n_rows_context: int | None = None,
) -> np.ndarray:
    """Array variant of augment_frame for callers that already have a numpy matrix.

    `index` supplies the dates corresponding to the rows of X. When the index has
    more entries than X (common for predict-time tail slices), the last len(X)
    entries are used.
    """
    if component == "none" or component is None:
        return X
    if component not in OPERATIONAL_COMPONENTS:
        raise ValueError(f"unsupported deterministic_components={component!r}")

    n = X.shape[0]
    if len(index) >= n:
        idx = index[-n:]
    else:
        raise ValueError(f"augment_array: index length {len(index)} < X rows {n}")

    if component == "constant_only":
        col = np.ones((n, 1), dtype=float)
        return np.hstack([X, col])

    if component == "linear_trend":
        col = np.arange(n, dtype=float).reshape(-1, 1)
        return np.hstack([X, col])

    if component == "monthly_seasonal":
        if not hasattr(idx, "month"):
            raise ValueError("monthly_seasonal requires a DatetimeIndex")
        months = np.asarray(idx.month)
        cols = np.stack([(months == m).astype(float) for m in range(1, 12)], axis=1)
        return np.hstack([X, cols])

    if component == "quarterly_seasonal":
        if not hasattr(idx, "quarter"):
            raise ValueError("quarterly_seasonal requires a DatetimeIndex")
        quarters = np.asarray(idx.quarter)
        cols = np.stack([(quarters == q).astype(float) for q in range(1, 4)], axis=1)
        return np.hstack([X, cols])

    if component == "break_dummies":
        if not break_dates:
            raise ValueError("break_dummies requires leaf_config.break_dates (non-empty list)")
        try:
            breaks = [pd.Timestamp(d) for d in break_dates]
        except Exception as exc:
            raise ValueError(f"break_dummies: invalid break_dates entry: {exc}") from exc
        idx_arr = np.asarray(idx)
        cols = np.stack([(idx_arr >= np.datetime64(b)).astype(float) for b in breaks], axis=1)
        return np.hstack([X, cols])

    return X
