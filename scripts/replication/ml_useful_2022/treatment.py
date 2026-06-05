"""ML-Useful treatment-effect regression (Eq. 11): pseudo-R2 ~ ML-feature dummies.

Given a pipeline forecast panel (one row per arm/target/horizon/origin) and the
arms' feature flags, compute the per-forecast pseudo-out-of-sample R2 and regress it
on the ML-feature dummies with (target, horizon, origin) fixed effects, using a
Newey-West HAC covariance for inference. The coefficient on each feature dummy is the
marginal pseudo-R2 attributable to that feature (the paper's alpha_F).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pseudo_r2(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-(target,horizon,origin,arm) pseudo-R2 = 1 - e^2 / Var_h(y).

    The denominator is the per-(target,horizon) variance of the realised target
    about its mean, matching R2_{t,h,v,m} = 1 - e^2 / ((1/T) sum (y - ybar)^2).
    """
    df = panel.dropna(subset=["prediction", "actual"]).copy()
    df["sq_err"] = (df["prediction"].astype(float) - df["actual"].astype(float)) ** 2
    denom = (
        df.groupby(["target", "horizon"])["actual"]
        .transform(lambda s: float(np.mean((s.astype(float) - s.astype(float).mean()) ** 2)))
    )
    df["pseudo_r2"] = 1.0 - df["sq_err"] / denom.replace(0.0, np.nan)
    return df


def treatment_effects(
    panel: pd.DataFrame,
    arms: list[Any],
    *,
    features: tuple[str, ...] = ("nonlinear",),
) -> dict[str, Any]:
    """Estimate alpha_F for the requested feature dummies via within-(target,horizon,
    origin) demeaning + Newey-West HAC OLS. ``features`` are binary flags on the arms."""
    from macroforecast.data_analysis import newey_west

    flags = {a.name: dict(a.metadata) for a in arms}
    df = _pseudo_r2(panel)
    df = df[df["pseudo_r2"].notna()].copy()
    # attach feature dummies from the arm metadata (contender == arm here)
    for feat in features:
        df[feat] = df["contender"].map(lambda c: float(bool(flags.get(c, {}).get(feat, False))))

    # (target, horizon, origin) fixed effects via within-group demeaning
    fe_keys = ["target", "horizon", "origin"]
    cols = ["pseudo_r2", *features]
    demeaned = df[cols] - df.groupby(fe_keys)[cols].transform("mean")
    y = demeaned["pseudo_r2"].to_numpy(dtype=float)
    X = demeaned[list(features)].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if X.shape[0] <= X.shape[1] + 1:
        return {"alpha": {}, "n_obs": int(X.shape[0]), "note": "insufficient rows"}
    res = newey_west(pd.DataFrame(X, columns=list(features)), y, add_intercept=False)
    alpha = {feat: res["coefficients"][i] for i, feat in enumerate(features)}
    return {"alpha": alpha, "n_obs": int(X.shape[0]), "newey_west": res}
