"""Pipeline evaluation: accuracy, DM/CW significance, MCS, cross-arm combinations (Stage 2)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import CombinationContender, PipelineSpec, contender_names

# combination methods that need realised values (estimated weights), and their lag
_ESTIMATED = {
    "inverse_mspe", "dmspe", "best_n", "bates_granger",
    "granger_ramanathan", "constrained_ls", "eigenvector", "regularized",
}


def _combine(method: str, frame: pd.DataFrame, y_true: pd.Series, *, horizon: int, **params: Any) -> pd.Series:
    """Dispatch a combination method name to the forecasting combine_* primitives."""
    from macroforecast import forecasting as F

    key = str(method).lower()
    simple = {
        "mean": F.combine_mean, "median": F.combine_median,
        "trimmed_mean": F.combine_trimmed_mean, "winsorized_mean": F.combine_winsorized_mean,
    }
    estimated = {
        "inverse_mspe": F.combine_inverse_mspe, "dmspe": F.combine_dmspe,
        "best_n": F.combine_best_n, "bates_granger": F.combine_bates_granger,
        "granger_ramanathan": F.combine_granger_ramanathan, "constrained_ls": F.combine_constrained_ls,
        "eigenvector": F.combine_eigenvector, "regularized": F.combine_regularized,
    }
    if key in simple:
        return simple[key](frame, **params)
    if key in estimated:
        return estimated[key](frame, y_true, horizon=int(horizon), **params)
    raise ValueError(f"unsupported combination method {method!r}")


def apply_combinations(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Append cross-arm combination contenders to the master forecast frame."""
    if master.empty or not spec.combinations:
        return master
    rows: list[pd.DataFrame] = []
    if "combined" in master.columns:
        base = master.loc[~master["combined"].fillna(False).astype(bool)]
    else:
        base = master
    existing = set(master["contender"].unique()) if "contender" in master.columns else set()
    for combo in spec.combinations:
        if combo.name in existing:
            continue  # already present -> idempotent re-application
        for (target, horizon), group in base.groupby(["target", "horizon"], dropna=False):
            wide = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean").sort_index()
            actual = group.groupby("origin")["actual"].first().reindex(wide.index)
            date = group.groupby("origin")["date"].first().reindex(wide.index)
            if isinstance(combo.over, (list, tuple)):
                keep = [c for c in combo.over if c in wide.columns]
                if len(keep) < 2:
                    continue
                wide = wide[keep]
            if wide.shape[1] < 2:
                continue
            params = dict(combo.params or {})
            if combo.shrink_to_equal is not None and str(combo.method).lower() in _ESTIMATED:
                params.setdefault("shrink_to_equal", combo.shrink_to_equal)
            if combo.weight_window is not None and str(combo.method).lower() in _ESTIMATED:
                params.setdefault("window", combo.weight_window)
            combined = _combine(combo.method, wide, actual, horizon=int(horizon), **params)
            rows.append(pd.DataFrame({
                "arm": combo.name, "model": combo.name, "contender": combo.name,
                "target": target, "horizon": horizon, "origin": wide.index,
                "date": date.to_numpy(), "prediction": combined.to_numpy(),
                "actual": actual.to_numpy(), "combined": True,
            }))
    if not rows:
        return master
    return pd.concat([master, *rows], ignore_index=True)


def accuracy_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """RMSE, relative-MSE (vs benchmark) and OOS-R2 per (target, horizon, contender)."""
    bench = spec.evaluation.benchmark
    out: list[dict[str, Any]] = []
    for (target, horizon), group in master.groupby(["target", "horizon"], dropna=False):
        # Enforce a COMMON sample: every contender is scored on the same origins
        # (those where all contenders and the realised target are observed), so
        # relative RMSE / OOS-R2 are not biased by ragged coverage. This matches
        # the listwise-deletion sample the MCS uses, keeping the three tables
        # mutually consistent.
        wide = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean")
        actual = group.groupby("origin")["actual"].first().reindex(wide.index)
        common = actual.notna() & wide.notna().all(axis=1)
        wide_c = wide.loc[common]
        y = actual.loc[common].to_numpy(dtype=float)
        n_common = int(common.sum())
        scored = {}
        for contender in wide_c.columns:
            err = wide_c[contender].to_numpy(dtype=float) - y
            scored[contender] = float(np.mean(err ** 2)) if err.size else np.nan
        bench_present = bench in scored
        bench_mse = scored.get(bench, np.nan)
        for contender, mse in scored.items():
            out.append({
                "target": target, "horizon": horizon, "contender": contender,
                "rmse": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
                "relative_mse": (mse / bench_mse) if (np.isfinite(mse) and np.isfinite(bench_mse) and bench_mse > 0) else np.nan,
                "r2_oos": (1.0 - mse / bench_mse) if (np.isfinite(mse) and np.isfinite(bench_mse) and bench_mse > 0) else np.nan,
                "n_common": n_common,
                "is_benchmark": contender == bench,
                "benchmark_present": bench_present,
            })
    return pd.DataFrame(out)


def significance_table(master: pd.DataFrame, spec: PipelineSpec) -> pd.DataFrame:
    """Diebold-Mariano and Clark-West p-values of each contender vs the benchmark."""
    from macroforecast.tests import clark_west_test, dm_test

    bench = spec.evaluation.benchmark
    use_cw = bool(spec.evaluation.cw_for_nested)
    # Clark-West is only licensed where the benchmark is nested in the contender.
    # Build the set of contenders whose arm declares nesting; CW is emitted only
    # for those. Non-nested contenders get DM only (CW would be an invalid test).
    nested: set[str] = set()
    for a in spec.arms:
        if getattr(a, "nested_in_benchmark", False):
            nested |= set(contender_names(a))
    out: list[dict[str, Any]] = []
    for (target, horizon), group in master.groupby(["target", "horizon"], dropna=False):
        pivot_f = group.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean")
        actual = group.groupby("origin")["actual"].first().reindex(pivot_f.index)
        if bench not in pivot_f.columns:
            continue
        fb = pivot_f[bench]
        for contender in pivot_f.columns:
            if contender == bench:
                continue
            fc = pivot_f[contender]
            common = actual.notna() & fb.notna() & fc.notna()
            if int(common.sum()) < 8:
                continue
            y = actual[common].to_numpy(float)
            loss_b = (fb[common].to_numpy(float) - y) ** 2
            loss_c = (fc[common].to_numpy(float) - y) ** 2
            row: dict[str, Any] = {"target": target, "horizon": horizon, "contender": contender}
            try:
                dm = dm_test(loss_c, loss_b, horizon=int(horizon), input_type="loss")
                row["dm_stat"] = float(dm.statistic); row["dm_p"] = float(dm.p_value)
            except Exception:
                row["dm_stat"] = np.nan; row["dm_p"] = np.nan
            if use_cw and contender in nested:
                try:
                    cw = clark_west_test(loss_b, loss_c, fb[common].to_numpy(float),
                                         fc[common].to_numpy(float), horizon=int(horizon))
                    row["cw_stat"] = float(cw.statistic); row["cw_p"] = float(cw.p_value)
                except Exception:
                    row["cw_stat"] = np.nan; row["cw_p"] = np.nan
            out.append(row)
    return pd.DataFrame(out)


def mcs_table(master: pd.DataFrame, spec: PipelineSpec, *, n_boot: int = 499) -> pd.DataFrame:
    """Model Confidence Set membership per (target, horizon)."""
    from macroforecast.tests import model_confidence_set

    panel = master.copy()
    panel = panel[panel["actual"].notna() & panel["prediction"].notna()]
    if panel.empty:
        return pd.DataFrame()
    panel = panel.assign(
        squared_error=(panel["prediction"].astype(float) - panel["actual"].astype(float)) ** 2,
        model_id=panel["contender"],
    )
    out: list[dict[str, Any]] = []
    for (target, horizon), group in panel.groupby(["target", "horizon"], dropna=False):
        if group["model_id"].nunique() < 2:
            continue
        try:
            res = model_confidence_set(
                group[["origin", "model_id", "squared_error"]],
                loss="squared_error", model="model_id", origin="origin",
                target="target", horizon="horizon", alpha=spec.evaluation.mcs_alpha,
                n_boot=n_boot, random_state=int(spec.seed or 0),
            )
        except Exception:
            continue
        included = set()
        for entry in (res.get("mcs_inclusion") or []):
            included |= set(entry.get("models", []))
        for contender in group["model_id"].unique():
            out.append({
                "target": target, "horizon": horizon, "contender": contender,
                "in_mcs": contender in included,
            })
    return pd.DataFrame(out)


def evaluate(master: pd.DataFrame, spec: PipelineSpec) -> dict[str, pd.DataFrame]:
    """Run the full evaluation: combinations -> accuracy + significance + MCS."""
    full = apply_combinations(master, spec)
    return {
        "forecasts": full,
        "accuracy": accuracy_table(full, spec),
        "significance": significance_table(full, spec),
        "mcs": mcs_table(full, spec),
    }
