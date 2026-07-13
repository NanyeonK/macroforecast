"""Run Hounyo-Li (2026) full Table 2 on the labeled author-oracle surface.

This is a replication-runner surface, not a package feature.  It ports the
author macro MATLAB row geometry and leaky block standardization used by the
per-target ``*_results/*_linear.m`` scripts to the three Table 2 macro targets.
Each target uses the author-shipped target workbook and target-specific
``Macrodataset.xls`` predictor panel.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf  # noqa: E402
from macroforecast.model_selection import explicit_folds, grid, select_params  # noqa: E402
from macroforecast.models.linear import pcr, pls, supervised_pca  # noqa: E402

from scripts.replication.hounyo_li_2026_pipeline.build_data import (  # noqa: E402
    build_data,
    target_panel_csv,
    target_series_csv,
)


T_ADJ = 240
HORIZONS = (1, 6, 12, 24)
METHODS = ("PCA", "SPCA", "sPCA", "SsPCA", "PLS")
SUPERVISED_METHODS = frozenset({"SPCA", "SsPCA"})
TARGETS: dict[str, dict[str, str]] = {
    "inflation": {"column": "CPIAUCSL", "label": "Inflation"},
    "ip_growth": {"column": "INDPRO", "label": "IP growth"},
    "unemployment": {"column": "UNRATE", "label": "Unemployment"},
}
HORIZON_START_DATES = {
    1: "1973-03-01",
    6: "1972-10-01",
    12: "1972-04-01",
    24: "1971-04-01",
}
K_GRID = tuple(range(1, 11))
QN_GRID = tuple(range(18, 109, 6))
CONTROL_COL = "__author_target_lag0__"
SUPERVISED_CHUNK_SIZE = 4

PAPER_TABLE2_PC: dict[tuple[str, int, str], float] = {
    ("inflation", 1, "PCA"): 0.970,
    ("inflation", 1, "SPCA"): 0.823,
    ("inflation", 1, "sPCA"): 0.768,
    ("inflation", 1, "SsPCA"): 0.738,
    ("inflation", 1, "PLS"): 0.861,
    ("inflation", 6, "PCA"): 0.928,
    ("inflation", 6, "SPCA"): 0.912,
    ("inflation", 6, "sPCA"): 0.855,
    ("inflation", 6, "SsPCA"): 0.848,
    ("inflation", 6, "PLS"): 1.082,
    ("inflation", 12, "PCA"): 1.076,
    ("inflation", 12, "SPCA"): 1.049,
    ("inflation", 12, "sPCA"): 0.959,
    ("inflation", 12, "SsPCA"): 0.983,
    ("inflation", 12, "PLS"): 1.208,
    ("inflation", 24, "PCA"): 0.987,
    ("inflation", 24, "SPCA"): 0.953,
    ("inflation", 24, "sPCA"): 0.902,
    ("inflation", 24, "SsPCA"): 0.858,
    ("inflation", 24, "PLS"): 1.180,
    ("ip_growth", 1, "PCA"): 1.148,
    ("ip_growth", 1, "SPCA"): 0.902,
    ("ip_growth", 1, "sPCA"): 1.071,
    ("ip_growth", 1, "SsPCA"): 0.844,
    ("ip_growth", 1, "PLS"): 1.219,
    ("ip_growth", 6, "PCA"): 0.930,
    ("ip_growth", 6, "SPCA"): 0.903,
    ("ip_growth", 6, "sPCA"): 0.923,
    ("ip_growth", 6, "SsPCA"): 0.886,
    ("ip_growth", 6, "PLS"): 0.948,
    ("ip_growth", 12, "PCA"): 1.025,
    ("ip_growth", 12, "SPCA"): 1.017,
    ("ip_growth", 12, "sPCA"): 0.972,
    ("ip_growth", 12, "SsPCA"): 0.984,
    ("ip_growth", 12, "PLS"): 1.054,
    ("ip_growth", 24, "PCA"): 1.107,
    ("ip_growth", 24, "SPCA"): 1.055,
    ("ip_growth", 24, "sPCA"): 1.045,
    ("ip_growth", 24, "SsPCA"): 1.000,
    ("ip_growth", 24, "PLS"): 1.149,
    ("unemployment", 1, "PCA"): 1.644,
    ("unemployment", 1, "SPCA"): 1.628,
    ("unemployment", 1, "sPCA"): 1.654,
    ("unemployment", 1, "SsPCA"): 1.411,
    ("unemployment", 1, "PLS"): 1.698,
    ("unemployment", 6, "PCA"): 0.825,
    ("unemployment", 6, "SPCA"): 0.806,
    ("unemployment", 6, "sPCA"): 0.798,
    ("unemployment", 6, "SsPCA"): 0.766,
    ("unemployment", 6, "PLS"): 0.831,
    ("unemployment", 12, "PCA"): 0.849,
    ("unemployment", 12, "SPCA"): 0.815,
    ("unemployment", 12, "sPCA"): 0.802,
    ("unemployment", 12, "SsPCA"): 0.778,
    ("unemployment", 12, "PLS"): 0.849,
    ("unemployment", 24, "PCA"): 0.842,
    ("unemployment", 24, "SPCA"): 0.800,
    ("unemployment", 24, "sPCA"): 0.812,
    ("unemployment", 24, "SsPCA"): 0.785,
    ("unemployment", 24, "PLS"): 0.853,
}

COMMON_SUPERVISED_PARAMS: dict[str, Any] = {
    "scale": False,
    "control_columns": (CONTROL_COL,),
    "include_constant": True,
    "drop_control_columns": True,
    "preselect": "none",
    "preselect_stage": "raw_before_standardize",
    "t_threshold": 1.28,
    "elastic_net_alpha": 0.0002,
    "elastic_net_l1_ratio": 0.5,
    "random_state": 0,
    "quadratic_factors": False,
}


def _read_target_source(target_key: str) -> tuple[pd.DataFrame, pd.Series]:
    panel_csv = target_panel_csv(target_key)
    target_csv = target_series_csv(target_key)
    if not panel_csv.exists() or not target_csv.exists():
        build_data()
    panel = pd.read_csv(panel_csv, parse_dates=["date"]).set_index("date").sort_index()
    target = pd.read_csv(target_csv, parse_dates=["date"]).set_index("date").sort_index()
    panel = panel.astype(float)
    target_col = TARGETS[target_key]["column"]
    if target_col not in target.columns:
        raise ValueError(f"{target_csv} did not contain expected {target_col} column")
    if target_col in panel.columns:
        raise ValueError(
            f"{panel_csv} contains target {target_col}; author Table 2 panels exclude the target"
        )
    y = target[target_col].astype(float)
    common = panel.index.intersection(y.index)
    if common.empty:
        raise ValueError(f"{target_key} panel and target have no overlapping dates")
    return panel.loc[common], y.loc[common]


def _series_for_cell(target_key: str, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    panel, target = _read_target_source(target_key)
    start = pd.Timestamp(HORIZON_START_DATES[horizon])
    x_loop = panel.loc[start:"2023-03-01"].copy()
    y_loop = target.loc[start:"2023-03-01"].copy()
    expected = T_ADJ + horizon + 360
    if x_loop.shape[0] != expected or y_loop.shape[0] != expected:
        raise ValueError(
            f"{target_key} h={horizon} author slice expected {expected} rows, "
            f"got predictors={x_loop.shape[0]} target={y_loop.shape[0]}"
        )
    return x_loop.astype(float), y_loop.astype(float)


def _finite_zscore(values: np.ndarray, *, axis: int) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    mean = np.nanmean(out, axis=axis, keepdims=True)
    sd = np.nanstd(out, axis=axis, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (out - mean) / sd
    return np.where(np.isfinite(out), out, 0.0)


def _matlab_movmean(values: np.ndarray, window: int) -> np.ndarray:
    before = window // 2
    after = window - before - 1
    out = np.empty_like(values, dtype=float)
    for i in range(values.size):
        lo = max(0, i - before)
        hi = min(values.size, i + after + 1)
        out[i] = float(values[lo:hi].mean())
    return out


def _author_lag_bic(yt: np.ndarray, max_lag: int = 12) -> int:
    vals: list[float] = []
    for lag in range(1, max_lag + 1):
        lag_matrix = np.column_stack([yt[lag - 1 - j : yt.size - j] for j in range(lag)])
        response = lag_matrix[:, 0]
        rhs = lag_matrix[:, 1:]
        if rhs.shape[1] == 0:
            resid = response - response.mean()
        else:
            design = np.column_stack([np.ones(response.size), rhs])
            beta = np.linalg.lstsq(design, response, rcond=None)[0]
            resid = response - design @ beta
        n_obs = lag_matrix.shape[0]
        k_params = lag * lag_matrix.shape[1]
        vals.append(float(n_obs * np.log(np.mean(resid**2)) + k_params * np.log(n_obs)))
    return int(np.argmin(vals) + 1)


def _ar_fit_forward_backward_no_intercept(y: np.ndarray, lag: int) -> np.ndarray:
    coeffs: list[list[float]] = []
    targets: list[float] = []
    y = np.asarray(y, dtype=float).reshape(-1)
    for series in (y, y[::-1]):
        for t in range(lag, series.size):
            coeffs.append([float(series[t - j]) for j in range(1, lag + 1)])
            targets.append(float(series[t]))
    return np.linalg.lstsq(np.asarray(coeffs), np.asarray(targets), rcond=None)[0]


def _author_ar_one(y_loop: pd.Series, origin_i: int, horizon: int) -> dict[str, Any]:
    y_values = y_loop.to_numpy(dtype=float)
    y_or = y_values[origin_i : origin_i + T_ADJ + horizon]
    yt1 = np.diff(y_or)
    yt = yt1 - _matlab_movmean(yt1, 12)
    yt_standardized = (yt - yt.mean()) / yt.std(ddof=1)
    ytplush = yt_standardized[horizon:]
    wt = yt_standardized[: ytplush.size]
    lag = _author_lag_bic(ytplush)
    phi = _ar_fit_forward_backward_no_intercept(wt, lag)
    pred = float((phi**horizon) @ wt[-lag:])
    actual = float(ytplush[-1])
    return {
        "origin": int(origin_i + 1),
        "oos_date": y_loop.index[origin_i + T_ADJ + horizon - 1],
        "actual": actual,
        "prediction": pred,
        "error2": float((actual - pred) ** 2),
        "lag": lag,
    }


def _author_surface_block(
    x_loop: pd.DataFrame,
    y_loop: pd.Series,
    origin_i: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    x_values = x_loop.to_numpy(dtype=float)
    y_values = y_loop.to_numpy(dtype=float)
    xt = _finite_zscore(x_values[origin_i : origin_i + T_ADJ, :].T, axis=1)
    y_or = y_values[origin_i : origin_i + T_ADJ + horizon]
    y_std = (y_or - y_or.mean()) / y_or.std(ddof=1)
    ytplush = y_std[horizon : horizon + T_ADJ]
    wt = np.vstack([y_std[:T_ADJ], np.ones(T_ADJ, dtype=float)])
    denom = np.sum(xt * xt, axis=1)
    slopes = np.divide(
        xt @ ytplush,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=np.abs(denom) > 0.0,
    )
    scaled_xt = np.where(np.isfinite(xt * slopes[:, None]), xt * slopes[:, None], 0.0)
    return {"xt": xt, "scaled_xt": scaled_xt, "ytplush": ytplush, "wt": wt}


def _author_pcr_like_tune(x_block: np.ndarray, ytplush: np.ndarray, wt: np.ndarray) -> tuple[int, np.ndarray]:
    fold_defs = ((80, 81, 130), (130, 131, 190), (190, 191, 240))
    fold_mse = np.zeros((len(K_GRID), len(fold_defs)), dtype=float)
    for fold_idx, (train_end, val_start, val_end) in enumerate(fold_defs):
        val_end = min(val_end, x_block.shape[1])
        preds = np.zeros((len(K_GRID), val_end - val_start + 1), dtype=float)
        actual = ytplush[val_start - 1 : val_end]
        for o, idx_out_1based in enumerate(range(val_start, val_end + 1)):
            train_last_1based = train_end + o
            train_slice = slice(0, train_last_1based)
            idx_out = idx_out_1based - 1
            x_train = x_block[:, train_slice]
            y_train = ytplush[train_slice]
            wt_train = wt[:, train_slice]
            alpha_w = y_train @ np.linalg.pinv(wt_train)
            y_res = y_train - alpha_w @ wt_train
            u, s, vt = np.linalg.svd(x_train, full_matrices=False)
            usable = min(len(K_GRID), s.size)
            factors = s[:usable, None] * vt[:usable, :]
            denom = np.where(s[:usable] > 0.0, s[:usable] ** 2, np.inf)
            alpha = (y_res @ factors.T) / denom
            factor_out = u[:, :usable].T @ x_block[:, idx_out]
            preds[:usable, o] = np.cumsum(alpha * factor_out) + float(alpha_w @ wt[:, idx_out])
            if usable < len(K_GRID):
                preds[usable:, o] = preds[usable - 1, o]
        fold_mse[:, fold_idx] = np.mean((actual[None, :] - preds) ** 2, axis=1)
    return int(np.argmin(np.mean(fold_mse, axis=1)) + 1), fold_mse


def _pls_x_weights_one_target(x: np.ndarray, y: np.ndarray, *, max_components: int) -> np.ndarray:
    x_work = np.asarray(x, dtype=float).copy()
    y_work = np.asarray(y, dtype=float).reshape(-1).copy()
    weights: list[np.ndarray] = []
    for _ in range(max_components):
        w = x_work.T @ y_work
        norm = float(np.linalg.norm(w))
        if norm <= 1e-14:
            break
        w = w / norm
        score = x_work @ w
        denom = float(score @ score)
        if denom <= 1e-14:
            break
        loading = (x_work.T @ score) / denom
        y_loading = float((y_work @ score) / denom)
        x_work = x_work - np.outer(score, loading)
        y_work = y_work - y_loading * score
        weights.append(w)
    if not weights:
        return np.empty((x.shape[1], 0), dtype=float)
    return np.column_stack(weights)


def _author_pls_tune(x_block: np.ndarray, ytplush: np.ndarray, wt: np.ndarray) -> tuple[int, np.ndarray]:
    fold_defs = ((80, 81, 130), (130, 131, 190), (190, 191, 240))
    fold_mse = np.zeros((len(K_GRID), len(fold_defs)), dtype=float)
    for fold_idx, (train_end, val_start, val_end) in enumerate(fold_defs):
        val_end = min(val_end, x_block.shape[1])
        actual = ytplush[val_start - 1 : val_end]
        preds = np.zeros((len(K_GRID), val_end - val_start + 1), dtype=float)
        for o, idx_out_1based in enumerate(range(val_start, val_end + 1)):
            train_last_1based = train_end + o
            idx_out = idx_out_1based - 1
            x_train = x_block[:, :train_last_1based].T
            y_train = ytplush[:train_last_1based]
            wt_train = wt[:, :train_last_1based].T
            control_coef = np.linalg.pinv(wt_train) @ y_train
            y_res = y_train - wt_train @ control_coef
            usable = min(len(K_GRID), x_train.shape[0], x_train.shape[1])
            weights = _pls_x_weights_one_target(x_train, y_res, max_components=usable)
            usable = min(usable, weights.shape[1])
            control_part = float(wt[:, idx_out] @ control_coef)
            if usable == 0:
                preds[:, o] = control_part
                continue
            x_out = x_block[:, idx_out]
            for k in K_GRID:
                kk = min(k, usable)
                factors = x_train @ weights[:, :kk]
                alpha = np.linalg.pinv(factors) @ y_res
                factor_part = float((x_out @ weights[:, :kk]) @ alpha)
                preds[k - 1, o] = factor_part + control_part
        fold_mse[:, fold_idx] = np.mean((actual[None, :] - preds) ** 2, axis=1)
    return int(np.argmin(np.mean(fold_mse, axis=1)) + 1), fold_mse


def _factor_frames(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_fit = pd.DataFrame(x_block[:, :-1].T, columns=predictors)
    x_test = pd.DataFrame(x_block[:, -1:].T, columns=predictors)
    x_fit[CONTROL_COL] = wt[0, :-1]
    x_test[CONTROL_COL] = [wt[0, -1]]
    y_fit = pd.Series(ytplush[:-1], index=x_fit.index)
    return x_fit, y_fit, x_test


def _forecast_with_pcr(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
    k: int,
) -> float:
    x_fit, y_fit, x_test = _factor_frames(x_block, ytplush, wt, predictors)
    fit = pcr(
        x_fit,
        y_fit,
        n_components=k,
        control_columns=(CONTROL_COL,),
        include_constant=True,
        drop_control_columns=True,
        standardize=False,
        nan_policy="zero_after_standardize",
        quadratic_factors=False,
    )
    return float(fit.estimator.predict(x_test)[0])


def _forecast_with_pls(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
    k: int,
) -> float:
    x_fit, y_fit, x_test = _factor_frames(x_block, ytplush, wt, predictors)
    fit = pls(
        x_fit,
        y_fit,
        n_components=k,
        scale=False,
        control_columns=(CONTROL_COL,),
        include_constant=True,
        drop_control_columns=True,
        quadratic_factors=False,
        score_projection="x_weights_raw",
    )
    return float(fit.estimator.predict(x_test)[0])


def _supervised_search_spec() -> Any:
    return grid(
        {"n_selected": QN_GRID, "n_components": K_GRID},
        validation_splitter=explicit_folds([80, 130, 190, 240], within_fold="expanding"),
        score_aggregation="mean_fold",
    )


def _bounded_identity_search_spec() -> Any:
    return grid(
        {"n_selected": QN_GRID, "n_components": K_GRID},
        validation_splitter=explicit_folds([80, 90, 100], within_fold="expanding"),
        score_aggregation="mean_fold",
    )


def _supervised_model_spec(*, grouped: bool) -> Any:
    spec = mf.models.get_model("supervised_pca")
    if grouped:
        return spec
    return replace(spec, name="supervised_pca_prefix_disabled", prefix_search=None)


def _select_supervised_params(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
    *,
    grouped: bool = True,
    bounded: bool = False,
) -> tuple[dict[str, int], pd.DataFrame]:
    x_frame = pd.DataFrame(x_block.T, columns=predictors)
    x_frame[CONTROL_COL] = wt[0]
    y = pd.Series(ytplush, index=x_frame.index)
    result = select_params(
        model=_supervised_model_spec(grouped=grouped),
        X=x_frame,
        y=y,
        search=_bounded_identity_search_spec() if bounded else _supervised_search_spec(),
        metric="mse",
        fixed_params=COMMON_SUPERVISED_PARAMS,
    )
    return {
        "n_components": int(result.best_params["n_components"]),
        "n_selected": int(result.best_params["n_selected"]),
    }, result.trials


def _forecast_with_supervised(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
    params: dict[str, int],
) -> float:
    x_fit, y_fit, x_test = _factor_frames(x_block, ytplush, wt, predictors)
    fit = supervised_pca(
        x_fit,
        y_fit,
        n_components=params["n_components"],
        n_selected=params["n_selected"],
        **COMMON_SUPERVISED_PARAMS,
    )
    return float(fit.estimator.predict(x_test)[0])


def _supervised_record(
    x_loop: pd.DataFrame,
    y_loop: pd.Series,
    predictors: list[str],
    target_key: str,
    horizon: int,
    method: str,
    origin_i: int,
) -> dict[str, Any]:
    block = _author_surface_block(x_loop, y_loop, origin_i, horizon)
    x_block = block["scaled_xt"] if method == "SsPCA" else block["xt"]
    params, _trials = _select_supervised_params(
        x_block,
        block["ytplush"],
        block["wt"],
        predictors,
        grouped=True,
    )
    prediction = _forecast_with_supervised(
        x_block,
        block["ytplush"],
        block["wt"],
        predictors,
        params,
    )
    actual = float(block["ytplush"][-1])
    return {
        "target": target_key,
        "horizon": horizon,
        "method": method,
        "origin": origin_i + 1,
        "oos_date": y_loop.index[origin_i + T_ADJ + horizon - 1],
        "actual": actual,
        "prediction": float(prediction),
        "error2": float((actual - prediction) ** 2),
        **params,
    }


def _cell_file(result_store: Path, target_key: str, horizon: int, method: str) -> Path:
    return result_store / "cells" / f"{target_key}_h{horizon}_{method}.csv"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _origin_chunks(origin_indices: list[int], chunk_size: int) -> list[tuple[int, ...]]:
    return [
        tuple(origin_indices[start : start + chunk_size])
        for start in range(0, len(origin_indices), chunk_size)
    ]


def _run_supervised_chunk(args: tuple[str, int, str, str, tuple[int, ...]]) -> dict[str, Any]:
    target_key, horizon, method, result_store_raw, origin_indices = args
    start = time.perf_counter()
    x_loop, y_loop = _series_for_cell(target_key, horizon)
    predictors = list(x_loop.columns)
    rows = [
        _supervised_record(x_loop, y_loop, predictors, target_key, horizon, method, origin_i)
        for origin_i in origin_indices
    ]
    return {
        "kind": "supervised_chunk",
        "target": target_key,
        "horizon": horizon,
        "method": method,
        "rows": len(rows),
        "records": rows,
        "path": str(_cell_file(Path(result_store_raw), target_key, horizon, method)),
        "runtime_seconds": time.perf_counter() - start,
    }


def _run_cell(args: tuple[str, int, str, str]) -> dict[str, Any]:
    target_key, horizon, method, result_store_raw = args
    result_store = Path(result_store_raw)
    out_path = _cell_file(result_store, target_key, horizon, method)
    start = time.perf_counter()
    x_loop, y_loop = _series_for_cell(target_key, horizon)
    predictors = list(x_loop.columns)
    n_origins = y_loop.shape[0] - T_ADJ - horizon + 1
    completed: set[int] = set()
    rows: list[dict[str, Any]] = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict(orient="records")
        completed = {int(row["origin"]) for row in rows}
    for origin_i in range(n_origins):
        origin = origin_i + 1
        if origin in completed:
            continue
        if method == "AR_BIC":
            record = _author_ar_one(y_loop, origin_i, horizon)
            record.update({"target": target_key, "horizon": horizon, "method": method})
        else:
            block = _author_surface_block(x_loop, y_loop, origin_i, horizon)
            x_block = block["scaled_xt"] if method in {"sPCA", "SsPCA"} else block["xt"]
            if method in {"PCA", "sPCA"}:
                k, _curves = _author_pcr_like_tune(x_block, block["ytplush"], block["wt"])
                prediction = _forecast_with_pcr(x_block, block["ytplush"], block["wt"], predictors, k)
                params = {"n_components": k}
            elif method == "PLS":
                k, _curves = _author_pls_tune(x_block, block["ytplush"], block["wt"])
                prediction = _forecast_with_pls(x_block, block["ytplush"], block["wt"], predictors, k)
                params = {"n_components": k}
            elif method in SUPERVISED_METHODS:
                record = _supervised_record(
                    x_loop,
                    y_loop,
                    predictors,
                    target_key,
                    horizon,
                    method,
                    origin_i,
                )
                rows.append(record)
                _atomic_write_csv(pd.DataFrame(rows).sort_values("origin"), out_path)
                continue
            else:
                raise ValueError(f"unknown method {method}")
            actual = float(block["ytplush"][-1])
            record = {
                "target": target_key,
                "horizon": horizon,
                "method": method,
                "origin": origin,
                "oos_date": y_loop.index[origin_i + T_ADJ + horizon - 1],
                "actual": actual,
                "prediction": float(prediction),
                "error2": float((actual - prediction) ** 2),
                **params,
            }
        rows.append(record)
        if method in SUPERVISED_METHODS or origin % 25 == 0 or origin == n_origins:
            _atomic_write_csv(pd.DataFrame(rows).sort_values("origin"), out_path)
    _atomic_write_csv(pd.DataFrame(rows).sort_values("origin"), out_path)
    return {
        "target": target_key,
        "horizon": horizon,
        "method": method,
        "rows": len(rows),
        "path": str(out_path),
        "runtime_seconds": time.perf_counter() - start,
    }


def run_kprefix_identity_gate(
    result_store: Path,
    *,
    n_origins: int = 5,
    target_key: str = "inflation",
    horizon: int = 1,
) -> dict[str, Any]:
    start = time.perf_counter()
    x_loop, y_loop = _series_for_cell(target_key, horizon)
    predictors = list(x_loop.columns)
    rows: list[dict[str, Any]] = []
    max_score_diff = 0.0
    max_forecast_diff = 0.0
    selected_match = True
    forecast_match = True
    for origin_i in range(n_origins):
        block = _author_surface_block(x_loop, y_loop, origin_i, horizon)
        # Author SsPCA = SPCA recursion applied to author-built scaleXs.
        x_block = block["scaled_xt"]
        enabled_params, enabled_trials = _select_supervised_params(
            x_block,
            block["ytplush"],
            block["wt"],
            predictors,
            grouped=True,
            bounded=True,
        )
        disabled_params, disabled_trials = _select_supervised_params(
            x_block,
            block["ytplush"],
            block["wt"],
            predictors,
            grouped=False,
            bounded=True,
        )
        enabled_sorted = enabled_trials.sort_values("trial").reset_index(drop=True)
        disabled_sorted = disabled_trials.sort_values("trial").reset_index(drop=True)
        finite = enabled_sorted["score"].notna() & disabled_sorted["score"].notna()
        if finite.any():
            score_diff = float(
                (enabled_sorted.loc[finite, "score"] - disabled_sorted.loc[finite, "score"])
                .abs()
                .max()
            )
        else:
            score_diff = 0.0
        max_score_diff = max(max_score_diff, score_diff)
        enabled_forecast = _forecast_with_supervised(
            x_block,
            block["ytplush"],
            block["wt"],
            predictors,
            enabled_params,
        )
        disabled_forecast = _forecast_with_supervised(
            x_block,
            block["ytplush"],
            block["wt"],
            predictors,
            disabled_params,
        )
        forecast_diff = abs(float(enabled_forecast) - float(disabled_forecast))
        max_forecast_diff = max(max_forecast_diff, forecast_diff)
        params_equal = enabled_params == disabled_params
        selected_match = selected_match and params_equal
        forecast_equal = forecast_diff <= 1e-12
        forecast_match = forecast_match and forecast_equal
        rows.append(
            {
                "origin": origin_i + 1,
                "enabled_K": enabled_params["n_components"],
                "enabled_qN": enabled_params["n_selected"],
                "disabled_K": disabled_params["n_components"],
                "disabled_qN": disabled_params["n_selected"],
                "selected_match": params_equal,
                "enabled_forecast": enabled_forecast,
                "disabled_forecast": disabled_forecast,
                "forecast_abs_diff": forecast_diff,
                "max_score_abs_diff": score_diff,
            }
        )
    proof = pd.DataFrame(rows)
    proof_path = result_store / "kprefix_identity_gate.csv"
    _atomic_write_csv(proof, proof_path)
    payload = {
        "gate": "P4 K-prefix identity",
        "model_surface": "author SsPCA scaleXs + package supervised_pca recursion",
        "target": target_key,
        "horizon": horizon,
        "n_origins": n_origins,
        "validation": "bounded author expanding folds [80,90,100]",
        "grid": {"K": list(K_GRID), "qN": list(QN_GRID)},
        "selected_identical": bool(selected_match),
        "forecasts_identical": bool(forecast_match),
        "max_score_abs_diff": max_score_diff,
        "max_forecast_abs_diff": max_forecast_diff,
        "passed": bool(selected_match and forecast_match and max_score_diff <= 1e-12),
        "proof_csv": str(proof_path),
        "runtime_seconds": time.perf_counter() - start,
    }
    (result_store / "kprefix_identity_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _resolve_n_jobs(value: str) -> int:
    if value == "auto":
        return os.cpu_count() or 1
    return max(1, int(value))


def _resolve_targets(value: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(TARGETS)
    targets = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown target(s): {unknown}; expected one of {sorted(TARGETS)}")
    if not targets:
        raise ValueError("--targets must be 'all' or a comma-separated target list")
    return targets


def run_full_table2(result_store: Path, *, n_jobs: int, run_targets: tuple[str, ...]) -> dict[str, Any]:
    start = time.perf_counter()
    result_store.mkdir(parents=True, exist_ok=True)
    cell_summaries: list[dict[str, Any]] = []
    supervised_rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    supervised_pending: dict[tuple[str, int, str], int] = {}
    supervised_started: dict[tuple[str, int, str], float] = {}
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures: dict[cf.Future[dict[str, Any]], tuple[str, tuple[str, int, str] | None]] = {}
        for target_key in run_targets:
            for horizon in HORIZONS:
                for method in ("AR_BIC",) + METHODS:
                    if method not in SUPERVISED_METHODS:
                        task = (target_key, horizon, method, str(result_store))
                        futures[executor.submit(_run_cell, task)] = ("cell", None)
                        continue

                    key = (target_key, horizon, method)
                    out_path = _cell_file(result_store, target_key, horizon, method)
                    rows: list[dict[str, Any]] = []
                    completed: set[int] = set()
                    if out_path.exists():
                        existing = pd.read_csv(out_path)
                        rows = existing.to_dict(orient="records")
                        completed = {int(row["origin"]) for row in rows}
                    _x_loop, y_loop = _series_for_cell(target_key, horizon)
                    n_origins = y_loop.shape[0] - T_ADJ - horizon + 1
                    missing = [
                        origin_i
                        for origin_i in range(n_origins)
                        if origin_i + 1 not in completed
                    ]
                    supervised_rows[key] = rows
                    supervised_pending[key] = 0
                    supervised_started[key] = time.perf_counter()
                    if not missing:
                        summary = {
                            "target": target_key,
                            "horizon": horizon,
                            "method": method,
                            "rows": len(rows),
                            "path": str(out_path),
                            "runtime_seconds": 0.0,
                        }
                        cell_summaries.append(summary)
                        print(
                            "completed {target} h={horizon} {method} rows={rows} seconds={runtime_seconds:.1f}".format(
                                **summary
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        continue

                    for chunk in _origin_chunks(missing, SUPERVISED_CHUNK_SIZE):
                        task = (target_key, horizon, method, str(result_store), chunk)
                        futures[executor.submit(_run_supervised_chunk, task)] = (
                            "supervised",
                            key,
                        )
                        supervised_pending[key] += 1

        for future in cf.as_completed(futures):
            kind, key = futures[future]
            summary = future.result()
            if kind == "cell":
                cell_summaries.append(summary)
                print(
                    "completed {target} h={horizon} {method} rows={rows} seconds={runtime_seconds:.1f}".format(
                        **summary
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if key is None:
                raise RuntimeError("supervised future missing cell key")
            target_key, horizon, method = key
            rows = supervised_rows[key]
            rows.extend(summary["records"])
            out_path = _cell_file(result_store, target_key, horizon, method)
            _atomic_write_csv(pd.DataFrame(rows).sort_values("origin"), out_path)
            supervised_pending[key] -= 1
            if supervised_pending[key] == 0:
                cell_summary = {
                    "target": target_key,
                    "horizon": horizon,
                    "method": method,
                    "rows": len(rows),
                    "path": str(out_path),
                    "runtime_seconds": time.perf_counter() - supervised_started[key],
                }
                cell_summaries.append(cell_summary)
                print(
                    "completed {target} h={horizon} {method} rows={rows} seconds={runtime_seconds:.1f}".format(
                        **cell_summary
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    parity_rows: list[dict[str, Any]] = []
    forecast_frames: list[pd.DataFrame] = []
    for target_key in TARGETS:
        for horizon in HORIZONS:
            ar_path = _cell_file(result_store, target_key, horizon, "AR_BIC")
            ar = pd.read_csv(ar_path)
            ar_mse = float(ar["error2"].mean())
            for method in METHODS:
                path = _cell_file(result_store, target_key, horizon, method)
                frame = pd.read_csv(path)
                forecast_frames.append(frame)
                ratio = float(frame["error2"].mean() / ar_mse)
                paper = PAPER_TABLE2_PC[(target_key, horizon, method)]
                delta = ratio - paper
                parity_rows.append(
                    {
                        "target": target_key,
                        "target_label": TARGETS[target_key]["label"],
                        "horizon": horizon,
                        "method": method,
                        "reproduced_ratio": ratio,
                        "paper_table2_ratio": paper,
                        "delta": delta,
                        "abs_delta": abs(delta),
                        "within_0p03": abs(delta) <= 0.03,
                        "verdict": "PASS" if abs(delta) <= 0.03 else "MISS",
                        "denominator": "AR_BIC",
                        "ar_mse": ar_mse,
                    }
                )
    parity = pd.DataFrame(parity_rows)
    parity_path = result_store / "author_oracle_table2_parity.csv"
    _atomic_write_csv(parity, parity_path)
    forecasts_path = result_store / "author_oracle_table2_forecasts.csv"
    _atomic_write_csv(pd.concat(forecast_frames, ignore_index=True), forecasts_path)
    elapsed = time.perf_counter() - start
    passed = int(parity["within_0p03"].sum())
    payload = {
        "scope": "Full Hounyo-Li Table 2 PC/no-threshold author-oracle grid",
        "surface": "author_oracle",
        "targets": list(TARGETS),
        "run_targets": list(run_targets),
        "horizons": list(HORIZONS),
        "methods": list(METHODS),
        "denominator": "AR_BIC",
        "paper_table2_column": "PC",
        "n_cells": int(len(parity)),
        "within_0p03": passed,
        "misses": int(len(parity) - passed),
        "reproduces": bool(passed == len(parity)),
        "runtime_seconds": elapsed,
        "n_jobs": n_jobs,
        "logical_cores": os.cpu_count(),
        "core_utilization_note": "ProcessPoolExecutor cell-level parallelism; one BLAS thread per worker.",
        "kprefix_speedup": "enabled through ModelSpec.prefix_search for supervised_pca cells",
        "outputs": {
            "parity": str(parity_path),
            "forecasts": str(forecasts_path),
            "cells": str(result_store / "cells"),
        },
        "cell_summaries": sorted(
            cell_summaries,
            key=lambda row: (row["target"], row["horizon"], row["method"]),
        ),
        "provenance_limits": [
            "Target-specific IPGrowth_results/ and Unemployment_results/ MATLAB scripts and workbooks are present and used for those targets.",
            "SsPCA author-oracle cells use author-built scaleXs followed by package supervised_pca recursion, matching SsPCA_emp002.m after slope scaling.",
        ],
    }
    (result_store / "author_oracle_table2_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-store", default="runs/hl2026_table2_author_oracle")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument(
        "--targets",
        default="all",
        help="Targets to run: all or comma-separated subset such as ip_growth,unemployment.",
    )
    parser.add_argument("--identity-origins", type=int, default=5)
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--only-identity", action="store_true")
    parser.add_argument(
        "--parallel-cell-timeout",
        default="none",
        help="Accepted for command provenance; this runner does not impose a parent-side timeout.",
    )
    args = parser.parse_args()
    result_store = REPO_ROOT / args.result_store
    result_store.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "result_store": str(result_store),
        "parallel_cell_timeout": args.parallel_cell_timeout,
    }
    if not args.skip_identity:
        gate = run_kprefix_identity_gate(result_store, n_origins=args.identity_origins)
        output["identity_gate"] = gate
        if not gate["passed"]:
            print(json.dumps(output, indent=2, sort_keys=True, default=str))
            raise SystemExit(2)
    if not args.only_identity:
        output["table2"] = run_full_table2(
            result_store,
            n_jobs=_resolve_n_jobs(args.n_jobs),
            run_targets=_resolve_targets(args.targets),
        )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
