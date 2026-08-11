"""Run Hounyo-Li (2026) Table 2 on the honest load_fred_md leak-free surface.

This runner is intentionally separate from the labeled author-oracle runner.
Primary numbers here use macroforecast's native FRED-MD loader, official
FRED-MD transforms, origin-available predictor standardization, and target
availability constraints for every direct h-step forecast.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
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
from macroforecast.models.linear import (  # noqa: E402
    pcr,
    pls,
    scaled_pca,
    supervised_pca,
    supervised_scaled_pca,
)
from macroforecast.models.timeseries import ar_bic  # noqa: E402


T_ADJ = 240
FRED_MD_VINTAGE = "2023-04"
SAMPLE_END = "2023-03-01"
HORIZONS = (1, 6, 12, 24)
METHODS = ("PCA", "SPCA", "sPCA", "SsPCA", "PLS")
SUPERVISED_METHODS = frozenset({"SPCA", "SsPCA"})
TARGETS: dict[str, dict[str, str]] = {
    "inflation": {
        "column": "CPIAUCSL",
        "label": "Inflation",
        "target_transform": "FRED-MD tcode 6: second log difference of CPIAUCSL",
    },
    "ip_growth": {
        "column": "INDPRO",
        "label": "IP growth",
        "target_transform": "FRED-MD tcode 5: log difference of INDPRO",
    },
    "unemployment": {
        "column": "UNRATE",
        "label": "Unemployment",
        "target_transform": "FRED-MD tcode 2: first difference of UNRATE",
    },
}
HORIZON_START_DATES = {
    1: "1973-03-01",
    6: "1972-10-01",
    12: "1972-04-01",
    24: "1971-04-01",
}
K_GRID = tuple(range(1, 11))
QN_GRID = tuple(range(18, 109, 6))
CONTROL_COL = "__target_lag0__"
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


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _prepare_data(result_store: Path, *, vintage: str, force_download: bool = False) -> dict[str, Any]:
    data_dir = result_store / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = mf.data.load_fred_md(vintage=vintage, force=force_download)
    processed = mf.preprocessing.reprocess(
        raw,
        transform="official",
        outliers="none",
        impute="zero",
        standardize="none",
        frame="keep",
    )
    panel = processed.panel.astype(float).sort_index()
    panel.index.name = "date"
    panel_path = data_dir / "fred_md_2023_04_official_transformed_zero_imputed.csv"
    panel.reset_index().to_csv(panel_path, index=False)
    target_path = data_dir / "table2_targets.csv"
    panel[[spec["column"] for spec in TARGETS.values()]].reset_index().to_csv(target_path, index=False)
    transform_codes = raw.metadata.get("transform_codes", {})
    artifact = raw.metadata.get("artifact", {})
    cells = []
    for target_key, spec in TARGETS.items():
        target_col = spec["column"]
        for horizon in HORIZONS:
            x_loop, y_loop = _series_for_cell(panel, target_key, horizon)
            cells.append(
                {
                    "target": target_key,
                    "horizon": horizon,
                    "start": HORIZON_START_DATES[horizon],
                    "end": SAMPLE_END,
                    "rows_in_slice": int(len(y_loop)),
                    "n_origins": int(len(y_loop) - T_ADJ - horizon + 1),
                    "strict_train_rows_per_origin": int(T_ADJ - horizon),
                    "predictor_columns": int(x_loop.shape[1]),
                    "target_column": target_col,
                }
            )
    manifest = {
        "dataset": "fred_md",
        "loader": "mf.data.load_fred_md",
        "vintage": vintage,
        "data_through": raw.metadata.get("data_through"),
        "raw_panel": {
            "rows": int(raw.panel.shape[0]),
            "columns": int(raw.panel.shape[1]),
            "start": str(raw.panel.index.min().date()),
            "end": str(raw.panel.index.max().date()),
        },
        "transformed_panel": {
            "path": str(panel_path),
            "rows": int(panel.shape[0]),
            "columns": int(panel.shape[1]),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "transform": "official",
            "outliers": "none",
            "impute": "zero",
            "standardize": "none",
        },
        "targets": {
            key: {
                "column": spec["column"],
                "label": spec["label"],
                "target_transform": spec["target_transform"],
                "tcode": int(transform_codes.get(spec["column"], -1)),
            }
            for key, spec in TARGETS.items()
        },
        "artifact": {
            "source_url": artifact.get("source_url"),
            "local_path": artifact.get("local_path"),
            "downloaded_at": artifact.get("downloaded_at"),
            "file_sha256": artifact.get("file_sha256"),
            "file_size_bytes": artifact.get("file_size_bytes"),
            "cache_hit": artifact.get("cache_hit"),
        },
        "paper_window": {
            "oos_actual_start": "1993-03-01",
            "oos_actual_end": SAMPLE_END,
            "rolling_window": T_ADJ,
            "horizon_start_dates": HORIZON_START_DATES,
        },
        "cell_slices": cells,
    }
    manifest_path = data_dir / "fred_md_load_manifest.json"
    _atomic_write_json(manifest, manifest_path)
    return {
        "panel_path": str(panel_path),
        "target_path": str(target_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _read_panel(panel_path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, parse_dates=["date"]).set_index("date").sort_index()
    panel.index.name = "date"
    return panel.astype(float)


def _series_for_cell(panel: pd.DataFrame, target_key: str, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    target_col = TARGETS[target_key]["column"]
    if target_col not in panel.columns:
        raise ValueError(f"transformed panel does not contain {target_col}")
    start = pd.Timestamp(HORIZON_START_DATES[horizon])
    end = pd.Timestamp(SAMPLE_END)
    x_loop = panel.drop(columns=[target_col]).loc[start:end].copy()
    y_loop = panel[target_col].loc[start:end].copy()
    expected = T_ADJ + horizon + 360
    if x_loop.shape[0] != expected or y_loop.shape[0] != expected:
        raise ValueError(
            f"{target_key} h={horizon} FRED-MD slice expected {expected} rows, "
            f"got predictors={x_loop.shape[0]} target={y_loop.shape[0]}"
        )
    return x_loop.astype(float), y_loop.astype(float)


def _cell_file(result_store: Path, target_key: str, horizon: int, method: str) -> Path:
    return result_store / "cells" / f"{target_key}_h{horizon}_{method}.csv"


def _finite_zscore(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    mean = np.nanmean(out, axis=0, keepdims=True)
    sd = np.nanstd(out, axis=0, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (out - mean) / sd
    return np.where(np.isfinite(z), z, 0.0)


def _origin_block(
    x_loop: pd.DataFrame,
    y_loop: pd.Series,
    origin_i: int,
    horizon: int,
) -> dict[str, Any]:
    x_values = x_loop.to_numpy(dtype=float)
    y_values = y_loop.to_numpy(dtype=float)
    x_origin = _finite_zscore(x_values[origin_i : origin_i + T_ADJ, :])
    y_origin = y_values[origin_i : origin_i + T_ADJ]
    y_future = y_values[origin_i + horizon : origin_i + T_ADJ + horizon]
    n_train = T_ADJ - horizon
    actual_pos = origin_i + T_ADJ + horizon - 1
    return {
        "x_origin": x_origin,
        "y_origin": y_origin,
        "y_future": y_future,
        "n_train": n_train,
        "x_train": x_origin[:n_train],
        "y_train": y_future[:n_train],
        "control_train": y_origin[:n_train],
        "x_test": x_origin[[T_ADJ - 1], :],
        "control_test": np.asarray([y_origin[T_ADJ - 1]], dtype=float),
        "actual": float(y_values[actual_pos]),
        "origin_date": x_loop.index[origin_i + T_ADJ - 1],
        "actual_date": y_loop.index[actual_pos],
    }


def _training_frame(block: dict[str, Any], predictors: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_fit = pd.DataFrame(block["x_train"], columns=predictors)
    x_fit[CONTROL_COL] = block["control_train"]
    x_test = pd.DataFrame(block["x_test"], columns=predictors)
    x_test[CONTROL_COL] = block["control_test"]
    y_fit = pd.Series(block["y_train"], index=x_fit.index)
    return x_fit, y_fit, x_test


def _control_fit(control: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([control, np.ones(control.shape[0], dtype=float)])
    coef = np.linalg.pinv(design) @ y
    residual = y - design @ coef
    return coef, residual


def _control_predict(control: float, coef: np.ndarray) -> float:
    return float(np.asarray([control, 1.0], dtype=float) @ coef)


def _pcr_curve_for_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    control_train: np.ndarray,
    x_out: np.ndarray,
    control_out: float,
) -> np.ndarray:
    control_coef, residual = _control_fit(control_train, y_train)
    try:
        u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    usable = min(len(K_GRID), s.size)
    preds = np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    if usable <= 0:
        return preds
    factors = x_train @ vt[:usable, :].T
    coefs = np.linalg.pinv(factors) @ residual
    factor_out = x_out @ vt[:usable, :].T
    preds[:usable] = preds[:usable] + np.cumsum(coefs * factor_out)
    if usable < len(K_GRID):
        preds[usable:] = preds[usable - 1]
    return preds


def _marginal_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)
    y_centered = y - float(np.nanmean(y))
    denom = np.nansum(x_centered * x_centered, axis=0)
    numer = np.nansum(x_centered * y_centered[:, None], axis=0)
    out = np.zeros(x.shape[1], dtype=float)
    np.divide(numer, denom, out=out, where=denom > 1e-12)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _absolute_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)
    y_centered = y - float(np.nanmean(y))
    x_scale = np.sqrt(np.nansum(x_centered * x_centered, axis=0))
    y_scale = float(np.sqrt(np.nansum(y_centered * y_centered)))
    denom = x_scale * y_scale
    numer = np.nansum(x_centered * y_centered[:, None], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(numer / denom)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _selected_indices(scores: np.ndarray, n_selected: int) -> np.ndarray:
    order = np.argsort(-scores)
    return np.asarray(order[: min(int(n_selected), scores.size)], dtype=int)


def _extract_supervised_arrays(
    x: np.ndarray,
    residual: np.ndarray,
    *,
    n_components: int,
    n_selected: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = x.shape
    n_out = min(max(1, int(n_components)), n_features, n_samples)
    work_x = np.asarray(x, dtype=float).copy()
    work_y = np.asarray(residual, dtype=float).copy()
    loadings = np.zeros((n_out, n_features), dtype=float)
    factor_coefs = np.zeros(n_out, dtype=float)
    active = 0
    for component in range(n_out):
        selected = _selected_indices(_absolute_correlations(work_x, work_y), n_selected)
        selected_x = work_x[:, selected]
        try:
            _u, _s, vt = np.linalg.svd(selected_x, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        loading_selected = np.asarray(vt[0], dtype=float)
        loading = np.zeros(n_features, dtype=float)
        loading[selected] = loading_selected
        factor = work_x @ loading
        denom = float(factor @ factor)
        if denom <= 1e-12:
            break
        alpha = float(work_y @ factor / denom)
        lambdas = work_x.T @ factor / denom
        loadings[component, :] = loading
        factor_coefs[component] = alpha
        active += 1
        work_y = work_y - alpha * factor
        work_x = work_x - np.outer(factor, lambdas)
    if active == 0:
        return np.zeros((0, n_features), dtype=float), np.zeros(0, dtype=float)
    return loadings[:active], factor_coefs[:active]


def _supervised_curve_for_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    control_train: np.ndarray,
    x_out: np.ndarray,
    control_out: float,
    *,
    n_selected: int,
    slope_scale: bool,
) -> np.ndarray:
    control_coef, residual = _control_fit(control_train, y_train)
    factor_train = np.asarray(x_train, dtype=float)
    factor_out = np.asarray(x_out, dtype=float)
    if slope_scale:
        slopes = _marginal_slopes(factor_train, y_train)
        factor_train = factor_train * slopes
        factor_out = factor_out * slopes
    loadings, coefs = _extract_supervised_arrays(
        factor_train,
        residual,
        n_components=max(K_GRID),
        n_selected=n_selected,
    )
    preds = np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    active = int(loadings.shape[0])
    if active == 0:
        return preds
    factor_prefix = factor_out @ loadings.T
    cumulative = np.cumsum(factor_prefix * coefs)
    for k in K_GRID:
        active_k = min(k, active)
        preds[k - 1] = preds[k - 1] + float(cumulative[active_k - 1])
    return preds


def _tune_supervised(block: dict[str, Any], method: str) -> tuple[dict[str, int], np.ndarray]:
    n_train = int(block["n_train"])
    boundaries = _cheap_fold_boundaries(n_train)
    scores = np.zeros((len(QN_GRID), len(K_GRID), len(boundaries) - 1), dtype=float)
    x_all = np.asarray(block["x_origin"], dtype=float)
    y_all = np.asarray(block["y_future"], dtype=float)
    control_all = np.asarray(block["y_origin"], dtype=float)
    slope_scale = method == "SsPCA"
    for fold_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        actual = y_all[start:end]
        preds = np.zeros((len(QN_GRID), len(K_GRID), end - start), dtype=float)
        for col, val_pos in enumerate(range(start, end)):
            x_train = x_all[:val_pos, :]
            y_train = y_all[:val_pos]
            control_train = control_all[:val_pos]
            x_out = x_all[val_pos, :]
            control_out = float(control_all[val_pos])
            for q_idx, qn in enumerate(QN_GRID):
                preds[q_idx, :, col] = _supervised_curve_for_validation(
                    x_train,
                    y_train,
                    control_train,
                    x_out,
                    control_out,
                    n_selected=qn,
                    slope_scale=slope_scale,
                )
        scores[:, :, fold_idx] = np.mean((actual[None, None, :] - preds) ** 2, axis=2)
    mean_scores = scores.mean(axis=2)
    flat_idx = int(np.argmin(mean_scores))
    q_idx, k_idx = np.unravel_index(flat_idx, mean_scores.shape)
    return {
        "n_components": int(K_GRID[k_idx]),
        "n_selected": int(QN_GRID[q_idx]),
    }, scores


def _scaled_pca_curve_for_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    control_train: np.ndarray,
    x_out: np.ndarray,
    control_out: float,
) -> np.ndarray:
    control_coef, residual = _control_fit(control_train, y_train)
    slopes = _marginal_slopes(x_train, y_train)
    scaled_train = x_train * slopes
    try:
        u, _, _ = np.linalg.svd(scaled_train, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    usable = min(len(K_GRID), u.shape[1])
    preds = np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    if usable <= 0:
        return preds
    factors = u[:, :usable] * np.sqrt(float(x_train.shape[0]))
    loadings = scaled_train.T @ factors / float(x_train.shape[0])
    projection = loadings @ np.linalg.pinv(loadings.T @ loadings)
    factor_out = (x_out * slopes) @ projection[:, :usable]
    coefs = np.linalg.pinv(factors) @ residual
    preds[:usable] = preds[:usable] + np.cumsum(coefs * factor_out)
    if usable < len(K_GRID):
        preds[usable:] = preds[usable - 1]
    return preds


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


def _pls_curve_for_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    control_train: np.ndarray,
    x_out: np.ndarray,
    control_out: float,
) -> np.ndarray:
    control_coef, residual = _control_fit(control_train, y_train)
    usable = min(len(K_GRID), x_train.shape[0], x_train.shape[1])
    weights = _pls_x_weights_one_target(x_train, residual, max_components=usable)
    usable = min(usable, weights.shape[1])
    preds = np.full(len(K_GRID), _control_predict(control_out, control_coef), dtype=float)
    if usable <= 0:
        return preds
    for k in K_GRID:
        kk = min(k, usable)
        factors = x_train @ weights[:, :kk]
        alpha = np.linalg.pinv(factors) @ residual
        preds[k - 1] = preds[k - 1] + float((x_out @ weights[:, :kk]) @ alpha)
    return preds


def _cheap_fold_boundaries(n_train: int) -> tuple[int, int, int, int]:
    return (80, 130, 190, n_train)


def _tune_cheap(block: dict[str, Any], method: str) -> tuple[int, np.ndarray]:
    n_train = int(block["n_train"])
    boundaries = _cheap_fold_boundaries(n_train)
    fold_mse = np.zeros((len(K_GRID), len(boundaries) - 1), dtype=float)
    x_all = np.asarray(block["x_origin"], dtype=float)
    y_all = np.asarray(block["y_future"], dtype=float)
    control_all = np.asarray(block["y_origin"], dtype=float)
    for fold_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        preds = np.zeros((len(K_GRID), end - start), dtype=float)
        actual = y_all[start:end]
        for col, val_pos in enumerate(range(start, end)):
            x_train = x_all[:val_pos, :]
            y_train = y_all[:val_pos]
            control_train = control_all[:val_pos]
            x_out = x_all[val_pos, :]
            control_out = float(control_all[val_pos])
            if method == "PCA":
                curve = _pcr_curve_for_validation(x_train, y_train, control_train, x_out, control_out)
            elif method == "sPCA":
                curve = _scaled_pca_curve_for_validation(x_train, y_train, control_train, x_out, control_out)
            elif method == "PLS":
                curve = _pls_curve_for_validation(x_train, y_train, control_train, x_out, control_out)
            else:
                raise ValueError(f"cheap tuning called with unsupported method {method}")
            preds[:, col] = curve
        fold_mse[:, fold_idx] = np.mean((actual[None, :] - preds) ** 2, axis=1)
    return int(np.argmin(np.mean(fold_mse, axis=1)) + 1), fold_mse


def _forecast_cheap(block: dict[str, Any], predictors: list[str], method: str, k: int) -> float:
    x_fit, y_fit, x_test = _training_frame(block, predictors)
    if method == "PCA":
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
    elif method == "sPCA":
        fit = scaled_pca(
            x_fit,
            y_fit,
            n_components=k,
            scale=False,
            control_columns=(CONTROL_COL,),
            include_constant=True,
            drop_control_columns=True,
            quadratic_factors=False,
        )
    elif method == "PLS":
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
    else:
        raise ValueError(f"unknown cheap method {method}")
    return float(fit.estimator.predict(x_test)[0])


def _supervised_search_spec(n_train: int) -> Any:
    return grid(
        {"n_selected": QN_GRID, "n_components": K_GRID},
        validation_splitter=explicit_folds(_cheap_fold_boundaries(n_train), within_fold="expanding"),
        score_aggregation="mean_fold",
    )


def _select_supervised(block: dict[str, Any], predictors: list[str], method: str) -> tuple[dict[str, int], pd.DataFrame]:
    params, scores = _tune_supervised(block, method)
    trials = pd.DataFrame(
        {
            "n_selected": np.repeat(QN_GRID, len(K_GRID)),
            "n_components": list(K_GRID) * len(QN_GRID),
            "score": scores.mean(axis=2).reshape(-1),
        }
    )
    return params, trials


def _forecast_supervised(block: dict[str, Any], predictors: list[str], method: str, params: dict[str, int]) -> float:
    x_fit, y_fit, x_test = _training_frame(block, predictors)
    fit_func = supervised_scaled_pca if method == "SsPCA" else supervised_pca
    fit = fit_func(
        x_fit,
        y_fit,
        n_components=params["n_components"],
        n_selected=params["n_selected"],
        **COMMON_SUPERVISED_PARAMS,
    )
    return float(fit.estimator.predict(x_test)[0])


def _ar_one(y_loop: pd.Series, origin_i: int, horizon: int) -> dict[str, Any]:
    y_train = y_loop.iloc[origin_i : origin_i + T_ADJ].astype(float)
    actual_pos = origin_i + T_ADJ + horizon - 1
    fit = ar_bic(
        y_train,
        min_lag=1,
        max_lag=12,
        criterion="bic",
        include_constant=True,
        ic_parameter_count="lag_square",
        estimator="matlab_ar",
        forecast_mode="coefficient_power",
        horizon=horizon,
    )
    prediction = float(fit.estimator.predict(pd.DataFrame({"__origin__": [0.0]}))[0])
    actual = float(y_loop.iloc[actual_pos])
    return {
        "origin": int(origin_i + 1),
        "origin_date": y_loop.index[origin_i + T_ADJ - 1],
        "actual_date": y_loop.index[actual_pos],
        "actual": actual,
        "prediction": prediction,
        "error2": float((actual - prediction) ** 2),
        "selected_lag": int(fit.metadata["selected_lag"]),
    }


def _method_record(
    x_loop: pd.DataFrame,
    y_loop: pd.Series,
    predictors: list[str],
    target_key: str,
    horizon: int,
    method: str,
    origin_i: int,
) -> dict[str, Any]:
    block = _origin_block(x_loop, y_loop, origin_i, horizon)
    if method in {"PCA", "sPCA", "PLS"}:
        k, _curve = _tune_cheap(block, method)
        prediction = _forecast_cheap(block, predictors, method, k)
        params = {"n_components": k}
    elif method in SUPERVISED_METHODS:
        params, _trials = _select_supervised(block, predictors, method)
        prediction = _forecast_supervised(block, predictors, method, params)
    else:
        raise ValueError(f"unknown method {method}")
    actual = float(block["actual"])
    return {
        "target": target_key,
        "horizon": horizon,
        "method": method,
        "origin": int(origin_i + 1),
        "origin_date": block["origin_date"],
        "actual_date": block["actual_date"],
        "actual": actual,
        "prediction": prediction,
        "error2": float((actual - prediction) ** 2),
        "train_rows": int(block["n_train"]),
        **params,
    }


def _run_supervised_chunk(args: tuple[str, int, str, str, str, tuple[int, ...]]) -> dict[str, Any]:
    target_key, horizon, method, panel_path, result_store_raw, origin_indices = args
    start = time.perf_counter()
    panel = _read_panel(panel_path)
    x_loop, y_loop = _series_for_cell(panel, target_key, horizon)
    predictors = list(x_loop.columns)
    rows = [
        _method_record(x_loop, y_loop, predictors, target_key, horizon, method, origin_i)
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


def _run_cell(args: tuple[str, int, str, str, str]) -> dict[str, Any]:
    target_key, horizon, method, panel_path, result_store_raw = args
    result_store = Path(result_store_raw)
    out_path = _cell_file(result_store, target_key, horizon, method)
    start = time.perf_counter()
    panel = _read_panel(panel_path)
    x_loop, y_loop = _series_for_cell(panel, target_key, horizon)
    predictors = list(x_loop.columns)
    n_origins = y_loop.shape[0] - T_ADJ - horizon + 1
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        rows = existing.to_dict(orient="records")
        completed = {int(row["origin"]) for row in rows}
    for origin_i in range(n_origins):
        origin = origin_i + 1
        if origin in completed:
            continue
        if method == "AR_BIC":
            record = _ar_one(y_loop, origin_i, horizon)
            record.update({"target": target_key, "horizon": horizon, "method": method})
        else:
            record = _method_record(x_loop, y_loop, predictors, target_key, horizon, method, origin_i)
        rows.append(record)
        if origin % 25 == 0 or origin == n_origins:
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


def _origin_chunks(origin_indices: list[int], chunk_size: int) -> list[tuple[int, ...]]:
    return [
        tuple(origin_indices[start : start + chunk_size])
        for start in range(0, len(origin_indices), chunk_size)
    ]


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


def _resolve_horizons(value: str) -> tuple[int, ...]:
    if value == "all":
        return HORIZONS
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    unknown = sorted(set(horizons) - set(HORIZONS))
    if unknown:
        raise ValueError(f"unknown horizon(s): {unknown}; expected one of {HORIZONS}")
    if not horizons:
        raise ValueError("--horizons must be 'all' or a comma-separated horizon list")
    return horizons


def _resolve_methods(value: str) -> tuple[str, ...]:
    if value == "all":
        return METHODS
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(methods) - set(METHODS) - {"AR_BIC"})
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; expected one of {METHODS}")
    if not methods:
        raise ValueError("--methods must be 'all' or a comma-separated method list")
    return methods


def _build_score_outputs(
    result_store: Path,
    *,
    run_targets: tuple[str, ...],
    run_horizons: tuple[int, ...],
    run_methods: tuple[str, ...],
) -> dict[str, Any]:
    parity_rows: list[dict[str, Any]] = []
    forecast_frames: list[pd.DataFrame] = []
    for target_key in run_targets:
        for horizon in run_horizons:
            ar_path = _cell_file(result_store, target_key, horizon, "AR_BIC")
            ar = pd.read_csv(ar_path, parse_dates=["origin_date", "actual_date"])
            ar_mse = float(ar["error2"].mean())
            for method in run_methods:
                if method == "AR_BIC":
                    continue
                path = _cell_file(result_store, target_key, horizon, method)
                frame = pd.read_csv(path, parse_dates=["origin_date", "actual_date"])
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
                        "leakfree_load_fred_md_ratio": ratio,
                        "paper_table2_ratio": paper,
                        "delta": delta,
                        "abs_delta": abs(delta),
                        "denominator": "AR_BIC",
                        "ar_mse": ar_mse,
                    }
                )
    parity = pd.DataFrame(parity_rows)
    parity_path = result_store / "leakfree_table2_parity.csv"
    _atomic_write_csv(parity, parity_path)
    forecasts_path = result_store / "leakfree_table2_forecasts.csv"
    if forecast_frames:
        _atomic_write_csv(pd.concat(forecast_frames, ignore_index=True), forecasts_path)
    return {
        "parity_path": str(parity_path),
        "forecasts_path": str(forecasts_path),
        "n_cells": int(len(parity)),
        "mean_abs_delta": float(parity["abs_delta"].mean()) if not parity.empty else None,
        "max_abs_delta": float(parity["abs_delta"].max()) if not parity.empty else None,
    }


def run_full_table2(
    result_store: Path,
    *,
    panel_path: str,
    n_jobs: int,
    run_targets: tuple[str, ...],
    run_horizons: tuple[int, ...],
    run_methods: tuple[str, ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    result_store.mkdir(parents=True, exist_ok=True)
    cell_summaries: list[dict[str, Any]] = []
    supervised_rows: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    supervised_pending: dict[tuple[str, int, str], int] = {}
    supervised_started: dict[tuple[str, int, str], float] = {}
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures: dict[cf.Future[dict[str, Any]], tuple[str, tuple[str, int, str] | None]] = {}
        for target_key in run_targets:
            for horizon in run_horizons:
                for method in ("AR_BIC",) + tuple(m for m in run_methods if m != "AR_BIC"):
                    if method not in SUPERVISED_METHODS:
                        task = (target_key, horizon, method, panel_path, str(result_store))
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
                    panel = _read_panel(panel_path)
                    _x_loop, y_loop = _series_for_cell(panel, target_key, horizon)
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
                        task = (target_key, horizon, method, panel_path, str(result_store), chunk)
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
    score_outputs = _build_score_outputs(
        result_store,
        run_targets=run_targets,
        run_horizons=run_horizons,
        run_methods=tuple(m for m in run_methods if m != "AR_BIC"),
    )
    elapsed = time.perf_counter() - start
    payload = {
        "scope": "Full Hounyo-Li Table 2 PC/no-threshold grid on load_fred_md leak-free data",
        "surface": "leak_free_load_fred_md",
        "targets": list(TARGETS),
        "run_targets": list(run_targets),
        "horizons": list(run_horizons),
        "methods": list(run_methods),
        "denominator": "AR_BIC",
        "paper_table2_column": "PC",
        "n_cells": int(score_outputs["n_cells"]),
        "runtime_seconds": elapsed,
        "n_jobs": n_jobs,
        "logical_cores": os.cpu_count(),
        "parallel_cell_timeout": "none",
        "data": {
            "loader": "mf.data.load_fred_md",
            "fred_md_vintage": FRED_MD_VINTAGE,
            "panel_path": panel_path,
        },
        "leak_free_contract": [
            "predictors are standardized with origin-available rows only",
            "future target values are never used in predictor scaling, target scaling, model selection, or final fit",
            "direct h-step factor models train only on rows whose realized target is available by the forecast origin",
            "AR_BIC uses target history through the origin date only",
        ],
        "kprefix_speedup": "enabled through ModelSpec.prefix_search for supervised_pca and supervised_scaled_pca cells",
        "outputs": {
            "parity": score_outputs["parity_path"],
            "forecasts": score_outputs["forecasts_path"],
            "cells": str(result_store / "cells"),
        },
        "score_summary": {
            "mean_abs_delta": score_outputs["mean_abs_delta"],
            "max_abs_delta": score_outputs["max_abs_delta"],
        },
        "cell_summaries": sorted(
            cell_summaries,
            key=lambda row: (row["target"], row["horizon"], row["method"]),
        ),
    }
    _atomic_write_json(payload, result_store / "leakfree_table2_report.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-store", default="runs/hl2026_table2_leakfree_load_fred_md")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--horizons", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--vintage", default=FRED_MD_VINTAGE)
    parser.add_argument("--force-data-download", action="store_true")
    parser.add_argument("--only-data", action="store_true")
    parser.add_argument(
        "--parallel-cell-timeout",
        default="none",
        help="Accepted for command provenance; this runner does not impose a parent-side timeout.",
    )
    args = parser.parse_args()
    if args.parallel_cell_timeout != "none":
        raise ValueError("this replication run requires --parallel-cell-timeout none")
    if args.vintage != FRED_MD_VINTAGE:
        raise ValueError(f"this Table 2 run is pinned to vintage {FRED_MD_VINTAGE}")
    result_store = REPO_ROOT / args.result_store
    result_store.mkdir(parents=True, exist_ok=True)
    data = _prepare_data(result_store, vintage=args.vintage, force_download=args.force_data_download)
    output: dict[str, Any] = {
        "result_store": str(result_store),
        "parallel_cell_timeout": args.parallel_cell_timeout,
        "data": data,
    }
    if not args.only_data:
        output["table2"] = run_full_table2(
            result_store,
            panel_path=data["panel_path"],
            n_jobs=_resolve_n_jobs(args.n_jobs),
            run_targets=_resolve_targets(args.targets),
            run_horizons=_resolve_horizons(args.horizons),
            run_methods=_resolve_methods(args.methods),
        )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
