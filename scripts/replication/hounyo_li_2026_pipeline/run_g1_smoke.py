"""Run the Hounyo-Li (2026) G1 macro/inflation smoke gate.

Scope is intentionally limited to Table 2, Inflation, PC, h=1, no threshold.
By default this runs the paper factor arms PCA/pcr, sPCA/scaled_pca,
SPCA/supervised_pca, SsPCA/supervised_scaled_pca, and PLS with raw x-weight
scores versus AR_BIC/ar_bic. Use --arms cheap to run only the cheap arms.

Use ``--surface author_oracle`` for the labeled B2 diagnostic that reproduces
the author's look-ahead standardization surface for cheap methods only. That
path is intentionally runner-local and does not add a leaky package feature.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="The behavior of array concatenation with empty entries is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated.*",
    category=FutureWarning,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import macroforecast as mf  # noqa: E402
from macroforecast.model_selection import SearchSpec  # noqa: E402
from macroforecast.models.linear import pcr, pls  # noqa: E402
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline  # noqa: E402

from scripts.replication.hounyo_li_2026_pipeline.build_data import (  # noqa: E402
    MERGED_CSV,
    build_data,
)


TARGET = "CPIAUCSL"
PAPER_TABLE2_INFLATION_H1 = {
    "PCA": 0.970,
    "SPCA": 0.823,
    "sPCA": 0.768,
    "SsPCA": 0.738,
    "PLS": 0.861,
}
METHOD_ORDER = ("PCA", "SPCA", "sPCA", "SsPCA", "PLS")
DEFAULT_ARM_ORDER = ("AR_BIC", "PCA", "sPCA", "SPCA", "SsPCA", "PLS")
CHEAP_ARM_ORDER = ("RW", "AR_BIC", "PCA", "sPCA", "PLS")
ALL_ARM_ORDER = ("RW",) + DEFAULT_ARM_ORDER
SUPERVISED_ARMS = frozenset({"SPCA", "SsPCA"})
AUTHOR_ORACLE_ARM_ORDER = ("AR_BIC", "PCA", "sPCA", "PLS")
AUTHOR_ORACLE_FACTOR_ORDER = ("PCA", "sPCA", "PLS")


def _canonical_arm_name(raw: str) -> str:
    token = raw.strip()
    if not token:
        raise ValueError("empty arm name in --arms/--exclude")
    exact = {name: name for name in ALL_ARM_ORDER}
    if token in exact:
        return exact[token]
    key = token.lower().replace("-", "_")
    aliases = {
        "ar": "AR_BIC",
        "ar_bic": "AR_BIC",
        "arbic": "AR_BIC",
        "naive": "RW",
        "random_walk": "RW",
        "rw": "RW",
        "pca": "PCA",
        "pcr": "PCA",
        "scaled_pca": "sPCA",
        "supervised_pca": "SPCA",
        "supervised_scaled_pca": "SsPCA",
        "sspca": "SsPCA",
        "pls": "PLS",
    }
    if key not in aliases:
        allowed = ", ".join(ALL_ARM_ORDER)
        raise ValueError(f"unknown arm {raw!r}; allowed arms: {allowed}, plus aliases cheap/full/all")
    return aliases[key]


def _dedupe_ordered(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _parse_arm_filter(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not raw.strip():
        return None
    key = raw.strip().lower().replace("-", "_")
    if key in {"cheap", "cheap_only", "cheap_methods"}:
        return CHEAP_ARM_ORDER
    if key in {"full", "default"}:
        return DEFAULT_ARM_ORDER
    if key == "all":
        return ALL_ARM_ORDER
    names: list[str] = []
    for chunk in raw.split(","):
        names.append(_canonical_arm_name(chunk))
    return _dedupe_ordered(names)


def _selected_arm_names(
    arms: tuple[str, ...] | None,
    exclude: tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = list(DEFAULT_ARM_ORDER if arms is None else arms)
    excluded = set(exclude or ())
    selected = [name for name in selected if name not in excluded]
    if "AR_BIC" not in selected:
        raise ValueError("AR_BIC must be selected because the G1 smoke scores relative_mse against it")
    return tuple(selected)


def _load_author_h1_panel() -> pd.DataFrame:
    if not MERGED_CSV.exists():
        build_data()
    frame = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    frame = frame[(frame["date"] >= "1973-03-01") & (frame["date"] <= "2023-03-01")]
    frame = frame.set_index("date").sort_index()
    frame = frame.astype(float)
    if frame.shape[0] != 601:
        raise ValueError(f"h=1 author full-sample slice should have 601 rows, got {frame.shape[0]}")
    return frame


def _author_expanding_folds_or_closest(index: pd.Index) -> list[tuple[np.ndarray, np.ndarray]]:
    """Author three-fold expanding validation, clipped only for the first origin.

    The package's h-step target-availability guard can expose 239 selection rows
    at the first h=1 origin, while the MATLAB oracle tunes on a 240-row block.
    For every full 240-row origin this is exactly {80,130,190,240}; when one row
    short, keep the first two validation blocks intact and clip the last endpoint.
    """

    n_obs = len(index)
    boundaries = [80, 130, 190, min(240, n_obs)]
    if n_obs < 190:
        raise ValueError("author fold approximation requires at least 190 selection rows")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        for val_pos in range(start, end):
            splits.append(
                (
                    np.arange(val_pos, dtype=int),
                    np.asarray([val_pos], dtype=int),
                )
            )
    return splits


_author_expanding_folds_or_closest.__mf_digest__ = "hl2026_author_folds_80_130_190_240_clip_first_origin"


def _zero_nonfinite_after_standardize(
    panel: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out.loc[:, numeric] = out.loc[:, numeric].where(np.isfinite(out.loc[:, numeric]), 0.0)
    return out, dict(metadata or {})


_zero_nonfinite_after_standardize.__mf_digest__ = "hl2026_zero_nonfinite_after_standardize_v1"


def _search_grid(param_grid: dict[str, Any]) -> SearchSpec:
    return mf.model_selection.grid(
        param_grid,
        validation_splitter=_author_expanding_folds_or_closest,
        score_aggregation="mean_fold",
    )


def _mse(actual: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return float(np.mean((actual_arr - pred_arr) ** 2))


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


def _author_ar_oracle(panel: pd.DataFrame) -> pd.DataFrame:
    """Runner-local port of the author's AR_BIC inflation denominator."""

    y_loop = panel[TARGET].to_numpy(dtype=float)
    out: list[dict[str, Any]] = []
    for i in range(y_loop.size - 240):
        y_or = y_loop[i : i + 241]
        yt1 = np.diff(y_or)
        yt = yt1 - _matlab_movmean(yt1, 12)
        yt_standardized = (yt - yt.mean()) / yt.std(ddof=1)
        ytplush = yt_standardized[1:]
        wt = yt_standardized[: ytplush.size]
        lag = _author_lag_bic(ytplush)
        phi = _ar_fit_forward_backward_no_intercept(wt, lag)
        pred = float(phi @ wt[-lag:])
        actual = float(ytplush[-1])
        out.append(
            {
                "i": i + 1,
                "date": panel.index[i + 240],
                "lag": lag,
                "actual": actual,
                "prediction": pred,
                "error2": (actual - pred) ** 2,
            }
        )
    return pd.DataFrame(out)


def _author_surface_block(
    panel: pd.DataFrame,
    predictors: list[str],
    origin_i: int,
) -> dict[str, np.ndarray]:
    """Build the author's leaky 240-row X / 241-row target block."""

    x_loop = panel[predictors].to_numpy(dtype=float)
    y_loop = panel[TARGET].to_numpy(dtype=float)
    xt = _finite_zscore(x_loop[origin_i : origin_i + 240, :].T, axis=1)
    y_or = y_loop[origin_i : origin_i + 241]
    y_std = (y_or - y_or.mean()) / y_or.std(ddof=1)
    ytplush = y_std[1:]
    wt = np.vstack([y_std[:240], np.ones(240, dtype=float)])

    # inflation_linear.m:199-211 estimates slopes on the full author block,
    # including the realized OOS target embedded in ytplush[-1].
    denom = np.sum(xt * xt, axis=1)
    slopes = np.divide(
        xt @ ytplush,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=np.abs(denom) > 0.0,
    )
    scaled_xt = np.where(np.isfinite(xt * slopes[:, None]), xt * slopes[:, None], 0.0)
    return {"xt": xt, "scaled_xt": scaled_xt, "ytplush": ytplush, "wt": wt}


def _author_pcr_like_tune(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Author three-fold expanding K selection for PCA/sPCA-style factors."""

    max_k = 10
    fold_defs = ((80, 81, 130), (130, 131, 190), (190, 191, 240))
    fold_mse = np.zeros((max_k, len(fold_defs)), dtype=float)
    for fold_idx, (train_end, val_start, val_end) in enumerate(fold_defs):
        val_end = min(val_end, x_block.shape[1])
        preds = np.zeros((max_k, val_end - val_start + 1), dtype=float)
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
            usable = min(max_k, s.size)
            factors = s[:usable, None] * vt[:usable, :]
            denom = np.where(s[:usable] > 0.0, s[:usable] ** 2, np.inf)
            alpha = (y_res @ factors.T) / denom
            factor_out = u[:, :usable].T @ x_block[:, idx_out]
            preds[:usable, o] = np.cumsum(alpha * factor_out) + float(
                alpha_w @ wt[:, idx_out]
            )
            if usable < max_k:
                preds[usable:, o] = preds[usable - 1, o]
        fold_mse[:, fold_idx] = np.mean((actual[None, :] - preds) ** 2, axis=1)
    return int(np.argmin(np.mean(fold_mse, axis=1)) + 1), fold_mse


def _pls_x_weights_one_target(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_components: int,
) -> np.ndarray:
    """NIPALS raw x-weights matching the package PLS raw-weight projection."""

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


def _author_pls_tune(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Author three-fold expanding K selection for raw-weight PLS."""

    max_k = 10
    fold_defs = ((80, 81, 130), (130, 131, 190), (190, 191, 240))
    fold_mse = np.zeros((max_k, len(fold_defs)), dtype=float)
    for fold_idx, (train_end, val_start, val_end) in enumerate(fold_defs):
        val_end = min(val_end, x_block.shape[1])
        actual = ytplush[val_start - 1 : val_end]
        preds = np.zeros((max_k, val_end - val_start + 1), dtype=float)
        for o, idx_out_1based in enumerate(range(val_start, val_end + 1)):
            train_last_1based = train_end + o
            idx_out = idx_out_1based - 1
            x_train = x_block[:, :train_last_1based].T
            y_train = ytplush[:train_last_1based]
            wt_train = wt[:, :train_last_1based].T
            control_coef = np.linalg.pinv(wt_train) @ y_train
            y_res = y_train - wt_train @ control_coef
            usable = min(max_k, x_train.shape[0], x_train.shape[1])
            weights = _pls_x_weights_one_target(
                x_train,
                y_res,
                max_components=usable,
            )
            usable = min(usable, weights.shape[1])
            control_part = float(wt[:, idx_out] @ control_coef)
            if usable == 0:
                preds[:, o] = control_part
                continue
            x_out = x_block[:, idx_out]
            for k in range(1, max_k + 1):
                kk = min(k, usable)
                factors = x_train @ weights[:, :kk]
                alpha = np.linalg.pinv(factors) @ y_res
                factor_part = float((x_out @ weights[:, :kk]) @ alpha)
                preds[k - 1, o] = factor_part + control_part
        fold_mse[:, fold_idx] = np.mean((actual[None, :] - preds) ** 2, axis=1)
    return int(np.argmin(np.mean(fold_mse, axis=1)) + 1), fold_mse


def _forecast_with_pcr(
    x_block: np.ndarray,
    ytplush: np.ndarray,
    wt: np.ndarray,
    predictors: list[str],
    k: int,
) -> float:
    control_col = f"{TARGET}_lag0"
    x_fit = pd.DataFrame(x_block[:, :-1].T, columns=predictors)
    x_test = pd.DataFrame(x_block[:, -1:].T, columns=predictors)
    x_fit[control_col] = wt[0, :-1]
    x_test[control_col] = [wt[0, -1]]
    fit = pcr(
        x_fit,
        pd.Series(ytplush[:-1], index=x_fit.index),
        n_components=k,
        control_columns=(control_col,),
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
    control_col = f"{TARGET}_lag0"
    x_fit = pd.DataFrame(x_block[:, :-1].T, columns=predictors)
    x_test = pd.DataFrame(x_block[:, -1:].T, columns=predictors)
    x_fit[control_col] = wt[0, :-1]
    x_test[control_col] = [wt[0, -1]]
    fit = pls(
        x_fit,
        pd.Series(ytplush[:-1], index=x_fit.index),
        n_components=k,
        scale=False,
        control_columns=(control_col,),
        include_constant=True,
        drop_control_columns=True,
        quadratic_factors=False,
        score_projection="x_weights_raw",
    )
    return float(fit.estimator.predict(x_test)[0])


def _arms(
    control_col: str,
    features: Any,
    feature_policy: Any,
    *,
    selected: tuple[str, ...],
) -> list[Arm]:
    k_grid = tuple(range(1, 11))
    qn_grid = tuple(range(18, 109, 6))
    common = {
        "scale": False,
        "control_columns": (control_col,),
        "include_constant": True,
        "drop_control_columns": True,
        "quadratic_factors": False,
    }
    pcr_common = {
        "control_columns": (control_col,),
        "include_constant": True,
        "drop_control_columns": True,
        "standardize": True,
        "nan_policy": "zero_after_standardize",
        "quadratic_factors": False,
    }
    supervised_common = {
        **common,
        "preselect": "none",
        "preselect_stage": "raw_before_standardize",
        "t_threshold": 1.28,
        "elastic_net_alpha": 0.0002,
        "elastic_net_l1_ratio": 0.5,
        "random_state": 0,
    }
    all_arms = {
        "RW": Arm(
            "RW",
            model="naive",
            features=features,
            feature_policy=feature_policy,
        ),
        "AR_BIC": Arm(
            "AR_BIC",
            model="ar_bic",
            features=features,
            feature_policy=feature_policy,
            params={
                "min_lag": 1,
                "max_lag": 12,
                "criterion": "bic",
                "ic_parameter_count": "lag_square",
                "estimator": "matlab_ar",
                "forecast_mode": "coefficient_power",
                "include_constant": True,
                "horizon": 1,
            },
            is_benchmark=True,
        ),
        "PCA": Arm(
            "PCA",
            model="pcr",
            features=features,
            feature_policy=feature_policy,
            params=pcr_common,
            model_selection=_search_grid({"n_components": k_grid}),
        ),
        "sPCA": Arm(
            "sPCA",
            model="scaled_pca",
            features=features,
            feature_policy=feature_policy,
            params=common,
            model_selection=_search_grid({"n_components": k_grid}),
        ),
        "SPCA": Arm(
            "SPCA",
            model="supervised_pca",
            features=features,
            feature_policy=feature_policy,
            params=supervised_common,
            model_selection=_search_grid({"n_components": k_grid, "n_selected": qn_grid}),
        ),
        "SsPCA": Arm(
            "SsPCA",
            model="supervised_scaled_pca",
            features=features,
            feature_policy=feature_policy,
            params=supervised_common,
            model_selection=_search_grid({"n_components": k_grid, "n_selected": qn_grid}),
        ),
        "PLS": Arm(
            "PLS",
            model="pls",
            features=features,
            feature_policy=feature_policy,
            params={**common, "score_projection": "x_weights_raw"},
            model_selection=_search_grid({"n_components": k_grid}),
        ),
    }
    return [all_arms[name] for name in selected]


def _build_spec(
    result_store: Path,
    *,
    n_jobs: int | str,
    parallel_cell_timeout: float | None,
    arms: tuple[str, ...] | None,
    exclude: tuple[str, ...] | None,
) -> Any:
    frame = _load_author_h1_panel()
    predictors = [column for column in frame.columns if column != TARGET]
    bundle = mf.data.custom_dataset(
        frame,
        transform_codes={column: 1 for column in frame.columns},
        metadata={
            "dataset": "hounyo_li_2026_author_macro",
            "source_csv": str(MERGED_CSV.relative_to(REPO_ROOT)),
        },
    )
    features = mf.feature_engineering.feature_spec(
        target=TARGET,
        predictors=predictors,
        lags=(0,),
        target_lags=(0,),
        target_transform="level",
        target_mode="direct",
        drop_missing=True,
    )
    control_col = f"{TARGET}_lag0"
    preprocessing = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="zero",
        standardize="zscore",
        standardize_ddof=1,
        standardize_scope="origin_available_predictors",
        custom_steps=mf.preprocessing.custom_preprocess_step(
            "zero_nonfinite_after_standardize",
            _zero_nonfinite_after_standardize,
        ),
    )
    window = mf.window.from_cutoffs(
        test_start="1993-02-01",
        test_end="2023-02-01",
        mode="rolling",
        estimation_size=240,
        embargo=0,
        retrain_every=1,
        val_method="expanding",
        val_min_train_size=80,
        val_horizon=1,
        val_embargo=0,
        retune_every=1,
        retune_on_retrain=True,
        reuse_params=False,
        horizon=1,
        step=1,
    )
    feature_policy = mf.window.stage_policy("fit_window", update="on_retrain")
    selected = _selected_arm_names(arms, exclude)
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(TARGET, transform="level", policy="direct")],
        horizons=[1],
        window=window,
        arms=_arms(control_col, features, feature_policy, selected=selected),
        preprocessing=preprocessing,
        preprocessing_policy=mf.window.stage_policy("origin_available", update="on_retrain"),
        evaluation=EvalSpec(benchmark="AR_BIC", metrics=("relative_mse",)),
        result_store=result_store,
        n_jobs=n_jobs,
        parallel_cell_timeout=parallel_cell_timeout,
        save_models=False,
        seed=42,
        on_unsupported_direct="warn",
    )


def _rank(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda key: values[key])


def _parity_table(accuracy: pd.DataFrame, selected_arms: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    got = {
        str(row["contender"]): float(row["relative_mse"])
        for _, row in accuracy.iterrows()
        if str(row["contender"]) in PAPER_TABLE2_INFLATION_H1 or str(row["contender"]) == "AR_BIC"
    }
    paper_methods = tuple(method for method in METHOD_ORDER if method in selected_arms)
    for method in tuple(name for name in selected_arms if name == "AR_BIC" or name in paper_methods):
        ratio = got.get(method)
        paper = 1.0 if method == "AR_BIC" else PAPER_TABLE2_INFLATION_H1[method]
        delta = None if ratio is None else ratio - paper
        verdict = "MISSING"
        if ratio is not None:
            close = abs(delta) <= 0.03
            verdict = "BENCHMARK" if method == "AR_BIC" else ("PASS" if close else "FAIL")
        rows.append(
            {
                "method": method,
                "ratio": ratio,
                "paper": paper,
                "delta": delta,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def run_g1_smoke(
    result_store: Path,
    *,
    n_jobs: int | str = "auto",
    parallel_cell_timeout: float | None = 3600.0,
    arms: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    spec = _build_spec(
        result_store,
        n_jobs=n_jobs,
        parallel_cell_timeout=parallel_cell_timeout,
        arms=arms,
        exclude=exclude,
    )
    report = run_pipeline(spec)
    elapsed = time.perf_counter() - start
    result_store.mkdir(parents=True, exist_ok=True)
    selected_arms = tuple(arm.name for arm in spec.arms)
    accuracy = report.accuracy.copy()
    parity = _parity_table(accuracy, selected_arms)
    selected_paper = {name: PAPER_TABLE2_INFLATION_H1[name] for name in METHOD_ORDER if name in selected_arms}
    paper_rank = _rank(selected_paper) if selected_paper else []
    got = {
        str(row["contender"]): float(row["relative_mse"])
        for _, row in accuracy.iterrows()
        if str(row["contender"]) in selected_paper
    }
    got_rank = _rank(got) if len(got) == len(selected_paper) else []
    smoke_pass = bool(parity["verdict"].isin({"PASS", "BENCHMARK"}).all())
    accuracy.to_csv(result_store / "g1_smoke_accuracy_raw.csv", index=False)
    parity.to_csv(result_store / "g1_smoke_parity.csv", index=False)
    payload = {
        "scope": "G1 smoke only",
        "target": "inflation",
        "horizon": 1,
        "table2_panel": "Inflation / PC / h=1",
        "paper_rank": paper_rank,
        "observed_rank": got_rank,
        "selected_arms": list(selected_arms),
        "excluded_arms": list(exclude or ()),
        "supervised_excluded": sorted(SUPERVISED_ARMS.difference(selected_arms)),
        "smoke_pass": smoke_pass,
        "runtime_seconds": elapsed,
        "result_store": str(result_store),
        "n_jobs_requested": n_jobs,
        "n_jobs_resolved": report.provenance.get("spec_echo", {}).get("n_jobs"),
        "parallel_cell_timeout": parallel_cell_timeout,
        "result_store_provenance": report.provenance.get("result_store"),
        "failed_cells": list(report.failed_cells),
        "empty_cells": list(report.empty_cells),
        "leakage_audit": report.leakage_audit,
        "parity": parity.to_dict(orient="records"),
        "pipeline_provenance": report.provenance,
    }
    (result_store / "g1_smoke_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _selected_author_oracle_names(
    arms: tuple[str, ...] | None,
    exclude: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if arms is None:
        selected = list(AUTHOR_ORACLE_ARM_ORDER)
    else:
        selected = [name for name in arms if name != "RW"]
    excluded = set(exclude or ())
    selected = [name for name in selected if name not in excluded]
    unsupported = [name for name in selected if name not in AUTHOR_ORACLE_ARM_ORDER]
    if unsupported:
        raise ValueError(
            "--surface author_oracle is scoped to cheap Table 2 methods only; "
            f"unsupported arms: {unsupported}"
        )
    if "AR_BIC" not in selected:
        selected.insert(0, "AR_BIC")
    return _dedupe_ordered(selected)


def run_author_oracle_surface(
    result_store: Path,
    *,
    arms: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the labeled author-methodology cheap-method diagnostic."""

    start = time.perf_counter()
    result_store.mkdir(parents=True, exist_ok=True)
    selected_arms = _selected_author_oracle_names(arms, exclude)
    selected_factors = tuple(
        name for name in AUTHOR_ORACLE_FACTOR_ORDER if name in selected_arms
    )
    panel = _load_author_h1_panel()
    predictors = [column for column in panel.columns if column != TARGET]
    author_ar = _author_ar_oracle(panel)
    author_ar_mse = float(author_ar["error2"].mean())

    rows: list[dict[str, Any]] = []
    k_counts: dict[str, dict[str, int]] = {}
    fold_curve_sample: dict[str, Any] = {}
    total = panel.shape[0] - 240
    for origin_i in range(total):
        if origin_i == 0 or (origin_i + 1) % 50 == 0 or origin_i + 1 == total:
            print(
                f"author_oracle cheap origin {origin_i + 1}/{total}",
                file=sys.stderr,
                flush=True,
            )
        block = _author_surface_block(panel, predictors, origin_i)
        ytplush = block["ytplush"]
        wt = block["wt"]
        row: dict[str, Any] = {
            "i": origin_i + 1,
            "date": panel.index[origin_i + 240],
            "actual": float(ytplush[-1]),
            "ar_actual": float(author_ar.loc[origin_i, "actual"]),
            "ar_prediction": float(author_ar.loc[origin_i, "prediction"]),
            "ar_error2": float(author_ar.loc[origin_i, "error2"]),
        }
        if "PCA" in selected_factors:
            k, curves = _author_pcr_like_tune(block["xt"], ytplush, wt)
            row["PCA_k"] = k
            row["PCA_prediction"] = _forecast_with_pcr(
                block["xt"], ytplush, wt, predictors, k
            )
            row["PCA_error2"] = (row["actual"] - row["PCA_prediction"]) ** 2
            if origin_i == 0:
                fold_curve_sample["PCA"] = curves.tolist()
        if "sPCA" in selected_factors:
            k, curves = _author_pcr_like_tune(block["scaled_xt"], ytplush, wt)
            row["sPCA_k"] = k
            # The author's sPCA leak includes slope scaling before the final
            # split. After that scaling, scaledPCA_emp002 is PCR on scaleXs.
            row["sPCA_prediction"] = _forecast_with_pcr(
                block["scaled_xt"], ytplush, wt, predictors, k
            )
            row["sPCA_error2"] = (row["actual"] - row["sPCA_prediction"]) ** 2
            if origin_i == 0:
                fold_curve_sample["sPCA"] = curves.tolist()
        if "PLS" in selected_factors:
            k, curves = _author_pls_tune(block["xt"], ytplush, wt)
            row["PLS_k"] = k
            row["PLS_prediction"] = _forecast_with_pls(
                block["xt"], ytplush, wt, predictors, k
            )
            row["PLS_error2"] = (row["actual"] - row["PLS_prediction"]) ** 2
            if origin_i == 0:
                fold_curve_sample["PLS"] = curves.tolist()
        rows.append(row)

    forecast_frame = pd.DataFrame(rows)
    table_rows: list[dict[str, Any]] = []
    for method in selected_factors:
        ratio = float(forecast_frame[f"{method}_error2"].mean() / author_ar_mse)
        paper = PAPER_TABLE2_INFLATION_H1[method]
        delta = ratio - paper
        reproduced = abs(delta) <= 0.03
        k_series = forecast_frame[f"{method}_k"].astype(int)
        k_counts[method] = {
            str(int(key)): int(value)
            for key, value in k_series.value_counts().sort_index().items()
        }
        table_rows.append(
            {
                "method": method,
                "author_methodology_ratio": ratio,
                "paper_table2_ratio": paper,
                "delta": delta,
                "abs_delta": abs(delta),
                "reproduced_within_0p03": reproduced,
                "verdict": "REPRODUCED" if reproduced else "MISS",
            }
        )
    reproduction = pd.DataFrame(table_rows)
    forecast_path = result_store / "author_oracle_forecasts.csv"
    table_path = result_store / "author_oracle_reproduction.csv"
    report_path = result_store / "author_oracle_report.json"
    forecast_frame.to_csv(forecast_path, index=False)
    reproduction.to_csv(table_path, index=False)
    payload = {
        "scope": "B2 labeled author-methodology diagnostic only",
        "surface": "author_oracle",
        "diagnostic_only": True,
        "look_ahead_standardization": {
            "predictor_block": "240-row origin predictor block standardized once before CV/forecast",
            "target_block": "241-row target block standardized once including realized y_{T+h}",
            "author_lines": "inflation_linear.m:192-197",
            "package_feature": False,
        },
        "target": "inflation",
        "horizon": 1,
        "table2_panel": "Inflation / PC / h=1",
        "selected_arms": list(selected_arms),
        "supervised_excluded": sorted(SUPERVISED_ARMS),
        "author_ar_mse": author_ar_mse,
        "k_counts": k_counts,
        "fold_curve_sample_origin_1": fold_curve_sample,
        "reproduction": reproduction.to_dict(orient="records"),
        "artifacts_reused": [
            "qa/hounyo_li_b2_pca_decomp.py",
            "qa/hounyo_li_b2_author_pca_port.csv",
            "qa/hounyo_li_b2_kdiag_sample_curves.json",
        ],
        "outputs": {
            "forecasts": str(forecast_path),
            "table": str(table_path),
            "report": str(report_path),
        },
        "runtime_seconds": time.perf_counter() - start,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-store", default="runs/hl2026_store")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument(
        "--arms",
        default=None,
        help='Comma-separated arm names or preset "cheap" / "full" / "all".',
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated arm names to drop after applying --arms/default.",
    )
    parser.add_argument(
        "--parallel-cell-timeout",
        default="3600",
        help='Parent-side cell heartbeat timeout in seconds; use "none" to disable.',
    )
    parser.add_argument(
        "--surface",
        default="leak_free",
        choices=("leak_free", "author_oracle"),
        help=(
            "leak_free runs the honest package pipeline; author_oracle runs the "
            "labeled leaky author-methodology diagnostic for cheap methods only."
        ),
    )
    args = parser.parse_args()
    n_jobs: int | str = int(args.n_jobs) if str(args.n_jobs).isdigit() else args.n_jobs
    arms = _parse_arm_filter(args.arms)
    exclude = _parse_arm_filter(args.exclude)
    timeout_arg = str(args.parallel_cell_timeout).strip().lower()
    if timeout_arg in {"none", "null"}:
        parallel_cell_timeout = None
    else:
        parallel_cell_timeout = float(timeout_arg)
    if args.surface == "author_oracle":
        payload = run_author_oracle_surface(
            REPO_ROOT / args.result_store,
            arms=arms,
            exclude=exclude,
        )
    else:
        payload = run_g1_smoke(
            REPO_ROOT / args.result_store,
            n_jobs=n_jobs,
            parallel_cell_timeout=parallel_cell_timeout,
            arms=arms,
            exclude=exclude,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
