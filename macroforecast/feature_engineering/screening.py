from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.feature_engineering._sparse_ic import (
    select_sparse_ic_params,
    sparse_ic_metadata,
)


PredictorScreenMethod = Literal["t_stat", "delta_r2", "lasso", "elastic_net"]


@dataclass(frozen=True)
class PredictorScreenResult:
    """Fitted predictor-screen result for runner-safe feature pipelines."""

    selected_columns: tuple[str, ...]
    candidate_columns: tuple[str, ...]
    controls: tuple[str, ...]
    scores: dict[str, float]
    method: str
    threshold: float
    top_k: int | None
    min_k: int | None
    n_fit_rows: int
    metadata: dict[str, Any]


def normalize_predictor_screen_method(value: str) -> PredictorScreenMethod:
    """Normalize predictor-screen method aliases."""

    aliases = {
        "t": "t_stat",
        "tstat": "t_stat",
        "t_stat": "t_stat",
        "t_statistics": "t_stat",
        "hard_tstat": "t_stat",
        "delta_r2": "delta_r2",
        "delta_r_squared": "delta_r2",
        "marginal_r2": "delta_r2",
        "incremental_r2": "delta_r2",
        "lasso": "lasso",
        "lasso_selection": "lasso",
        "elasticnet": "elastic_net",
        "elastic_net": "elastic_net",
        "enet": "elastic_net",
    }
    key = str(value).lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            "predictor screen method must be one of "
            f"{sorted(aliases)}; got {value!r}"
        )
    return aliases[key]  # type: ignore[return-value]


def fit_predictor_screen(
    source: pd.DataFrame,
    target: pd.Series,
    *,
    columns: Iterable[str] | None = None,
    method: str = "t_stat",
    threshold: float | None = None,
    top_k: int | None = None,
    min_k: int | None = None,
    controls: Iterable[str] | None = None,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    lambda_search: Any | None = None,
    max_iter: int = 20000,
    random_state: int | None = 0,
    min_train_size: int | None = None,
) -> PredictorScreenResult:
    """Fit a supervised predictor screen on one training window."""

    if source.empty:
        raise ValueError("predictor_screen requires at least one source column")
    method_value = normalize_predictor_screen_method(method)
    selected_input = _resolve_columns(source, columns)
    control_columns = _resolve_controls(source, controls)
    all_columns = tuple(
        str(column)
        for column in source.columns
        if str(column) in set(selected_input).union(control_columns)
    )
    control_set = set(control_columns)
    candidate_columns = tuple(column for column in all_columns if column not in control_set)
    if not candidate_columns and not control_columns:
        raise ValueError("predictor_screen requires at least one candidate or control column")
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive when provided")
    if min_k is not None and int(min_k) < 0:
        raise ValueError("min_k must be non-negative when provided")

    joined = pd.concat(
        [source.loc[:, all_columns].astype(float), target.rename("__target__").astype(float)],
        axis=1,
    ).dropna()
    min_rows = _screen_min_train_size(min_train_size, method_value, n_controls=len(control_columns))
    if len(joined) < min_rows:
        raise ValueError(
            f"predictor_screen requires at least {min_rows} target-aligned complete rows"
        )

    y = joined["__target__"].to_numpy(dtype=float)
    x = joined.loc[:, candidate_columns].to_numpy(dtype=float) if candidate_columns else np.empty((len(joined), 0))
    controls_matrix = (
        joined.loc[:, control_columns].to_numpy(dtype=float)
        if control_columns
        else np.empty((len(joined), 0))
    )
    resolved_threshold = _default_threshold(method_value) if threshold is None else float(threshold)
    scores_array, score_name, fit_metadata = _screen_scores(
        x,
        y,
        controls_matrix=controls_matrix,
        method=method_value,
        alpha=alpha,
        l1_ratio=l1_ratio,
        lambda_search=lambda_search,
        max_iter=max_iter,
        random_state=random_state,
    )
    scores = {
        column: float(score)
        for column, score in zip(candidate_columns, scores_array, strict=True)
    }
    selected_candidates = _select_screen_candidates(
        candidate_columns,
        scores_array,
        threshold=resolved_threshold,
        top_k=top_k,
        min_k=min_k,
    )
    selected_set = set(selected_candidates).union(control_columns)
    selected_columns = tuple(column for column in all_columns if column in selected_set)
    if not selected_columns:
        raise ValueError("predictor_screen selected no columns; set min_k or controls")
    return PredictorScreenResult(
        selected_columns=selected_columns,
        candidate_columns=candidate_columns,
        controls=control_columns,
        scores=scores,
        method=method_value,
        threshold=resolved_threshold,
        top_k=None if top_k is None else int(top_k),
        min_k=None if min_k is None else int(min_k),
        n_fit_rows=int(len(joined)),
        metadata={
            "score": score_name,
            "threshold": resolved_threshold,
            "top_k": None if top_k is None else int(top_k),
            "min_k": None if min_k is None else int(min_k),
            "controls": list(control_columns),
            **fit_metadata,
        },
    )


def marginal_t_stats(
    X: np.ndarray,
    y: np.ndarray,
    *,
    controls: np.ndarray | None = None,
) -> np.ndarray:
    """Return marginal t-statistics, optionally partialling out controls."""

    x_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    control_values = (
        np.empty((len(y_values), 0), dtype=float)
        if controls is None
        else np.asarray(controls, dtype=float)
    )
    n_rows = x_values.shape[0]
    out = np.zeros(x_values.shape[1], dtype=float)
    if n_rows <= control_values.shape[1] + 2:
        return out
    for idx in range(x_values.shape[1]):
        design = _regression_design(control_values, x_values[:, [idx]])
        coef = np.linalg.pinv(design) @ y_values
        resid = y_values - design @ coef
        dof = max(n_rows - design.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(max(cov[-1, -1], 0.0)))
        out[idx] = 0.0 if se <= 1e-12 else float(coef[-1] / se)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _screen_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    controls_matrix: np.ndarray,
    method: PredictorScreenMethod,
    alpha: float,
    l1_ratio: float,
    lambda_search: Any | None,
    max_iter: int,
    random_state: int | None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if x.shape[1] == 0:
        return np.empty(0, dtype=float), "none", {}
    if method == "t_stat":
        return (
            np.abs(marginal_t_stats(x, y, controls=controls_matrix)),
            "absolute_partial_t_stat",
            {},
        )
    if method == "delta_r2":
        return (
            _delta_r2_scores(x, y, controls_matrix=controls_matrix),
            "incremental_r_squared",
            {},
        )
    selected_alpha = float(alpha)
    selected_l1_ratio = float(l1_ratio)
    metadata: dict[str, Any] = {}
    x_resid, y_resid = _partial_residualize(x, y, controls_matrix)
    x_scaled, _center, _scale = _standardize_matrix(x_resid)
    if lambda_search is not None:
        model_name = "lasso" if method == "lasso" else "elastic_net"
        columns = [f"x{idx}" for idx in range(x_scaled.shape[1])]
        result = select_sparse_ic_params(
            model_name,
            pd.DataFrame(x_scaled, columns=columns),
            pd.Series(y_resid, name="target"),
            lambda_search,
            allowed_params={"alpha"} if method == "lasso" else {"alpha", "l1_ratio"},
            fixed_params={
                "max_iter": int(max_iter),
                "random_state": random_state,
                **({} if method == "lasso" else {"l1_ratio": selected_l1_ratio}),
            },
        )
        selected = dict(result.best_params)
        selected_alpha = float(selected["alpha"])
        if method != "lasso" and "l1_ratio" in selected:
            selected_l1_ratio = float(selected["l1_ratio"])
        metadata["lambda_selection"] = sparse_ic_metadata(result)
    if method == "lasso":
        from sklearn.linear_model import Lasso

        model = Lasso(
            alpha=selected_alpha,
            max_iter=int(max_iter),
            random_state=random_state,
        )
    else:
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(
            alpha=selected_alpha,
            l1_ratio=selected_l1_ratio,
            max_iter=int(max_iter),
            random_state=random_state,
        )
    model.fit(x_scaled, y_resid)
    scores = np.abs(np.asarray(model.coef_, dtype=float))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores, "absolute_standardized_sparse_coefficient", {
        "alpha": selected_alpha,
        "l1_ratio": None if method == "lasso" else selected_l1_ratio,
        "max_iter": int(max_iter),
        "random_state": random_state,
        **metadata,
    }


def _delta_r2_scores(
    x: np.ndarray,
    y: np.ndarray,
    *,
    controls_matrix: np.ndarray,
) -> np.ndarray:
    base_design = _regression_design(controls_matrix)
    base_r2 = _r_squared(base_design, y)
    out = np.zeros(x.shape[1], dtype=float)
    for idx in range(x.shape[1]):
        full_design = _regression_design(controls_matrix, x[:, [idx]])
        out[idx] = max(0.0, _r_squared(full_design, y) - base_r2)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _partial_residualize(
    x: np.ndarray,
    y: np.ndarray,
    controls_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = _regression_design(controls_matrix)
    projection = np.linalg.pinv(design) @ x
    x_resid = x - design @ projection
    y_resid = y - design @ (np.linalg.pinv(design) @ y)
    return x_resid, y_resid


def _regression_design(
    controls_matrix: np.ndarray,
    candidate: np.ndarray | None = None,
) -> np.ndarray:
    n_rows = (
        controls_matrix.shape[0]
        if controls_matrix.ndim == 2 and controls_matrix.shape[0] > 0
        else (candidate.shape[0] if candidate is not None else 0)
    )
    parts = [np.ones((n_rows, 1), dtype=float)]
    if controls_matrix.size:
        parts.append(np.asarray(controls_matrix, dtype=float))
    if candidate is not None and candidate.size:
        parts.append(np.asarray(candidate, dtype=float))
    return np.column_stack(parts)


def _r_squared(design: np.ndarray, y: np.ndarray) -> float:
    coef = np.linalg.pinv(design) @ y
    fitted = design @ coef
    resid = y - fitted
    centered = y - float(np.mean(y))
    tss = float(centered @ centered)
    if tss <= 1e-12:
        return 0.0
    return float(1.0 - (resid @ resid) / tss)


def _standardize_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return (values - center) / scale, center, scale


def _select_screen_candidates(
    columns: tuple[str, ...],
    scores: np.ndarray,
    *,
    threshold: float,
    top_k: int | None,
    min_k: int | None,
) -> tuple[str, ...]:
    if not columns:
        return ()
    order = sorted(range(len(columns)), key=lambda idx: (-float(scores[idx]), idx))
    selected = [idx for idx in order if float(scores[idx]) > threshold]
    if top_k is not None:
        selected = selected[: min(int(top_k), len(columns))]
    if min_k is not None and len(selected) < int(min_k):
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min(int(min_k), len(columns)):
                break
    selected_set = set(selected)
    return tuple(column for idx, column in enumerate(columns) if idx in selected_set)


def _default_threshold(method: PredictorScreenMethod) -> float:
    if method == "t_stat":
        return 1.28
    if method == "delta_r2":
        return 0.0
    return 1e-12


def _screen_min_train_size(
    value: int | None,
    method: PredictorScreenMethod,
    *,
    n_controls: int,
) -> int:
    minimum = max(3, n_controls + 3) if method in {"t_stat", "delta_r2"} else max(2, n_controls + 2)
    if value is None:
        return minimum
    out = int(value)
    if out < minimum:
        raise ValueError(f"min_train_size must be >= {minimum}")
    return out


def _resolve_columns(source: pd.DataFrame, columns: Iterable[str] | None) -> tuple[str, ...]:
    if columns is None:
        return tuple(str(column) for column in source.columns)
    selected = tuple(str(column) for column in columns)
    missing = [column for column in selected if column not in source.columns]
    if missing:
        raise ValueError(f"predictor_screen columns are not in the source: {missing}")
    if not selected:
        raise ValueError("predictor_screen columns must not be empty")
    return selected


def _resolve_controls(source: pd.DataFrame, controls: Iterable[str] | None) -> tuple[str, ...]:
    if controls is None:
        return ()
    selected = tuple(str(column) for column in controls)
    missing = [column for column in selected if column not in source.columns]
    if missing:
        raise ValueError(f"predictor_screen controls are not in the source: {missing}")
    return selected


__all__ = [
    "PredictorScreenMethod",
    "PredictorScreenResult",
    "fit_predictor_screen",
    "marginal_t_stats",
    "normalize_predictor_screen_method",
]
