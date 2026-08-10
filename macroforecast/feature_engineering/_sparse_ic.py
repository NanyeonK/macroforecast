from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def select_sparse_ic_params(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search: Any,
    *,
    allowed_params: set[str],
    fixed_params: dict[str, Any] | None = None,
) -> Any:
    """Run opt-in sparse IC selection and return the SearchResult."""

    from macroforecast.model_selection import select_params

    spec = normalize_sparse_ic_search(search, allowed_params=allowed_params)
    return select_params(
        model_name,
        X,
        y,
        search=spec,
        fixed_params=dict(fixed_params or {}),
    )


def normalize_sparse_ic_search(
    search: Any,
    *,
    allowed_params: set[str],
) -> Any:
    from macroforecast.model_selection import SearchSpec

    if isinstance(search, SearchSpec):
        raw = search
    elif isinstance(search, Mapping):
        raw = _search_spec_from_mapping(search, allowed_params=allowed_params)
    else:
        raise TypeError("lambda_search must be a SearchSpec or mapping")

    method = str(raw.method).lower().replace("-", "_")
    if method == "informationcriterion":
        method = "information_criterion"
    if method not in {"information_criterion", "grid", "cv_path"}:
        raise ValueError(
            "lambda_search must use method='information_criterion', 'grid', or 'cv_path'"
        )
    param_grid = {key: _as_tuple(value) for key, value in raw.param_grid.items()}
    _validate_sparse_ic_grid(param_grid, allowed_params=allowed_params)
    metadata = dict(raw.metadata)
    if method != "information_criterion":
        metadata.setdefault("normalized_from_method", raw.method)
    return SearchSpec(
        method="information_criterion",
        param_grid=param_grid,
        criterion=str(raw.criterion or "aicc").lower(),
        random_state=raw.random_state,
        metadata=metadata,
    )


def sparse_ic_metadata(result: Any) -> dict[str, Any]:
    trials = result.trials
    n_successful = (
        int((trials["status"] == "ok").sum()) if "status" in trials else int(len(trials))
    )
    return {
        "method": result.method,
        "criterion": result.metric,
        "selected_params": dict(result.best_params),
        "best_score": float(result.best_score),
        "n_trials": int(len(trials)),
        "n_successful": n_successful,
    }


def _search_spec_from_mapping(
    search: Mapping[str, Any],
    *,
    allowed_params: set[str],
) -> Any:
    from macroforecast.model_selection import SearchSpec

    values = dict(search)
    method = values.pop("method", "information_criterion")
    criterion = values.pop("criterion", "aicc")
    random_state = values.pop("random_state", None)
    metadata = dict(values.pop("metadata", {}) or {})
    values.pop("n_iter", None)
    values.pop("population_size", None)
    values.pop("generations", None)
    values.pop("mutation_rate", None)
    values.pop("custom_search", None)
    values.pop("validation_splitter", None)
    values.pop("score_aggregation", None)
    param_distributions = values.pop("param_distributions", {})
    if param_distributions:
        raise ValueError("lambda_search requires param_grid, not distributions")
    param_grid = values.pop("param_grid", None)
    if param_grid is None:
        param_grid = {
            key: values.pop(key)
            for key in tuple(values)
            if key in allowed_params
        }
    if values:
        raise ValueError(f"unsupported lambda_search key(s): {sorted(values)}")
    return SearchSpec(
        method=str(method),
        param_grid=dict(param_grid),
        criterion=str(criterion),
        random_state=None if random_state is None else int(random_state),
        metadata=metadata,
    )


def _validate_sparse_ic_grid(
    param_grid: dict[str, tuple[Any, ...]],
    *,
    allowed_params: set[str],
) -> None:
    extra = set(param_grid) - set(allowed_params)
    if extra:
        raise ValueError(f"unsupported lambda_search parameter(s): {sorted(extra)}")
    if "alpha" not in param_grid:
        raise ValueError("lambda_search requires an alpha grid")
    if not param_grid["alpha"]:
        raise ValueError("lambda_search alpha grid cannot be empty")
    for alpha in param_grid["alpha"]:
        if float(alpha) <= 0.0 or not np.isfinite(float(alpha)):
            raise ValueError("lambda_search alpha values must be finite and > 0")
    if "l1_ratio" in param_grid:
        for l1_ratio in param_grid["l1_ratio"]:
            value = float(l1_ratio)
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError("lambda_search l1_ratio values must be in (0, 1]")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, range):
        return tuple(value)
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    return (value,)


__all__ = [
    "normalize_sparse_ic_search",
    "select_sparse_ic_params",
    "sparse_ic_metadata",
]
