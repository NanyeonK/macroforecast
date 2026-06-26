from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


AlbaMAMode = Literal["one_sided", "two_sided"]
InbagRule = Literal["single", "positive"]


@dataclass(frozen=True)
class AlbaMAResult:
    """Result returned by the AlbaMA adaptive moving-average feature builder."""

    smoothed: pd.Series
    weights: pd.DataFrame
    mode: str
    backend: str
    params: dict[str, Any]
    metadata: dict[str, Any]


def albama(
    y: Any,
    *,
    dates: Any | None = None,
    mode: AlbaMAMode | str = "one_sided",
    n_estimators: int = 500,
    min_samples_leaf: int = 6,
    sample_fraction: float = 0.6,
    random_state: int | None = 42,
    replace: bool = True,
    inbag_rule: InbagRule | str = "single",
    min_train_size: int = 2,
    name: str | None = None,
) -> AlbaMAResult:
    """Create Goulet Coulombe-Klieber AlbaMA features for one time series.

    AlbaMA is a learned feature transform, not a multivariate forecast model.
    It fits bagged CART trees with a deterministic time trend as the only
    regressor. The output is the tree-averaged adaptive moving average plus a
    date-by-date observation-weight matrix.

    R alignment:
    - Goulet Coulombe and Klieber's `AlbaMA/AMA_main.R` uses
      `ranger(MAIN ~ Time_Trend, keep.inbag = TRUE, terminalNodes)`.
    - This port uses explicit `DecisionTreeRegressor` bagging so the in-bag
      counts needed for terminal-node co-membership weights are stored directly
      instead of relying on private sklearn RandomForest attributes.
    - `mode="two_sided"` mirrors `Albama_center`; `mode="one_sided"` mirrors
      the recursive `Albama_right` loop.
    """

    series = _coerce_series(y, dates=dates, name=name)
    mode_value = _normalize_mode(mode)
    inbag_value = _normalize_inbag_rule(inbag_rule)
    params = _validate_params(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        sample_fraction=sample_fraction,
        min_train_size=min_train_size,
    )
    values = series.to_numpy(dtype=float)
    index = series.index
    finite = np.isfinite(values)
    positions: np.ndarray = np.arange(len(series), dtype=float).reshape(-1, 1)
    smoothed: np.ndarray = np.full(len(series), np.nan, dtype=float)
    weights: np.ndarray = np.zeros((len(series), len(series)), dtype=float)

    if mode_value == "two_sided":
        train_positions = np.flatnonzero(finite)
        if len(train_positions) >= params["min_train_size"]:
            fit = _fit_tree_average(
                positions=positions,
                values=values,
                train_positions=train_positions,
                target_positions=np.arange(len(series)),
                n_estimators=cast(int, params["n_estimators"]),
                min_samples_leaf=cast(int, params["min_samples_leaf"]),
                sample_fraction=params["sample_fraction"],
                random_state=random_state,
                replace=replace,
                inbag_rule=inbag_value,
            )
            smoothed[:] = fit.predictions
            weights[:, :] = fit.weights
        elif len(train_positions) == 1:
            pos = int(train_positions[0])
            smoothed[pos] = values[pos]
            weights[pos, pos] = 1.0
    else:
        all_positions = np.arange(len(series))
        for target_pos in all_positions:
            train_positions = np.flatnonzero(finite & (all_positions <= target_pos))
            if len(train_positions) >= params["min_train_size"]:
                fit = _fit_tree_average(
                    positions=positions,
                    values=values,
                    train_positions=train_positions,
                    target_positions=np.array([target_pos]),
                    n_estimators=cast(int, params["n_estimators"]),
                    min_samples_leaf=cast(int, params["min_samples_leaf"]),
                    sample_fraction=params["sample_fraction"],
                    random_state=random_state,
                    replace=replace,
                    inbag_rule=inbag_value,
                )
                smoothed[target_pos] = fit.predictions[0]
                weights[:, target_pos] = fit.weights[:, 0]
            elif len(train_positions) == 1:
                pos = int(train_positions[0])
                smoothed[target_pos] = values[pos]
                weights[pos, target_pos] = 1.0

    weight_frame = pd.DataFrame(weights, index=index, columns=index)
    smoothed_series = pd.Series(smoothed, index=index, name=f"{series.name}_albama")
    result_params = {
        **params,
        "random_state": random_state,
        "replace": bool(replace),
        "inbag_rule": inbag_value,
    }
    metadata = {
        "kind": "albama",
        "version": 1,
        "paper": "Goulet Coulombe and Klieber (2025), arXiv:2501.13222",
        "mode": mode_value,
        "backend": "manual_sklearn_decision_tree_bagging",
        "params": result_params,
        "source_series": str(series.name),
        "weight_extraction": "terminal_node_co_membership_with_inbag_filter",
        "r_reference": "AlbaMA/AMA_main.R ranger keep.inbag terminalNodes loop",
        "analysis": "Use macroforecast.feature_analysis.effective_window and recent_weight_share on weights.",
    }
    return AlbaMAResult(
        smoothed=smoothed_series,
        weights=weight_frame,
        mode=mode_value,
        backend="manual_sklearn_decision_tree_bagging",
        params=result_params,
        metadata=metadata,
    )


class AdaptiveMovingAverage:
    """Reusable AlbaMA adaptive moving-average feature builder."""

    def __init__(
        self,
        *,
        mode: AlbaMAMode | str = "one_sided",
        n_estimators: int = 500,
        min_samples_leaf: int = 6,
        sample_fraction: float = 0.6,
        random_state: int | None = 42,
        replace: bool = True,
        inbag_rule: InbagRule | str = "single",
        min_train_size: int = 2,
    ) -> None:
        self.mode = mode
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.replace = replace
        self.inbag_rule = inbag_rule
        self.min_train_size = min_train_size
        self.result_: AlbaMAResult | None = None

    def fit(
        self, y: Any, *, dates: Any | None = None, name: str | None = None
    ) -> "AdaptiveMovingAverage":
        self.result_ = albama(
            y,
            dates=dates,
            mode=self.mode,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            sample_fraction=self.sample_fraction,
            random_state=self.random_state,
            replace=self.replace,
            inbag_rule=self.inbag_rule,
            min_train_size=self.min_train_size,
            name=name,
        )
        return self

    def fit_transform(
        self, y: Any, *, dates: Any | None = None, name: str | None = None
    ) -> AlbaMAResult:
        return self.fit(y, dates=dates, name=name).result()

    def result(self) -> AlbaMAResult:
        if self.result_ is None:
            raise RuntimeError("AdaptiveMovingAverage has not been fit")
        return self.result_


class AlbaMA(AdaptiveMovingAverage):
    """Alias class for the Goulet Coulombe-Klieber AlbaMA smoother."""


@dataclass(frozen=True)
class _TreeAverageFit:
    predictions: np.ndarray
    weights: np.ndarray


def _fit_tree_average(
    *,
    positions: np.ndarray,
    values: np.ndarray,
    train_positions: np.ndarray,
    target_positions: np.ndarray,
    n_estimators: int,
    min_samples_leaf: int,
    sample_fraction: float,
    random_state: int | None,
    replace: bool,
    inbag_rule: str,
) -> _TreeAverageFit:
    x_train = positions[train_positions]
    y_train = values[train_positions]
    x_target = positions[target_positions]
    n_train = len(train_positions)
    rng = np.random.default_rng(random_state)
    sample_size = max(1, min(n_train, int(np.floor(sample_fraction * n_train))))
    prediction_sum: np.ndarray = np.zeros(len(target_positions), dtype=float)
    weight_matrix: np.ndarray = np.zeros((len(values), len(target_positions)), dtype=float)
    train_position_set = set(int(pos) for pos in train_positions)
    col: int | np.signedinteger[Any]

    for _tree_idx in range(n_estimators):
        sampled = rng.choice(n_train, size=sample_size, replace=replace)
        counts = np.bincount(sampled, minlength=n_train).astype(float)
        unique = np.flatnonzero(counts > 0)
        tree = DecisionTreeRegressor(
            min_samples_leaf=max(1, min(int(min_samples_leaf), len(unique))),
            random_state=None,
        )
        tree.fit(x_train[unique], y_train[unique], sample_weight=counts[unique])
        prediction_sum += tree.predict(x_target)

        train_leaves = tree.apply(x_train)
        target_leaves = tree.apply(x_target)
        if inbag_rule == "single":
            inbag = counts == 1
        else:
            inbag = counts > 0
        for col, leaf in enumerate(target_leaves):
            used = (train_leaves == leaf) & inbag
            if not np.any(used):
                continue
            original_rows = train_positions[used]
            weight_matrix[original_rows, col] += 1.0 / float(len(original_rows))

    predictions = prediction_sum / float(n_estimators)
    normalizer = weight_matrix.sum(axis=0)
    valid = normalizer > 0.0
    weight_matrix[:, valid] = weight_matrix[:, valid] / normalizer[valid]
    empty = ~valid
    if np.any(empty):
        for col in np.flatnonzero(empty):
            target = int(target_positions[col])
            if target in train_position_set:
                weight_matrix[target, col] = 1.0
            elif len(train_positions):
                weight_matrix[int(train_positions[-1]), col] = 1.0
    return _TreeAverageFit(predictions=predictions, weights=weight_matrix)


def _coerce_series(y: Any, *, dates: Any | None, name: str | None) -> pd.Series:
    if isinstance(y, pd.Series):
        series = y.astype(float).copy()
        if dates is not None:
            series.index = pd.Index(dates)
    else:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional")
        index = pd.Index(dates) if dates is not None else pd.RangeIndex(len(values))
        series = pd.Series(values, index=index)
    if len(series.index) != len(series):
        raise ValueError("dates must have the same length as y")
    if name is not None:
        series.name = str(name)
    elif series.name is None:
        series.name = "series"
    return series


def _normalize_mode(mode: AlbaMAMode | str) -> str:
    value = str(mode).lower().replace("-", "_")
    aliases = {
        "one": "one_sided",
        "right": "one_sided",
        "two": "two_sided",
        "center": "two_sided",
    }
    value = aliases.get(value, value)
    if value not in {"one_sided", "two_sided"}:
        raise ValueError("mode must be 'one_sided' or 'two_sided'")
    return value


def _normalize_inbag_rule(rule: InbagRule | str) -> str:
    value = str(rule).lower()
    if value not in {"single", "positive"}:
        raise ValueError("inbag_rule must be 'single' or 'positive'")
    return value


def _validate_params(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    sample_fraction: float,
    min_train_size: int,
) -> dict[str, int | float]:
    n_tree = int(n_estimators)
    min_leaf = int(min_samples_leaf)
    min_train = int(min_train_size)
    fraction = float(sample_fraction)
    if n_tree <= 0:
        raise ValueError("n_estimators must be positive")
    if min_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if min_train <= 0:
        raise ValueError("min_train_size must be positive")
    return {
        "n_estimators": n_tree,
        "min_samples_leaf": min_leaf,
        "sample_fraction": fraction,
        "min_train_size": min_train,
    }


__all__ = ["AlbaMA", "AlbaMAResult", "AdaptiveMovingAverage", "albama"]
